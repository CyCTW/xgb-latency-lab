"""Exact threshold-rank encoding for selected high-use features."""
from collections import Counter, defaultdict
import struct

from llvmlite import ir


def plan_ranks(forest, counts, limit):
    """Rank by estimated path visits saved minus binary-search comparisons.

    This is only a selection heuristic. All model thresholds are retained,
    including thresholds never visited during calibration.
    """
    if not limit:
        return {}
    thresholds = defaultdict(set)
    visits = Counter()
    for ti, tree in enumerate(forest.trees):
        population = sum(counts[ti][0]) if tree.left[0] != -1 else 0
        for node, left in enumerate(tree.left):
            if left == -1:
                continue
            feature = tree.feature[node]
            thresholds[feature].add(tree.value[node])
            visits[feature] += sum(counts[ti][node]) / population
    savings = {f: visits[f] - len(ts).bit_length() for f, ts in thresholds.items()}
    chosen = sorted((f for f in thresholds if savings[f] > 0),
                    key=lambda f: (-savings[f], f))[:limit]
    return {f: tuple(sorted(thresholds[f])) for f in chosen}


def bucket_plan(values, prefix_bits=12):
    """Model-known IEEE ordering prefixes; all thresholds remain exact."""
    def key(value):
        bits = struct.unpack("<I", struct.pack("<f", 0.0 if value == 0 else value))[0]
        return bits ^ (0xffffffff if bits >> 31 else 0x80000000)
    prefixes = [key(value) >> (32-prefix_bits) for value in values]
    first, last = prefixes[0], prefixes[-1]
    counts = Counter(prefixes)
    starts, position = [], 0
    for prefix in range(first, last+1):
        starts.append(position)
        position += counts[prefix]
    return first, starts, max(counts.values()).bit_length()


def split_bucket_plan(values, prefix_bits=12):
    """Compact positive/negative prefix ranges into separate directories."""
    negative = [value for value in values if value < 0]
    positive = [value for value in values if value >= 0]
    plans = []
    for part, base in ((negative, 0), (positive, len(negative))):
        if part:
            first, starts, rounds = bucket_plan(part, prefix_bits)
            plans.append((first, [base+start for start in starts], rounds))
        else:
            plans.append((0, [base], 1))
    return plans


def emit_ranks(builder, features, thresholds, strategy="binary", bucket_bits=12):
    """Emit exact rank computation; NaN has the signed i32 sentinel -1.

    For finite/infinite non-NaN x, rank = number of model thresholds <= x.
    Thus x < threshold[i] is exactly rank <= i. The sentinel sorts below all
    ranks with a signed comparison and above all ranks with an unsigned one.
    """
    f32, i32 = ir.FloatType(), ir.IntType(32)
    ranks = {}
    for feature, values in thresholds.items():
        value = builder.load(builder.gep(features, [ir.Constant(i32, feature)]))
        if strategy == "simd":
            # Dense vector comparisons trade extra work for independent loads.
            # Ordered >= gives false for NaN and infinity padding is avoided.
            vector = ir.VectorType(f32, 4)
            lane_values = ir.Constant(vector, ir.Undefined)
            for lane in range(4):
                lane_values = builder.insert_element(lane_values, value, ir.Constant(i32, lane))
            total = ir.Constant(ir.VectorType(i32, 4), [0] * 4)
            for start in range(0, len(values) - len(values) % 4, 4):
                limits = ir.Constant(vector, [ir.Constant(f32, v) for v in values[start:start+4]])
                flags = builder.fcmp_ordered(">=", lane_values, limits)
                total = builder.add(total, builder.zext(flags, ir.VectorType(i32, 4)))
            position = ir.Constant(i32, 0)
            for lane in range(4):
                position = builder.add(position, builder.extract_element(total, ir.Constant(i32, lane)))
            for threshold in values[len(values) - len(values) % 4:]:
                position = builder.add(position, builder.zext(
                    builder.fcmp_ordered(">=", value, ir.Constant(f32, threshold)), i32))
            ranks[feature] = builder.select(builder.fcmp_unordered("uno", value, value),
                                            ir.Constant(i32, -1), position, name=f"rank_f{feature}")
            continue
        if strategy in {"bucket", "bucket_split"}:
            # Canonicalize both signed zeros: numerical comparisons treat them
            # equally. Ordered IEEE keys otherwise preserve float32 ordering.
            bits = builder.bitcast(value, i32)
            bits = builder.select(builder.fcmp_ordered("==", value, ir.Constant(f32, 0.0)),
                                  ir.Constant(i32, 0), bits)
            negative = builder.icmp_signed("<", bits, ir.Constant(i32, 0))
            mask = builder.select(negative, ir.Constant(i32, -1), ir.Constant(i32, 0x80000000))
            prefix = builder.lshr(builder.xor(bits, mask), ir.Constant(i32, 32-bucket_bits))
            if strategy == "bucket_split":
                neg_plan, pos_plan = split_bucket_plan(values, bucket_bits)
                first = builder.select(negative, ir.Constant(i32, neg_plan[0]), ir.Constant(i32, pos_plan[0]))
                length = builder.select(negative, ir.Constant(i32, len(neg_plan[1])), ir.Constant(i32, len(pos_plan[1])))
                base = builder.select(negative, ir.Constant(i32, 0), ir.Constant(i32, len(neg_plan[1])))
                starts = neg_plan[1] + pos_plan[1]
                rounds = max(neg_plan[2], pos_plan[2])
            else:
                first, starts, rounds = bucket_plan(values, bucket_bits)
                first, length, base = ir.Constant(i32, first), ir.Constant(i32, len(starts)), ir.Constant(i32, 0)
            # Signed clamping handles out-of-range prefixes, including an
            # entirely absent sign partition and the canonical zero bucket.
            offset = builder.sub(prefix, first)
            offset = builder.select(builder.icmp_signed("<", offset, ir.Constant(i32, 0)),
                                    ir.Constant(i32, 0), offset)
            offset = builder.select(builder.icmp_unsigned(">=", offset, length),
                                    builder.sub(length, ir.Constant(i32, 1)), offset)
            offset = builder.add(offset, base)
            entry_bits = 8 if max(starts) <= 255 else (16 if max(starts) <= 65535 else 32)
            start_type = ir.ArrayType(ir.IntType(entry_bits), len(starts))
            directory = ir.GlobalVariable(builder.function.module, start_type,
                                           name=f"rank_directory_f{feature}")
            directory.linkage = "internal"
            directory.global_constant = True
            directory.unnamed_addr = True
            directory.align = 4
            directory.initializer = ir.Constant(start_type, starts)
            position = builder.load(builder.gep(directory, [ir.Constant(i32, 0), offset], inbounds=True))
            if entry_bits < 32:
                position = builder.zext(position, i32)
            # Probes may cross into the next bucket; those thresholds are > x,
            # so they compare false. End padding is clamped for +infinity.
            padded = list(values) + [float("inf")] * ((1 << rounds)-1)
            array_type = ir.ArrayType(f32, len(padded))
            table = ir.GlobalVariable(builder.function.module, array_type, name=f"rank_thresholds_f{feature}")
            table.linkage = "internal"
            table.global_constant = True
            table.unnamed_addr = True
            table.align = 4
            table.initializer = ir.Constant(array_type, [ir.Constant(f32, v) for v in padded])
            for bit in reversed(range(rounds)):
                step = 1 << bit
                index = builder.add(position, ir.Constant(i32, step-1))
                threshold = builder.load(builder.gep(table, [ir.Constant(i32, 0), index], inbounds=True))
                right = builder.fcmp_ordered(">=", value, threshold)
                position = builder.add(position, builder.select(right, ir.Constant(i32, step), ir.Constant(i32, 0)))
            position = builder.select(builder.icmp_unsigned(">", position, ir.Constant(i32, len(values))),
                                      ir.Constant(i32, len(values)), position)
            ranks[feature] = builder.select(builder.fcmp_unordered("uno", value, value),
                                            ir.Constant(i32, -1), position, name=f"rank_f{feature}")
            continue
        rounds = len(values).bit_length()
        padded_size = (1 << rounds) - 1
        data = [ir.Constant(f32, x) for x in values]
        data.extend([ir.Constant(f32, float("inf"))] * (padded_size - len(values)))
        if strategy == "eytzinger":
            ordered = data
            data = [None] * padded_size
            def arrange(lo, hi, node):
                if lo == hi:
                    return
                mid = (lo + hi) // 2
                data[node - 1] = ordered[mid]
                arrange(lo, mid, node * 2)
                arrange(mid + 1, hi, node * 2 + 1)
            arrange(0, padded_size, 1)
        array_type = ir.ArrayType(f32, padded_size)
        table = ir.GlobalVariable(builder.function.module, array_type,
                                  name=f"rank_thresholds_f{feature}")
        table.linkage = "internal"
        table.global_constant = True
        table.unnamed_addr = True
        table.align = 4
        table.initializer = ir.Constant(array_type, data)
        position = ir.Constant(i32, 1 if strategy == "eytzinger" else 0)
        for bit in reversed(range(rounds)):
            step = 1 << bit
            # Previous steps bound this index to [0, padded_size - 1].
            index = builder.add(position, ir.Constant(i32, -1 if strategy == "eytzinger" else step - 1))
            threshold = builder.load(builder.gep(table, [ir.Constant(i32, 0), index],
                                                 inbounds=True))
            right = builder.fcmp_ordered(">=", value, threshold)
            if strategy == "eytzinger":
                position = builder.add(builder.shl(position, ir.Constant(i32, 1)), builder.zext(right, i32))
            else:
                position = builder.add(position, builder.select(right, ir.Constant(i32, step),
                                                                  ir.Constant(i32, 0)))
        if strategy == "eytzinger":
            position = builder.sub(position, ir.Constant(i32, 1 << rounds))
        if len(values) != padded_size:
            # +inf also passes padding entries; clamp to the true model count.
            position = builder.select(builder.icmp_unsigned(">", position, ir.Constant(i32, len(values))),
                                      ir.Constant(i32, len(values)), position)
        missing = builder.fcmp_unordered("uno", value, value)
        ranks[feature] = builder.select(missing, ir.Constant(i32, -1), position,
                                       name=f"rank_f{feature}")
    return ranks
