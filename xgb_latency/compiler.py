"""Lower a forest to native code with branch/select and load-placement controls."""

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from collections import Counter

import llvmlite
from llvmlite import binding as llvm, ir
import numpy as np

from .model import Forest
from .hybrid import emit_hybrid, plan_hybrid
from .interleaved import emit_interleaved
from .separated import emit_separated
from .rank import bucket_plan, emit_ranks, plan_ranks


def _cost_select_nodes(tree, counts, max_depth, branch_penalty):
    """Bottom-up heuristic in comparison-equivalent units, not measured cycles.

    Eager cost counts all split nodes. Branch cost includes conditional child
    costs and min(p_left, p_right) times a tunable penalty. Calibration chooses
    only representation; unobserved branches are retained for arbitrary input.
    """
    selected = set()

    def plan(node):
        if tree.left[node] == -1:
            return 0, 0.0
        ln, lc = plan(tree.left[node])
        rn, rc = plan(tree.right[node])
        nodes = 1 + ln + rn
        nl, nr = counts[node]
        population = nl + nr
        p = nl / population if population else .5
        branch_cost = 1 + p * lc + (1 - p) * rc + min(p, 1 - p) * branch_penalty
        if population and tree.height[node] <= max_depth and nodes < branch_cost:
            selected.add(node)
            return nodes, float(nodes)
        return nodes, branch_cost

    plan(0)
    return selected


def _emit_trees(fn, indexed_trees, initial, *, select_depth, preload, counts, select_policy,
                predicate_hoist_limit, leaf_table_bits, select_branch_penalty,
                rank_thresholds, rank_strategy, rank_bucket_bits, compact_leaf_depth, accumulation_batch,
                hybrid_plan, hybrid_fn, rank_buffer=None, accumulation_order="sequential"):
    f32, i32 = ir.FloatType(), ir.IntType(32)
    x = fn.args[0]
    trees_only = [tree for _, tree in indexed_trees]
    b = ir.IRBuilder(fn.append_basic_block("entry"))
    used = sorted({t.feature[n] for t in trees_only for n in range(len(t.left)) if t.left[n] != -1})
    loaded = {f: b.load(b.gep(x, [ir.Constant(i32, f)]), name=f"f{f}") for f in used} if preload else {}
    ranks = (emit_ranks(b, x, rank_thresholds, rank_strategy, rank_bucket_bits) if rank_buffer is None else
             {f: b.load(b.gep(rank_buffer, [ir.Constant(i32, i)]))
              for i, f in enumerate(rank_thresholds) if f in used})
    rank_indices = {f: {value: i for i, value in enumerate(values)}
                    for f, values in rank_thresholds.items()}

    def build_condition(f, threshold_value, default_left):
        if f in ranks:
            threshold = ir.Constant(i32, rank_indices[f][threshold_value])
            return (b.icmp_signed("<=", ranks[f], threshold) if default_left else
                    b.icmp_unsigned("<=", ranks[f], threshold))
        v = loaded[f] if preload else b.load(b.gep(x, [ir.Constant(i32, f)]))
        threshold = ir.Constant(f32, threshold_value)
        # XGBoost numerical split is strictly <. Unordered < folds NaN-default-left.
        return (b.fcmp_unordered("<", v, threshold) if default_left
                else b.fcmp_ordered("<", v, threshold))

    # Hoist a bounded number of frequently reused predicates. Scores reflect
    # path visits in calibration, not the number of static references alone.
    # Input loads are valid for every dense row; eager comparison preserves NaN.
    scores, occurrences = Counter(), Counter()
    if predicate_hoist_limit:
        for ti, tree in indexed_trees:
            if tree.left[0] == -1:
                continue
            population = sum(counts[ti][0])
            for node, left in enumerate(tree.left):
                if left == -1:
                    continue
                key = (tree.feature[node],tree.value[node],tree.default_left[node])
                occurrences[key] += 1
                scores[key] += sum(counts[ti][node]) / population
    keys = sorted((k for k in scores if scores[k] > 1.5 and occurrences[k] > 1),
                  key=lambda k: (-scores[k],k))[:predicate_hoist_limit]
    cached = {key: build_condition(*key) for key in keys}
    table_stats = {"count": 0, "bytes": 0, "eager_regions": 0, "compact_count": 0}

    def condition(tree,node):
        key = (tree.feature[node],tree.value[node],tree.default_left[node])
        return cached[key] if key in cached else build_condition(*key)

    def leaf_table(tree, root):
        # Specialize a subtree's complete Boolean truth table at compile time.
        # Correlated/impossible combinations are harmless: every runtime index
        # still maps to the leaf reached by exactly the original predicates.
        predicates = {}
        stack = [root]
        while stack:
            node = stack.pop()
            if tree.left[node] == -1:
                continue
            key = (tree.feature[node], tree.value[node], tree.default_left[node])
            if key not in predicates:
                predicates[key] = len(predicates)
                if len(predicates) > leaf_table_bits:
                    return None
            stack.extend([tree.right[node], tree.left[node]])
        if len(predicates) < 2:
            return None  # A single comparison already lowers well as select.
        values = []
        for mask in range(1 << len(predicates)):
            node = root
            while tree.left[node] != -1:
                key = (tree.feature[node], tree.value[node], tree.default_left[node])
                node = tree.left[node] if mask & (1 << predicates[key]) else tree.right[node]
            values.append(ir.Constant(f32, tree.value[node]))
        table_type = ir.ArrayType(f32, len(values))
        table = ir.GlobalVariable(fn.module, table_type,
                                  name=f"{fn.name}_leaf_table_{table_stats['count']}")
        table.linkage = "internal"
        table.global_constant = True
        table.unnamed_addr = True
        table.align = 4
        table.initializer = ir.Constant(table_type, values)
        index = ir.Constant(i32, 0)
        for key, bit in predicates.items():
            comparison = cached[key] if key in cached else build_condition(*key)
            flag = b.zext(comparison, i32)
            if bit:
                flag = b.shl(flag, ir.Constant(i32, bit))
            index = b.or_(index, flag)
        table_stats["count"] += 1
        table_stats["bytes"] += 4 * len(values)
        return b.load(b.gep(table, [ir.Constant(i32, 0), index], inbounds=True))

    def compact_leaf(tree, root):
        leaves = []
        def index(node):
            if tree.left[node] == -1:
                result = ir.Constant(i32, len(leaves))
                leaves.append(ir.Constant(f32, tree.value[node]))
                return result
            cond = condition(tree, node)
            left, right = index(tree.left[node]), index(tree.right[node])
            return b.select(cond, left, right)
        idx = index(root)
        array_type = ir.ArrayType(f32, len(leaves))
        table = ir.GlobalVariable(fn.module, array_type,
                                  name=f"{fn.name}_compact_{table_stats['compact_count']}")
        table.linkage = "internal"
        table.global_constant = True
        table.unnamed_addr = True
        table.align = 4
        table.initializer = ir.Constant(array_type, leaves)
        table_stats["compact_count"] += 1
        table_stats["bytes"] += 4 * len(leaves)
        return b.load(b.gep(table, [ir.Constant(i32, 0), idx], inbounds=True))

    def eager(tree, node):
        if tree.left[node] == -1:
            return ir.Constant(f32, tree.value[node])
        if compact_leaf_depth and 1 < tree.height[node] <= compact_leaf_depth:
            return compact_leaf(tree, node)
        if leaf_table_bits:
            value = leaf_table(tree, node)
            if value is not None:
                return value
        c = condition(tree, node)
        lv, rv = eager(tree, tree.left[node]), eager(tree, tree.right[node])
        return b.select(c, lv, rv)

    total = initial
    pending_values = []
    for ti, tree in indexed_trees:
        cost_nodes = (_cost_select_nodes(tree, counts[ti], select_depth, select_branch_penalty)
                      if select_policy == "cost" else None)
        merge = fn.append_basic_block(f"t{ti}_end")
        incoming = []

        def emit(node):
            if (ti, node) in hybrid_plan.roots:
                value = b.call(hybrid_fn, [x, ir.Constant(i32, hybrid_plan.roots[ti, node])])
                incoming.append((value, b.block))
                b.branch(merge)
                return
            use_select = tree.height[node] <= select_depth
            if select_policy == "cost" and tree.left[node] != -1:
                use_select = node in cost_nodes
            if use_select and tree.left[node] != -1 and select_policy == "profile":
                nl, nr = counts[ti][node]
                # Keep very predictable or unobserved decisions as control flow.
                use_select = nl + nr > 0 and min(nl, nr) / (nl + nr) >= 0.15
            # Do not let an eager ancestor absorb a planned data subtree.
            if (ti, node) in hybrid_plan.ancestors:
                use_select = False
            if use_select:
                if tree.left[node] != -1:
                    table_stats["eager_regions"] += 1
                value = eager(tree, node)
                incoming.append((value, b.block))
                b.branch(merge)
                return
            c = condition(tree, node)
            left = fn.append_basic_block(f"t{ti}_n{node}_left")
            right = fn.append_basic_block(f"t{ti}_n{node}_right")
            branch = b.cbranch(c, left, right)
            if counts is not None:
                nl, nr = counts[ti][node]
                branch.set_weights([max(1, min(nl, 2**31-1)), max(1, min(nr, 2**31-1))])
            b.position_at_end(left)
            emit(tree.left[node])
            b.position_at_end(right)
            emit(tree.right[node])

        emit(0)
        b.position_at_end(merge)
        value = b.phi(f32, name=f"tree_{ti}")
        for v, block in incoming:
            value.add_incoming(v, block)
        pending_values.append(value)
        if accumulation_order == "pairwise_inexact":
            continue
        if len(pending_values) >= accumulation_batch:
            for pending in pending_values:
                total = b.fadd(total, pending)
            pending_values.clear()
    if accumulation_order == "pairwise_inexact":
        # Diagnostic only: reassociates the sum to expose the serial fadd chain's cost.
        while len(pending_values) > 1:
            pending_values = [b.fadd(pending_values[i], pending_values[i + 1]) if i + 1 < len(pending_values)
                              else pending_values[i] for i in range(0, len(pending_values), 2)]
    for pending in pending_values:
        total = b.fadd(total, pending)
    return b, total, len(cached), table_stats


def compile_model(model_path: str | Path, output_dir: str | Path, *,
                  select_depth: int = 1, preload: bool = False,
                  calibration: np.ndarray | None = None, cc: str = "clang",
                  backend: str = "llvmlite", tree_block_size: int = 0,
                  select_policy: str = "height", predicate_hoist_limit: int = 0,
                  leaf_table_bits: int = 0, select_branch_penalty: float = 4.0,
                  rank_feature_limit: int = 0, rank_strategy: str = "binary", rank_bucket_bits: int = 12,
                  compact_leaf_depth: int = 0, accumulation_batch: int = 1,
                  optimization: str = "O3", machine_outliner: bool = False,
                  hybrid_depth: int = 0, hybrid_max_probability: float = 1.0,
                  hybrid_layout: str = "wide", traversal_lanes: int = 0,
                  traversal_mode: str = "scalar", traversal_leaf_layout: str = "sentinel",
                  traversal_data_layout: str = "aos", traversal_alignment: int = 16,
                  traversal_load_schedule: str = "staged", traversal_prefetch: str = "none",
                  traversal_prefetch_distance: int = 1, traversal_prefetch_locality: int = 3,
                  traversal_leaf_state: str = "split", accumulation_order: str = "sequential") -> Path:
    """Emit .o/.so or .dylib, LLVM IR, assembly and metadata for the host CPU.

    Small subtrees become eager LLVM selects; others remain control flow.
    No fast-math, reassociation, missing-value assumptions, or tree reordering
    (except the explicitly inexact accumulation_order="pairwise_inexact" diagnostic).
    Output: void predict_row(const float *features, float *raw_margin).
    """
    if accumulation_order not in {"sequential", "pairwise_inexact"}:
        raise ValueError("accumulation_order must be sequential or pairwise_inexact")
    if accumulation_order != "sequential" and (traversal_lanes or tree_block_size or accumulation_batch != 1):
        raise ValueError("pairwise_inexact accumulation is a single-function diagnostic; disable traversal, blocks and batching")
    if traversal_prefetch not in {"none", "roots", "next", "both"}:
        raise ValueError("traversal_prefetch must be none, roots, next or both")
    if traversal_prefetch != "none" and not traversal_lanes:
        raise ValueError("traversal_prefetch requires interleaved traversal")
    if type(traversal_prefetch_distance) is not int or traversal_prefetch_distance not in (1,2,4):
        raise ValueError("traversal_prefetch_distance must be 1, 2 or 4 groups")
    if type(traversal_prefetch_locality) is not int or traversal_prefetch_locality not in (0,1,2,3):
        raise ValueError("traversal_prefetch_locality must be 0, 1, 2 or 3")
    if traversal_load_schedule not in {"staged", "direct", "lane"}:
        raise ValueError("traversal_load_schedule must be staged, direct or lane")
    if traversal_data_layout not in {"aos", "soa", "soa8"}:
        raise ValueError("traversal_data_layout must be aos, soa or soa8")
    if isinstance(traversal_alignment, bool) or not isinstance(traversal_alignment, int) or traversal_alignment not in (16,64,128,4096):
        raise ValueError("traversal_alignment must be 16, 64, 128 or 4096")
    if traversal_leaf_layout not in {"sentinel", "self_loop", "separate"}:
        raise ValueError("traversal_leaf_layout must be sentinel, self_loop or separate")
    if traversal_leaf_layout == "separate" and (not traversal_lanes or traversal_prefetch != "none"):
        raise ValueError("Separate traversal leaves require lanes and prefetch none")
    if traversal_leaf_state not in {"split", "packed"}:
        raise ValueError("traversal_leaf_state must be split or packed")
    if traversal_leaf_state == "packed" and traversal_leaf_layout != "separate":
        raise ValueError("Packed traversal_leaf_state requires separate leaves")
    if isinstance(traversal_lanes, bool) or traversal_lanes not in (0, 1, 2, 4, 8, 12, 16, 24, 32) or not isinstance(traversal_lanes, int):
        raise ValueError("traversal_lanes must be 0, 1, 2, 4, 8, 12, 16, 24 or 32")
    if traversal_mode not in {"scalar", "vector"}:
        raise ValueError("traversal_mode must be scalar or vector")
    if traversal_mode == "vector" and traversal_lanes not in (4, 8, 12, 16, 24, 32):
        raise ValueError("Vector traversal requires 4, 8, 12, 16, 24 or 32 lanes")
    if traversal_lanes and (hybrid_depth or rank_feature_limit or predicate_hoist_limit or leaf_table_bits or
                            compact_leaf_depth or tree_block_size or preload or accumulation_batch != 1):
        raise ValueError("Interleaved traversal is a separate lowering; disable hybrid, rank, table, block, preload and batching options")
    if hybrid_layout not in {"wide", "compact"}:
        raise ValueError("hybrid_layout must be wide or compact")
    if not isinstance(hybrid_depth, int) or isinstance(hybrid_depth, bool) or not 0 <= hybrid_depth <= 64:
        raise ValueError("hybrid_depth must be an integer in [0, 64]")
    if (not isinstance(hybrid_max_probability, (int, float)) or
            not math.isfinite(hybrid_max_probability) or not 0 <= hybrid_max_probability <= 1):
        raise ValueError("hybrid_max_probability must be finite and in [0, 1]")
    if hybrid_depth and hybrid_max_probability < 1 and calibration is None:
        raise ValueError("Probability-based hybrid selection requires calibration")
    if not isinstance(machine_outliner, bool):
        raise ValueError("machine_outliner must be boolean")
    if machine_outliner and backend != "clang":
        raise ValueError("machine_outliner requires the clang backend")
    if not isinstance(select_depth, int) or not 0 <= select_depth <= 6:
        raise ValueError("select_depth must be an integer from 0 to 6")
    if backend not in {"llvmlite", "clang"}:
        raise ValueError("backend must be llvmlite or clang")
    if not isinstance(tree_block_size, int) or tree_block_size < 0:
        raise ValueError("tree_block_size must be a nonnegative integer")
    if select_policy not in {"height", "profile", "cost"}:
        raise ValueError("select_policy must be height, profile or cost")
    if select_policy in {"profile", "cost"} and calibration is None:
        raise ValueError("profile/cost select_policy requires calibration data")
    if (not isinstance(select_branch_penalty, (int, float)) or
            not math.isfinite(select_branch_penalty) or select_branch_penalty < 0):
        raise ValueError("select_branch_penalty must be finite and nonnegative")
    if not isinstance(predicate_hoist_limit,int) or not 0 <= predicate_hoist_limit <= 256:
        raise ValueError("predicate_hoist_limit must be an integer in [0, 256]")
    if predicate_hoist_limit and calibration is None:
        raise ValueError("predicate hoisting requires calibration data")
    if not isinstance(leaf_table_bits, int) or not 0 <= leaf_table_bits <= 8:
        raise ValueError("leaf_table_bits must be an integer in [0, 8]")
    if not isinstance(rank_feature_limit, int) or not 0 <= rank_feature_limit <= 256:
        raise ValueError("rank_feature_limit must be an integer in [0, 256]")
    if rank_feature_limit and calibration is None:
        raise ValueError("rank encoding requires calibration data")
    if not isinstance(rank_bucket_bits, int) or not 8 <= rank_bucket_bits <= 16:
        raise ValueError("rank_bucket_bits must be an integer in [8, 16]")
    if rank_strategy not in {"binary", "eytzinger", "simd", "bucket", "bucket_split"}:
        raise ValueError("rank_strategy must be binary, eytzinger, simd, bucket or bucket_split")
    if not isinstance(compact_leaf_depth, int) or not 0 <= compact_leaf_depth <= 6:
        raise ValueError("compact_leaf_depth must be an integer in [0, 6]")
    if not isinstance(accumulation_batch, int) or not 1 <= accumulation_batch <= 64:
        raise ValueError("accumulation_batch must be an integer in [1, 64]")
    if backend == "llvmlite" and optimization in {"Os", "Oz"}:
        raise ValueError("Os and Oz require the clang backend")
    if optimization not in {"O3", "O2", "Os", "Oz"}:
        raise ValueError("optimization must be O3, O2, Os or Oz")
    started = time.perf_counter()
    forest = Forest.load(model_path)
    counts = forest.branch_counts(calibration) if calibration is not None else None
    hybrid_plan = plan_hybrid(forest, counts, hybrid_depth, hybrid_max_probability)
    rank_thresholds = plan_ranks(forest, counts, rank_feature_limit)
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    cpu = llvm.get_host_cpu_name()
    features = llvm.get_host_cpu_features().flatten()
    tm = llvm.Target.from_default_triple().create_target_machine(
        cpu=cpu, features=features, opt=3, reloc="pic", codemodel="small")
    module = ir.Module(name="xgb_latency")
    module.triple = llvm.get_default_triple()
    module.data_layout = str(tm.target_data)
    hybrid_fn, _ = emit_hybrid(module, hybrid_plan, hybrid_layout)
    f32, i32 = ir.FloatType(), ir.IntType(32)
    fn = ir.Function(module, ir.FunctionType(ir.VoidType(), [f32.as_pointer(), f32.as_pointer()]),
                     name="predict_row")
    fn.attributes.add("nounwind")
    x, output = fn.args
    x.name, output.name = "features", "raw_margin"
    indexed_trees = list(enumerate(forest.trees))
    lowering = dict(select_depth=select_depth, preload=preload, counts=counts,
                    select_policy=select_policy,predicate_hoist_limit=predicate_hoist_limit,
                    leaf_table_bits=leaf_table_bits, select_branch_penalty=select_branch_penalty)
    lowering["rank_thresholds"] = rank_thresholds
    lowering.update(rank_strategy=rank_strategy, rank_bucket_bits=rank_bucket_bits, compact_leaf_depth=compact_leaf_depth,
                    accumulation_batch=accumulation_batch, hybrid_plan=hybrid_plan, hybrid_fn=hybrid_fn,
                    accumulation_order=accumulation_order)
    hoisted_count = 0
    table_count = table_bytes = 0
    eager_regions = 0
    compact_count = 0
    initial = ir.Constant(f32, forest.base_margin)
    traversal_stats = {}
    if traversal_lanes and traversal_leaf_layout == "separate":
        b, total, traversal_stats = emit_separated(fn, forest, traversal_lanes, traversal_mode,
                                                   traversal_data_layout, traversal_alignment, traversal_load_schedule,
                                                   traversal_leaf_state)
    elif traversal_lanes:
        b, total, traversal_stats = emit_interleaved(fn, forest, traversal_lanes, traversal_mode, traversal_leaf_layout,
                                                    traversal_data_layout, traversal_alignment, traversal_load_schedule,
                                                    traversal_prefetch, traversal_prefetch_distance, traversal_prefetch_locality)
    elif tree_block_size and indexed_trees:
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        rank_buffer = None
        if rank_thresholds:
            storage = b.alloca(ir.ArrayType(i32, len(rank_thresholds)), name="row_ranks")
            rank_buffer = b.gep(storage, [ir.Constant(i32, 0), ir.Constant(i32, 0)])
            for i, value in enumerate(emit_ranks(b, x, rank_thresholds, rank_strategy, rank_bucket_bits).values()):
                b.store(value, b.gep(rank_buffer, [ir.Constant(i32, i)]))
        total = initial
        for start in range(0, len(indexed_trees), tree_block_size):
            arguments = [f32.as_pointer(), f32] + ([i32.as_pointer()] if rank_thresholds else [])
            group = ir.Function(module, ir.FunctionType(f32, arguments),
                                name=f"tree_block_{start}")
            group.linkage = "internal"
            group.attributes.add("noinline")
            group.attributes.add("nounwind")
            gb, subtotal, hoisted, tables = _emit_trees(group, indexed_trees[start:start+tree_block_size],
                                       group.args[1], rank_buffer=group.args[2] if rank_thresholds else None,
                                       **lowering)
            hoisted_count += hoisted
            table_count += tables["count"]
            table_bytes += tables["bytes"]
            eager_regions += tables["eager_regions"]
            compact_count += tables["compact_count"]
            gb.ret(subtotal)
            # Pass the accumulator through each block: no partial-sum reassociation.
            total = b.call(group, [x, total] + ([rank_buffer] if rank_thresholds else []))
    else:
        b, total, hoisted_count, tables = _emit_trees(fn, indexed_trees, initial, **lowering)
        table_count, table_bytes = tables["count"], tables["bytes"]
        eager_regions = tables["eager_regions"]
        compact_count = tables["compact_count"]
    b.store(total, output)
    b.ret_void()
    raw_ir = str(module)
    (out / "model.ll").write_text(raw_ir)
    mod = llvm.parse_assembly(raw_ir)
    mod.verify()
    if backend == "llvmlite":
        speed = {"O3": 3, "O2": 2}[optimization]
        with llvm.create_pipeline_tuning_options(speed_level=speed) as pto:
            with llvm.create_pass_builder(tm, pto) as pb:
                with pb.getModulePassManager() as pm:
                    pm.run(mod, pb)
        mod.verify()
        (out / "model.opt.ll").write_text(str(mod))
        (out / "model.s").write_text(tm.emit_assembly(mod))
        (out / "model.o").write_bytes(tm.emit_object(mod))
    else:
        # Control experiment: use the same optimizer/code generator as TL2cgen.
        native_flag = "-mcpu=native" if module.triple.startswith("arm64") or module.triple.startswith("aarch64") else "-march=native"
        command = [cc, "-" + optimization, native_flag, "-fPIC", "-fno-fast-math", "-ffp-contract=off",
                   *(["-mllvm", "-enable-machine-outliner=always"] if machine_outliner else []),
                   "-x", "ir", str(out / "model.ll")]
        for options, filename in [(["-c"], "model.o"), (["-S"], "model.s"),
                                  (["-S", "-emit-llvm"], "model.opt.ll")]:
            subprocess.run([*command, *options, "-o", str(out / filename)],
                           check=True, capture_output=True)
    lib = out / ("model.dylib" if sys.platform == "darwin" else "model.so")
    subprocess.run([cc, "-dynamiclib" if sys.platform == "darwin" else "-shared",
                    str(out / "model.o"), "-o", str(lib)], check=True, capture_output=True)
    (out / "model.h").write_text(
        '#pragma once\n#ifdef __cplusplus\nextern "C" {\n#endif\n'
        '/* Dense float32, NaN missing; caller provides one output float. */\n'
        'void predict_row(const float *features, float *raw_margin);\n'
        '#ifdef __cplusplus\n}\n#endif\n')
    metadata = {"format_version": 1, "num_feature": forest.num_feature,
                "num_trees": len(forest.trees), "objective": forest.objective,
                "output": "raw_margin", "base_margin": forest.base_margin,
                "source_sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
                "xgboost_model_version": forest.version, "cpu": cpu, "features": features,
                "target": module.triple, "llvmlite": llvmlite.__version__,
                "ir_library_llvm": llvm.llvm_version_info,
                "backend": backend,
                "codegen_version": (list(llvm.llvm_version_info) if backend == "llvmlite" else
                                    subprocess.check_output([cc,"--version"],text=True).splitlines()[0]),
                "select_depth": select_depth, "select_policy": select_policy,
                "select_branch_penalty": select_branch_penalty,
                "eager_regions_in_ir": eager_regions,
                "rank_feature_limit": rank_feature_limit,
                "rank_strategy": rank_strategy, "rank_bucket_bits": rank_bucket_bits, "compact_leaf_depth": compact_leaf_depth,
                "compact_leaf_tables_in_ir": compact_count, "accumulation_batch": accumulation_batch,
                "accumulation_order": accumulation_order, "exact_accumulation_order": accumulation_order == "sequential",
                "optimization": optimization, "machine_outliner": machine_outliner,
                "hybrid_depth": hybrid_depth, "hybrid_max_probability": hybrid_max_probability,
                "hybrid_subtrees": len(hybrid_plan.roots), "hybrid_nodes": len(hybrid_plan.records),
                "hybrid_layout": hybrid_layout, "hybrid_node_bytes": 8 if hybrid_layout == "compact" else 16,
                "traversal_lanes": traversal_lanes, "traversal_mode": traversal_mode, "traversal_stats": traversal_stats,
                "traversal_leaf_layout": traversal_leaf_layout,
                "traversal_leaf_state": traversal_leaf_state,
                "traversal_data_layout": traversal_data_layout, "traversal_alignment": traversal_alignment,
                "traversal_load_schedule": traversal_load_schedule,
                "traversal_prefetch": traversal_prefetch, "traversal_prefetch_distance": traversal_prefetch_distance,
                "traversal_prefetch_locality": traversal_prefetch_locality,
                "hybrid_table_bytes": (8 if hybrid_layout == "compact" else 16) * len(hybrid_plan.records),
                "rank_features": [{"feature": f, "threshold_count": len(values),
                                   "search_rounds": None if rank_strategy == "simd" else (bucket_plan(values, rank_bucket_bits)[2] if rank_strategy in {"bucket", "bucket_split"} else len(values).bit_length()),
                                   "vector_groups": len(values) // 4 if rank_strategy == "simd" else 0}
                                  for f, values in rank_thresholds.items()],
                "tree_block_size": tree_block_size,
                "predicate_hoist_limit": predicate_hoist_limit,
                "predicates_hoisted_in_ir": hoisted_count,
                "leaf_table_bits": leaf_table_bits,
                "leaf_tables_in_ir": table_count,
                "leaf_table_bytes_in_ir": table_bytes,
                "preload": preload, "profiled": counts is not None,
                "compile_seconds": time.perf_counter() - started,
                "library_bytes": lib.stat().st_size}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return lib
