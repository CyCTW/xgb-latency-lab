"""Exact data traversal for selected subtrees, shared by generated tree code."""

from dataclasses import dataclass

from llvmlite import ir


@dataclass
class HybridPlan:
    # Records: tagged feature, original float32 threshold/leaf, left, right.
    records: list
    roots: dict
    ancestors: set


def plan_hybrid(forest, counts, max_height, max_probability):
    """Choose representation using calibration visits, never remove paths.

    Probability is unconditional within each tree, relative to its root.
    Height-one stumps remain generated code because they already lower cheaply.
    """
    plan = HybridPlan([], {}, set())
    if not max_height:
        return plan

    def pack(tree, node):
        index = len(plan.records)
        plan.records.append(None)
        if tree.left[node] == -1:
            record = (-1, tree.value[node], -1, -1)
        else:
            left, right = pack(tree, tree.left[node]), pack(tree, tree.right[node])
            record = (2 * tree.feature[node] + int(tree.default_left[node]),
                      tree.value[node], left, right)
        plan.records[index] = record
        return index

    for ti, tree in enumerate(forest.trees):
        population = sum(counts[ti][0]) if counts is not None and tree.left[0] != -1 else 0

        def visit(node):
            if tree.left[node] == -1:
                return False
            probability = sum(counts[ti][node]) / population if population else 1.0
            if 2 <= tree.height[node] <= max_height and probability <= max_probability:
                plan.roots[ti, node] = pack(tree, node)
                return True
            left, right = visit(tree.left[node]), visit(tree.right[node])
            if left or right:
                plan.ancestors.add((ti, node))
            return left or right

        visit(0)
    # LLVM's traversal indices and tagged features use signed i32.
    if len(plan.records) >= 2**31 or any(r[0] >= 2**31 for r in plan.records):
        raise ValueError("Hybrid table exceeds signed i32 representation")
    return plan


def emit_hybrid(module, plan, layout):
    """One noinline loop for all packed subtrees; returns a leaf, never a sum."""
    if not plan.roots:
        return None, None
    f32, i32 = ir.FloatType(), ir.IntType(32)
    c = lambda value: ir.Constant(i32, value)
    compact = layout == 'compact'
    feature_bits = max(1, max((r[0] >> 1).bit_length() for r in plan.records if r[0] >= 0))
    records = []
    for index, (tag, value, left, right) in enumerate(plan.records):
        if compact and tag >= 0:
            assert left == index + 1  # Preorder layout makes the left child implicit.
            tag |= (right-index) << (feature_bits+1)
            if tag >= 2**31:
                raise ValueError('Hybrid compact node cannot encode feature and child offset')
        records.append([c(tag), ir.Constant(f32, value)] + ([] if compact else [c(left), c(right)]))
    record_type = ir.LiteralStructType([i32, f32] + ([] if compact else [i32, i32]))
    array_type = ir.ArrayType(record_type, len(plan.records))
    data = ir.GlobalVariable(module, array_type, name="hybrid_nodes")
    data.linkage, data.global_constant, data.unnamed_addr = "internal", True, True
    data.align = 16
    data.initializer = ir.Constant(array_type, [ir.Constant(record_type, r) for r in records])
    fn = ir.Function(module, ir.FunctionType(f32, [f32.as_pointer(), i32]), name="hybrid_walk")
    fn.linkage = "internal"
    fn.attributes.add("noinline")
    fn.attributes.add("nounwind")
    x, root = fn.args
    entry, loop, split, done = [fn.append_basic_block(n) for n in ("entry", "walk", "split", "leaf")]
    b = ir.IRBuilder(entry)
    b.branch(loop)
    b.position_at_end(loop)
    index = b.phi(i32, name="node")
    index.add_incoming(root, entry)
    ptr = b.gep(data, [c(0), index], inbounds=True)
    tag = b.load(b.gep(ptr, [c(0), c(0)], inbounds=True))
    value = b.load(b.gep(ptr, [c(0), c(1)], inbounds=True))
    b.cbranch(b.icmp_signed("<", tag, c(0)), done, split)
    b.position_at_end(split)
    feature = b.lshr(tag, c(1))
    if compact:
        feature = b.and_(feature, c((1 << feature_bits)-1))
    v = b.load(b.gep(x, [feature], inbounds=True))
    missing_left = b.icmp_unsigned("!=", b.and_(tag, c(1)), c(0))
    left_condition = b.or_(b.fcmp_ordered("<", v, value),
                           b.and_(b.fcmp_unordered("!=", v, v), missing_left))
    if compact:
        left = b.add(index, c(1))
        right = b.add(index, b.lshr(tag, c(feature_bits+1)))
    else:
        left = b.load(b.gep(ptr, [c(0), c(2)], inbounds=True))
        right = b.load(b.gep(ptr, [c(0), c(3)], inbounds=True))
    child = b.select(left_condition, left, right)
    index.add_incoming(child, split)
    b.branch(loop)
    b.position_at_end(done)
    b.ret(value)
    return fn, data
