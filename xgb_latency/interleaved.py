"""Advance several trees for one row, preserving the original sum order."""
from llvmlite import ir


def emit_interleaved(fn, forest, lanes, mode, leaf_layout, data_layout, alignment, load_schedule,
                     prefetch, prefetch_distance, prefetch_locality):
    f32, i32 = ir.FloatType(), ir.IntType(32)
    c = lambda n: ir.Constant(i32, n)
    feature_bits = max(1, (forest.num_feature-1).bit_length())
    self_loop = leaf_layout == 'self_loop'
    records, roots = [], []

    def pack(tree, node):
        index = len(records)
        records.append(None)
        if tree.left[node] == -1:
            records[index] = (0 if self_loop else -1, tree.value[node])
        else:
            left, right = pack(tree, tree.left[node]), pack(tree, tree.right[node])
            assert left == index+1
            tag = ((right-index) << (feature_bits+1)) | (tree.feature[node] << 1) | int(tree.default_left[node])
            if tag >= 2**31:
                raise ValueError('Interleaved node cannot encode feature and child offset')
            records[index] = (tag | (0x80000000 if self_loop else 0), tree.value[node])
        return index

    for tree in forest.trees:
        roots.append(pack(tree, 0))
    if len(records) >= 2**31:
        raise ValueError('Interleaved table exceeds signed i32 representation')
    b = ir.IRBuilder(fn.append_basic_block('entry'))
    total = ir.Constant(f32, forest.base_margin)
    if not roots:
        return b, total, dict(nodes=0, node_bytes=8, table_bytes=0, groups=0, prefetch_sites=0)
    def constant_array(name, typ, values):
        array_type = ir.ArrayType(typ, len(values))
        array = ir.GlobalVariable(fn.module, array_type, name=name)
        array.linkage, array.global_constant, array.unnamed_addr = 'internal', True, True
        array.align = alignment
        array.initializer = ir.Constant(array_type, values)
        return array

    padded_nodes = len(records)
    if data_layout == 'soa':
        tags_data = constant_array('interleaved_controls', i32, [c(tag) for tag, _ in records])
        values_data = constant_array('interleaved_values', f32, [ir.Constant(f32, v) for _, v in records])
    elif data_layout == 'soa8':
        controls_type, values_type = ir.ArrayType(i32, 8), ir.ArrayType(f32, 8)
        tile_type = ir.LiteralStructType([controls_type, values_type])
        padded_nodes = (len(records)+7)//8*8
        padded = records + [(0, 0.)] * (padded_nodes-len(records))
        tiles = [ir.Constant(tile_type, [ir.Constant(controls_type, [c(tag) for tag, _ in padded[i:i+8]]),
                                        ir.Constant(values_type, [ir.Constant(f32, v) for _, v in padded[i:i+8]])])
                 for i in range(0, padded_nodes, 8)]
        data = constant_array('interleaved_tiles', tile_type, tiles)
    else:
        node_type = ir.LiteralStructType([i32, f32])
        data = constant_array('interleaved_nodes', node_type,
                              [ir.Constant(node_type, [c(tag), ir.Constant(f32, v)]) for tag, v in records])

    def field_ptr(builder, index, field):
        if data_layout == 'soa':
            return builder.gep(tags_data if field == 0 else values_data, [c(0), index], inbounds=True)
        if data_layout == 'soa8':
            return builder.gep(data, [c(0), builder.lshr(index, c(3)), c(field), builder.and_(index, c(7))], inbounds=True)
        if load_schedule == 'direct':
            return builder.gep(data, [c(0), index, c(field)], inbounds=True)
        node = builder.gep(data, [c(0), index], inbounds=True)
        return builder.gep(node, [c(0), c(field)], inbounds=True)
    roots_type = ir.ArrayType(i32, len(roots))
    root_data = ir.GlobalVariable(fn.module, roots_type, name='interleaved_roots')
    root_data.linkage, root_data.global_constant, root_data.unnamed_addr = 'internal', True, True
    root_data.align = 16
    root_data.initializer = ir.Constant(roots_type, [c(n) for n in roots])
    prefetch_sites = 0
    if prefetch != 'none':
        i8p = ir.IntType(8).as_pointer()
        hint = ir.Function(fn.module, ir.FunctionType(ir.VoidType(), [i8p, i32, i32, i32]),
                           name='llvm.prefetch.p0i8')

    def prefetch_node(builder, index):
        nonlocal prefetch_sites
        # AoS control and threshold share an aligned 8-byte record. SoA variants
        # need both field addresses; hardware may merge redundant hints.
        for field in ((0,) if data_layout == 'aos' else (0,1)):
            pointer = builder.bitcast(field_ptr(builder, index, field), i8p)
            builder.call(hint, [pointer, c(0), c(prefetch_locality), c(1)])
            prefetch_sites += 1

    helpers = {}

    def helper(width, height):
        key = width, height
        if key in helpers:
            return helpers[key]
        h = ir.Function(fn.module, ir.FunctionType(f32, [f32.as_pointer(), f32, i32]),
                        name=f'interleaved_{mode}_{width}_{height}')
        h.linkage = 'internal'
        h.attributes.add('noinline')
        h.attributes.add('nounwind')
        helpers[key] = h
        x, accumulator, start = h.args
        entry = h.append_basic_block('entry')
        hb = ir.IRBuilder(entry)
        indices = [hb.load(hb.gep(root_data, [c(0), hb.add(start, c(i))], inbounds=True)) for i in range(width)]
        if height:
            loop, done = h.append_basic_block('walk'), h.append_basic_block('done')
            hb.branch(loop)
            hb.position_at_end(loop)
            counter = hb.phi(i32, 'step'); counter.add_incoming(c(0), entry)
            positions = [hb.phi(i32, f'node_{i}') for i in range(width)]
            for pos, root in zip(positions, indices):
                pos.add_incoming(root, entry)
            if load_schedule == 'lane':
                # Emit each dependent feature load next to its node loads.
                # LLVM remains free to reschedule; this is a code-generation candidate.
                thresholds, leaves, controls, values = [], [], [], []
                for pos in positions:
                    tag = hb.load(field_ptr(hb, pos, 0))
                    thresholds.append(hb.load(field_ptr(hb, pos, 1)))
                    if not self_loop:
                        leaves.append(hb.icmp_signed('<', tag, c(0)))
                        tag = hb.select(leaves[-1], c(0), tag)
                    controls.append(tag)
                    feature = hb.and_(hb.lshr(tag, c(1)), c((1 << feature_bits)-1))
                    values.append(hb.load(hb.gep(x, [feature], inbounds=True)))
            else:
                # Preserve the original AoS node-address staging. Equivalent GEP
                # forms changed instruction scheduling and regressed the M3 control.
                if data_layout == 'aos' and load_schedule == 'staged':
                    nodes = [hb.gep(data, [c(0), pos], inbounds=True) for pos in positions]
                    tags = [hb.load(hb.gep(node, [c(0), c(0)], inbounds=True)) for node in nodes]
                    thresholds = [hb.load(hb.gep(node, [c(0), c(1)], inbounds=True)) for node in nodes]
                else:
                    tags = [hb.load(field_ptr(hb, pos, 0)) for pos in positions]
                    thresholds = [hb.load(field_ptr(hb, pos, 1)) for pos in positions]
                leaves = [] if self_loop else [hb.icmp_signed('<', tag, c(0)) for tag in tags]
                controls = tags if self_loop else [hb.select(leaf, c(0), tag) for leaf, tag in zip(leaves, tags)]
                features = [hb.and_(hb.lshr(tag, c(1)), c((1 << feature_bits)-1)) for tag in controls]
                # Finished lanes read valid feature zero and stay at their leaf.
                values = [hb.load(hb.gep(x, [feature], inbounds=True)) for feature in features]
            conditions = []
            for start_lane in range(0, width, 4 if mode == 'vector' else 1):
                count = min(4, width-start_lane) if mode == 'vector' else 1
                if count == 4:
                    vf = ir.VectorType(f32, 4)
                    lhs, rhs = ir.Constant(vf, ir.Undefined), ir.Constant(vf, ir.Undefined)
                    for j in range(4):
                        lhs = hb.insert_element(lhs, values[start_lane+j], c(j))
                        rhs = hb.insert_element(rhs, thresholds[start_lane+j], c(j))
                    less, missing = hb.fcmp_ordered('<', lhs, rhs), hb.fcmp_unordered('!=', lhs, lhs)
                    for j in range(4):
                        default = hb.icmp_unsigned('!=', hb.and_(controls[start_lane+j], c(1)), c(0))
                        conditions.append(hb.or_(hb.extract_element(less, c(j)), hb.and_(hb.extract_element(missing, c(j)), default)))
                else:
                    for j in range(start_lane, start_lane+count):
                        default = hb.icmp_unsigned('!=', hb.and_(controls[j], c(1)), c(0))
                        conditions.append(hb.or_(hb.fcmp_ordered('<', values[j], thresholds[j]),
                                                 hb.and_(hb.fcmp_unordered('!=', values[j], values[j]), default)))
            next_indices = []
            for i, pos in enumerate(positions):
                if self_loop:
                    # Split: left delta=1, right delta>0. Leaf: both deltas=0.
                    left_delta = hb.lshr(controls[i], c(31))
                    right_delta = hb.lshr(hb.and_(controls[i], c(0x7fffffff)), c(feature_bits+1))
                    next_node = hb.add(pos, hb.select(conditions[i], left_delta, right_delta))
                else:
                    right = hb.add(pos, hb.lshr(controls[i], c(feature_bits+1)))
                    child = hb.select(conditions[i], hb.add(pos, c(1)), right)
                    next_node = hb.select(leaves[i], pos, child)
                next_indices.append(next_node)
                pos.add_incoming(next_node, loop)
                if prefetch in ('next', 'both'):
                    # The exact chosen child (or self-loop leaf) is always valid.
                    prefetch_node(hb, next_node)
            step = hb.add(counter, c(1)); counter.add_incoming(step, loop)
            branch = hb.cbranch(hb.icmp_unsigned('<', step, c(height)), loop, done)
            # Keep code shared across levels rather than replicating the gather body.
            unroll = fn.module.add_metadata([ir.MetaDataString(fn.module, 'llvm.loop.unroll.disable')])
            loop_md = fn.module.add_metadata([ir.MetaDataString(fn.module, h.name)])
            loop_md.operands = (loop_md, unroll)
            branch.set_metadata('llvm.loop', loop_md)
            hb.position_at_end(done)
            # The backedge phi contains the prior level; select the final next node.
            indices = next_indices
        for index in indices:
            value = hb.load(field_ptr(hb, index, 1))
            accumulator = hb.fadd(accumulator, value)
        hb.ret(accumulator)
        return h

    if prefetch in ('roots', 'both'):
        # Seed the first distance groups. Each later group is hinted exactly once,
        # distance groups before its helper call; no past-the-end root accesses.
        for root in roots[:lanes*prefetch_distance]:
            prefetch_node(b, c(root))
    for start in range(0, len(roots), lanes):
        if prefetch in ('roots', 'both'):
            lookahead = start + lanes*prefetch_distance
            for root in roots[lookahead:lookahead+lanes]:
                prefetch_node(b, c(root))
        trees = forest.trees[start:start+lanes]
        h = helper(len(trees), max(t.height[0] for t in trees))
        total = b.call(h, [fn.args[0], total, c(start)])
    return b, total, dict(nodes=len(records), padded_nodes=padded_nodes, node_bytes=8, table_bytes=8*padded_nodes+4*len(roots),
                          groups=(len(roots)+lanes-1)//lanes, helpers=len(helpers), prefetch_sites=prefetch_sites)
