"""Compact split records and exact leaves, without padding sparse subtrees.

For a binary subtree with k internal nodes there are k+1 leaves. Therefore
the right-child displacement in internal preorder is also the number of
leaves skipped by taking the right branch. Carrying a leaf prefix index
avoids an extra leaf pointer in every split record.
"""
from llvmlite import ir


def emit_separated(fn, forest, lanes, mode, layout, alignment, schedule, state_layout='split'):
    f32,i32=ir.FloatType(),ir.IntType(32)
    c=lambda n:ir.Constant(i32,n)
    i64=ir.IntType(64);c64=lambda n:ir.Constant(i64,n)
    packed=state_layout=='packed'
    feature_bits=max(1,(forest.num_feature-1).bit_length())
    right_leaf_bit=1 << (feature_bits+1)
    shift=feature_bits+2
    # Record zero is a self-loop for finished lanes, not a model split.
    records=[(0,0.)];leaves=[];roots=[];leaf_starts=[]

    def pack(tree,node):
        if tree.left[node]==-1:
            leaves.append(tree.value[node])
            return 0
        index=len(records);records.append(None)
        left=pack(tree,tree.left[node])
        delta=len(records)-index
        right=pack(tree,tree.right[node])
        assert (left==0 and delta==1) or left==index+1
        assert right==0 or right==index+delta
        tag=(delta<<shift)|(int(right==0)*right_leaf_bit)|(tree.feature[node]<<1)|int(tree.default_left[node])
        if tag>=2**31:raise ValueError('Separated node cannot encode feature and child displacement')
        records[index]=(tag|0x80000000,tree.value[node])
        return index

    for tree in forest.trees:
        leaf_starts.append(len(leaves));roots.append(pack(tree,0))
    if max(len(records),len(leaves))>=2**31:
        raise ValueError('Separated tables exceed signed i32 indexing')
    b=ir.IRBuilder(fn.append_basic_block('entry'))
    total=ir.Constant(f32,forest.base_margin)
    if not roots:
        return b,total,dict(nodes=0,internal_nodes=0,leaves=0,table_bytes=0,groups=0)

    def array(name,typ,values,align=alignment):
        at=ir.ArrayType(typ,len(values));g=ir.GlobalVariable(fn.module,at,name=name)
        g.linkage,g.global_constant,g.unnamed_addr='internal',True,True
        g.align=align;g.initializer=ir.Constant(at,values)
        return g

    padded=len(records)
    if layout=='soa':
        controls=array('separated_controls',i32,[c(tag) for tag,_ in records])
        thresholds=array('separated_thresholds',f32,[ir.Constant(f32,v) for _,v in records])
    elif layout=='soa8':
        ct,vt=ir.ArrayType(i32,8),ir.ArrayType(f32,8)
        tile=ir.LiteralStructType([ct,vt]);padded=(len(records)+7)//8*8
        values=records+[(0,0.)]*(padded-len(records))
        data=array('separated_tiles',tile,[ir.Constant(tile,[
            ir.Constant(ct,[c(tag) for tag,_ in values[i:i+8]]),
            ir.Constant(vt,[ir.Constant(f32,v) for _,v in values[i:i+8]])]) for i in range(0,padded,8)])
    else:
        typ=ir.LiteralStructType([i32,f32])
        data=array('separated_nodes',typ,[ir.Constant(typ,[c(tag),ir.Constant(f32,v)]) for tag,v in records])
    leaf_data=array('separated_leaves',f32,[ir.Constant(f32,v) for v in leaves])
    root_data=array('separated_roots',i32,[c(v) for v in roots],16)
    start_data=array('separated_leaf_starts',i32,[c(v) for v in leaf_starts],16)

    def field(builder,index,which):
        if layout=='soa':return builder.gep(controls if which==0 else thresholds,[c(0),index],inbounds=True)
        if layout=='soa8':return builder.gep(data,[c(0),builder.lshr(index,c(3)),c(which),builder.and_(index,c(7))],inbounds=True)
        if schedule=='direct':return builder.gep(data,[c(0),index,c(which)],inbounds=True)
        node=builder.gep(data,[c(0),index],inbounds=True)
        return builder.gep(node,[c(0),c(which)],inbounds=True)

    helpers={}
    def helper(width,height):
        key=width,height
        if key in helpers:return helpers[key]
        h=ir.Function(fn.module,ir.FunctionType(f32,[f32.as_pointer(),f32,i32]),name=f'separated_{mode}_{width}_{height}')
        h.linkage='internal';h.attributes.add('noinline');h.attributes.add('nounwind');helpers[key]=h
        x,acc,start=h.args;entry=h.append_basic_block('entry');hb=ir.IRBuilder(entry)
        indices=[hb.load(hb.gep(root_data,[c(0),hb.add(start,c(i))],inbounds=True)) for i in range(width)]
        prefixes=[hb.load(hb.gep(start_data,[c(0),hb.add(start,c(i))],inbounds=True)) for i in range(width)]
        if height:
            if packed:
                states=[hb.or_(hb.zext(index,i64),hb.shl(hb.zext(leaf,i64),c64(32)))
                        for index,leaf in zip(indices,prefixes)]
            loop,done=h.append_basic_block('walk'),h.append_basic_block('done');hb.branch(loop);hb.position_at_end(loop)
            step=hb.phi(i32,'step');step.add_incoming(c(0),entry)
            if packed:
                packed_positions=[hb.phi(i64,f'state_{i}') for i in range(width)]
                for pos,state in zip(packed_positions,states):pos.add_incoming(state,entry)
                positions=[hb.trunc(state,i32) for state in packed_positions]
            else:
                positions=[hb.phi(i32,f'node_{i}') for i in range(width)]
                leaf_positions=[hb.phi(i32,f'leaf_{i}') for i in range(width)]
                for pos,index in zip(positions,indices):pos.add_incoming(index,entry)
                for pos,index in zip(leaf_positions,prefixes):pos.add_incoming(index,entry)
            if schedule=='lane':
                tags,limits,values=[],[],[]
                for pos in positions:
                    tag=hb.load(field(hb,pos,0));tags.append(tag)
                    limits.append(hb.load(field(hb,pos,1)))
                    feature=hb.and_(hb.lshr(tag,c(1)),c((1<<feature_bits)-1))
                    values.append(hb.load(hb.gep(x,[feature],inbounds=True)))
            else:
                if layout=='aos' and schedule=='staged':
                    nodes=[hb.gep(data,[c(0),pos],inbounds=True) for pos in positions]
                    tags=[hb.load(hb.gep(node,[c(0),c(0)],inbounds=True)) for node in nodes]
                    limits=[hb.load(hb.gep(node,[c(0),c(1)],inbounds=True)) for node in nodes]
                else:
                    tags=[hb.load(field(hb,pos,0)) for pos in positions]
                    limits=[hb.load(field(hb,pos,1)) for pos in positions]
                features=[hb.and_(hb.lshr(tag,c(1)),c((1<<feature_bits)-1)) for tag in tags]
                values=[hb.load(hb.gep(x,[feature],inbounds=True)) for feature in features]
            conditions=[]
            for first in range(0,width,4 if mode=='vector' else 1):
                count=min(4,width-first) if mode=='vector' else 1
                if count==4:
                    vf=ir.VectorType(f32,4);lhs,rhs=ir.Constant(vf,ir.Undefined),ir.Constant(vf,ir.Undefined)
                    for j in range(4):
                        lhs=hb.insert_element(lhs,values[first+j],c(j));rhs=hb.insert_element(rhs,limits[first+j],c(j))
                    less,missing=hb.fcmp_ordered('<',lhs,rhs),hb.fcmp_unordered('!=',lhs,lhs)
                    for j in range(4):
                        default=hb.icmp_unsigned('!=',hb.and_(tags[first+j],c(1)),c(0))
                        conditions.append(hb.or_(hb.extract_element(less,c(j)),hb.and_(hb.extract_element(missing,c(j)),default)))
                else:
                    for j in range(first,first+count):
                        default=hb.icmp_unsigned('!=',hb.and_(tags[j],c(1)),c(0))
                        conditions.append(hb.or_(hb.fcmp_ordered('<',values[j],limits[j]),hb.and_(hb.fcmp_unordered('!=',values[j],values[j]),default)))
            final_leaves=[]
            for i,(pos,tag,left) in enumerate(zip(positions,tags,conditions)):
                delta=hb.lshr(hb.and_(tag,c(0x7fffffff)),c(shift))
                right_is_leaf=hb.icmp_unsigned('!=',hb.and_(tag,c(right_leaf_bit)),c(0))
                child_is_leaf=hb.select(left,hb.icmp_unsigned('==',delta,c(1)),right_is_leaf)
                if packed:
                    # Each table has <2^31 entries; even a leaf-child one-past
                    # internal index fits the low word, so it cannot carry.
                    d64=hb.zext(delta,i64)
                    right_delta=hb.or_(d64,hb.shl(d64,c64(32)))
                    child=hb.add(packed_positions[i],hb.select(left,hb.zext(hb.lshr(tag,c(31)),i64),right_delta))
                    state=hb.select(child_is_leaf,hb.and_(child,c64(0xffffffff00000000)),child)
                    packed_positions[i].add_incoming(state,loop)
                    final_leaves.append(state)
                else:
                    child=hb.add(pos,hb.select(left,hb.lshr(tag,c(31)),delta))
                    # Dummy record: zero deltas/flags keep node=0 and leaf unchanged.
                    node=hb.select(child_is_leaf,c(0),child)
                    leaf=leaf_positions[i];next_leaf=hb.add(leaf,hb.select(left,c(0),delta))
                    pos.add_incoming(node,loop);leaf.add_incoming(next_leaf,loop);final_leaves.append(next_leaf)
            next_step=hb.add(step,c(1));step.add_incoming(next_step,loop)
            branch=hb.cbranch(hb.icmp_unsigned('<',next_step,c(height)),loop,done)
            unroll=fn.module.add_metadata([ir.MetaDataString(fn.module,'llvm.loop.unroll.disable')])
            md=fn.module.add_metadata([ir.MetaDataString(fn.module,h.name)]);md.operands=(md,unroll);branch.set_metadata('llvm.loop',md)
            hb.position_at_end(done);prefixes=final_leaves
            if packed:prefixes=[hb.trunc(hb.lshr(state,c64(32)),i32) for state in prefixes]
        for index in prefixes:
            value=hb.load(hb.gep(leaf_data,[c(0),index],inbounds=True));acc=hb.fadd(acc,value)
        hb.ret(acc)
        return h

    for start in range(0,len(roots),lanes):
        trees=forest.trees[start:start+lanes]
        h=helper(len(trees),max(t.height[0] for t in trees));total=b.call(h,[fn.args[0],total,c(start)])
    return b,total,dict(nodes=len(records)-1+len(leaves),internal_nodes=len(records)-1,leaves=len(leaves),
        padded_internal_records=padded,internal_node_bytes=8,leaf_bytes=4,table_bytes=8*padded+4*len(leaves)+8*len(roots),
        groups=(len(roots)+lanes-1)//lanes,helpers=len(helpers))
