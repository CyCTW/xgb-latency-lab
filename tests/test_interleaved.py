import json

import numpy as np
import pytest
import xgboost as xgb

from xgb_latency import Forest,Predictor,compile_model


@pytest.fixture(scope='module')
def cases(tmp_path_factory):
    root=tmp_path_factory.mktemp('interleaved')
    rng=np.random.default_rng(318)
    train=rng.normal(size=(512,7)).astype(np.float32)
    labels=train[:,0]+train[:,1]*train[:,2]
    train[rng.random(train.shape)<.1]=np.nan
    result={}
    for objective in ['reg:squarederror','binary:logistic']:
        dtrain=xgb.DMatrix(train,label=(labels>0).astype(np.float32) if 'logistic' in objective else labels)
        booster=xgb.Booster({'objective':objective,'base_score':.23,'nthread':1,'seed':318},[dtrain])
        for i in range(37):
            booster.set_param({'max_depth':1+i%4,'gamma':1e9 if i==36 else 0})
            booster.update(dtrain,i)
        path=root/(objective.replace(':','_')+'.json');booster.save_model(path)
        forest=Forest.load(path)
        rows=[*rng.normal(size=(128,7)).astype(np.float32),np.full(7,np.nan,dtype=np.float32),
              np.zeros(7,dtype=np.float32),np.full(7,-0.,dtype=np.float32)]
        for tree in forest.trees:
            for n,left in enumerate(tree.left):
                if left==-1:continue
                threshold=np.float32(tree.value[n])
                for value in [threshold,np.nextafter(threshold,np.float32(-np.inf)),
                              np.nextafter(threshold,np.float32(np.inf)),np.nan,np.inf,-np.inf]:
                    row=rng.normal(size=7).astype(np.float32);row[tree.feature[n]]=value;rows.append(row)
        rows=np.array(rows,dtype=np.float32)
        expected=[]
        for row in rows:
            total=np.float32(forest.base_margin)
            for tree in forest.trees:
                n=0
                while tree.left[n]!=-1:
                    value=row[tree.feature[n]]
                    left=tree.default_left[n] if np.isnan(value) else value<tree.value[n]
                    n=tree.left[n] if left else tree.right[n]
                total=np.float32(total+np.float32(tree.value[n]))
            expected.append(total)
        expected=np.array(expected,dtype=np.float32)
        valid=~np.isinf(rows).any(axis=1)
        np.testing.assert_array_equal(expected[valid].view(np.uint32),
            booster.predict(xgb.DMatrix(rows[valid]),output_margin=True).view(np.uint32))
        assert forest.trees[-1].height[0]==0
        assert len({t.height[0] for t in forest.trees})>=3
        result[objective]=(path,rows,expected)
    return result


@pytest.mark.parametrize('objective',['reg:squarederror','binary:logistic'])
@pytest.mark.parametrize('backend',['clang','llvmlite'])
@pytest.mark.parametrize('leaf_layout',['sentinel','self_loop'])
@pytest.mark.parametrize('mode,lanes',[('scalar',1),('scalar',2),('scalar',4),('scalar',8),('scalar',16),
                                       ('vector',4),('vector',8),('vector',16)])
def test_interleaved_exact(cases,tmp_path,objective,backend,leaf_layout,mode,lanes):
    path,rows,expected=cases[objective];before=rows.copy()
    lib=compile_model(path,tmp_path,backend=backend,traversal_lanes=lanes,traversal_mode=mode,traversal_leaf_layout=leaf_layout)
    predictor=Predictor(lib)
    np.testing.assert_array_equal(predictor.predict(rows).view(np.uint32),expected.view(np.uint32))
    np.testing.assert_array_equal(rows.view(np.uint32),before.view(np.uint32))
    assert predictor.metadata['traversal_stats']['groups']==(37+lanes-1)//lanes
    assert 'llvm.loop.unroll.disable' in (tmp_path/'model.ll').read_text()
    if mode=='vector':assert 'fcmp olt <4 x float>' in (tmp_path/'model.ll').read_text()


@pytest.mark.parametrize('empty',[False,True])
@pytest.mark.parametrize('leaf_layout',['sentinel','self_loop'])
def test_interleaved_constant_and_empty(cases,tmp_path,empty,leaf_layout):
    original,_,_=cases['reg:squarederror'];doc=json.loads(original.read_text())
    model=doc['learner']['gradient_booster']['model']
    model['trees']=[] if empty else [model['trees'][-1]]
    model['tree_info']=[] if empty else [0]
    path=tmp_path/'constant.json';path.write_text(json.dumps(doc))
    forest=Forest.load(path)
    expected=np.float32(forest.base_margin)
    if not empty:expected=np.float32(expected+np.float32(forest.trees[0].value[0]))
    lib=compile_model(path,tmp_path/'compiled',traversal_lanes=4,traversal_mode='vector',traversal_leaf_layout=leaf_layout)
    rows=np.full((3,7),np.nan,dtype=np.float32)
    np.testing.assert_array_equal(Predictor(lib).predict(rows),np.full(3,expected,dtype=np.float32))


@pytest.mark.parametrize('options',[dict(traversal_leaf_layout='bad'),dict(traversal_lanes=-1),dict(traversal_lanes=True),dict(traversal_lanes=3),
    dict(traversal_lanes=4.),dict(traversal_mode='bad'),dict(traversal_mode='vector'),
    dict(traversal_mode='vector',traversal_lanes=2),dict(traversal_lanes=4,rank_feature_limit=1),
    dict(traversal_lanes=4,hybrid_depth=2),dict(traversal_lanes=4,tree_block_size=4)])
def test_interleaved_invalid(tmp_path,options):
    with pytest.raises(ValueError,match='[Tt]raversal|Interleaved'):
        compile_model(tmp_path/'unused.json',tmp_path,**options)


@pytest.mark.parametrize('backend',['clang','llvmlite'])
@pytest.mark.parametrize('layout',['aos','soa','soa8'])
@pytest.mark.parametrize('leaf_layout',['sentinel','self_loop'])
@pytest.mark.parametrize('mode,lanes',[('scalar',12),('scalar',24),('scalar',32),('vector',12),('vector',32)])
def test_wide_interleaved_layouts(cases,tmp_path,backend,layout,leaf_layout,mode,lanes):
    path,rows,expected=cases['binary:logistic']
    alignment=4096 if lanes==32 else 64
    lib=compile_model(path,tmp_path,backend=backend,traversal_lanes=lanes,traversal_mode=mode,
                      traversal_leaf_layout=leaf_layout,traversal_data_layout=layout,traversal_alignment=alignment)
    predictor=Predictor(lib)
    np.testing.assert_array_equal(predictor.predict(rows).view(np.uint32),expected.view(np.uint32))
    stats=predictor.metadata['traversal_stats']
    assert stats['groups']==(37+lanes-1)//lanes
    padded=(stats['nodes']+7)//8*8 if layout=='soa8' else stats['nodes']
    assert stats['table_bytes']==8*padded+4*37
    assert predictor.metadata['traversal_alignment']==alignment


@pytest.mark.parametrize('options',[dict(traversal_data_layout='bad'),dict(traversal_alignment=32),
                                    dict(traversal_alignment=True),dict(traversal_alignment=64.),
                                    dict(traversal_load_schedule='bad')])
def test_invalid_traversal_layout(tmp_path,options):
    with pytest.raises(ValueError,match='traversal_'):
        compile_model(tmp_path/'unused.json',tmp_path,**options)


@pytest.mark.parametrize('backend',['clang','llvmlite'])
@pytest.mark.parametrize('layout',['aos','soa','soa8'])
@pytest.mark.parametrize('leaf_layout',['sentinel','self_loop'])
@pytest.mark.parametrize('schedule',['direct','lane'])
@pytest.mark.parametrize('mode',['scalar','vector'])
def test_interleaved_load_schedules(cases,tmp_path,backend,layout,leaf_layout,schedule,mode):
    path,rows,expected=cases['reg:squarederror'];before=rows.copy()
    lib=compile_model(path,tmp_path,backend=backend,traversal_lanes=12,traversal_mode=mode,
                      traversal_leaf_layout=leaf_layout,traversal_data_layout=layout,
                      traversal_load_schedule=schedule)
    predictor=Predictor(lib)
    np.testing.assert_array_equal(predictor.predict(rows).view(np.uint32),expected.view(np.uint32))
    np.testing.assert_array_equal(rows.view(np.uint32),before.view(np.uint32))
    assert predictor.metadata['traversal_load_schedule']==schedule


@pytest.mark.parametrize('backend',['clang','llvmlite'])
@pytest.mark.parametrize('layout',['aos','soa','soa8'])
@pytest.mark.parametrize('leaf_layout',['sentinel','self_loop'])
@pytest.mark.parametrize('prefetch',['roots','next','both'])
@pytest.mark.parametrize('mode',['scalar','vector'])
def test_interleaved_prefetch(cases,tmp_path,backend,layout,leaf_layout,prefetch,mode):
    path,rows,expected=cases['binary:logistic'];before=rows.copy()
    distance=4 if mode=='vector' else 1
    locality={'roots':0,'next':2,'both':3}[prefetch]
    lib=compile_model(path,tmp_path,backend=backend,traversal_lanes=12,traversal_mode=mode,
                      traversal_leaf_layout=leaf_layout,traversal_data_layout=layout,
                      traversal_load_schedule='lane',traversal_prefetch=prefetch,
                      traversal_prefetch_distance=distance,traversal_prefetch_locality=locality)
    p=Predictor(lib)
    np.testing.assert_array_equal(p.predict(rows).view(np.uint32),expected.view(np.uint32))
    np.testing.assert_array_equal(rows.view(np.uint32),before.view(np.uint32))
    assert p.metadata['traversal_prefetch']==prefetch
    assert p.metadata['traversal_stats']['prefetch_sites']>0
    assert 'llvm.prefetch' in (tmp_path/'model.ll').read_text()


@pytest.mark.parametrize('empty',[False,True])
def test_prefetch_empty_and_constant(cases,tmp_path,empty):
    original,_,_=cases['reg:squarederror'];doc=json.loads(original.read_text())
    model=doc['learner']['gradient_booster']['model']
    model['trees']=[] if empty else [model['trees'][-1]]
    model['tree_info']=[] if empty else [0]
    path=tmp_path/'constant.json';path.write_text(json.dumps(doc))
    forest=Forest.load(path);expected=np.float32(forest.base_margin)
    if not empty:expected=np.float32(expected+np.float32(forest.trees[0].value[0]))
    lib=compile_model(path,tmp_path/'compiled',traversal_lanes=32,traversal_prefetch='both',traversal_prefetch_distance=4)
    rows=np.full((3,7),np.nan,dtype=np.float32)
    np.testing.assert_array_equal(Predictor(lib).predict(rows),np.full(3,expected,dtype=np.float32))


@pytest.mark.parametrize('options',[dict(traversal_prefetch='bad'),dict(traversal_prefetch='roots'),
    dict(traversal_prefetch_distance=0),dict(traversal_prefetch_distance=True),dict(traversal_prefetch_distance=1.),
    dict(traversal_prefetch_locality=-1),dict(traversal_prefetch_locality=4),dict(traversal_prefetch_locality=True)])
def test_invalid_prefetch(tmp_path,options):
    with pytest.raises(ValueError,match='traversal_prefetch'):
        compile_model(tmp_path/'unused.json',tmp_path,**options)


@pytest.mark.parametrize('objective',['reg:squarederror','binary:logistic'])
@pytest.mark.parametrize('backend',['clang','llvmlite'])
@pytest.mark.parametrize('layout',['aos','soa','soa8'])
@pytest.mark.parametrize('mode,lanes',[('scalar',1),('scalar',4),('scalar',8),('scalar',12),('scalar',16),
    ('scalar',24),('scalar',32),('vector',4),('vector',12),('vector',32)])
@pytest.mark.parametrize('state',['split','packed'])
def test_separate_leaves_exact(cases,tmp_path,objective,backend,layout,mode,lanes,state):
    path,rows,expected=cases[objective];before=rows.copy()
    schedule='direct' if lanes<=4 else 'lane' if lanes<=12 else 'staged'
    lib=compile_model(path,tmp_path,backend=backend,traversal_lanes=lanes,traversal_mode=mode,
                      traversal_data_layout=layout,traversal_load_schedule=schedule,traversal_leaf_layout='separate',
                      traversal_leaf_state=state)
    p=Predictor(lib)
    np.testing.assert_array_equal(p.predict(rows).view(np.uint32),expected.view(np.uint32))
    np.testing.assert_array_equal(rows.view(np.uint32),before.view(np.uint32))
    stats=p.metadata['traversal_stats'];forest=Forest.load(path)
    assert p.metadata['traversal_leaf_state']==state
    internal=sum(sum(left!=-1 for left in t.left) for t in forest.trees)
    leaves=sum(sum(left==-1 for left in t.left) for t in forest.trees)
    assert stats['internal_nodes']==internal and stats['leaves']==leaves
    assert stats['groups']==(37+lanes-1)//lanes
    assert stats['table_bytes']<8*(internal+leaves)+4*37


@pytest.mark.parametrize('empty',[False,True])
@pytest.mark.parametrize('layout',['aos','soa','soa8'])
@pytest.mark.parametrize('state',['split','packed'])
def test_separate_constant_and_empty(cases,tmp_path,empty,layout,state):
    original,_,_=cases['reg:squarederror'];doc=json.loads(original.read_text())
    model=doc['learner']['gradient_booster']['model']
    model['trees']=[] if empty else [model['trees'][-1]]*5
    model['tree_info']=[] if empty else [0]*5
    # Distinct constants exercise the per-tree leaf prefix even with no splits.
    if not empty:
        model['trees']=json.loads(json.dumps(model['trees']))
        for i,t in enumerate(model['trees']):
            t['id']=i;t['split_conditions'][0]=float(i-.5)
    path=tmp_path/'constant.json';path.write_text(json.dumps(doc))
    forest=Forest.load(path);expected=np.float32(forest.base_margin)
    for tree in forest.trees:expected=np.float32(expected+np.float32(tree.value[0]))
    lib=compile_model(path,tmp_path/'compiled',traversal_lanes=4,traversal_mode='vector',
                      traversal_data_layout=layout,traversal_leaf_layout='separate',traversal_leaf_state=state)
    np.testing.assert_array_equal(Predictor(lib).predict(np.full((3,7),np.nan,dtype=np.float32)),np.full(3,expected,dtype=np.float32))


@pytest.mark.parametrize('options',[dict(traversal_leaf_layout='separate'),
    dict(traversal_lanes=4,traversal_leaf_layout='separate',traversal_prefetch='roots')])
def test_invalid_separate_leaves(tmp_path,options):
    with pytest.raises(ValueError,match='[Ss]eparate'):
        compile_model(tmp_path/'unused.json',tmp_path,**options)


@pytest.mark.parametrize('options',[dict(traversal_leaf_state='bad'),dict(traversal_leaf_state='packed'),
    dict(traversal_lanes=4,traversal_leaf_layout='self_loop',traversal_leaf_state='packed')])
def test_invalid_leaf_state(tmp_path,options):
    with pytest.raises(ValueError,match='traversal_leaf_state'):
        compile_model(tmp_path/'unused.json',tmp_path,**options)
