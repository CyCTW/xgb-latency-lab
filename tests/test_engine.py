import json

import numpy as np
import pytest
import xgboost as xgb

from xgb_latency import Forest, Predictor, compile_model


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("models")
    rng = np.random.default_rng(19)
    x = rng.normal(size=(512,6)).astype(np.float32)
    y = (x[:,0] + .5*x[:,1]*x[:,2]).astype(np.float32)
    x[rng.random(x.shape)<.1] = np.nan
    models = {}
    for objective, base in [("reg:squarederror",.37),("binary:logistic",.23),
                            ("reg:logistic",.61),("reg:absoluteerror",.19)]:
        labels = (y>0).astype(np.float32) if "logistic" in objective else y
        b = xgb.train({"objective":objective,"base_score":base,"max_depth":3,
                       "nthread":1,"seed":19},xgb.DMatrix(x,label=labels),12)
        path = root / (objective.replace(":","_")+".json")
        b.save_model(path)
        models[objective] = (b,path)
    return models, x


@pytest.mark.parametrize("objective",["reg:squarederror","binary:logistic","reg:logistic","reg:absoluteerror"])
@pytest.mark.parametrize("depth,preload,profile",[(0,False,False),(1,False,False),(2,True,False),(3,False,True)])
def test_xgboost_agreement(trained,tmp_path,objective,depth,preload,profile):
    models,calibration = trained
    booster,path = models[objective]
    rng = np.random.default_rng(71)
    rows = rng.normal(size=(256,6)).astype(np.float32)
    rows[rng.random(rows.shape)<.1] = np.nan
    rows[0] = np.nan
    rows[1] = 0
    forest = Forest.load(path)
    # Probe exact thresholds and the adjacent representable values.
    probes = []
    for tree in forest.trees:
        for node,left in enumerate(tree.left):
            if left == -1:
                continue
            threshold = np.float32(tree.value[node])
            for value in (threshold,np.nextafter(threshold,np.float32(-np.inf)),
                          np.nextafter(threshold,np.float32(np.inf))):
                row = rng.normal(size=6).astype(np.float32)
                row[tree.feature[node]] = value
                probes.append(row)
    rows = np.vstack([rows,probes])
    lib = compile_model(path,tmp_path,select_depth=depth,preload=preload,
                        calibration=calibration if profile else None)
    actual = Predictor(lib).predict(rows)
    expected = booster.predict(xgb.DMatrix(rows),output_margin=True)
    np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-6)


@pytest.mark.parametrize("objective",["reg:squarederror","binary:logistic","reg:logistic","reg:absoluteerror"])
def test_same_clang_backend(trained,tmp_path,objective):
    models,rows = trained
    booster,path = models[objective]
    lib = compile_model(path,tmp_path,backend="clang",calibration=rows)
    expected = booster.predict(xgb.DMatrix(rows),output_margin=True)
    np.testing.assert_allclose(Predictor(lib).predict(rows),expected,rtol=1e-6,atol=1e-6)


@pytest.mark.parametrize("backend",["clang","llvmlite"])
@pytest.mark.parametrize("policy",["height","profile","cost"])
@pytest.mark.parametrize("block",[1,5,64])
def test_ordered_blocks(trained,tmp_path,backend,policy,block):
    models,calibration = trained
    booster,path = models["binary:logistic"]
    lib = compile_model(path,tmp_path,backend=backend,select_depth=2,
                        tree_block_size=block,select_policy=policy,calibration=calibration)
    rows = np.vstack([calibration,np.full((1,6),np.nan,dtype=np.float32)])
    expected = booster.predict(xgb.DMatrix(rows),output_margin=True)
    np.testing.assert_array_equal(Predictor(lib).predict(rows),expected)


@pytest.mark.parametrize("backend",["clang","llvmlite"])
@pytest.mark.parametrize("block",[0,2])
def test_hoisted_repeated_predicates(trained,tmp_path,backend,block):
    import copy
    models,calibration = trained
    original = models["binary:logistic"][1]
    forest = Forest.load(original)
    doc = json.loads(original.read_text())
    model = doc["learner"]["gradient_booster"]["model"]
    model["trees"] = [copy.deepcopy(model["trees"][0]) for _ in range(3)]
    model["tree_info"] = [0]*3
    model["gbtree_model_param"]["num_trees"] = "3"
    model["iteration_indptr"] = [0,1,2,3]
    for i,tree in enumerate(model["trees"]):
        tree["id"] = i
    path = tmp_path/"repeated.json"
    path.write_text(json.dumps(doc))
    rows = np.vstack([calibration,np.full((1,6),np.nan,dtype=np.float32)])
    tree = forest.trees[0]
    expected = []
    for row in rows:
        node = 0
        while tree.left[node] != -1:
            value = row[tree.feature[node]]
            left = tree.default_left[node] if np.isnan(value) else value < tree.value[node]
            node = tree.left[node] if left else tree.right[node]
        result = np.float32(forest.base_margin)
        for _ in range(3):
            result = np.float32(result + np.float32(tree.value[node]))
        expected.append(result)
    lib = compile_model(path,tmp_path/"compiled",backend=backend,tree_block_size=block,
                        calibration=calibration,predicate_hoist_limit=8,select_depth=3,
                        select_policy="profile")
    assert Predictor(lib).metadata["predicates_hoisted_in_ir"] > 0
    np.testing.assert_array_equal(Predictor(lib).predict(rows),expected)


@pytest.mark.parametrize("preset", ["initial", "tables", "cost", "ranks", "explore"])
def test_tuning_freezes_before_evaluation(trained,tmp_path,monkeypatch,preset):
    from benchmarks import optimize
    models,rows = trained
    booster,path = models["reg:squarederror"]
    files = []
    for name,data in zip(("calibration","tuning","evaluation"),np.array_split(rows,3)):
        dest = tmp_path/f"{name}.npy"
        np.save(dest,data)
        files.append(dest)
    output = tmp_path/"tuned"
    original = optimize.load_rows
    def audited_load(path,nf):
        if path == files[2]:
            assert (output/"selection.json").exists()
        return original(path,nf)
    monkeypatch.setattr(optimize,"load_rows",audited_load)
    monkeypatch.setattr("sys.argv",["optimize","--model",str(path),"--output",str(output),
        "--preset",preset,
        "--calibration",str(files[0]),"--tuning",str(files[1]),"--evaluation",str(files[2]),
        "--rounds","3","--samples","64"])
    optimize.main()
    report = json.loads((output/"report.json").read_text())
    tuning = json.loads((output/"tuning.json").read_text())
    assert report["selection"]["selected"] == optimize.choose(tuning,report["entries"],"block_median_ns_per_row")
    evaluation = np.load(files[2])
    np.testing.assert_array_equal(Predictor(report["selected_library"]).predict(evaluation),
                                   booster.predict(xgb.DMatrix(evaluation),output_margin=True))


@pytest.mark.parametrize("backend", ["clang", "llvmlite"])
@pytest.mark.parametrize("unobserved", [False, True])
@pytest.mark.parametrize("penalty", [0., 8.])
def test_cost_policy_unseen_inputs(trained, tmp_path, backend, unobserved, penalty):
    models, calibration = trained
    booster, path = models["reg:squarederror"]
    if unobserved:
        calibration = np.full((4, 6), np.nan, dtype=np.float32)
    forest = Forest.load(path)
    rng = np.random.default_rng(539)
    rows = rng.normal(size=(256, 6)).astype(np.float32)
    rows[rng.random(rows.shape) < .2] = np.nan
    probes = [np.full(6, np.nan, dtype=np.float32)]
    for tree in forest.trees:
        for node, left in enumerate(tree.left):
            if left == -1:
                continue
            threshold = np.float32(tree.value[node])
            for v in (threshold, np.nextafter(threshold, np.float32(-np.inf)),
                      np.nextafter(threshold, np.float32(np.inf))):
                row = rng.normal(size=6).astype(np.float32)
                row[tree.feature[node]] = v
                probes.append(row)
    rows = np.vstack([rows, probes])
    lib = compile_model(path, tmp_path / "native", backend=backend, calibration=calibration,
                        select_depth=6, select_policy="cost", select_branch_penalty=penalty,
                        leaf_table_bits=3, predicate_hoist_limit=4, tree_block_size=3)
    expected = booster.predict(xgb.DMatrix(rows), output_margin=True)
    np.testing.assert_array_equal(Predictor(lib).predict(rows).view(np.uint32),
                                  expected.view(np.uint32))


@pytest.mark.parametrize("backend", ["clang", "llvmlite"])
@pytest.mark.parametrize("block", [0, 3])
def test_ranked_forest(trained, tmp_path, backend, block):
    models, calibration = trained
    booster, path = models["binary:logistic"]
    forest = Forest.load(path)
    rng = np.random.default_rng(493)
    rows = rng.normal(size=(256, 6)).astype(np.float32)
    rows[rng.random(rows.shape) < .15] = np.nan
    probes = [np.full(6, np.nan, dtype=np.float32)]
    for tree in forest.trees:
        for node, left in enumerate(tree.left):
            if left == -1:
                continue
            threshold = np.float32(tree.value[node])
            for value in (threshold, np.nextafter(threshold, np.float32(-np.inf)),
                          np.nextafter(threshold, np.float32(np.inf))):
                row = rng.normal(size=6).astype(np.float32)
                row[tree.feature[node]] = value
                probes.append(row)
    rows = np.vstack([rows, probes])
    lib = compile_model(path, tmp_path / "native", backend=backend, calibration=calibration,
                        rank_feature_limit=4, tree_block_size=block, select_depth=4,
                        select_policy="cost", predicate_hoist_limit=4, leaf_table_bits=3)
    predictor = Predictor(lib)
    assert predictor.metadata["rank_features"]
    np.testing.assert_array_equal(predictor.predict(rows).view(np.uint32),
                                  booster.predict(xgb.DMatrix(rows), output_margin=True).view(np.uint32))


@pytest.mark.parametrize("backend", ["clang", "llvmlite"])
@pytest.mark.parametrize("strategy,compact,batch,opt", [
    ("eytzinger", 0, 8, "O3"), ("simd", 3, 4, "O3"),
    ("binary", 4, 16, "Os"), ("binary", 0, 64, "Oz"), ("bucket", 2, 4, "O3"), ("bucket_split", 2, 4, "O3")])
def test_exploratory_lowerings(trained, tmp_path, backend, strategy, compact, batch, opt):
    if backend == "llvmlite" and opt in ("Os", "Oz"):
        opt = "O2"
    models, calibration = trained
    booster, path = models["binary:logistic"]
    rng = np.random.default_rng(730)
    rows = rng.normal(size=(256, 6)).astype(np.float32)
    rows[rng.random(rows.shape) < .2] = np.nan
    rows[0] = np.nan
    lib = compile_model(path, tmp_path, backend=backend, calibration=calibration,
                        select_depth=4, rank_feature_limit=4, tree_block_size=5,
                        rank_strategy=strategy, compact_leaf_depth=compact,
                        accumulation_batch=batch, optimization=opt)
    np.testing.assert_array_equal(Predictor(lib).predict(rows).view(np.uint32),
                                  booster.predict(xgb.DMatrix(rows), output_margin=True).view(np.uint32))


def test_root_split_edges_and_defaults(trained,tmp_path):
    models,_ = trained
    _,path = models["reg:squarederror"]
    doc = json.loads(path.read_text())
    model = doc["learner"]["gradient_booster"]["model"]
    template = model["trees"][0]
    # A hand-built fixture tests routing independently of XGBoost predictions.
    template.update(left_children=[1,-1,-1],right_children=[2,-1,-1],
                    split_indices=[0,0,0],split_conditions=[0.,-2.,3.],
                    default_left=[1,0,0],split_type=[0,0,0])
    template["tree_param"].update(num_nodes="3",num_deleted="0")
    model["trees"],model["tree_info"] = [template],[0]
    doc["learner"]["learner_model_param"]["base_score"] = "[0E0]"
    fixture = tmp_path/"fixture.json"
    x = np.zeros((7,6),dtype=np.float32)
    x[:,0] = [-np.inf,-np.finfo(np.float32).tiny,-0.,0.,np.finfo(np.float32).tiny,np.inf,np.nan]
    for default in (False,True):
        template["default_left"][0] = int(default)
        fixture.write_text(json.dumps(doc))
        for depth in (0,1):
            lib = compile_model(fixture,tmp_path/f"d{default}s{depth}",select_depth=depth)
            np.testing.assert_array_equal(Predictor(lib).predict(x),[-2,-2,3,3,3,3,-2 if default else 3])


@pytest.mark.parametrize("change,message",[
    ("categorical","Categorical"),("dart","gbtree"),("multi","single-output"),
    ("objective","objective"),("cycle","cycle"),("child","out-of-bounds"),
    ("feature","Invalid split"),("array","array lengths")])
def test_reject_unsupported_or_malformed(trained,tmp_path,change,message):
    models,_ = trained
    doc = json.loads(models["reg:squarederror"][1].read_text())
    learner = doc["learner"]
    tree = learner["gradient_booster"]["model"]["trees"][0]
    if change=="categorical": tree["split_type"][0] = 1
    if change=="dart": learner["gradient_booster"]["name"] = "dart"
    if change=="multi": learner["learner_model_param"]["num_target"] = "2"
    if change=="objective": learner["objective"]["name"] = "count:poisson"
    if change=="cycle": tree["left_children"][0] = 0
    if change=="child": tree["left_children"][0] = 100000
    if change=="feature": tree["split_indices"][0] = 100000
    if change=="array": tree["default_left"].pop()
    path = tmp_path/"bad.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match=message): Forest.load(path)


def test_shape_and_dump_rejected(trained,tmp_path):
    models,_ = trained
    path = models["reg:squarederror"][1]
    lib = compile_model(path,tmp_path/"compiled")
    with pytest.raises(ValueError,match="shape"):
        Predictor(lib).predict(np.zeros((2,5)))
    with pytest.raises(ValueError,match="Calibration"):
        compile_model(path,tmp_path/"bad",calibration=np.zeros((0,6)))
    dumped = tmp_path/"dump.json"
    dumped.write_text("[]")
    with pytest.raises(ValueError,match="save_model"):
        Forest.load(dumped)


def test_scalar_base_and_constant_model(trained,tmp_path):
    models,_ = trained
    doc = json.loads(models["reg:squarederror"][1].read_text())
    doc["learner"]["learner_model_param"]["base_score"] = "3.7E-1"
    model = doc["learner"]["gradient_booster"]["model"]
    model["trees"],model["tree_info"] = [],[]
    path = tmp_path/"constant.json"
    path.write_text(json.dumps(doc))
    lib = compile_model(path,tmp_path/"compiled")
    np.testing.assert_array_equal(Predictor(lib).predict(np.zeros((2,6))),np.full(2,.37,dtype=np.float32))


def test_llvmlite_rejects_size_options(tmp_path):
    with pytest.raises(ValueError, match="require the clang backend"):
        compile_model(tmp_path / "unused.json", tmp_path, backend="llvmlite", optimization="Oz")


@pytest.mark.parametrize("objective", ["reg:squarederror", "binary:logistic"])
def test_machine_outliner(trained,tmp_path,objective):
    models,rows=trained
    booster,path=models[objective]
    lib=compile_model(path,tmp_path,backend="clang",calibration=rows,
                      rank_feature_limit=4,rank_strategy="bucket_split",rank_bucket_bits=16,
                      select_depth=4,compact_leaf_depth=4,tree_block_size=8,machine_outliner=True)
    assert Predictor(lib).metadata["machine_outliner"]
    np.testing.assert_array_equal(Predictor(lib).predict(rows).view(np.uint32),
        booster.predict(xgb.DMatrix(rows),output_margin=True).view(np.uint32))


def test_outliner_requires_clang(tmp_path):
    with pytest.raises(ValueError,match="requires the clang backend"):
        compile_model(tmp_path/"unused.json",tmp_path,machine_outliner=True)


@pytest.mark.parametrize('objective', ['reg:squarederror', 'binary:logistic'])
@pytest.mark.parametrize('backend', ['clang', 'llvmlite'])
@pytest.mark.parametrize('layout', ['wide', 'compact'])
@pytest.mark.parametrize('depth,probability,block,rank', [(2,1.,0,0),(3,.2,5,4),(64,1.,0,4),(3,0.,5,0)])
def test_hybrid_exact_boundaries(trained,tmp_path,objective,backend,layout,depth,probability,block,rank):
    models,calibration=trained
    booster,path=models[objective]
    rng=np.random.default_rng(761)
    rows=[*rng.normal(size=(128,6)).astype(np.float32),np.full(6,np.nan,dtype=np.float32),
          np.zeros(6,dtype=np.float32),np.full(6,-0.,dtype=np.float32)]
    for tree in Forest.load(path).trees:
        for node,left in enumerate(tree.left):
            if left==-1:continue
            threshold=np.float32(tree.value[node])
            for value in (threshold,np.nextafter(threshold,np.float32(-np.inf)),
                          np.nextafter(threshold,np.float32(np.inf)),np.nan,np.inf,-np.inf):
                row=rng.normal(size=6).astype(np.float32)
                row[tree.feature[node]]=value
                rows.append(row)
    rows=np.array(rows,dtype=np.float32);before=rows.copy()
    lib=compile_model(path,tmp_path,backend=backend,calibration=calibration,select_depth=6,
                      select_policy='cost',hybrid_depth=depth,hybrid_max_probability=probability,hybrid_layout=layout,
                      compact_leaf_depth=4,tree_block_size=block,rank_feature_limit=rank,
                      rank_strategy='bucket_split',predicate_hoist_limit=2,preload=True)
    predictor=Predictor(lib)
    assert predictor.metadata['hybrid_table_bytes']==predictor.metadata['hybrid_nodes']*(8 if layout=='compact' else 16)
    # XGBoost DMatrix rejects infinity; use the separately verified generated
    # baseline for those probes and XGBoost directly for all remaining rows.
    finite=~np.isinf(rows).any(axis=1)
    np.testing.assert_array_equal(predictor.predict(rows[finite]).view(np.uint32),
        booster.predict(xgb.DMatrix(rows[finite]),output_margin=True).view(np.uint32))
    baseline=Predictor(compile_model(path,tmp_path/'baseline',backend=backend,select_depth=0))
    np.testing.assert_array_equal(predictor.predict(rows).view(np.uint32),baseline.predict(rows).view(np.uint32))
    np.testing.assert_array_equal(rows.view(np.uint32),before.view(np.uint32))
    if probability==1:
        assert predictor.metadata['hybrid_subtrees']>0
        assert 'hybrid_walk' in (tmp_path/'model.opt.ll').read_text()


@pytest.mark.parametrize('options', [dict(hybrid_layout='unknown'),dict(hybrid_depth=-1),dict(hybrid_depth=65),dict(hybrid_depth=True),
                                    dict(hybrid_depth=1.5),dict(hybrid_max_probability=-.1),
                                    dict(hybrid_max_probability=1.1),dict(hybrid_max_probability=float('nan')),
                                    dict(hybrid_depth=2,hybrid_max_probability=.1)])
def test_hybrid_invalid_options(tmp_path,options):
    with pytest.raises(ValueError,match='hybrid|Hybrid'):
        compile_model(tmp_path/'unused.json',tmp_path,**options)
