"""Exercise compiled lookup tables against independent tree traversal and XGBoost."""
import itertools
import numpy as np
import pytest
import xgboost as xgb

from xgb_latency import Forest, Predictor, compile_model
from xgb_latency.model import Tree


@pytest.mark.parametrize("backend", ["clang", "llvmlite"])
@pytest.mark.parametrize("block", [0, 1])
@pytest.mark.parametrize("repeated", [False, True])
def test_truth_table_edges(tmp_path, monkeypatch, backend, block, repeated):
    # Balanced depth-two tree; repeated predicates use different NaN directions.
    tree = Tree((1, 3, 5, -1, -1, -1, -1), (2, 4, 6, -1, -1, -1, -1),
                (0, 0 if repeated else 1, 0 if repeated else 2, 0, 0, 0, 0),
                (0., 0. if repeated else 1., 0. if repeated else -1.,
                 -0., 1.25, -3.5, 17.),
                (True, False, True, False, False, False, False),
                (2, 1, 1, 0, 0, 0, 0))
    forest = Forest((tree, tree), 3, -0., "reg:squarederror", (3, 4, 1))
    monkeypatch.setattr(Forest, "load", classmethod(lambda cls, path: forest))
    path = tmp_path / "model.json"
    path.write_text("{}")
    axes = []
    for threshold in (0., 1., -1.):
        t = np.float32(threshold)
        axes.append([np.nan, -np.inf, np.inf, -0., 0., t,
                     np.nextafter(t, np.float32(-np.inf)),
                     np.nextafter(t, np.float32(np.inf))])
    rows = np.asarray(list(itertools.product(*axes)), dtype=np.float32)
    expected = []
    for row in rows:
        total = np.float32(forest.base_margin)
        for t in forest.trees:
            node = 0
            while t.left[node] != -1:
                v = row[t.feature[node]]
                left = t.default_left[node] if np.isnan(v) else v < t.value[node]
                node = t.left[node] if left else t.right[node]
            total = np.float32(total + np.float32(t.value[node]))
        expected.append(total)
    lib = compile_model(path, tmp_path / "native", backend=backend, select_depth=2,
                        tree_block_size=block, leaf_table_bits=3)
    predictor = Predictor(lib)
    # Compare bit patterns too, including signed zero and the ordered accumulator.
    np.testing.assert_array_equal(predictor.predict(rows).view(np.uint32),
                                  np.asarray(expected, dtype=np.float32).view(np.uint32))
    assert predictor.metadata["leaf_tables_in_ir"] == 2


@pytest.mark.parametrize("backend", ["clang", "llvmlite"])
@pytest.mark.parametrize("bits", [2, 3, 5])
def test_trained_table_agreement(tmp_path, backend, bits):
    rng = np.random.default_rng(727)
    train = rng.normal(size=(512, 5)).astype(np.float32)
    labels = train[:, 0] * train[:, 1] + train[:, 2]
    train[rng.random(train.shape) < .1] = np.nan
    booster = xgb.train({"max_depth": 4, "nthread": 1, "base_score": .37},
                        xgb.DMatrix(train, label=labels), 8)
    path = tmp_path / "model.json"
    booster.save_model(path)
    forest = Forest.load(path)
    rows = rng.normal(size=(256, 5)).astype(np.float32)
    rows[rng.random(rows.shape) < .15] = np.nan
    probes = [np.full(5, np.nan, dtype=np.float32)]
    for tree in forest.trees:
        for node, left in enumerate(tree.left):
            if left == -1:
                continue
            threshold = np.float32(tree.value[node])
            for v in (threshold, np.nextafter(threshold, np.float32(-np.inf)),
                      np.nextafter(threshold, np.float32(np.inf))):
                row = rng.normal(size=5).astype(np.float32)
                row[tree.feature[node]] = v
                probes.append(row)
    rows = np.vstack([rows, probes])
    lib = compile_model(path, tmp_path / "native", backend=backend, select_depth=4,
                        select_policy="profile", calibration=train, leaf_table_bits=bits,
                        predicate_hoist_limit=4, tree_block_size=3)
    expected = booster.predict(xgb.DMatrix(rows), output_margin=True)
    np.testing.assert_array_equal(Predictor(lib).predict(rows).view(np.uint32),
                                  expected.view(np.uint32))


@pytest.mark.parametrize("bits", [-1, 9, 2.5, "3"])
def test_reject_unbounded_tables(tmp_path, bits):
    with pytest.raises(ValueError, match="leaf_table_bits"):
        compile_model(tmp_path / "unused.json", tmp_path, leaf_table_bits=bits)
