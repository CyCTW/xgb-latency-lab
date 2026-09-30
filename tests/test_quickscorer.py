import json

import numpy as np
import pytest
import xgboost as xgb

from xgb_latency import Forest, Predictor, compile_model
from xgb_latency.quickscorer import compile_quickscorer, generate_c


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    root = tmp_path_factory.mktemp("qs-models")
    rng = np.random.default_rng(23)
    x = rng.normal(size=(1024, 7)).astype(np.float32)
    y = (x[:, 0] + .5 * x[:, 1] * x[:, 2] + np.sin(2 * x[:, 3])).astype(np.float32)
    x[rng.random(x.shape) < .1] = np.nan
    out = {}
    for name, objective, depth, trees in [("reg_d3", "reg:squarederror", 3, 15),
                                           ("bin_d6", "binary:logistic", 6, 12),
                                           ("abs_d2", "reg:absoluteerror", 2, 9)]:
        labels = (y > 0).astype(np.float32) if "logistic" in objective else y
        booster = xgb.train({"objective": objective, "max_depth": depth, "nthread": 1, "seed": 23},
                            xgb.DMatrix(x, label=labels), trees)
        path = root / f"{name}.json"
        booster.save_model(path)
        out[name] = (booster, path)
    return out


def probe_rows(path, nf, seed=5):
    rng = np.random.default_rng(seed)
    rows = rng.normal(size=(192, nf)).astype(np.float32)
    rows[rng.random(rows.shape) < .15] = np.nan
    rows[0] = np.nan
    rows[1] = np.inf
    rows[2] = -np.inf
    rows[3] = -0.0
    for tree in Forest.load(path).trees:
        for node, left in enumerate(tree.left):
            if left == -1:
                continue
            threshold = np.float32(tree.value[node])
            for value in (threshold, np.nextafter(threshold, np.float32(-np.inf)),
                          np.nextafter(threshold, np.float32(np.inf))):
                row = rng.normal(size=nf).astype(np.float32)
                row[tree.feature[node]] = value
                rows = np.vstack([rows, row])
    return rows


@pytest.mark.parametrize("name", ["reg_d3", "bin_d6", "abs_d2"])
@pytest.mark.parametrize("stride,linear,search", [(0, 64, "two_level"), (1, 0, "two_level"), (1, 4, "binary"),
                                                  (1, 1 << 20, "two_level"), (2, 64, "two_level"),
                                                  (8, 0, "two_level"), (8, 3, "binary"), (1024, 64, "two_level")])
def test_bitwise_equal_to_llvm_prototype(models, tmp_path, name, stride, linear, search):
    booster, path = models[name]
    rows = probe_rows(path, 7)
    reference = Predictor(compile_model(path, tmp_path / "llvm")).predict(rows)
    lib = compile_quickscorer(path, tmp_path / "qs", stride=stride, rank_linear_max=linear, rank_search=search)
    actual = Predictor(lib).predict(rows)
    np.testing.assert_array_equal(actual.view(np.uint32), reference.view(np.uint32))
    finite = ~np.isinf(rows).any(axis=1)  # XGBoost rejects infinite inputs.
    expected = booster.predict(xgb.DMatrix(rows[finite]), output_margin=True)
    np.testing.assert_allclose(actual[finite], expected, rtol=1e-6, atol=1e-6)
    meta = json.loads((lib.parent / "metadata.json").read_text())
    assert meta["exact_accumulation_order"] and meta["stride"] == stride
    assert meta["word_bits"] == (64 if name == "bin_d6" else 8)


def test_budget_selects_smallest_fitting_stride(models):
    _, path = models["bin_d6"]
    forest = Forest.load(path)
    _, dense = generate_c(forest, stride=1)
    total = dense["checkpoint_table_bytes"]
    _, unlimited = generate_c(forest, dense_budget_bytes=1 << 30)
    assert unlimited["stride"] == 1
    _, tight = generate_c(forest, dense_budget_bytes=total // 4)
    assert tight["stride"] > 1 and tight["checkpoint_table_bytes"] <= total // 4


def test_pairwise_diagnostic_is_flagged(models, tmp_path):
    booster, path = models["reg_d3"]
    rows = probe_rows(path, 7)
    lib = compile_quickscorer(path, tmp_path, stride=1, summation="pairwise_inexact")
    meta = json.loads((lib.parent / "metadata.json").read_text())
    assert meta["summation"] == "pairwise_inexact" and not meta["exact_accumulation_order"]
    rows = rows[~np.isinf(rows).any(axis=1)]
    expected = booster.predict(xgb.DMatrix(rows), output_margin=True)
    np.testing.assert_allclose(Predictor(lib).predict(rows), expected, rtol=1e-5, atol=1e-5)


def test_llvm_pairwise_diagnostic(models, tmp_path):
    booster, path = models["reg_d3"]
    rows = probe_rows(path, 7)
    lib = compile_model(path, tmp_path, accumulation_order="pairwise_inexact")
    meta = json.loads((lib.parent / "metadata.json").read_text())
    assert meta["accumulation_order"] == "pairwise_inexact" and not meta["exact_accumulation_order"]
    rows = rows[~np.isinf(rows).any(axis=1)]
    expected = booster.predict(xgb.DMatrix(rows), output_margin=True)
    np.testing.assert_allclose(Predictor(lib).predict(rows), expected, rtol=1e-5, atol=1e-5)
    with pytest.raises(ValueError, match="accumulation_order"):
        compile_model(path, tmp_path / "bad", accumulation_order="fast")
    with pytest.raises(ValueError, match="pairwise_inexact"):
        compile_model(path, tmp_path / "bad2", accumulation_order="pairwise_inexact", tree_block_size=4)


def test_constant_model_without_splits(tmp_path):
    x = np.zeros((64, 3), dtype=np.float32)
    booster = xgb.train({"objective": "reg:squarederror", "nthread": 1}, xgb.DMatrix(x, label=np.ones(64)), 3)
    path = tmp_path / "constant.json"
    booster.save_model(path)
    rows = np.array([[0, np.nan, 1], [5, 5, 5]], dtype=np.float32)
    for stride in (0, 1, 4):
        actual = Predictor(compile_quickscorer(path, tmp_path / f"qs{stride}", stride=stride)).predict(rows)
        np.testing.assert_allclose(actual, booster.predict(xgb.DMatrix(rows), output_margin=True), rtol=1e-6)


def test_rejects_more_than_64_leaves(tmp_path):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(4096, 4)).astype(np.float32)
    booster = xgb.train({"objective": "reg:squarederror", "max_depth": 8, "min_child_weight": 0,
                         "nthread": 1, "seed": 3}, xgb.DMatrix(x, label=rng.normal(size=4096)), 1)
    path = tmp_path / "wide.json"
    booster.save_model(path)
    assert max(sum(c == -1 for c in t.left) for t in Forest.load(path).trees) > 64
    with pytest.raises(ValueError, match="64 leaves"):
        compile_quickscorer(path, tmp_path / "qs")


@pytest.mark.parametrize("kwargs,match", [(dict(stride=3), "stride"), (dict(stride=-1), "stride"),
                                          (dict(rank_search="linear"), "rank_search"),
                                          (dict(summation="kahan"), "summation"),
                                          (dict(rank_linear_max=-1), "rank_linear_max"),
                                          (dict(dense_budget_bytes=-1), "dense_budget_bytes")])
def test_rejects_invalid_options(models, kwargs, match):
    _, path = models["reg_d3"]
    with pytest.raises(ValueError, match=match):
        generate_c(Forest.load(path), **kwargs)
