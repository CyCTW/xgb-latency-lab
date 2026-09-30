import json
import platform

import numpy as np
import pytest
import xgboost as xgb

from xgb_latency import Forest, Predictor, compile_model
from xgb_latency.blockmix import compile_blockmix, compile_spec, equal_blocks, parse_spec
from xgb_latency.direct import compile_direct
from xgb_latency.packed import LAYOUTS, compile_packed
from xgb_latency.rapidscorer import FULL, _epitome, compile_rapidscorer
from xgb_latency.tiled import compile_tiled, lut, schedule
from xgb_latency.vpred import compile_vpred


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    root = tmp_path_factory.mktemp("gen-models")
    rng = np.random.default_rng(31)
    x = rng.normal(size=(2048, 6)).astype(np.float32)
    y = (x[:, 0] + .5 * x[:, 1] * x[:, 2] + np.sin(2 * x[:, 3])).astype(np.float32)
    x[rng.random(x.shape) < .1] = np.nan
    calibration = rng.normal(size=(512, 6)).astype(np.float32)
    calibration[rng.random(calibration.shape) < .1] = np.nan
    out = {}
    for name, params, trees in [
        ("reg_d3", dict(objective="reg:squarederror", max_depth=3), 11),
        ("bin_d6", dict(objective="binary:logistic", max_depth=6), 7),
        # Uneven heights and more than 64 leaves per tree.
        ("lossguide", dict(objective="reg:squarederror", grow_policy="lossguide", max_leaves=100,
                           max_depth=12, min_child_weight=0), 5),
    ]:
        labels = (y > 0).astype(np.float32) if "logistic" in params["objective"] else y
        booster = xgb.train({**params, "nthread": 1, "seed": 31}, xgb.DMatrix(x, label=labels), trees)
        path = root / f"{name}.json"
        booster.save_model(path)
        out[name] = (booster, path)
    return out, calibration


def probe_rows(path, seed=9):
    rng = np.random.default_rng(seed)
    rows = rng.normal(size=(128, 6)).astype(np.float32)
    rows[rng.random(rows.shape) < .15] = np.nan
    rows[0] = np.nan
    rows[1] = np.inf
    rows[2] = -np.inf
    rows[3] = -0.0
    extra = []
    for tree in Forest.load(path).trees:
        for node, left in enumerate(tree.left):
            if left != -1:
                t = np.float32(tree.value[node])
                for value in (t, np.nextafter(t, np.float32(-np.inf)), np.nextafter(t, np.float32(np.inf))):
                    row = rng.normal(size=6).astype(np.float32)
                    row[tree.feature[node]] = value
                    extra.append(row)
    return np.vstack([rows, *extra]) if extra else rows


@pytest.fixture(scope="module")
def references(models, tmp_path_factory):
    trained, _ = models
    root = tmp_path_factory.mktemp("gen-ref")
    refs = {}
    for name, (booster, path) in trained.items():
        rows = probe_rows(path)
        refs[name] = (rows, Predictor(compile_model(path, root / name)).predict(rows))
        finite = ~np.isinf(rows).any(axis=1)
        np.testing.assert_allclose(refs[name][1][finite], booster.predict(xgb.DMatrix(rows[finite]), output_margin=True),
                                   rtol=1e-6, atol=1e-6)
    return refs


def assert_bitwise(lib, rows, reference):
    np.testing.assert_array_equal(Predictor(lib).predict(rows).view(np.uint32), reference.view(np.uint32))
    assert json.loads((lib.parent / "metadata.json").read_text())["exact_accumulation_order"]


NAMES = ["reg_d3", "bin_d6", "lossguide"]


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("lanes,layout,grouped", [(1, "tree", True), (4, "level", True), (8, "tree", False),
                                                  (16, "level", False), (32, "tree", True)])
def test_vpred(models, references, tmp_path, name, lanes, layout, grouped):
    path = models[0][name][1]
    rows, ref = references[name]
    assert_bitwise(compile_vpred(path, tmp_path, lanes=lanes, layout=layout, group_by_height=grouped), rows, ref)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("lanes", [1, 8])
def test_packed(models, references, tmp_path, name, layout, lanes):
    (trained, calibration) = models
    rows, ref = references[name]
    lib = compile_packed(trained[name][1], tmp_path, lanes=lanes, layout=layout, calibration=calibration,
                         group_by_height=lanes == 8)
    assert_bitwise(lib, rows, ref)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("budget", [0, 4096, 1 << 30])
def test_rapidscorer(models, references, tmp_path, name, budget):
    rows, ref = references[name]
    lib = compile_rapidscorer(models[0][name][1], tmp_path, dense_budget_bytes=budget)
    assert_bitwise(lib, rows, ref)
    meta = json.loads((lib.parent / "metadata.json").read_text())
    assert meta["merged_nodes"] <= meta["splits"]
    if name == "lossguide":
        assert meta["max_leaves_per_tree"] > 64 and meta["max_words_per_tree"] >= 2


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("depth,profiled", [(0, False), (0, True), (2, True)])
def test_direct(models, references, tmp_path, name, depth, profiled):
    trained, calibration = models
    rows, ref = references[name]
    lib = compile_direct(trained[name][1], tmp_path, select_depth=depth, calibration=calibration if profiled else None)
    assert_bitwise(lib, rows, ref)


@pytest.mark.parametrize("name", NAMES)
def test_blockmix_and_specs(models, references, tmp_path, name):
    trained, calibration = models
    path = trained[name][1]
    rows, ref = references[name]
    specs = ["vpred:lanes=4,layout=level", "rs", "packed:lanes=2,layout=forest", "direct:select_depth=1"]
    if name != "lossguide":
        specs.insert(0, "qs:stride=1")
    n = len(Forest.load(path).trees)
    ranges = equal_blocks(n, min(len(specs), n))
    blocks = [(a, b, specs[i % len(specs)]) for i, (a, b) in enumerate(ranges)]
    assert_bitwise(compile_blockmix(path, tmp_path / "mix", blocks, calibration=calibration), rows, ref)
    for spec in specs:
        assert_bitwise(compile_spec(path, tmp_path / spec.replace(":", "_").replace(",", "_").replace("=", ""),
                                    spec, calibration=calibration), rows, ref)


def test_sub_model_matches_iteration_range(models, tmp_path):
    booster, path = models[0]["reg_d3"]
    rows = probe_rows(path)
    rows = rows[~np.isinf(rows).any(axis=1)]
    lib = compile_spec(path, tmp_path, "vpred:lanes=4", start=3, stop=8)
    expected = booster.predict(xgb.DMatrix(rows), output_margin=True, iteration_range=(3, 8))
    np.testing.assert_allclose(Predictor(lib).predict(rows), expected, rtol=1e-5, atol=1e-5)


def test_epitome_masks():
    wa, wb, ma, mb = _epitome(3, 7)
    assert (wa, wb) == (0, 0) and ma == mb == FULL & ~(0b1111 << 3)
    wa, wb, ma, mb = _epitome(60, 130)
    assert (wa, wb) == (0, 2) and ma == (1 << 60) - 1 and mb == FULL & ~0b11  # bits 128, 129
    assert _epitome(64, 128) == (1, 1, 0, 0)


@pytest.mark.parametrize("call,match", [
    (lambda p, t: compile_vpred(p, t, lanes=3), "lanes"),
    (lambda p, t: compile_vpred(p, t, layout="diag"), "layout"),
    (lambda p, t: compile_packed(p, t, layout="hot_dfs"), "calibration"),
    (lambda p, t: compile_packed(p, t, layout="zigzag"), "layout"),
    (lambda p, t: compile_rapidscorer(p, t, dense_budget_bytes=-1), "dense_budget_bytes"),
    (lambda p, t: compile_direct(p, t, select_depth=9), "select_depth"),
    (lambda p, t: compile_blockmix(p, t, [(0, 3, "rs")]), "blocks"),
    (lambda p, t: compile_blockmix(p, t, [(0, 5, "rs"), (6, 11, "rs")]), "blocks"),
    (lambda p, t: compile_spec(p, t, "magic"), "unknown strategy"),
    (lambda p, t: compile_spec(p, t, "rs:dense"), "bad option"),
])
def test_rejects_invalid(models, tmp_path, call, match):
    with pytest.raises(ValueError, match=match):
        call(models[0]["reg_d3"][1], tmp_path)


def test_parse_spec():
    assert parse_spec("vpred:lanes=8,layout=level,group_by_height=false") == (
        "vpred", {"lanes": 8, "layout": "level", "group_by_height": False})
    assert parse_spec("rs") == ("rs", {})


VECTOR_OK = platform.machine() in ("x86_64", "AMD64")


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("tile_levels", [2, 3])
@pytest.mark.parametrize("mode", ["scalar", "gather", "insert"])
@pytest.mark.parametrize("lanes,grouped", [(1, True), (8, False)])
def test_tiled(models, references, tmp_path, name, tile_levels, mode, lanes, grouped):
    if mode != "scalar" and not VECTOR_OK:
        pytest.skip("vector tile modes need x86-64 AVX2")
    rows, ref = references[name]
    lib = compile_tiled(models[0][name][1], tmp_path, lanes=lanes, tile_levels=tile_levels, mode=mode,
                        group_by_height=grouped)
    assert_bitwise(lib, rows, ref)


def test_tile_schedule_and_lut():
    assert schedule(8, 3) == [3, 3, 2] and schedule(6, 2) == [2, 2, 2] and schedule(0, 3) == []
    assert lut(1) == [0, 1]
    # Bit 0 = root goes right; bits 1/2 = its left/right child go right.
    assert lut(2) == [0, 2, 1, 2, 0, 3, 1, 3]
    table = lut(3)
    assert len(table) == 128 and sorted(set(table)) == list(range(8))
    assert table[0] == 0 and table[0b1000101] == 7  # right, right, right: nodes 0, 2, 6


def test_tiled_in_blockmix(models, references, tmp_path):
    trained, calibration = models
    rows, ref = references["bin_d6"]
    path = trained["bin_d6"][1]
    blocks = [(0, 3, "tiled:tile_levels=2,mode=scalar"), (3, 7, "tiled:lanes=2,tile_levels=3")]
    assert_bitwise(compile_blockmix(path, tmp_path, blocks, calibration=calibration), rows, ref)


@pytest.mark.parametrize("kwargs,match", [(dict(tile_levels=4), "tile_levels"), (dict(mode="avx9"), "mode"),
                                          (dict(lanes=5), "lanes")])
def test_tiled_rejects_invalid(models, tmp_path, kwargs, match):
    with pytest.raises(ValueError, match=match):
        compile_tiled(models[0]["reg_d3"][1], tmp_path, **kwargs)
