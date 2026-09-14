"""Validate the exact TL2cgen entry points used by the native comparison."""
import numpy as np
import pytest
import treelite
import tl2cgen
import xgboost as xgb

from benchmarks.run import build_tl, check_tl


@pytest.mark.parametrize("quantize",[False,True])
@pytest.mark.parametrize("profiled",[False,True])
@pytest.mark.parametrize("patched",[False,True])
def test_tl_adapters(tmp_path,quantize,profiled,patched):
    rng = np.random.default_rng(52)
    train = rng.normal(size=(256,4)).astype(np.float32)
    label = (train[:,0] + train[:,1] > 0).astype(np.float32)
    train[::7,0] = np.nan
    booster = xgb.train({"objective":"binary:logistic","base_score":.23,
                         "max_depth":3,"nthread":1},xgb.DMatrix(train,label=label),9)
    path = tmp_path/"model.json"
    booster.save_model(path)
    annotation = None
    if profiled:
        annotation = tmp_path/"annotation.json"
        tl2cgen.annotate_branch(treelite.frontend.load_xgboost_model(str(path)),
                               tl2cgen.DMatrix(train),annotation,nthread=1)
    lib,_ = build_tl(path,tmp_path/"compiled",4,annotation=annotation,quantize=quantize,
                    f32_accumulation=patched)
    rows = rng.normal(size=(128,4)).astype(np.float32)
    rows[rng.random(rows.shape)<.15] = np.nan
    rows[0] = np.nan
    expected = booster.predict(xgb.DMatrix(rows),output_margin=True)
    # check_tl verifies both predictions and bitwise preservation of caller input.
    check_tl(lib,rows,expected)
    if not quantize:
        check_tl(lib,rows,expected,prepared=True)
