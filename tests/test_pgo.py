"""Exercise real instrumentation, profile use, and deployed native artifacts."""
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pytest
import xgboost as xgb

from benchmarks.pgo import build_pgo
from benchmarks.run import build_tl, check_tl
from xgb_latency import Predictor, compile_model


@pytest.mark.parametrize('family', ['engine', 'tl', 'tl_quantized'])
def test_instrumentation_pgo(tmp_path, family):
    engine = family == 'engine'
    rng = np.random.default_rng(297)
    calibration = rng.normal(size=(128, 4)).astype(np.float32)
    calibration[::7, 1] = np.nan
    booster = xgb.train({'nthread': 1, 'max_depth': 3, 'base_score': .23},
                        xgb.DMatrix(calibration, label=np.nan_to_num(calibration[:, 1])), 4)
    model = tmp_path / 'source.json'
    booster.save_model(model)
    shared = '-dynamiclib' if sys.platform == 'darwin' else '-shared'
    native = '-mcpu=native' if platform.machine() in ('arm64', 'aarch64') else '-march=native'
    if engine:
        lib = compile_model(model, tmp_path/'base', backend='clang', calibration=calibration,
                            tree_block_size=2, rank_feature_limit=4)
        command = ['clang', '-O3', native, '-fPIC', '-fno-fast-math', '-ffp-contract=off',
                   '-x', 'ir', str(lib.parent/'model.ll'), shared, '-o', str(lib)]
    else:
        lib, command = build_tl(model, tmp_path/'base', 4, quantize=family == 'tl_quantized', lto=True,
                                f32_accumulation=True)
    source_bits = lib.read_bytes()
    harness = tmp_path/'native'
    subprocess.run(['clang++', '-O3', '-std=c++17', str(Path(__file__).parents[1]/'benchmarks/native.cc'),
                    '-o', str(harness), *([] if sys.platform == 'darwin' else ['-ldl'])], check=True)
    entry = dict(library=str(lib), symbol='predict_row', prepared=False,
                 family='engine' if engine else 'tl_f32_modified')
    interfaces = [False, True] if family == 'tl' else [False]
    output, metadata = build_pgo(command, entry, tmp_path/'pgo', harness, calibration,
                                 training_interfaces=interfaces)
    assert lib.read_bytes() == source_bits
    assert metadata['pgo'] and metadata['profile_training'] == 'calibration only'
    assert (output.parent/'model.profdata').stat().st_size > 0
    rows = rng.normal(size=(128, 4)).astype(np.float32)
    rows[::5, 0] = np.nan
    rows[0] = np.nan
    expected = booster.predict(xgb.DMatrix(rows), output_margin=True)
    if engine:
        np.testing.assert_array_equal(Predictor(output).predict(rows).view(np.uint32), expected.view(np.uint32))
        linked = output.parent/('relinked' + output.suffix)
        subprocess.run(['clang', shared, str(output.parent/'model.o'), '-o', str(linked)], check=True)
        np.testing.assert_array_equal(Predictor(linked).predict(rows).view(np.uint32), expected.view(np.uint32))
    else:
        check_tl(output, rows, expected)
        if family == 'tl':
            check_tl(output, rows, expected, prepared=True)
