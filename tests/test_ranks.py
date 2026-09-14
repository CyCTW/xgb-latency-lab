import ctypes

import numpy as np
import pytest
from llvmlite import binding as llvm, ir

from xgb_latency import compile_model
from xgb_latency.rank import emit_ranks


@pytest.mark.parametrize("count", [1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 255, 256])
@pytest.mark.parametrize("strategy,bucket_bits", [("binary",12),("eytzinger",12),("simd",12),
                                                     ("bucket",8),("bucket",12),("bucket",16),
                                                     ("bucket_split",8),("bucket_split",12),("bucket_split",16)])
@pytest.mark.parametrize("distribution", ["linear", "bit_patterns"])
def test_rank_encoder_exact_edges(count, strategy, bucket_bits, distribution):
    thresholds = np.linspace(-1., 1., count, dtype=np.float32)
    if distribution == "bit_patterns":
        rng = np.random.default_rng(814+count)
        random_bits = rng.integers(0, 2**32, size=count*2, dtype=np.uint32).view(np.float32)
        edges = np.array([0, 0x80000000, 1, 0x80000001, 0x00800000, 0x80800000,
                          0x7f7fffff, 0xff7fffff], dtype=np.uint32).view(np.float32)
        thresholds = np.unique(np.concatenate([random_bits[np.isfinite(random_bits)][:count], edges]))
    with np.errstate(over="ignore"):
        below = np.nextafter(thresholds, np.float32(-np.inf))
        above = np.nextafter(thresholds, np.float32(np.inf))
    probes = np.concatenate([
        thresholds,
        below,
        above,
        np.array([-np.inf, np.inf, -0., 0.], dtype=np.float32),
        np.array([0x7fc00000, 0xffc00001, 0x7f800001], dtype=np.uint32).view(np.float32),
    ])
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    tm = llvm.Target.from_default_triple().create_target_machine()
    module = ir.Module(name="test_rank")
    module.triple = llvm.get_default_triple()
    module.data_layout = str(tm.target_data)
    f32, i32 = ir.FloatType(), ir.IntType(32)
    fn = ir.Function(module, ir.FunctionType(i32, [f32.as_pointer()]), name="rank")
    b = ir.IRBuilder(fn.append_basic_block("entry"))
    b.ret(emit_ranks(b, fn.args[0], {0: tuple(map(float, thresholds))}, strategy, bucket_bits)[0])
    mod = llvm.parse_assembly(str(module))
    mod.verify()
    with llvm.create_pipeline_tuning_options(speed_level=3) as pto:
        with llvm.create_pass_builder(tm, pto) as pb:
            with pb.getModulePassManager() as pm:
                pm.run(mod, pb)
    with llvm.create_mcjit_compiler(mod, tm) as engine:
        engine.finalize_object()
        function = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(ctypes.c_float))(
            engine.get_function_address("rank"))
        actual = [function(probes[i:i+1].ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
                  for i in range(len(probes))]
    expected = np.searchsorted(thresholds, probes, side="right").astype(np.int32)
    expected[np.isnan(probes)] = -1
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("limit", [-1, 257, 2.5, "3"])
def test_invalid_rank_limit(tmp_path, limit):
    with pytest.raises(ValueError, match="rank_feature_limit"):
        compile_model(tmp_path / "unused.json", tmp_path, rank_feature_limit=limit)


def test_rank_requires_calibration(tmp_path):
    with pytest.raises(ValueError, match="calibration"):
        compile_model(tmp_path / "unused.json", tmp_path, rank_feature_limit=1)
