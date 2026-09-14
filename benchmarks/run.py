"""Reproducible synthetic experiment or benchmark a supplied model + .npy rows."""
import argparse
import ctypes
import importlib.metadata
import json
import platform
import re
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import treelite
import tl2cgen
import xgboost as xgb

from xgb_latency import Forest, Predictor, compile_model


def build_tl(model_path, directory, nf, *, annotation=None, quantize=False, lto=False,
             f32_accumulation=False):
    directory.mkdir(parents=True, exist_ok=True)
    model = treelite.frontend.load_xgboost_model(str(model_path))
    params = {"quantize": int(quantize), "parallel_comp": 0}
    if annotation is not None:
        params["annotate_in"] = str(annotation)
    tl2cgen.generate_c_code(model, dirpath=directory, params=params)
    if f32_accumulation:
        # Explicitly modified baseline: round every additive result constant,
        # including the final base score, to a C99 float hexadecimal literal.
        # Leave split thresholds and quantization logic untouched.
        pattern = re.compile(r"(result\[\d+\]\s*\+=\s*)([-+0-9.eE]+)(\s*;)")
        replacements = 0
        for source in sorted(directory.glob("*.c")):
            def replace(match):
                return match[1] + float(np.float32(match[2])).hex() + "f" + match[3]
            code, count = pattern.subn(replace, source.read_text())
            source.write_text(code)
            replacements += count
        if not replacements:
            raise ValueError("No TL2cgen accumulation literals matched; inspect generated code")
        (directory / "f32_patch.json").write_text(json.dumps({
            "modified_baseline": True, "accumulation_constants_replaced": replacements,
            "split_thresholds_modified": False}, indent=2))
    # Entry preparation belongs inside the dense-input comparison. It also makes
    # a fresh mutable copy for TL2cgen's in-place threshold quantization.
    wrapper = '''#include "header.h"
void predict_row(const float *x, float *out) {
  union Entry data[NUM_FEATURE];
  for (int i=0; i<NUM_FEATURE; ++i) {
    if (isnan(x[i])) data[i].missing = -1;
    else data[i].fvalue = x[i];
  }
  *out = 0.0f;
  predict(data, 1, out);
}
'''.replace("NUM_FEATURE", str(nf))
    (directory / "adapter.c").write_text(wrapper)
    lib = directory / ("model.dylib" if sys.platform == "darwin" else "model.so")
    command = ["clang", "-O3", "-mcpu=native" if platform.machine() in ("arm64", "aarch64") else "-march=native",
               "-fno-fast-math", "-ffp-contract=off", "-fPIC",
               *(["-flto"] if lto else []),
               "-dynamiclib" if sys.platform == "darwin" else "-shared",
               *map(str, sorted(directory.glob("*.c"))), "-o", str(lib)]
    subprocess.run(command, check=True, capture_output=True)
    return lib, command


def check_tl(lib, rows, expected, prepared=False):
    dll = ctypes.CDLL(str(lib))
    fn = dll.predict if prepared else dll.predict_row
    ptr = ctypes.POINTER(ctypes.c_float)
    fn.argtypes = [ptr,ctypes.c_int,ptr] if prepared else [ptr,ptr]
    fn.restype = None
    data = rows.copy()
    if prepared:
        data.view(np.int32)[np.isnan(data)] = -1
    original_bits = data.view(np.uint32).copy()
    actual = np.zeros(len(data), dtype=np.float32)
    for i, row in enumerate(data):
        x, y = row.ctypes.data_as(ptr), ctypes.cast(actual.ctypes.data+i*4, ptr)
        if prepared:
            fn(x,1,y)
        else:
            fn(x,y)
    # TL2cgen sums trees then base_score; XGBoost begins at base_score.
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    np.testing.assert_array_equal(data.view(np.uint32), original_bits,
                                  err_msg="Native entry point mutated the caller's input")
    return float(np.max(np.abs(actual-expected)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/synthetic")
    p.add_argument("--model", help="XGBoost save_model JSON; requires --data")
    p.add_argument("--data", help="Held-out dense float32-compatible .npy")
    p.add_argument("--calibration", help="Separate .npy for branch profiling; required with --model")
    p.add_argument("--trees", type=int, default=100)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--features", type=int, default=32)
    p.add_argument("--samples", type=int, default=4096)
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--tl-lto", action="store_true", help="Also test TL2cgen builds with link-time optimization")
    p.add_argument("--clang-control", action="store_true", help="Also compile our IR through the same Clang used for TL2cgen")
    args = p.parse_args()
    if min(args.samples, args.rounds, args.features, args.trees, args.depth) < 1:
        p.error("Counts must be positive")
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    if args.model:
        if not args.data or not args.calibration:
            p.error("--model requires --data and separate --calibration")
        model_path = Path(args.model).resolve()
        rows = np.ascontiguousarray(np.load(args.data, allow_pickle=False), dtype=np.float32)
        calibration = np.ascontiguousarray(np.load(args.calibration, allow_pickle=False), dtype=np.float32)
        booster = xgb.Booster(model_file=str(model_path))
        dataset = "user_supplied"
    else:
        if args.features < 4:
            p.error("Synthetic data requires at least 4 features")
        train = rng.normal(size=(8192,args.features)).astype(np.float32)
        target = (2*train[:,0] + train[:,1]*train[:,2] + np.sin(3*train[:,3])
                  + rng.normal(scale=.1,size=len(train))).astype(np.float32)
        train[rng.random(train.shape)<.03] = np.nan
        booster = xgb.train({"objective":"reg:squarederror", "max_depth":args.depth,
                             "eta":.1, "seed":args.seed, "nthread":1, "tree_method":"hist"},
                            xgb.DMatrix(train,label=target), args.trees)
        model_path = out / "model.json"
        booster.save_model(model_path)
        calibration = rng.normal(size=(4096,args.features)).astype(np.float32)
        rows = rng.normal(size=(4096,args.features)).astype(np.float32)
        for a in (calibration,rows):
            a[rng.random(a.shape)<.03] = np.nan
        dataset = "synthetic_regression_normal_3pct_missing"
    booster.set_param({"nthread":1})
    forest = Forest.load(model_path)
    if rows.ndim != 2 or rows.shape[1] != forest.num_feature or len(rows)==0:
        p.error("Invalid benchmark data shape")
    forest.branch_counts(calibration)  # Validate before building any variants.
    np.save(out / "data.npy",rows)
    np.save(out / "calibration.npy",calibration)
    rows.tofile(out / "data.f32")
    expected = booster.predict(xgb.DMatrix(rows), output_margin=True)
    entries, builds = [], []
    # Every choice is reported; do not select a winner using timed test rows.
    llvm_variants = [(d,p,b,"llvmlite") for d,p,b in [(0,False,False),(1,False,False),
                     (2,False,False),(1,True,False),(1,False,True)]]
    if args.clang_control:
        llvm_variants += [(1,False,False,"clang"),(1,False,True,"clang")]
    for depth, preload, profiled, backend in llvm_variants:
        name = f"llvm_select{depth}" + ("_preload" if preload else "") + ("_profiled" if profiled else "") + ("_clang" if backend=="clang" else "")
        print(f"Building {name}", flush=True)
        lib = compile_model(model_path,out/name,select_depth=depth,preload=preload,
                            calibration=calibration if profiled else None,backend=backend)
        actual = Predictor(lib).predict(rows)
        np.testing.assert_allclose(actual,expected,rtol=2e-6,atol=2e-6)
        meta = json.loads((lib.parent / "metadata.json").read_text())
        builds.append({"name":name, "max_abs_error":float(np.max(np.abs(actual-expected))), **meta})
        entries.append((name,str(lib),"predict_row","0"))
    annotation = out / "tl_annotation.json"
    tl2cgen.annotate_branch(treelite.frontend.load_xgboost_model(str(model_path)),
                           tl2cgen.DMatrix(calibration),annotation,nthread=1)
    variants = [(p,q,False) for p,q in [(False,False),(True,False),(False,True),(True,True)]]
    if args.tl_lto:
        variants += [(p,q,True) for p,q in [(False,False),(True,False),(False,True),(True,True)]]
    for profiled, quantize, lto in variants:
        name = "tl2cgen" + ("_profiled" if profiled else "") + ("_quantized" if quantize else "") + ("_lto" if lto else "")
        print(f"Building {name}",flush=True)
        started = time.perf_counter()
        lib,command = build_tl(model_path,out/name,forest.num_feature,
                              annotation=annotation if profiled else None,quantize=quantize,lto=lto)
        compile_seconds = time.perf_counter()-started
        error = check_tl(lib,rows,expected)
        builds.append({"name":name,"max_abs_error":error,"command":command,
                       "compile_seconds":compile_seconds,"library_bytes":lib.stat().st_size})
        entries.append((name+"_dense",str(lib),"predict_row","0"))
        if not quantize:
            check_tl(lib,rows,expected,prepared=True)
            entries.append((name+"_prepared",str(lib),"predict","1"))
    harness = out / "native"
    source = Path(__file__).with_name("native.cc")
    command = ["clang++","-O3","-std=c++17",str(source),"-o",str(harness)]
    if sys.platform != "darwin":
        command.append("-ldl")
    subprocess.run(command,check=True,capture_output=True)
    result = subprocess.run([str(harness),str(out/"data.f32"),str(len(rows)),str(forest.num_feature),
                             str(args.samples),str(args.rounds),str(args.seed),
                             *[v for entry in entries for v in entry]],
                            check=True,capture_output=True,text=True)
    report = {"dataset":dataset,"platform":platform.platform(),"machine":platform.machine(),
              "compiler":subprocess.check_output(["clang","--version"],text=True),
              "versions":{p:importlib.metadata.version(p) for p in
                          ["numpy","xgboost","llvmlite","treelite","tl2cgen"]},
              "model":{"num_trees":len(forest.trees),"num_features":forest.num_feature,
                       "max_depth":max((t.height[0] for t in forest.trees),default=0)},
              "seed":args.seed,"rounds":args.rounds,"builds":builds,
              "tl_lto_included":args.tl_lto,
              "clang_control_included":args.clang_control,
              "measurement":json.loads(result.stdout),
              "caveats":["Warm steady-state single-thread calls; not request/service latency",
                         "Per-call percentiles include timer and indirect-call overhead",
                         "Block measurements include loop/index/checksum overhead",
                         "Prepared TL2cgen excludes Entry conversion; dense includes it",
                         "No CPU affinity/frequency isolation on this run",
                         "LLVM and Clang may bundle different LLVM versions",
                         "No inference from this synthetic workload to other models or CPUs"]}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("\nengine                                      p50 ns     p99 ns    block ns/row")
    for e in report["measurement"]["engines"]:
        print(f'{e["name"]:42} {e["p50_ns"]:9.1f} {e["p99_ns"]:10.1f} {e["block_median_ns_per_row"]:15.1f}')
    print(f"\nReport: {out/'report.json'}")


if __name__ == "__main__":
    main()
