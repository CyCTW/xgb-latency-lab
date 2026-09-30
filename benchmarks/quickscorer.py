"""Warm single-row comparison of QuickScorer lowering, LLVM prototype and TL2cgen.

Also builds explicitly inexact pairwise-summation diagnostics that bound how
much latency the exact serial float32 accumulation costs. Those engines are
reported separately and must never be selected for deployment.
"""
import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import treelite
import tl2cgen
import xgboost as xgb

from xgb_latency import Forest, Predictor, compile_model
from xgb_latency.quickscorer import compile_quickscorer, plan as qs_plan, table_bytes as qs_table_bytes
from benchmarks.run import build_tl, check_tl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True, help="Held-out rows used for timing")
    p.add_argument("--calibration", required=True, help="Separate rows for branch weights / annotation")
    p.add_argument("--output", required=True)
    p.add_argument("--samples", type=int, default=8192)
    p.add_argument("--rounds", type=int, default=11)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--strides", default="0,1,4,16,64",
                   help="Comma-separated QuickScorer strides (0 = classic loop, 1 = dense); "
                        "STRIDE/N sets rank_linear_max (default 64); STRIDE/Nb uses binary search above N")
    p.add_argument("--max-table-bytes", type=int, default=8 << 20,
                   help="Skip strides whose checkpoint table exceeds this size")
    p.add_argument("--skip-tl", action="store_true")
    args = p.parse_args()
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        p.error("output must be a new or empty directory")
    out.mkdir(parents=True, exist_ok=True)
    model = Path(args.model).resolve()
    rows = np.ascontiguousarray(np.load(args.data, allow_pickle=False), dtype=np.float32)
    calibration = np.ascontiguousarray(np.load(args.calibration, allow_pickle=False), dtype=np.float32)
    forest = Forest.load(model)
    rows.tofile(out / "data.f32")
    booster = xgb.Booster(model_file=str(model))
    booster.set_param({"nthread": 1})
    expected = booster.predict(xgb.DMatrix(rows), output_margin=True)
    entries, builds, reference = [], [], None

    def add(name, lib, exact=True):
        nonlocal reference
        actual = Predictor(lib).predict(rows)
        meta = json.loads((Path(lib).parent / "metadata.json").read_text())
        error = float(np.max(np.abs(actual - expected)))
        if exact:
            np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
            if reference is None:
                reference = actual
            bitwise = bool(np.array_equal(actual.view(np.uint32), reference.view(np.uint32)))
            if not bitwise:
                raise AssertionError(f"{name} is not bitwise equal to the exact LLVM reference")
        else:
            bitwise = bool(np.array_equal(actual.view(np.uint32), reference.view(np.uint32)))
        builds.append({"name": name, "exact": exact, "bitwise_equal_reference": bitwise,
                       "max_abs_error_vs_xgboost": error, **meta})
        entries.append((name, str(lib), "predict_row", "0"))
        print(f"built {name}: max|err|={error:.3g} bitwise={bitwise}", flush=True)

    llvm = [("llvm_select1", dict()),
            ("llvm_select1_profiled_clang", dict(calibration=calibration, backend="clang")),
            ("llvm_interleaved_self_scalar16", dict(traversal_lanes=16, traversal_leaf_layout="self_loop"))]
    for name, options in llvm:
        add(name, compile_model(model, out / name, **options))
    word_bits, _, _, features = qs_plan(forest)
    for spec in args.strides.split(","):
        stride, _, linear = spec.partition("/")
        search = "binary" if linear.endswith("b") else "two_level"
        stride, linear = int(stride), int(linear.rstrip("b") or 64)
        name = f"qs_stride{stride}" + (f"_lin{linear}" if "/" in spec else "") + ("_bin" if search == "binary" else "")
        if stride and qs_table_bytes(features, word_bits, stride) > args.max_table_bytes:
            print(f"skip {name}: checkpoint table exceeds --max-table-bytes", flush=True)
            continue
        add(name, compile_quickscorer(model, out / name, stride=stride, rank_linear_max=linear,
                                             rank_search=search))
    diag = [("diag_llvm_select1_profiled_clang_pairwise",
             lambda d: compile_model(model, d, calibration=calibration, backend="clang",
                                     accumulation_order="pairwise_inexact")),
            ("diag_qs_stride1_pairwise",
             lambda d: compile_quickscorer(model, d, stride=1, rank_linear_max=1 << 20,
                                             summation="pairwise_inexact"))]
    for name, build in diag:
        add(name, build(out / name), exact=False)
    if not args.skip_tl:
        annotation = out / "tl_annotation.json"
        tl2cgen.annotate_branch(treelite.frontend.load_xgboost_model(str(model)),
                               tl2cgen.DMatrix(calibration), annotation, nthread=1)
        for name, profiled in [("tl2cgen_lto", False), ("tl2cgen_profiled_lto", True)]:
            lib, command = build_tl(model, out / name, forest.num_feature,
                                    annotation=annotation if profiled else None, lto=True)
            error = check_tl(lib, rows, expected, prepared=True)
            builds.append({"name": name + "_prepared", "max_abs_error_vs_xgboost": error, "command": command,
                           "library_bytes": lib.stat().st_size})
            entries.append((name + "_prepared", str(lib), "predict", "1"))
    harness = out / "native"
    command = ["clang++", "-O3", "-std=c++17", str(Path(__file__).with_name("native.cc")), "-o", str(harness)]
    if sys.platform != "darwin":
        command.append("-ldl")
    subprocess.run(command, check=True, capture_output=True)
    result = subprocess.run([str(harness), str(out / "data.f32"), str(len(rows)), str(forest.num_feature),
                             str(args.samples), str(args.rounds), str(args.seed),
                             *[v for entry in entries for v in entry]],
                            check=True, capture_output=True, text=True)
    report = {"platform": platform.platform(), "machine": platform.machine(),
              "cpu": next((l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
                           if l.startswith("model name")), None) if Path("/proc/cpuinfo").exists() else None,
              "compiler": subprocess.check_output(["clang", "--version"], text=True).splitlines()[0],
              "model": {"num_trees": len(forest.trees), "num_features": forest.num_feature,
                        "max_depth": max(t.height[0] for t in forest.trees)},
              "samples": args.samples, "rounds": args.rounds, "seed": args.seed,
              "builds": builds, "measurement": json.loads(result.stdout),
              "caveats": ["Warm steady-state single-thread calls; no CPU pinning or frequency isolation",
                          "diag_* engines reassociate the float32 sum and are NOT exact; bound only",
                          "Engines are enumerated, not selected on evaluation rows"]}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n{'engine':44} {'p50':>7} {'p99':>7} {'block ns/row':>13}")
    for e in report["measurement"]["engines"]:
        print(f'{e["name"]:44} {e["p50_ns"]:7.0f} {e["p99_ns"]:7.0f} {e["block_median_ns_per_row"]:13.1f}')
    print(f"\nReport: {out / 'report.json'}")


if __name__ == "__main__":
    main()
