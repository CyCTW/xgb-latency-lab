"""Tune compiler choices before opening final evaluation data.

Each run uses a fresh artifact directory; no previously loaded binary is replaced.
The explicitly modified TL2cgen float32 baseline is reported as its own family.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import numpy as np
import tl2cgen
import treelite
import xgboost as xgb

from xgb_latency import Forest, Predictor, compile_model
from .run import build_tl, check_tl
from .pgo import build_pgo


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rows(path, nf):
    rows = np.ascontiguousarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] != nf or not len(rows):
        raise ValueError("Data must be a nonempty (rows, num_feature) matrix")
    return rows


def choose(measurement, entries, metric):
    scores = {e["name"]: e[metric] for e in measurement["engines"]}
    selected = {}
    for family in sorted({e["family"] for e in entries}):
        candidates = [e for e in entries if e["family"] == family]
        selected[family] = min(candidates, key=lambda e: (scores[e["name"]], e["name"]))
    return selected


def measure(harness, rows, entries, output, *, samples, rounds, seed):
    data = output.with_suffix(".f32")
    rows.tofile(data)
    arguments = [str(harness), str(data), str(len(rows)), str(rows.shape[1]),
                 str(samples), str(rounds), str(seed)]
    for e in entries:
        arguments.extend([e["name"],e["library"],e["symbol"],str(int(e["prepared"]))])
    run = subprocess.run(arguments,check=True,capture_output=True,text=True)
    result = json.loads(run.stdout)
    output.write_text(json.dumps(result,indent=2)+"\n")
    return result


def validate(booster, rows, entries):
    expected = booster.predict(xgb.DMatrix(rows),output_margin=True)
    errors = {}
    for e in entries:
        if e["family"] == "engine":
            actual = Predictor(e["library"]).predict(rows)
            np.testing.assert_allclose(actual,expected,rtol=2e-6,atol=2e-6)
            errors[e["name"]] = float(np.max(np.abs(actual-expected)))
        else:
            errors[e["name"]] = check_tl(e["library"],rows,expected,prepared=e["prepared"])
    return errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model","calibration","tuning","evaluation","output"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--samples",type=int,default=8192)
    p.add_argument("--rounds",type=int,default=11)
    p.add_argument("--seed",type=int,default=8128)
    p.add_argument("--preset",choices=["initial","expanded","tables","cost","ranks","explore"],default="initial")
    p.add_argument("--pgo-training", choices=["dense", "interfaces"], default="interfaces",
                   help="PGO calibration entry points; dense reproduces the initial exploration")
    p.add_argument("--metric",choices=["block_median_ns_per_row","p50_ns","p99_ns"],
                   default="block_median_ns_per_row")
    args = p.parse_args()
    if min(args.samples,args.rounds) < 1:
        p.error("samples and rounds must be positive")
    paths = [Path(getattr(args,n)).resolve() for n in ("calibration","tuning","evaluation")]
    if len(set(paths)) != 3:
        p.error("calibration, tuning and evaluation must be separate files")
    model_path = Path(args.model).resolve()
    forest = Forest.load(model_path)
    calibration, tuning = (load_rows(path,forest.num_feature) for path in paths[:2])
    if fingerprint(paths[0]) == fingerprint(paths[1]):
        p.error("Calibration and tuning files contain identical data")
    # Evaluation path has not been read, hashed, or opened at this point.
    out = Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    booster = xgb.Booster(model_file=str(model_path))
    booster.set_param({"nthread":1})
    harness = out/"native"
    command = ["clang++","-O3","-std=c++17",str(Path(__file__).with_name("native.cc")),"-o",str(harness)]
    if sys.platform != "darwin":
        command.append("-ldl")
    subprocess.run(command,check=True,capture_output=True)
    entries, builds = [], []
    pgo_jobs = []
    candidates = [
        ("clang_d1", "clang", 1, 0, "height"),
        ("clang_d2", "clang", 2, 0, "height"),
        ("clang_d3", "clang", 3, 0, "height"),
        ("clang_d2_adaptive", "clang", 2, 0, "profile"),
        ("clang_d3_adaptive", "clang", 3, 0, "profile"),
        ("clang_d1_block16", "clang", 1, 16, "height"),
        ("clang_d1_block32", "clang", 1, 32, "height"),
        ("clang_d1_block64", "clang", 1, 64, "height"),
        ("clang_d2_block32", "clang", 2, 32, "height"),
        ("clang_d2_block64", "clang", 2, 64, "height"),
        ("llvm_d1", "llvmlite", 1, 0, "height"),
        ("llvm_d1_block32", "llvmlite", 1, 32, "height"),
    ]
    extra_options = {}
    reference_name = "clang_d1"
    if args.preset == "expanded":
        reference_name = "clang_d3_adaptive"
        candidates = [
            ("clang_d3_adaptive","clang",3,0,"profile"),
            ("clang_d4_adaptive","clang",4,0,"profile"),
            ("clang_d5_adaptive","clang",5,0,"profile"),
            ("clang_d4","clang",4,0,"height"),
            ("clang_d3_preload","clang",3,0,"profile"),
            ("clang_d3_hoist4","clang",3,0,"profile"),
            ("clang_d3_hoist8","clang",3,0,"profile"),
            ("clang_d3_hoist16","clang",3,0,"profile"),
            ("clang_d3_hoist32","clang",3,0,"profile"),
            ("clang_d4_hoist8","clang",4,0,"profile"),
            ("clang_d3_block32","clang",3,32,"profile"),
            ("llvm_d3_adaptive","llvmlite",3,0,"profile"),
        ]
        extra_options = {f"clang_d3_hoist{n}":dict(predicate_hoist_limit=n) for n in (4,8,16,32)}
        extra_options["clang_d4_hoist8"] = dict(predicate_hoist_limit=8)
        extra_options["clang_d3_preload"] = dict(preload=True)
    if args.preset == "tables":
        reference_name = "clang_d4_adaptive"
        candidates = [
            ("clang_d4_adaptive", "clang", 4, 0, "profile"),
            ("clang_d1_block32", "clang", 1, 32, "height"),
        ]
        for backend, prefix in (("clang", "clang"), ("llvmlite", "llvm")):
            for depth, bits, policy in ((2, 3, "height"), (3, 3, "profile"),
                                         (4, 3, "profile"), (4, 5, "profile")):
                name = f"{prefix}_d{depth}_table{bits}_{policy}"
                candidates.append((name, backend, depth, 0, policy))
                extra_options[name] = dict(leaf_table_bits=bits)
    if args.preset == "cost":
        reference_name = "clang_d4_adaptive"
        candidates = [
            ("clang_d4_adaptive", "clang", 4, 0, "profile"),
            ("clang_d1_block32", "clang", 1, 32, "height"),
        ]
        for penalty in (2, 4, 8, 16):
            name = f"clang_cost{penalty}"
            candidates.append((name, "clang", 6, 0, "cost"))
            extra_options[name] = dict(select_branch_penalty=penalty)
        candidates.extend([("llvm_cost4", "llvmlite", 6, 0, "cost"),
                           ("clang_cost4_block32", "clang", 6, 32, "cost")])
    if args.preset == "ranks":
        reference_name = "clang_d4_adaptive"
        candidates = [("clang_d4_adaptive", "clang", 4, 0, "profile"),
                      ("clang_cost4_block32", "clang", 6, 32, "cost")]
        for limit in (2, 4, 8, 32):
            name = f"clang_d4_rank{limit}"
            candidates.append((name, "clang", 4, 0, "profile"))
            extra_options[name] = dict(rank_feature_limit=limit)
        candidates.extend([("clang_cost4_block32_rank4", "clang", 6, 32, "cost"),
                           ("llvm_d4_rank4", "llvmlite", 4, 0, "profile")])
        extra_options["clang_cost4_block32_rank4"] = dict(rank_feature_limit=4)
        extra_options["llvm_d4_rank4"] = dict(rank_feature_limit=4)
    if args.preset == "explore":
        reference_name = "clang_cost4_block32_rank4"
        candidates = []
        for base, depth, block, policy, rank in (
            ("clang_cost4_block32_rank4", 6, 32, "cost", 4),
            ("clang_d4_rank32", 4, 0, "profile", 32),
        ):
            variants = [("", {}), ("_eytzinger", dict(rank_strategy="eytzinger")),
                        ("_simd", dict(rank_strategy="simd")),
                        ("_compact2", dict(compact_leaf_depth=2)),
                        ("_compact4", dict(compact_leaf_depth=4)),
                        ("_batch4", dict(accumulation_batch=4)),
                        ("_batch16", dict(accumulation_batch=16)),
                        ("_O2", dict(optimization="O2")),
                        ("_Os", dict(optimization="Os")), ("_Oz", dict(optimization="Oz"))]
            for suffix, options in variants:
                name = base + suffix
                candidates.append((name, "clang", depth, block, policy))
                extra_options[name] = dict(rank_feature_limit=rank, **options)
    for name, backend, depth, block, policy in candidates:
        print(f"Building {name}",flush=True)
        library = compile_model(model_path,out/name,backend=backend,select_depth=depth,
                                tree_block_size=block,select_policy=policy,calibration=calibration,
                                **extra_options.get(name,{}))
        entries.append(dict(name=name,family="engine",library=str(library),symbol="predict_row",prepared=False))
        builds.append(dict(name=name,**json.loads((library.parent/"metadata.json").read_text())))
        if args.preset == "explore" and name in ("clang_cost4_block32_rank4", "clang_d4_rank32"):
            native_flag = "-mcpu=native" if platform.machine() in ("arm64", "aarch64") else "-march=native"
            command = ["clang", "-O3", native_flag, "-fPIC", "-fno-fast-math", "-ffp-contract=off",
                       "-x", "ir", str(library.parent/"model.ll"),
                       "-dynamiclib" if sys.platform == "darwin" else "-shared", "-o", str(library)]
            pgo_jobs.append((name, command, entries[-1], [False]))
    annotation = out/"annotation.json"
    tl2cgen.annotate_branch(treelite.frontend.load_xgboost_model(str(model_path)),
                           tl2cgen.DMatrix(calibration),annotation,nthread=1)
    for patched in (False,True):
        for quantized in (False,True):
            # Include no-annotation and annotated options for both families.
            for profiled in (False,True):
                family = "tl_f32_modified" if patched else "tl_stock"
                name = family + ("_q" if quantized else "") + ("_profiled" if profiled else "")
                print(f"Building {name}",flush=True)
                library,command = build_tl(model_path,out/name,forest.num_feature,
                    annotation=annotation if profiled else None,quantize=quantized,lto=True,
                    f32_accumulation=patched)
                builds.append(dict(name=name,command=command,library_bytes=library.stat().st_size))
                for prepared in ([False] if quantized else [False,True]):
                    entries.append(dict(name=name+("_prepared" if prepared else "_dense"),
                        family=family,library=str(library),symbol="predict" if prepared else "predict_row",
                        prepared=prepared))
                if args.preset == "explore":
                    train_entry = dict(name=name, family=family, library=str(library), symbol="predict_row", prepared=False)
                    pgo_jobs.append((name, command, train_entry, [False] if quantized else [False, True]))
    for name, command, entry, interfaces in pgo_jobs:
        print(f"Training calibration-only PGO for {name}", flush=True)
        library, metadata = build_pgo(command, entry, out/(name+"_pgo"), harness, calibration,
                                      training_interfaces=interfaces if args.pgo_training == "interfaces" else [False])
        builds.append(dict(name=name+"_pgo", **metadata))
        for prepared in interfaces:
            family = entry["family"]
            suffix = "" if family == "engine" else ("_prepared" if prepared else "_dense")
            entries.append(dict(name=name+"_pgo"+suffix, family=family if family == "engine" else family+"_pgo",
                                library=str(library), symbol="predict" if prepared else "predict_row", prepared=prepared))
    tuning_errors = validate(booster,tuning,entries)
    (out/"builds.json").write_text(json.dumps(builds,indent=2)+"\n")
    print("Measuring tuning data",flush=True)
    timing_args = dict(samples=args.samples,rounds=args.rounds,seed=args.seed)
    tuning_measurement = measure(harness,tuning,entries,out/"tuning.json",**timing_args)
    selected = choose(tuning_measurement,entries,args.metric)
    reference_names = (["clang_d4_adaptive", "clang_d1_block32"]
                       if args.preset == "cost" else [reference_name])
    if args.preset == "ranks":
        reference_names = ["clang_d4_adaptive", "clang_cost4_block32"]
    if args.preset == "explore":
        reference_names = ["clang_cost4_block32_rank4", "clang_d4_rank32"]
    selection = dict(metric=args.metric,selected=selected,
                     preset=args.preset,pgo_training=args.pgo_training,reference_name=reference_name,reference_names=reference_names,
                     model_sha256=fingerprint(model_path),
                     calibration_sha256=fingerprint(paths[0]),tuning_sha256=fingerprint(paths[1]),
                     seed=args.seed,samples=args.samples,rounds=args.rounds)
    # Freeze and persist choices BEFORE accessing final evaluation data.
    (out/"selection.json").write_text(json.dumps(selection,indent=2)+"\n")
    for family,e in selected.items():
        print(f"Selected {family}: {e['name']}",flush=True)
    evaluation = load_rows(paths[2],forest.num_feature)
    evaluation_hash = fingerprint(paths[2])
    if evaluation_hash in (selection["calibration_sha256"],selection["tuning_sha256"]):
        raise ValueError("Evaluation data duplicates calibration or tuning data")
    # Fixed previous configuration is a reference, never selected on evaluation.
    prior = [e for e in entries if e["name"] in reference_names]
    final_entries = list({e["name"]:e for e in [*selected.values(),*prior]}.values())
    final_errors = validate(booster,evaluation,final_entries)
    print("Measuring untouched evaluation data",flush=True)
    timing_args["seed"] += 1
    final = measure(harness,evaluation,final_entries,out/"evaluation.json",**timing_args)
    # Copy the exact binary chosen on tuning; never recompile or reselect here.
    deploy = out/"selected_model"
    deploy.mkdir()
    chosen_library = Path(selected["engine"]["library"])
    for filename in (chosen_library.name,"model.o","model.h","metadata.json"):
        shutil.copy2(chosen_library.parent/filename,deploy/filename)
    report = dict(selection=selection,evaluation=final,evaluation_sha256=evaluation_hash,
                  calibration_rows=len(calibration),tuning_rows=len(tuning),evaluation_rows=len(evaluation),
                  tuning_errors=tuning_errors,evaluation_errors=final_errors,entries=entries,
                  platform=platform.platform(),
                  compiler=subprocess.check_output(["clang","--version"],text=True),
                  versions={n:importlib.metadata.version(n) for n in
                            ("numpy","xgboost","llvmlite","treelite","tl2cgen")},
                  selected_library=str(deploy/chosen_library.name),
                  caveats=["Warm desktop measurements without CPU/frequency isolation",
                           "Tuning winner is not reselected using evaluation results",
                           "tl_f32_modified changes accumulation constants, not stock TL2cgen",
                           "Calibration data must represent deployment; no automatic drift handling"])
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    for e in final["engines"]:
        print(f'{e["name"]:40} {e["block_median_ns_per_row"]:9.1f} ns/row; p99 {e["p99_ns"]:.0f} ns')
    print(f"Report: {out/'report.json'}")


if __name__ == "__main__":
    main()
