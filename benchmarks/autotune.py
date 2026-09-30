"""Offline autotuning over whole-model lowerings and per-block mixed lowerings.

Protocol (no evaluation rows are read before the selection is frozen):

1. Build every whole-model candidate; each must be bitwise equal to the
   reference LLVM prototype on the tuning rows.
2. For each block count K, split the trees into K contiguous blocks, time
   every block strategy as a standalone sub-model on the tuning rows, keep
   the fastest strategy per block and build the mixed model (bitwise checked).
3. Time all candidates on tuning rows; re-time a shortlist in independent
   processes; select the lowest median metric; write selection.json.
4. Only then load the evaluation rows and time the selection against fixed
   references in independent processes.

The default metric is time-stamp-counter ticks per row from the block loop
(``block_median_ticks_per_row``). On x86 the TSC runs at a constant nominal
rate, so ticks are proportional to wall time, not core cycles.
"""
import argparse
import json
import platform
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
import treelite
import tl2cgen
import xgboost as xgb

from xgb_latency import Forest, Predictor, compile_model
from xgb_latency.blockmix import compile_blockmix, compile_spec, equal_blocks
from benchmarks.run import build_tl, check_tl

TILE_MODES = ("scalar", "gather", "insert") if platform.machine() in ("x86_64", "AMD64") else ("scalar",)

BLOCK_SPECS = ["qs", "vpred:lanes=8,layout=level", "vpred:lanes=16,layout=tree",
               "packed:lanes=16,layout=forest", "packed:lanes=8,layout=hot_dfs", "rs",
               "direct:select_depth=1", "tiled:lanes=8,tile_levels=3,mode=scalar"]


def whole_candidates(max_leaves: int):
    llvm = [("llvm_select1_profiled_clang", dict(backend="clang", calibrated=True)),
            ("llvm_cost4_block32_rank4", dict(backend="clang", select_depth=6, select_policy="cost",
                                              tree_block_size=32, rank_feature_limit=4, calibrated=True)),
            ("llvm_interleaved_self_scalar16", dict(traversal_lanes=16, traversal_leaf_layout="self_loop")),
            ("llvm_interleaved_self_vector8", dict(traversal_lanes=8, traversal_mode="vector",
                                                   traversal_leaf_layout="self_loop")),
            ("llvm_interleaved_self_vector16", dict(traversal_lanes=16, traversal_mode="vector",
                                                    traversal_leaf_layout="self_loop"))]
    specs = ["vpred:lanes=8,layout=tree", "vpred:lanes=8,layout=level", "vpred:lanes=16,layout=tree",
             "vpred:lanes=16,layout=level", "vpred:lanes=16,layout=tree,group_by_height=false",
             "packed:lanes=16,layout=dfs", "packed:lanes=16,layout=bfs", "packed:lanes=16,layout=hot_dfs",
             "packed:lanes=16,layout=frames", "packed:lanes=16,layout=forest", "packed:lanes=8,layout=forest",
             "rs", "rs:dense_budget_bytes=262144", "direct", "direct:select_depth=1"]
    specs += [f"tiled:lanes=8,tile_levels={k},mode={m}" for k in (2, 3) for m in TILE_MODES]
    specs += [f"tiled:lanes=16,tile_levels=3,mode={m}" for m in ("scalar", "insert") if m in TILE_MODES]
    vector = "gather" if "gather" in TILE_MODES else "scalar"
    specs += [f"tiled:lanes=8,tile_levels=3,mode={vector},top_levels={t}" for t in (6, 9)]
    specs += [f"probtiled:lanes=8,mode={vector}", f"probtiled:lanes=8,mode={vector},early_exit=false",
              "probtiled:lanes=8,mode=insert" if "insert" in TILE_MODES else "probtiled:lanes=8,mode=scalar,max_nodes=3"]
    if max_leaves <= 64:
        specs = ["qs", "qs:stride=0"] + specs
    return llvm, specs


def safe(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name)


class Runner:
    def __init__(self, out: Path, nf: int, samples: int, rounds: int, metric: str):
        self.out, self.nf, self.samples, self.rounds, self.metric = out, nf, samples, rounds, metric
        self.harness = out / "native"
        command = ["clang++", "-O3", "-std=c++17", str(Path(__file__).with_name("native.cc")), "-o", str(self.harness)]
        subprocess.run(command + ([] if sys.platform == "darwin" else ["-ldl"]), check=True, capture_output=True)

    def run(self, rows: np.ndarray, entries: list[dict], seed: int, label: str) -> dict:
        data = self.out / "timing" / f"{label}.f32"
        data.parent.mkdir(exist_ok=True)
        rows.tofile(data)
        args = [str(self.harness), str(data), str(len(rows)), str(self.nf), str(self.samples), str(self.rounds), str(seed)]
        for e in entries:
            args += [e["name"], e["library"], e["symbol"], str(int(e["prepared"]))]
        result = json.loads(subprocess.run(args, check=True, capture_output=True, text=True).stdout)
        (self.out / "timing" / f"{label}.json").write_text(json.dumps(result, indent=2) + "\n")
        return {e["name"]: e for e in result["engines"]}

    def score(self, row: dict) -> float:
        value = row.get(self.metric, 0.0)
        return value if value > 0 else row["block_median_ns_per_row"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "calibration", "tuning", "evaluation", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--blocks", default="2,4", help="Comma-separated block counts for mixed lowerings")
    p.add_argument("--metric", choices=["block_median_ticks_per_row", "block_median_ns_per_row", "p99_ns"],
                   default="block_median_ticks_per_row")
    p.add_argument("--shortlist", type=int, default=6)
    p.add_argument("--confirm-runs", type=int, default=3)
    p.add_argument("--eval-runs", type=int, default=3)
    p.add_argument("--samples", type=int, default=4096)
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--seed", type=int, default=30300)
    p.add_argument("--skip-llvm", default="", help="Comma-separated LLVM candidate names to skip")
    args = p.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    model = Path(args.model).resolve()
    tuning_path, evaluation_path = Path(args.tuning).resolve(), Path(args.evaluation).resolve()
    if tuning_path == evaluation_path:
        p.error("tuning and evaluation must be different files")
    forest = Forest.load(model)
    nf = forest.num_feature
    calibration = np.ascontiguousarray(np.load(args.calibration, allow_pickle=False), dtype=np.float32)
    tuning = np.ascontiguousarray(np.load(tuning_path, allow_pickle=False), dtype=np.float32)
    booster = xgb.Booster(model_file=str(model))
    booster.set_param({"nthread": 1})
    expected = booster.predict(xgb.DMatrix(tuning), output_margin=True)
    runner = Runner(out, nf, args.samples, args.rounds, args.metric)
    max_leaves = max(sum(1 for c in t.left if c == -1) for t in forest.trees)
    skip = set(filter(None, args.skip_llvm.split(",")))
    entries, builds, reference = [], {}, None

    def add(name, lib, meta_extra=None):
        nonlocal reference
        actual = Predictor(lib).predict(tuning)
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
        if reference is None:
            reference = actual
        if not np.array_equal(actual.view(np.uint32), reference.view(np.uint32)):
            raise AssertionError(f"{name} is not bitwise equal to the reference")
        entries.append(dict(name=name, library=str(lib), symbol="predict_row", prepared=False))
        builds[name] = {**json.loads((Path(lib).parent / "metadata.json").read_text()), **(meta_extra or {})}
        print(f"built {name}", flush=True)

    llvm, specs = whole_candidates(max_leaves)
    for name, options in llvm:
        if name in skip:
            continue
        options = dict(options)
        calibrated = options.pop("calibrated", False)
        add(name, compile_model(model, out / "whole" / name, calibration=calibration if calibrated else None, **options))
    for spec in specs:
        add(safe(spec), compile_spec(model, out / "whole" / safe(spec), spec, calibration=calibration))
    # Per-block search on tuning rows only.
    block_search = {}
    for count in [int(v) for v in args.blocks.split(",") if v]:
        ranges = equal_blocks(len(forest.trees), count)
        sub_entries, sub_meta = [], {}
        for bi, (start, stop) in enumerate(ranges):
            sub_expected = booster.predict(xgb.DMatrix(tuning), output_margin=True, iteration_range=(start, stop))
            for spec in BLOCK_SPECS:
                if spec == "qs" and max(sum(1 for c in t.left if c == -1) for t in forest.trees[start:stop]) > 64:
                    continue
                name = f"k{count}_b{bi}_{safe(spec)}"
                lib = compile_spec(model, out / "blocks" / name, spec, calibration=calibration, start=start, stop=stop)
                np.testing.assert_allclose(Predictor(lib).predict(tuning), sub_expected, rtol=2e-5, atol=2e-5)
                sub_entries.append(dict(name=name, library=str(lib), symbol="predict_row", prepared=False))
                sub_meta[name] = (bi, spec)
        timing = runner.run(tuning, sub_entries, args.seed + count, f"blocks-k{count}")
        choice = {}
        for name, (bi, spec) in sub_meta.items():
            if bi not in choice or runner.score(timing[name]) < runner.score(timing[choice[bi]]):
                choice[bi] = name
        blocks = [(start, stop, sub_meta[choice[bi]][1]) for bi, (start, stop) in enumerate(ranges)]
        block_search[count] = {"blocks": blocks, "block_scores": {n: runner.score(timing[n]) for n in sub_meta}}
        add(f"mix_k{count}", compile_blockmix(model, out / "whole" / f"mix_k{count}", blocks, calibration=calibration))
    # Baseline TL2cgen (separate family; reported, not selectable).
    annotation = out / "tl_annotation.json"
    tl2cgen.annotate_branch(treelite.frontend.load_xgboost_model(str(model)), tl2cgen.DMatrix(calibration),
                           annotation, nthread=1)
    tl_lib, _ = build_tl(model, out / "tl2cgen_profiled_lto", nf, annotation=annotation, lto=True)
    check_tl(tl_lib, tuning, expected, prepared=True)
    tl_entry = dict(name="tl2cgen_profiled_lto_prepared", library=str(tl_lib), symbol="predict", prepared=True)
    # Tuning: one full run, then independent confirmation of the shortlist.
    first = runner.run(tuning, entries + [tl_entry], args.seed, "tuning-all")
    ranked = sorted(entries, key=lambda e: (runner.score(first[e["name"]]), e["name"]))
    shortlist = ranked[:args.shortlist]
    confirm = [runner.run(tuning, shortlist, args.seed + 100 + i, f"tuning-confirm-{i}") for i in range(args.confirm_runs)]
    medians = {e["name"]: statistics.median(runner.score(r[e["name"]]) for r in confirm) for e in shortlist}
    winner = min(shortlist, key=lambda e: (medians[e["name"]], e["name"]))
    selection = {"metric": args.metric, "winner": winner, "shortlist_medians": medians,
                 "first_pass": {e["name"]: runner.score(first[e["name"]]) for e in entries + [tl_entry]},
                 "block_search": {str(k): v for k, v in block_search.items()},
                 "tuning_file": str(tuning_path), "seed": args.seed}
    (out / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(f"Selected {winner['name']} ({medians[winner['name']]:.1f} {args.metric})", flush=True)
    # Evaluation (frozen selection).
    evaluation = np.ascontiguousarray(np.load(evaluation_path, allow_pickle=False), dtype=np.float32)
    eval_expected = booster.predict(xgb.DMatrix(evaluation), output_margin=True)
    references = [e for e in entries if e["name"] in ("llvm_select1_profiled_clang", "llvm_cost4_block32_rank4",
                                                      "llvm_interleaved_self_scalar16", "qs")]
    # The best tuning-pass member of each new family is also carried into evaluation.
    for family in ("vpred", "tiled_lanes_8_tile_levels_3_mode_gather_top", "probtiled", "tiled", "packed"):
        members = [e for e in entries if e["name"].startswith(family)]
        if members:
            references.append(min(members, key=lambda e: (runner.score(first[e["name"]]), e["name"])))
    final = list({e["name"]: e for e in [winner, *references, *shortlist[:3]]}.values())
    for e in final:
        actual = Predictor(e["library"]).predict(evaluation)
        np.testing.assert_allclose(actual, eval_expected, rtol=2e-6, atol=2e-6)
    check_tl(tl_lib, evaluation, eval_expected, prepared=True)
    runs = [runner.run(evaluation, final + [tl_entry], args.seed + 200 + i, f"evaluation-{i}")
            for i in range(args.eval_runs)]
    summary = {}
    for name in [e["name"] for e in final] + [tl_entry["name"]]:
        summary[name] = {k: [r[name][k] for r in runs] for k in
                         ("block_median_ticks_per_row", "block_median_ns_per_row", "p50_ns", "p99_ns")}
    report = {"platform": platform.platform(), "machine": platform.machine(),
              "compiler": subprocess.check_output(["clang", "--version"], text=True).splitlines()[0],
              "model": {"path": str(model), "num_trees": len(forest.trees), "max_leaves": max_leaves,
                        "max_height": max(t.height[0] for t in forest.trees)},
              "selection": selection, "evaluation": summary, "builds": builds,
              "caveats": ["TSC ticks are nominal-rate wall-clock ticks, not core cycles",
                          "Warm single-thread native calls; no pinning or frequency isolation",
                          "Per-block choices use standalone sub-model timings on tuning rows"]}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n{'engine':44} {'ticks/row (eval runs)':>28} {'ns/row':>24}")
    for name, v in sorted(summary.items(), key=lambda kv: statistics.median(kv[1]['block_median_ticks_per_row'])):
        print(f"{name:44} {'/'.join(f'{x:.0f}' for x in v['block_median_ticks_per_row']):>28} "
              f"{'/'.join(f'{x:.0f}' for x in v['block_median_ns_per_row']):>24}")
    print(f"\nReport: {out / 'report.json'}")


if __name__ == "__main__":
    main()
