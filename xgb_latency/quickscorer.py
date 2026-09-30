"""QuickScorer-style feature-major lowering (Lucchese et al., SIGIR 2015).

Each tree keeps a leaf bitvector, one bit per leaf in left-to-right order. For
every split that sends the row right, the bits of that split's left subtree are
cleared; the exit leaf is then the lowest remaining bit. The result is exact:
the exit leaf is never cleared (a split that clears it would have to be an
ancestor sending the row left), and every leaf to its left is cleared by the
lowest common ancestor. XGBoost routes a non-missing value right iff
``not (x < threshold)``; missing values follow ``default_left``.

Two per-feature strategies are generated:

* ``classic`` (``stride=0``): thresholds sorted ascending with a NaN sentinel;
  the loop clears masks while ``threshold <= x``. The trip count depends on
  the row, so each feature typically costs a mispredicted loop exit.
* checkpointed: an exact SIMD-countable rank ``r`` of ``x`` among the
  feature's sorted split thresholds selects a precomputed AND of the first
  ``(r // stride) * stride`` masks over the contiguous tree span that uses the
  feature; the remaining ``< stride`` masks are applied by a fixed-length,
  branch-free loop. ``stride=1`` is a fully dense table. Missing values apply
  the masks of splits whose default goes right.

Leaf values are then added to the base margin in tree order with float32
arithmetic, the same accumulation contract as ``compile_model``.
``summation="pairwise_inexact"`` exists only to measure the cost of that serial
dependency chain; its output is not bitwise-comparable and is flagged in
metadata.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from .model import Forest, Tree


@dataclass(frozen=True)
class _Split:
    tree: int
    feature: int
    threshold: float
    default_left: bool
    mask: int


def _hexf(value: float) -> str:
    value = float(np.float32(value))
    if math.isnan(value):
        return "__builtin_nanf(\"\")"
    return value.hex() + "f"


def _tree_layout(tree: Tree, word_bits: int) -> tuple[list[float], list[tuple[int, float, bool, int]]]:
    """In-order leaf values and (feature, threshold, default_left, mask) per split."""
    leaves: list[float] = []
    splits = []
    full = (1 << word_bits) - 1

    def walk(node: int) -> tuple[int, int]:
        if tree.left[node] == -1:
            leaves.append(tree.value[node])
            return len(leaves) - 1, len(leaves)
        start, middle = walk(tree.left[node])
        _, stop = walk(tree.right[node])
        left_bits = ((1 << (middle - start)) - 1) << start
        splits.append((tree.feature[node], tree.value[node], tree.default_left[node], full & ~left_bits))
        return start, stop

    # Explicit recursion depth is bounded by the frontend's depth limit (64).
    walk(0)
    return leaves, splits


def plan(forest: Forest):
    max_leaves = max(sum(1 for c in t.left if c == -1) for t in forest.trees)
    if max_leaves > 64:
        raise ValueError("QuickScorer lowering supports at most 64 leaves per tree")
    word_bits = max(8, 1 << math.ceil(math.log2(max_leaves)))
    leaves, offsets, splits = [], [], []
    for ti, tree in enumerate(forest.trees):
        tree_leaves, tree_splits = _tree_layout(tree, word_bits)
        offsets.append(len(leaves))
        leaves.extend(tree_leaves)
        splits.extend(_Split(ti, f, thr, dl, mask) for f, thr, dl, mask in tree_splits)
    by_feature: dict[int, list[_Split]] = {}
    for s in splits:
        by_feature.setdefault(s.feature, []).append(s)
    features = {}
    for f, items in sorted(by_feature.items()):
        items.sort(key=lambda s: (s.threshold, s.tree))
        t0 = min(s.tree for s in items)
        features[f] = {"splits": items, "tree_start": t0, "span": max(s.tree for s in items) - t0 + 1}
    return word_bits, leaves, offsets, features


def _unique(info: dict) -> list[float]:
    return sorted({s.threshold for s in info["splits"]})


def _checkpoint_rows(info: dict, stride: int) -> int:
    if stride == 1:  # Dense: one row per unique-threshold rank plus a missing-value row.
        return len(_unique(info)) + 2
    n = len(info["splits"])
    return 0 if n < stride else n // stride + 1


def table_bytes(features: dict, word_bits: int, stride: int) -> int:
    return sum(_checkpoint_rows(i, stride) * i["span"] * word_bits // 8 for i in features.values())


def choose_stride(features: dict, word_bits: int, budget: int) -> int:
    """Smallest power-of-two checkpoint stride whose prefix tables fit the budget."""
    stride = 1
    while stride < 1024 and table_bytes(features, word_bits, stride) > budget:
        stride *= 2
    return stride


def _array(ctype: str, name: str, values, fmt=str, align: int = 64) -> str:
    body = ",".join(fmt(v) for v in values) or "0"
    return f"static const {ctype} {name}[] __attribute__((aligned({align}))) = {{{body}}};\n"


def generate_c(forest: Forest, *, stride: int | None = None, dense_budget_bytes: int = 256 * 1024,
               rank_linear_max: int = 64, rank_search: str = "two_level",
               summation: str = "sequential") -> tuple[str, dict]:
    """``stride=0`` emits the classic row-dependent loop for every feature.

    Otherwise each feature stores the AND of every ``stride``-th prefix of its
    threshold-sorted split masks (``stride=1`` is a fully dense table) and
    applies the remaining ``< stride`` masks with a fixed-length branch-free
    loop. ``stride=None`` picks the smallest power of two within the budget.
    """
    if summation not in {"sequential", "pairwise_inexact"}:
        raise ValueError("summation must be sequential or pairwise_inexact")
    if not isinstance(rank_linear_max, int) or rank_linear_max < 0:
        raise ValueError("rank_linear_max must be a nonnegative integer")
    if rank_search not in {"two_level", "binary"}:
        raise ValueError("rank_search must be two_level or binary")
    if not isinstance(dense_budget_bytes, int) or dense_budget_bytes < 0:
        raise ValueError("dense_budget_bytes must be a nonnegative integer")
    if stride is not None and (not isinstance(stride, int) or stride < 0 or (stride and stride & (stride - 1))):
        raise ValueError("stride must be None, 0 (classic) or a power of two")
    word_bits, leaves, offsets, features = plan(forest)
    if stride is None:
        stride = choose_stride(features, word_bits, dense_budget_bytes)
    word = f"uint{word_bits}_t"
    full = (1 << word_bits) - 1
    n_trees = len(forest.trees)
    ctz = "__builtin_ctzll" if word_bits == 64 else "__builtin_ctz"
    hexw = lambda v: hex(v) + ("ull" if word_bits == 64 else "u")
    nan = float("nan")
    src = ["/* Generated by xgb_latency.quickscorer; exact float32 thresholds and leaves. */\n",
           "#include <stdint.h>\n#include <string.h>\n\n",
           _array("float", "LEAF", leaves, _hexf),
           _array("uint32_t", "LEAF_OFFSET", offsets)]
    thr, coarse, s_tree, s_mask, n_tree, n_mask, cp = [], [], [], [], [], [], []
    body = []
    for f, info in features.items():
        items = info["splits"]
        n = len(items)
        ns, s_off = len(n_tree), len(s_tree)
        if stride != 1:  # Dense tables carry the missing-value row themselves.
            for s in items:
                if not s.default_left:
                    n_tree.append(s.tree)
                    n_mask.append(s.mask)
                s_tree.append(s.tree)
                s_mask.append(s.mask)
        missing = (f"    if (__builtin_expect(x != x, 0))\n"
                   f"      for (uint32_t k = {ns}u; k < {len(n_tree)}u; ++k) bv[NTREE[k]] &= NMASK[k];\n")
        if stride == 0:
            s_tree.append(0)
            s_mask.append(full)
            t_off = len(thr)
            thr.extend([s.threshold for s in items] + [nan])  # NaN sentinel ends the scan.
            body.append(f"  {{ /* feature {f}: classic, {n} splits */\n"
                        f"    const float x = row[{f}];\n{missing}"
                        f"    else {{ uint32_t k = 0;\n"
                        f"      while (THR[{t_off}u + k] <= x) {{ bv[STREE[{s_off}u + k]] &= SMASK[{s_off}u + k]; ++k; }} }}\n  }}\n")
            continue
        if stride > 1:
            s_tree.extend([0] * stride)
            s_mask.extend([full] * stride)
        # Exact rank r = #{thresholds <= x}; NaN compares false, giving r = 0.
        t_off = len(thr)
        values = _unique(info) if stride == 1 else [s.threshold for s in items]
        n = len(values)
        blocks = n // 16
        thr.extend(values + [nan] * (16 * (blocks + 1) - n))
        if n <= rank_linear_max:
            rank = (f"    uint32_t r = 0;\n"
                    f"    for (uint32_t j = 0; j < {16 * (blocks + 1)}u; ++j) r += THR[{t_off}u + j] <= x;\n")
        elif rank_search == "binary":
            # Branchless upper bound; trip count depends only on n.
            rank = (f"    const float *base = THR + {t_off}u; uint32_t len = {n}u;\n"
                    f"    while (len > 1) {{ const uint32_t half = len / 2; base = base[half - 1] <= x ? base + half : base; len -= half; }}\n"
                    f"    uint32_t r = (uint32_t)(base - (THR + {t_off}u)) + (*base <= x);\n")
        else:
            c_off = len(coarse)
            coarse.extend([values[16 * i + 15] for i in range(blocks)] + [nan] * ((-blocks) % 16))
            rank = (f"    uint32_t c = 0;\n"
                    f"    for (uint32_t j = 0; j < {blocks + (-blocks) % 16}u; ++j) c += COARSE[{c_off}u + j] <= x;\n"
                    f"    uint32_t r = 16 * c;\n"
                    f"    for (uint32_t j = 0; j < 16u; ++j) r += THR[{t_off}u + 16 * c + j] <= x;\n")
        rows = _checkpoint_rows(info, stride)
        t0, span = info["tree_start"], info["span"]
        apply = ""
        if stride == 1:
            cp_off = len(cp)
            current, k = [full] * span, 0
            cp.extend(current)
            for value in values:
                while k < len(items) and items[k].threshold <= value:
                    current[items[k].tree - t0] &= items[k].mask
                    k += 1
                cp.extend(current)
            missing_row = [full] * span
            for s in items:
                if not s.default_left:
                    missing_row[s.tree - t0] &= s.mask
            cp.extend(missing_row)
            body.append(f"  {{ /* feature {f}: dense, {n} unique thresholds, trees {t0}..{t0 + span - 1} */\n"
                        f"    const float x = row[{f}];\n{rank}"
                        f"    r = x != x ? {n + 1}u : r;\n"
                        f"    const {word} *m = CP + {cp_off}u + r * {span}u;\n"
                        f"    for (uint32_t i = 0; i < {span}u; ++i) bv[{t0}u + i] &= m[i];\n  }}\n")
            continue
        if rows:
            cp_off = len(cp)
            current = [full] * span
            for k in range(rows):
                if k:
                    for s in items[(k - 1) * stride:k * stride]:
                        current[s.tree - t0] &= s.mask
                cp.extend(current)
            apply += (f"    const {word} *m = CP + {cp_off}u + (r / {stride}u) * {span}u;\n"
                      f"    for (uint32_t i = 0; i < {span}u; ++i) bv[{t0}u + i] &= m[i];\n")
        tail = min(stride, n) if stride > 1 or not rows else 0
        if tail:
            base = f"(r / {stride}u) * {stride}u" if rows else "0u"
            apply += (f"    const uint32_t k0 = {base};\n"
                      f"    for (uint32_t i = 0; i < {tail}u; ++i) {{\n"
                      f"      const uint32_t k = k0 + i;\n"
                      f"      bv[STREE[{s_off}u + k]] &= k < r ? SMASK[{s_off}u + k] : ({word}){hexw(full)};\n    }}\n")
        body.append(f"  {{ /* feature {f}: stride {stride}, {n} splits, trees {t0}..{t0 + span - 1} */\n"
                    f"    const float x = row[{f}];\n{rank}{apply}{missing}  }}\n")
    tree_index = "uint16_t" if n_trees <= 65536 else "uint32_t"
    for ctype, name, values, fmt in [("float", "THR", thr, _hexf), ("float", "COARSE", coarse, _hexf),
                                     (tree_index, "STREE", s_tree, str), (word, "SMASK", s_mask, hexw),
                                     (tree_index, "NTREE", n_tree, str), (word, "NMASK", n_mask, hexw),
                                     (word, "CP", cp, hexw)]:
        src.append(_array(ctype, name, values, fmt))
    src.append("\n__attribute__((visibility(\"default\")))\n"
               "void predict_row(const float *row, float *raw_margin) {\n"
               f"  {word} bv[{n_trees}] __attribute__((aligned(64)));\n"
               "  memset(bv, 0xff, sizeof bv);\n")
    src.extend(body)
    if summation == "sequential":
        src.append(f"  float acc = {_hexf(forest.base_margin)};\n"
                   f"  for (uint32_t t = 0; t < {n_trees}u; ++t) acc += LEAF[LEAF_OFFSET[t] + {ctz}(bv[t])];\n"
                   "  *raw_margin = acc;\n}\n")
    else:
        size = 1 << max(0, (n_trees - 1).bit_length())
        src.append(f"  float v[{size}] __attribute__((aligned(64)));\n"
                   f"  for (uint32_t t = 0; t < {n_trees}u; ++t) v[t] = LEAF[LEAF_OFFSET[t] + {ctz}(bv[t])];\n"
                   f"  for (uint32_t t = {n_trees}u; t < {size}u; ++t) v[t] = 0.0f;\n"
                   f"  for (uint32_t w = {size}u / 2; w > 0; w /= 2)\n"
                   "    for (uint32_t t = 0; t < w; ++t) v[t] += v[t + w];\n"
                   f"  *raw_margin = {_hexf(forest.base_margin)} + v[0];\n}}\n")
    stats = {"word_bits": word_bits, "num_leaves": len(leaves), "stride": stride, "rank_search": rank_search,
             "checkpoint_table_bytes": len(cp) * word_bits // 8,
             "split_table_bytes": len(s_tree) * (2 + word_bits // 8) + len(n_tree) * (2 + word_bits // 8),
             "threshold_table_bytes": 4 * (len(thr) + len(coarse)),
             "leaf_table_bytes": 4 * len(leaves) + 4 * n_trees,
             "bitvector_bytes": n_trees * word_bits // 8}
    return "".join(src), stats


def compile_quickscorer(model_path: str | Path, output_dir: str | Path, *, cc: str = "clang",
                        optimization: str = "O3", stride: int | None = None,
                        dense_budget_bytes: int = 256 * 1024, rank_linear_max: int = 64,
                        rank_search: str = "two_level", summation: str = "sequential") -> Path:
    """Emit model.c, model.so/.dylib and metadata; exports predict_row like compile_model."""
    if optimization not in {"O3", "O2", "Os"}:
        raise ValueError("optimization must be O3, O2 or Os")
    started = time.perf_counter()
    forest = Forest.load(model_path)
    source, stats = generate_c(forest, stride=stride, dense_budget_bytes=dense_budget_bytes,
                               rank_linear_max=rank_linear_max, rank_search=rank_search, summation=summation)
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.c").write_text(source)
    native = "-mcpu=native" if platform.machine() in ("arm64", "aarch64") else "-march=native"
    flags = ["-" + optimization, native, "-fPIC", "-fno-fast-math", "-ffp-contract=off", "-std=c11"]
    lib = out / ("model.dylib" if sys.platform == "darwin" else "model.so")
    subprocess.run([cc, *flags, "-c", str(out / "model.c"), "-o", str(out / "model.o")], check=True, capture_output=True)
    subprocess.run([cc, *flags, "-S", str(out / "model.c"), "-o", str(out / "model.s")], check=True, capture_output=True)
    subprocess.run([cc, "-dynamiclib" if sys.platform == "darwin" else "-shared",
                    str(out / "model.o"), "-o", str(lib)], check=True, capture_output=True)
    (out / "model.h").write_text(
        '#pragma once\n#ifdef __cplusplus\nextern "C" {\n#endif\n'
        '/* Dense float32, NaN missing; caller provides one output float. */\n'
        'void predict_row(const float *features, float *raw_margin);\n'
        '#ifdef __cplusplus\n}\n#endif\n')
    metadata = {"format_version": 1, "lowering": "quickscorer", "num_feature": forest.num_feature,
                "num_trees": len(forest.trees), "objective": forest.objective, "output": "raw_margin",
                "base_margin": forest.base_margin,
                "source_sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
                "compiler": subprocess.check_output([cc, "--version"], text=True).splitlines()[0],
                "flags": flags, "dense_budget_bytes": dense_budget_bytes, "rank_linear_max": rank_linear_max,
                "summation": summation, "exact_accumulation_order": summation == "sequential",
                **stats, "compile_seconds": time.perf_counter() - started,
                "library_bytes": lib.stat().st_size}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return lib
