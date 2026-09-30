"""RapidScorer, batch-one subset (Ye et al., KDD 2018), for trees of any leaf count.

QuickScorer's leaf bitvector becomes ``ceil(leaves / 64)`` 64-bit words per
tree. A split's left subtree is a contiguous leaf range, so its "false" mask
is stored as an *epitome* ``(first word, last word, first mask, last mask)``:
AND the two boundary words and zero any words in between. Splits with the same
feature and float32 threshold (in any tree) are *merged* into one node whose
single comparison applies all their epitomes. The SIMD-across-documents part
of RapidScorer is omitted, since there is only one row.

Per feature, either:

* ``loop``: scan merged nodes in ascending threshold order while
  ``threshold <= x`` (row-dependent trip count, NaN sentinel), or
* ``dense``: an exact rank among the unique thresholds (or a missing-value
  row) selects a precomputed AND over the word span of trees using the
  feature. Chosen greedily within ``dense_budget_bytes``.

The exit leaf of each tree is the lowest set bit across its words, found with
a fixed, branch-free scan. Leaf values are summed in original tree order.
"""

from pathlib import Path
import time

from .cgen import array, build, hexf, signature, sequential_sum
from .model import Forest, Tree

FULL = (1 << 64) - 1


def _hexw(v: int) -> str:
    return hex(v) + "ull"


def _layout(tree: Tree):
    """In-order leaf values and (feature, threshold, default_left, first, stop) left-leaf ranges."""
    leaves, splits = [], []

    def walk(node):
        if tree.left[node] == -1:
            leaves.append(tree.value[node])
            return len(leaves) - 1, len(leaves)
        start, middle = walk(tree.left[node])
        _, stop = walk(tree.right[node])
        splits.append((tree.feature[node], tree.value[node], tree.default_left[node], start, middle))
        return start, stop

    walk(0)
    return leaves, splits


def _epitome(bit_start: int, bit_stop: int) -> tuple[int, int, int, int]:
    """Clear global bits [bit_start, bit_stop): (first word, last word, first mask, last mask)."""
    wa, wb = bit_start // 64, (bit_stop - 1) // 64
    lo, hi = bit_start % 64, (bit_stop - 1) % 64
    keep_below = (1 << lo) - 1
    keep_above = FULL ^ ((1 << (hi + 1)) - 1)
    if wa == wb:
        mask = keep_below | keep_above
        return wa, wb, mask, mask
    return wa, wb, keep_below, keep_above


def plan(forest: Forest):
    leaves, leaf_offset, word_base, word_count, by_feature = [], [], [], [], {}
    words = 0
    for t, tree in enumerate(forest.trees):
        tree_leaves, splits = _layout(tree)
        leaf_offset.append(len(leaves))
        leaves.extend(tree_leaves)
        n = (len(tree_leaves) + 63) // 64
        word_base.append(words)
        word_count.append(n)
        for f, thr, dl, a, b in splits:
            by_feature.setdefault(f, []).append((thr, t, dl, _epitome(64 * words + a, 64 * words + b)))
        words += n
    features = {}
    for f, items in sorted(by_feature.items()):
        items.sort(key=lambda s: (s[0], s[1]))
        merged = {}
        for thr, t, dl, ep in items:
            merged.setdefault(thr, []).append((dl, ep))
        w0 = min(ep[0] for *_, ep in items)
        w1 = max(ep[1] for *_, ep in items)
        features[f] = {"merged": merged, "splits": len(items), "w0": w0, "span": w1 - w0 + 1,
                       "dense_bytes": (len(merged) + 2) * (w1 - w0 + 1) * 8}
    return leaves, leaf_offset, word_base, word_count, words, features


def _apply(row: list[int], w0: int, ep) -> None:
    wa, wb, ma, mb = ep
    row[wa - w0] &= ma
    for w in range(wa + 1, wb):
        row[w - w0] = 0
    row[wb - w0] &= mb


def generate_c(forest: Forest, *, dense_budget_bytes: int = 0,
               entry: str = "predict_row", chained: bool = False) -> tuple[str, dict]:
    if not isinstance(dense_budget_bytes, int) or dense_budget_bytes < 0:
        raise ValueError("dense_budget_bytes must be a nonnegative integer")
    leaves, leaf_offset, word_base, word_count, words, features = plan(forest)
    dense, used = set(), 0
    for f in sorted(features, key=lambda f: (features[f]["dense_bytes"] / features[f]["splits"], f)):
        if used + features[f]["dense_bytes"] <= dense_budget_bytes:
            dense.add(f)
            used += features[f]["dense_bytes"]
    thr, node_off, ep_wa, ep_wb, ep_ma, ep_mb, nan_off, cp = [], [], [], [], [], [], [], []
    epitomes = merged_nodes = 0
    body = []

    def emit(entries):
        start = len(ep_wa)
        for wa, wb, ma, mb in entries:
            ep_wa.append(wa)
            ep_wb.append(wb)
            ep_ma.append(ma)
            ep_mb.append(mb)
        return start, len(ep_wa)

    for f, info in features.items():
        values = sorted(info["merged"])
        missing = [ep for v in values for dl, ep in info["merged"][v] if not dl]
        merged_nodes += len(values)
        epitomes += info["splits"]
        if f in dense:
            w0, span = info["w0"], info["span"]
            t_off, c_off = len(thr), len(cp)
            thr.extend(values)
            row = [FULL] * span
            cp.extend(row)
            for v in values:
                for _, ep in info["merged"][v]:
                    _apply(row, w0, ep)
                cp.extend(row)
            nan_row = [FULL] * span
            for ep in missing:
                _apply(nan_row, w0, ep)
            cp.extend(nan_row)
            n = len(values)
            body.append(f"  {{ /* feature {f}: dense, {n} merged nodes, words {w0}..{w0 + span - 1} */\n"
                        f"    const float x = row[{f}]; uint32_t r = 0;\n"
                        f"    for (uint32_t j = 0; j < {n}u; ++j) r += THR[{t_off}u + j] <= x;\n"
                        f"    r = x != x ? {n + 1}u : r;\n"
                        f"    const uint64_t *m = CP + {c_off}u + r * {span}u;\n"
                        f"    for (uint32_t i = 0; i < {span}u; ++i) bv[{w0}u + i] &= m[i];\n  }}\n")
            continue
        t_off, k_off = len(thr), len(node_off)
        for v in values:
            thr.append(v)
            node_off.append(emit(ep for _, ep in info["merged"][v])[0])
        thr.append(float("nan"))  # Sentinel: the scan stops here for every x.
        node_off.append(len(ep_wa))
        nan_start, nan_stop = emit(missing)
        body.append(f"  {{ /* feature {f}: loop, {len(values)} merged nodes from {info['splits']} splits */\n"
                    f"    const float x = row[{f}];\n"
                    f"    if (__builtin_expect(x != x, 0)) {{\n"
                    f"      for (uint32_t e = {nan_start}u; e < {nan_stop}u; ++e) CLEAR(e);\n"
                    f"    }} else {{\n"
                    f"      uint32_t k = 0;\n"
                    f"      while (THR[{t_off}u + k] <= x) {{\n"
                    f"        for (uint32_t e = NODE_OFF[{k_off}u + k]; e < NODE_OFF[{k_off}u + k + 1]; ++e) CLEAR(e);\n"
                    f"        ++k;\n      }}\n    }}\n  }}\n")
    n_trees = len(forest.trees)
    src = ["/* Generated by xgb_latency.rapidscorer; exact float32 thresholds and leaves. */\n",
           "#include <stdint.h>\n#include <string.h>\n\n",
           array("float", "LEAF", leaves or [0.0], hexf), array("uint32_t", "LEAF_OFFSET", leaf_offset),
           array("uint32_t", "WORD_BASE", word_base), array("uint32_t", "WORD_COUNT", word_count),
           array("float", "THR", thr, hexf), array("uint32_t", "NODE_OFF", node_off),
           array("uint32_t", "EP_WA", ep_wa), array("uint32_t", "EP_WB", ep_wb),
           array("uint64_t", "EP_MA", ep_ma, _hexw), array("uint64_t", "EP_MB", ep_mb, _hexw),
           array("uint64_t", "CP", cp, _hexw),
           "#define CLEAR(e) do { const uint32_t wa = EP_WA[e], wb = EP_WB[e]; bv[wa] &= EP_MA[e];"
           " for (uint32_t w = wa + 1; w < wb; ++w) bv[w] = 0; bv[wb] &= EP_MB[e]; } while (0)\n",
           signature(entry, chained),
           f"  uint64_t bv[{max(words, 1)}] __attribute__((aligned(64)));\n",
           "  memset(bv, 0xff, sizeof bv);\n"]
    src.extend(body)
    src.append(f"  float val[{n_trees}];\n"
               f"  for (uint32_t t = 0; t < {n_trees}u; ++t) {{\n"
               "    const uint64_t *b = bv + WORD_BASE[t]; uint32_t idx = 0;\n"
               "    for (uint32_t w = WORD_COUNT[t]; w-- > 0;) {\n"
               "      const uint64_t v = b[w];\n"
               "      const uint32_t c = 64 * w + (uint32_t)__builtin_ctzll(v | 0x8000000000000000ull);\n"
               "      idx = v ? c : idx;\n    }\n"
               "    val[t] = LEAF[LEAF_OFFSET[t] + idx];\n  }\n")
    src.append(sequential_sum(forest, "val", chained) + "}\n")
    stats = {"words": words, "max_words_per_tree": max(word_count, default=0),
             "max_leaves_per_tree": max((sum(1 for c in t.left if c == -1) for t in forest.trees), default=0),
             "splits": epitomes, "merged_nodes": merged_nodes, "dense_features": sorted(dense),
             "dense_table_bytes": used, "epitome_bytes": 24 * len(ep_wa), "dense_budget_bytes": dense_budget_bytes}
    return "".join(src), stats


def compile_rapidscorer(model_path: str | Path, output_dir: str | Path, *, dense_budget_bytes: int = 0,
                        cc: str = "clang", optimization: str = "O3") -> Path:
    """Exports ``void predict_row(const float *, float *)`` like ``compile_model``."""
    started = time.perf_counter()
    forest = Forest.load(model_path)
    source, stats = generate_c(forest, dense_budget_bytes=dense_budget_bytes)
    return build(source, forest, model_path, output_dir, lowering="rapidscorer", cc=cc,
                 optimization=optimization, started=started, extra={"exact_accumulation_order": True, **stats})
