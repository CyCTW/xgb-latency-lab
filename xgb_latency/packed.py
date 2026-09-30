"""Explicit-pointer interleaved traversal with probability-aware node layouts.

Every node is a 16-byte record ``{threshold, feature|default_right, left, right}``
(four per 64-byte line). Leaves store their value in ``threshold`` and point
both children at themselves, so each lane group runs a fixed number of
branch-free steps (its tallest tree) and then reads ``threshold`` of the final
node. Trees are evaluated ``lanes`` at a time, optionally grouped by height,
and leaf values are summed in original tree order: the result is exact.

Only the order of records in memory changes between layouts:

* ``bfs`` / ``dfs``: per-tree breadth-first / preorder.
* ``hot_dfs``: preorder that visits the more frequently taken child first
  (calibration counts), so the hot path is contiguous.
* ``frames``: per-tree cache-line frames (tree framing): each 64-byte frame is
  filled greedily with the most frequently reached nodes connected to the
  frame root; frames are emitted hottest first and each tree starts on a new
  line.
* ``forest``: Forest-Packing-style: all nodes of a lane group, from every tree
  in it, sorted by calibration reach count, so the hot upper nodes of the whole
  group share the fewest cache lines.

Unobserved nodes (reach count 0) are placed last, never removed.
"""

import heapq
from pathlib import Path
import time

import numpy as np

from .cgen import array, build, hexf, signature, reach_counts, sequential_sum
from .model import Forest, Tree
from .vpred import plan_groups

_DEFAULT_RIGHT = 1 << 31
LAYOUTS = ("bfs", "dfs", "hot_dfs", "frames", "forest")


def _order(tree: Tree, reach: list[float], layout: str) -> list[int]:
    if layout == "bfs":
        order = [0]
        for node in order:
            if tree.left[node] != -1:
                order += [tree.left[node], tree.right[node]]
        return order
    if layout in ("dfs", "hot_dfs"):
        order, stack = [], [0]
        while stack:
            node = stack.pop()
            order.append(node)
            if tree.left[node] != -1:
                first, second = tree.left[node], tree.right[node]
                if layout == "hot_dfs" and reach[second] > reach[first]:
                    first, second = second, first
                stack += [second, first]
        return order
    if layout == "frames":
        order, roots = [], [(-reach[0], 0)]
        while roots:
            _, root = heapq.heappop(roots)
            frame, frontier = [], [(-reach[root], root)]
            while frontier and len(frame) < 4:
                _, node = heapq.heappop(frontier)
                frame.append(node)
                if tree.left[node] != -1:
                    for child in (tree.left[node], tree.right[node]):
                        heapq.heappush(frontier, (-reach[child], child))
            order.extend(frame)
            for item in frontier:  # Children not in this frame start new frames.
                heapq.heappush(roots, item)
        return order
    raise ValueError(f"unknown per-tree layout {layout}")


def generate_c(forest: Forest, *, lanes: int = 8, layout: str = "hot_dfs", group_by_height: bool = True,
               calibration: np.ndarray | None = None,
               entry: str = "predict_row", chained: bool = False) -> tuple[str, dict]:
    if isinstance(lanes, bool) or lanes not in (1, 2, 4, 8, 16, 32):
        raise ValueError("lanes must be 1, 2, 4, 8, 16 or 32")
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {', '.join(LAYOUTS)}")
    if layout in ("hot_dfs", "frames", "forest") and calibration is None:
        raise ValueError(f"layout {layout} requires calibration data")
    counts = forest.branch_counts(calibration) if calibration is not None else [None] * len(forest.trees)
    n_trees = len(forest.trees)
    groups = plan_groups(forest, lanes, group_by_height)
    records, roots, out_index, segments = [], [], [], []
    for g, members in enumerate(groups):
        height = max(forest.trees[t].height[0] for t in members)
        if segments and segments[-1][0] == height:
            segments[-1][2] = g + 1
        else:
            segments.append([height, g, g + 1])
        if layout == "forest":
            items = []
            for t in members:
                reach = reach_counts(forest.trees[t], counts[t])
                depth = {0: 0}
                for node in _order(forest.trees[t], reach, "bfs"):
                    if forest.trees[t].left[node] != -1:
                        depth[forest.trees[t].left[node]] = depth[forest.trees[t].right[node]] = depth[node] + 1
                    items.append((-reach[node], depth[node], t, node))
            placement = [(t, node) for *_, t, node in sorted(items)]
        else:
            placement = []
            if layout == "frames" and len(records) % 4:
                records.extend([None] * (4 - len(records) % 4))  # Start each tree on a new line.
            for t in members:
                reach = reach_counts(forest.trees[t], counts[t])
                placement.extend((t, node) for node in _order(forest.trees[t], reach, layout))
                if layout == "frames" and t != members[-1]:
                    pending = (len(records) + len(placement)) % 4
                    placement.extend([(None, None)] * ((4 - pending) % 4))
        base = len(records)
        index = {}
        for offset, (t, node) in enumerate(placement):
            if t is not None:
                index[t, node] = base + offset
        for t, node in placement:
            if t is None:
                records.append(None)
                continue
            tree = forest.trees[t]
            me = index[t, node]
            if tree.left[node] == -1:
                records.append((tree.value[node], 0, me, me))
            else:
                flag = 0 if tree.default_left[node] else _DEFAULT_RIGHT
                records.append((tree.value[node], tree.feature[node] | flag,
                                index[t, tree.left[node]], index[t, tree.right[node]]))
        roots.extend(index[t, 0] for t in members)
        roots.extend([index[members[0], 0]] * (lanes - len(members)))
        out_index.extend(members + [n_trees] * (lanes - len(members)))
    records = [r if r is not None else (0.0, 0, i, i) for i, r in enumerate(records)] or [(0.0, 0, 0, 0)]
    src = ["/* Generated by xgb_latency.packed; exact float32 thresholds and leaves. */\n",
           "#include <stdint.h>\n\n",
           "typedef struct { float t; uint32_t fd; uint32_t l, r; } node_t;\n",
           "static const node_t NODE[] __attribute__((aligned(64))) = {"
           + ",".join(f"{{{hexf(v)},{fd}u,{l}u,{r}u}}" for v, fd, l, r in records) + "};\n",
           array("uint32_t", "ROOT", roots), array("uint32_t", "OUT", out_index),
           signature(entry, chained),
           f"  float val[{n_trees + 1}];\n"]
    for height, g0, g1 in segments:
        src.append(f"  for (uint32_t g = {g0}u; g < {g1}u; ++g) {{ /* height {height} */\n"
                   f"    const uint32_t *root = ROOT + g * {lanes}u;\n")
        src.append("".join(f"    uint32_t i{w} = root[{w}];\n" for w in range(lanes)))
        for _ in range(height):
            for w in range(lanes):
                src.append(f"    {{ const node_t *n = &NODE[i{w}]; const float x = row[n->fd & 0x7fffffffu];\n"
                           f"      i{w} = ((uint32_t)(x >= n->t) | ((uint32_t)(x != x) & (n->fd >> 31))) ? n->r : n->l; }}\n")
        src.append("".join(f"    val[OUT[g * {lanes}u + {w}u]] = NODE[i{w}].t;\n" for w in range(lanes)))
        src.append("  }\n")
    src.append(sequential_sum(forest, "val", chained) + "}\n")
    stats = {"lanes": lanes, "layout": layout, "group_by_height": group_by_height, "groups": len(groups),
             "height_segments": [s[0] for s in segments], "node_records": len(records),
             "node_table_bytes": 16 * len(records), "profiled": calibration is not None}
    return "".join(src), stats


def compile_packed(model_path: str | Path, output_dir: str | Path, *, lanes: int = 8, layout: str = "hot_dfs",
                   group_by_height: bool = True, calibration: np.ndarray | None = None,
                   cc: str = "clang", optimization: str = "O3") -> Path:
    """Exports ``void predict_row(const float *, float *)`` like ``compile_model``."""
    started = time.perf_counter()
    forest = Forest.load(model_path)
    source, stats = generate_c(forest, lanes=lanes, layout=layout, group_by_height=group_by_height,
                               calibration=calibration)
    return build(source, forest, model_path, output_dir, lowering="packed", cc=cc, optimization=optimization,
                 started=started, extra={"exact_accumulation_order": True, **stats})
