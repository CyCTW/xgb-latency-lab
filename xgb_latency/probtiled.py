"""Probability-guided tree tiling (Treebeard, PACT 2022), batch-one variant.

Tiles are grown greedily from a tile root: repeatedly add the internal frontier
node with the highest calibration reach count, up to ``max_nodes`` (<= 7) nodes.
Hot paths therefore cross fewer tiles, and cold regions still form valid tiles.
A tile with ``n`` nodes has ``n + 1`` exits (children outside the tile),
numbered left to right. Tiles of the same *shape* share one lookup table that
maps the node go-right mask to an exit; each tile stores a child pointer per
exit. Leaves are one-node "leaf tiles" (shape 0) whose exits all point to
themselves and whose ``t[0]`` holds the leaf value.

Every lane group runs up to its longest tile path; with ``early_exit`` the loop
stops as soon as every lane sits on a leaf tile (one well-predicted branch per
step). Go-right is ``(x >= t) | (isnan(x) & default_right)``; leaf values are
summed in original tree order, so results are bitwise identical to
``compile_model``.
"""

from pathlib import Path
import platform
import time

import numpy as np

from .cgen import array, build, hexf, reach_counts, sequential_sum, signature
from .model import Forest, Tree
from .tiled import MODES, _step
from .vpred import plan_groups


def _tiles(tree: Tree, reach: list[float], max_nodes: int):
    """Return tile list [(slots, exits)] where slots are tree nodes (preorder in tile) and
    exits are the out-of-tile children, left to right. Tile 0 holds the root."""
    tiles, pending, owner = [], [0], {}
    while pending:
        root = pending.pop(0)
        if tree.left[root] == -1:
            owner[root] = len(tiles)
            tiles.append(([root], []))  # Leaf tile.
            continue
        members, frontier = {root}, [root]
        candidates = [c for c in (tree.left[root], tree.right[root]) if tree.left[c] != -1]
        while candidates and len(members) < max_nodes:
            best = max(candidates, key=lambda n: (reach[n], -n))
            candidates.remove(best)
            members.add(best)
            candidates += [c for c in (tree.left[best], tree.right[best]) if tree.left[c] != -1]
        slots, exits = [], []

        def walk(node):  # Preorder slots; exits appear left to right.
            if node not in members:
                exits.append(node)
                return
            slots.append(node)
            walk(tree.left[node])
            walk(tree.right[node])

        walk(root)
        owner[root] = len(tiles)
        tiles.append((slots, exits))
        pending.extend(exits)
    return tiles, owner


def _shape(tree: Tree, slots: list[int], exits: list[int]) -> tuple:
    slot = {n: i for i, n in enumerate(slots)}
    exit_ = {n: i for i, n in enumerate(exits)}
    ref = lambda c: ("s", slot[c]) if c in slot else ("e", exit_[c])
    return tuple((ref(tree.left[n]), ref(tree.right[n])) for n in slots)


def _lut(shape: tuple) -> list[int]:
    if not shape:
        return [0] * 128
    table = []
    for mask in range(128):
        node = 0
        while True:
            kind, index = shape[node][(mask >> node) & 1]
            if kind == "e":
                table.append(index)
                break
            node = index
    return table


def generate_c(forest: Forest, *, calibration: np.ndarray | None = None, lanes: int = 8, max_nodes: int = 7,
               mode: str = "gather", early_exit: bool = True, group_by_height: bool = True,
               entry: str = "predict_row", chained: bool = False) -> tuple[str, dict]:
    if calibration is None:
        raise ValueError("probability-guided tiling requires calibration data")
    if isinstance(lanes, bool) or lanes not in (1, 2, 4, 8, 16, 32):
        raise ValueError("lanes must be 1, 2, 4, 8, 16 or 32")
    if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or not 1 <= max_nodes <= 7:
        raise ValueError("max_nodes must be an integer from 1 to 7")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    if mode != "scalar" and platform.machine() not in ("x86_64", "AMD64"):
        raise ValueError("vector tile modes require x86-64 with AVX2")
    counts = forest.branch_counts(calibration)
    n_trees = len(forest.trees)
    shapes, luts = {(): 0}, [_lut(())]
    records, shape_of, roots, out_index, segments = [], [], [], [], []
    max_depth_seen = 0
    for g, members in enumerate(plan_groups(forest, lanes, group_by_height)):
        depth = 0
        for t in members:
            tree = forest.trees[t]
            tiles, owner = _tiles(tree, reach_counts(tree, counts[t]), max_nodes)
            base = len(records)
            tile_depth = {0: 1}
            for i, (slots, exits) in enumerate(tiles):
                for e in exits:
                    tile_depth[owner[e]] = tile_depth[i] + 1
            # A leaf tile needs no step; the path length is the number of internal tiles on it.
            depth = max(depth, max(tile_depth[i] - 1 for i in range(len(tiles))) if len(tiles) > 1 else 0)
            for slots, exits in tiles:
                shape = _shape(tree, slots, exits) if exits else ()
                if shape not in shapes:
                    shapes[shape] = len(luts)
                    luts.append(_lut(shape))
                if exits:
                    t_row = [tree.value[n] for n in slots] + [0.0] * (8 - len(slots))
                    f_row = [tree.feature[n] | (0 if tree.default_left[n] else 1 << 31) for n in slots]
                    f_row += [0] * (8 - len(slots))
                    c_row = [base + owner[e] for e in exits] + [0] * (8 - len(exits))
                else:
                    me = len(records)  # Leaf tiles point every exit at themselves.
                    t_row, f_row, c_row = [tree.value[slots[0]]] + [0.0] * 7, [0] * 8, [me] * 8
                records.append((t_row, f_row, c_row))
                shape_of.append(shapes[shape])
            roots.append(base)
        roots.extend([roots[-len(members)]] * (lanes - len(members)))
        out_index.extend(members + [n_trees] * (lanes - len(members)))
        max_depth_seen = max(max_depth_seen, depth)
        if segments and segments[-1][0] == depth:
            segments[-1][2] = g + 1
        else:
            segments.append([depth, g, g + 1])
    lut_type = "uint8_t"
    src = ["/* Generated by xgb_latency.probtiled; exact float32 thresholds and leaves. */\n",
           "#include <stdint.h>\n"]
    if mode != "scalar":
        src.append("#include <immintrin.h>\n#ifndef __AVX2__\n#error \"probtiled vector modes need AVX2\"\n#endif\n")
    src.append("typedef struct { float t[8]; uint32_t f[8]; uint32_t c[8]; } ptile_t;\n"
               "static const ptile_t P[] __attribute__((aligned(32))) = {"
               + ",".join("{{" + ",".join(hexf(v) for v in t) + "},{" + ",".join(f"{v}u" for v in f) + "},{"
                          + ",".join(f"{v}u" for v in c) + "}}" for t, f, c in records) + "};\n")
    src += [array("uint16_t", "SHAPE", shape_of), array(lut_type, "PLUT", [v for table in luts for v in table]),
            array("uint32_t", "ROOT", roots), array("uint32_t", "OUT", out_index),
            signature(entry, chained), f"  float val[{n_trees + 1}];\n"]
    for depth, g0, g1 in segments:
        src.append(f"  for (uint32_t g = {g0}u; g < {g1}u; ++g) {{ /* longest tile path {depth} */\n"
                   f"    const uint32_t *root = ROOT + g * {lanes}u;\n")
        src.append("".join(f"    uint32_t i{w} = root[{w}], m{w};\n" for w in range(lanes)))
        src.append(f"    for (uint32_t step = 0; step < {depth}u; ++step) {{\n")
        if early_exit:
            src.append("      if (!(" + " | ".join(f"SHAPE[i{w}]" for w in range(lanes)) + ")) break;\n")
        for w in range(lanes):
            src.append(_step(mode, w, 3, f"&P[i{w}]", "ptile_t"))
        src.append("".join(f"      i{w} = P[i{w}].c[PLUT[SHAPE[i{w}] * 128u + m{w}]];\n" for w in range(lanes)))
        src.append("    }\n")
        src.append("".join(f"    val[OUT[g * {lanes}u + {w}u]] = P[i{w}].t[0];\n" for w in range(lanes)))
        src.append("  }\n")
    src.append(sequential_sum(forest, "val", chained) + "}\n")
    stats = {"lanes": lanes, "max_nodes": max_nodes, "mode": mode, "early_exit": early_exit,
             "group_by_height": group_by_height, "tiles": len(records), "shapes": len(luts),
             "tile_table_bytes": 96 * len(records), "lut_bytes": 128 * len(luts),
             "longest_tile_path": max_depth_seen, "path_segments": [s[0] for s in segments]}
    return "".join(src), stats


def compile_probtiled(model_path: str | Path, output_dir: str | Path, *, calibration: np.ndarray, lanes: int = 8,
                      max_nodes: int = 7, mode: str = "gather", early_exit: bool = True,
                      group_by_height: bool = True, cc: str = "clang", optimization: str = "O3") -> Path:
    """Exports ``void predict_row(const float *, float *)`` like ``compile_model``."""
    started = time.perf_counter()
    forest = Forest.load(model_path)
    source, stats = generate_c(forest, calibration=calibration, lanes=lanes, max_nodes=max_nodes, mode=mode,
                               early_exit=early_exit, group_by_height=group_by_height)
    return build(source, forest, model_path, output_dir, lowering="probtiled", cc=cc, optimization=optimization,
                 started=started, extra={"exact_accumulation_order": True, **stats})
