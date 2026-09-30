"""Treebeard-style tree tiling on top of VPred's implicit complete trees.

A tile covers ``k`` consecutive levels of a (padded) complete tree: ``2**k - 1``
split nodes stored in breadth-first order inside one record (k=2: 3 nodes in a
32-byte record; k=3: 7 nodes in one 64-byte line). One step evaluates every
node of the tile, packs the "go right" bits into a mask, and a universal lookup
table (it depends only on ``k``, since every tile is complete) maps the mask to
the child tile ``0 .. 2**k - 1``. The dependent chain per tree shrinks from
``height`` steps to ``ceil(height / k)`` steps, at the price of evaluating
nodes that are not on the path.

Go-right is ``(x >= t) | (isnan(x) & default_right)``, the exact negation of
XGBoost's routing, so results stay bitwise identical. Trees run ``lanes`` at a
time (grouped by height), with leaf values summed in original order.

Modes:

* ``scalar``: per-node scalar compares OR'ed into the mask.
* ``gather``: AVX2 ``vgatherdps`` loads the tile's feature values, then one
  vector compare and ``movemask``.
* ``insert``: scalar feature loads assembled into a vector, then as ``gather``.

The vector modes require AVX2 (x86-64); generation fails elsewhere.
"""

from pathlib import Path
import platform
import time

from .cgen import array, build, hexf, sequential_sum, signature
from .model import Forest
from .vpred import _complete, plan_groups

MODES = ("scalar", "gather", "insert")


def schedule(height: int, k: int) -> list[int]:
    """Levels covered by each tile step, e.g. height 8, k 3 -> [3, 3, 2]."""
    return [k] * (height // k) + ([height % k] if height % k else [])


def lut(k: int) -> list[int]:
    """Exit child for every mask of a complete k-level tile (bit i = node i goes right)."""
    table = []
    for mask in range(1 << ((1 << k) - 1)):
        i = 0
        for _ in range(k):
            i = 2 * i + 1 + ((mask >> i) & 1)
        table.append(i - ((1 << k) - 1))
    return table


def _step(mode: str, w: int, k: int, rec: str, ctype: str | None = None) -> str:
    """C code setting m{w}: go-right bits of the k-level tile at ``rec``."""
    width, nodes = (4 if k <= 2 else 8), (1 << k) - 1
    keep = (1 << nodes) - 1
    ctype = ctype or f"tile{width}_t"
    if mode == "scalar":
        bits = " | ".join(
            f"((((uint32_t)(x{w}_{i} >= tp->t[{i}])) | ((uint32_t)(x{w}_{i} != x{w}_{i}) & (tp->f[{i}] >> 31))) << {i})"
            for i in range(nodes))
        loads = "".join(f"      const float x{w}_{i} = row[tp->f[{i}] & 0x7fffffffu];\n" for i in range(nodes))
        return f"    {{ const {ctype} *tp = {rec};\n{loads}      m{w} = {bits}; }}\n"
    v, ps, si, cast, sra, cmp, mv, band, bor, andi = (
        ("__m256", "_mm256_load_ps", "__m256i", "_mm256_castsi256_ps", "_mm256_srai_epi32", "_mm256_cmp_ps",
         "_mm256_movemask_ps", "_mm256_and_ps", "_mm256_or_ps", "_mm256_and_si256") if width == 8 else
        ("__m128", "_mm_load_ps", "__m128i", "_mm_castsi128_ps", "_mm_srai_epi32", "_mm_cmp_ps",
         "_mm_movemask_ps", "_mm_and_ps", "_mm_or_ps", "_mm_and_si128"))
    load_i = "_mm256_load_si256((const __m256i *)tp->f)" if width == 8 else "_mm_load_si128((const __m128i *)tp->f)"
    set1 = "_mm256_set1_epi32" if width == 8 else "_mm_set1_epi32"
    if mode == "gather":
        gather = "_mm256_i32gather_ps" if width == 8 else "_mm_i32gather_ps"
        xs = f"{gather}(row, {andi}(fd, {set1}(0x7fffffff)), 4)"
    else:
        setr = "_mm256_setr_ps" if width == 8 else "_mm_setr_ps"
        xs = f"{setr}(" + ", ".join(
            f"row[tp->f[{i}] & 0x7fffffffu]" if i < nodes else "0.0f" for i in range(width)) + ")"
    return (f"    {{ const {ctype} *tp = {rec}; const {si} fd = {load_i};\n"
            f"      const {v} x = {xs}, th = {ps}(tp->t);\n"
            f"      const {v} r = {bor}({cmp}(x, th, _CMP_GE_OQ), {band}({cmp}(x, x, _CMP_UNORD_Q), {cast}({sra}(fd, 31))));\n"
            f"      m{w} = (uint32_t){mv}(r) & {keep}u; }}\n")


def _complete_top(tree, height: int):
    """Complete top ``height`` levels as in ``_complete``; frontier[j] is the node at level ``height``.

    Early leaves above the frontier repeat into every frontier slot beneath them."""
    levels = [[None] * (1 << l) for l in range(height)]
    frontier = [0] * (1 << height)
    stack = [(0, 0, 0)]
    while stack:
        node, level, j = stack.pop()
        if level == height:
            frontier[j] = node
            continue
        if tree.left[node] == -1:
            levels[level][j] = (0.0, 0)
            stack += [(node, level + 1, 2 * j), (node, level + 1, 2 * j + 1)]
        else:
            flag = 0 if tree.default_left[node] else 1 << 31
            levels[level][j] = (tree.value[node], tree.feature[node] | flag)
            stack += [(tree.left[node], level + 1, 2 * j), (tree.right[node], level + 1, 2 * j + 1)]
    return levels, frontier


def generate_c(forest: Forest, *, lanes: int = 8, tile_levels: int = 3, mode: str = "scalar",
               group_by_height: bool = True, top_levels: int | None = None,
               entry: str = "predict_row", chained: bool = False) -> tuple[str, dict]:
    """``top_levels=T`` tiles only the top T levels (padding at most 2**T slots per tree); each
    tile exit then maps to an explicit node and the rest of the tree runs as fixed branch-free
    explicit-pointer steps (16-byte nodes, leaves loop to themselves)."""
    if isinstance(lanes, bool) or lanes not in (1, 2, 4, 8, 16, 32):
        raise ValueError("lanes must be 1, 2, 4, 8, 16 or 32")
    if isinstance(tile_levels, bool) or tile_levels not in (2, 3):
        raise ValueError("tile_levels must be 2 or 3")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    if mode != "scalar" and platform.machine() not in ("x86_64", "AMD64"):
        raise ValueError("vector tile modes require x86-64 with AVX2")
    if top_levels is not None and (isinstance(top_levels, bool) or not isinstance(top_levels, int) or
                                   not 1 <= top_levels <= 24):
        raise ValueError("top_levels must be None or an integer from 1 to 24")
    hybrid = top_levels is not None
    nodes, front, front_base = [], [], []
    n_trees = len(forest.trees)
    groups = plan_groups(forest, lanes, group_by_height)
    tiles = {4: [], 8: []}  # record width -> [(thresholds, features)]
    base = {4: [], 8: []}
    leaves, leaf_base, out_index, segments = [], [], [], []
    for g, members in enumerate(groups):
        height = max(forest.trees[t].height[0] for t in members)
        tiled_height = min(height, top_levels) if hybrid else height
        key = (tiled_height, height - tiled_height)
        if segments and segments[-1][0] == key:
            segments[-1][2] = g + 1
        else:
            segments.append([key, g, g + 1])
        steps = schedule(tiled_height, tile_levels)
        lane_trees = members + [members[0]] * (lanes - len(members))  # Dummy lanes write a scratch slot.
        if hybrid:
            built = []
            for t in lane_trees:
                tree = forest.trees[t]
                levels, frontier = _complete_top(tree, tiled_height)
                index, order = {}, []
                for node in frontier:  # Explicit records for everything at or below the frontier.
                    if node not in index:
                        index[node] = None
                        order.append(node)
                for node in order:
                    if tree.left[node] != -1:
                        for child in (tree.left[node], tree.right[node]):
                            if child not in index:
                                index[child] = None
                                order.append(child)
                start = len(nodes)
                for offset, node in enumerate(order):
                    index[node] = start + offset
                for node in order:
                    if tree.left[node] == -1:
                        nodes.append((tree.value[node], 0, index[node], index[node]))
                    else:
                        flag = 0 if tree.default_left[node] else 1 << 31
                        nodes.append((tree.value[node], tree.feature[node] | flag,
                                      index[tree.left[node]], index[tree.right[node]]))
                front_base.append(len(front))
                front.extend(index[node] for node in frontier)
                built.append((levels, []))
        else:
            built = [_complete(forest.trees[t], height) for t in lane_trees]
        out_index.extend(members + [n_trees] * (lanes - len(members)))
        for levels, tree_leaves in built:
            for width in (4, 8):
                base[width].append(len(tiles[width]))
            level0 = 0
            for k in steps:
                width = 4 if k <= 2 else 8
                for j in range(1 << level0):
                    t_row, f_row = [0.0] * width, [0] * width
                    for d in range(k):
                        for p in range(1 << d):
                            t_row[(1 << d) - 1 + p], f_row[(1 << d) - 1 + p] = levels[level0 + d][j * (1 << d) + p]
                    tiles[width].append((t_row, f_row))
                level0 += k
            leaf_base.append(len(leaves))
            leaves.extend(tree_leaves)
    src = ["/* Generated by xgb_latency.tiled; exact float32 thresholds and leaves. */\n",
           "#include <stdint.h>\n"]
    if mode != "scalar":
        src.append("#include <immintrin.h>\n#ifndef __AVX2__\n#error \"tiled vector modes need AVX2\"\n#endif\n")
    src.append("typedef struct { float t[4]; uint32_t f[4]; } tile4_t;\n"
               "typedef struct { float t[8]; uint32_t f[8]; } tile8_t;\n")
    for width in (4, 8):
        rows = tiles[width] or [([0.0] * width, [0] * width)]
        src.append(f"static const tile{width}_t T{width}[] __attribute__((aligned(64))) = {{"
                   + ",".join("{{" + ",".join(hexf(v) for v in t) + "},{" + ",".join(f"{v}u" for v in f) + "}}"
                              for t, f in rows) + "};\n")
        src.append(array("uint32_t", f"B{width}", base[width]))
    for k in (1, 2, 3):
        src.append(array("uint8_t", f"LUT{k}", lut(k)))
    if hybrid:
        src.append("typedef struct { float t; uint32_t fd; uint32_t l, r; } node_t;\n"
                   "static const node_t NODE[] __attribute__((aligned(64))) = {"
                   + ",".join(f"{{{hexf(v)},{fd}u,{l}u,{r}u}}" for v, fd, l, r in (nodes or [(0.0, 0, 0, 0)])) + "};\n")
        src += [array("uint32_t", "FRONT", front or [0]), array("uint32_t", "FBASE", front_base or [0])]
    src += [array("float", "LEAF", leaves or [0.0], hexf), array("uint32_t", "LBASE", leaf_base or [0]),
            array("uint32_t", "OUT", out_index), signature(entry, chained), f"  float val[{n_trees + 1}];\n"]

    for (height, bottom), g0, g1 in segments:
        steps = schedule(height, tile_levels)
        src.append(f"  for (uint32_t g = {g0}u; g < {g1}u; ++g) {{ /* tiled {height}, explicit {bottom}, tiles {steps} */\n"
                   f"    const uint32_t *b4 = B4 + g * {lanes}u, *b8 = B8 + g * {lanes}u;\n"
                   + (f"    const uint32_t *fb = FBASE + g * {lanes}u;\n" if hybrid else
                      f"    const uint32_t *lb = LBASE + g * {lanes}u;\n"))
        src.append("".join(f"    uint32_t j{w} = 0, m{w};\n" for w in range(lanes)))
        offsets = {4: 0, 8: 0}
        level0 = 0
        for k in steps:
            width = 4 if k <= 2 else 8
            for w in range(lanes):
                src.append(_step(mode, w, k, f"&T{width}[b{width}[{w}] + {offsets[width]}u + j{w}]"))
            src.append("".join(f"    j{w} = j{w} * {1 << k}u + LUT{k}[m{w}];\n" for w in range(lanes)))
            offsets[width] += 1 << level0
            level0 += k
        if hybrid:
            src.append("".join(f"    uint32_t i{w} = FRONT[fb[{w}] + j{w}];\n" for w in range(lanes)))
            for _ in range(bottom):
                for w in range(lanes):
                    src.append(f"    {{ const node_t *n = &NODE[i{w}]; const float x = row[n->fd & 0x7fffffffu];\n"
                               f"      i{w} = ((uint32_t)(x >= n->t) | ((uint32_t)(x != x) & (n->fd >> 31))) ? n->r : n->l; }}\n")
            src.append("".join(f"    val[OUT[g * {lanes}u + {w}u]] = NODE[i{w}].t;\n" for w in range(lanes)))
        else:
            src.append("".join(f"    val[OUT[g * {lanes}u + {w}u]] = LEAF[lb[{w}] + j{w}];\n" for w in range(lanes)))
        src.append("  }\n")
    src.append(sequential_sum(forest, "val", chained) + "}\n")
    stats = {"lanes": lanes, "tile_levels": tile_levels, "mode": mode, "group_by_height": group_by_height,
             "top_levels": top_levels, "height_segments": [list(s[0]) for s in segments],
             "tile_schedules": [schedule(s[0][0], tile_levels) for s in segments],
             "explicit_node_bytes": 16 * len(nodes),
             "tile4_records": len(tiles[4]), "tile8_records": len(tiles[8]),
             "tile_table_bytes": 32 * len(tiles[4]) + 64 * len(tiles[8]), "leaf_table_bytes": 4 * len(leaves)}
    return "".join(src), stats


def compile_tiled(model_path: str | Path, output_dir: str | Path, *, lanes: int = 8, tile_levels: int = 3,
                  mode: str = "scalar", group_by_height: bool = True, top_levels: int | None = None,
                  cc: str = "clang", optimization: str = "O3") -> Path:
    """Exports ``void predict_row(const float *, float *)`` like ``compile_model``."""
    started = time.perf_counter()
    forest = Forest.load(model_path)
    source, stats = generate_c(forest, lanes=lanes, tile_levels=tile_levels, mode=mode,
                               group_by_height=group_by_height, top_levels=top_levels)
    return build(source, forest, model_path, output_dir, lowering="tiled", cc=cc, optimization=optimization,
                 started=started, extra={"exact_accumulation_order": True, **stats})
