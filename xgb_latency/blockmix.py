"""Per-block mixed lowering: each contiguous range of trees uses its own strategy.

Blocks are contiguous in the original tree order and each block function takes
and returns the running float32 accumulator (``acc = block_k(row, acc)``), so
the sum is exactly base margin + tree 0 + tree 1 + ... regardless of how each
block evaluates its trees internally. Every block lives in its own translation
unit, keeping its static tables private.

Strategy specs are strings ``kind`` or ``kind:key=value,...``, for example
``qs:stride=1``, ``vpred:lanes=8,layout=level``, ``packed:lanes=16,layout=forest``,
``rs:dense_budget_bytes=0`` or ``direct:select_depth=1``.
"""

from pathlib import Path
import time

import numpy as np

from . import direct, packed, probtiled, quickscorer, rapidscorer, tiled, vpred
from .cgen import build, hexf, subforest
from .model import Forest

KINDS = {"qs": quickscorer.generate_c, "vpred": vpred.generate_c, "packed": packed.generate_c,
         "rs": rapidscorer.generate_c, "direct": direct.generate_c, "tiled": tiled.generate_c,
         "probtiled": probtiled.generate_c}
_CALIBRATED = {"packed", "direct", "probtiled"}


def parse_spec(spec: str) -> tuple[str, dict]:
    kind, _, rest = spec.partition(":")
    if kind not in KINDS:
        raise ValueError(f"unknown strategy kind {kind!r}; expected one of {', '.join(KINDS)}")
    options = {}
    for item in filter(None, rest.split(",")):
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"bad option {item!r} in {spec!r}")
        if value in ("true", "false"):
            options[key] = value == "true"
        elif value.lstrip("-").isdigit():
            options[key] = int(value)
        else:
            options[key] = value
    return kind, options


def generate_block(forest: Forest, spec: str, calibration: np.ndarray | None, entry: str,
                   chained: bool) -> tuple[str, dict]:
    kind, options = parse_spec(spec)
    if kind in _CALIBRATED:
        options.setdefault("calibration", calibration)
    return KINDS[kind](forest, entry=entry, chained=chained, **options)


def equal_blocks(n_trees: int, count: int) -> list[tuple[int, int]]:
    bounds = [round(i * n_trees / count) for i in range(count + 1)]
    return [(a, b) for a, b in zip(bounds, bounds[1:]) if b > a]


def compile_blockmix(model_path: str | Path, output_dir: str | Path, blocks: list[tuple[int, int, str]], *,
                     calibration: np.ndarray | None = None, cc: str = "clang", optimization: str = "O3") -> Path:
    """``blocks`` = [(start, stop, spec), ...] covering every tree in order."""
    started = time.perf_counter()
    forest = Forest.load(model_path)
    if not blocks or blocks[0][0] != 0 or blocks[-1][1] != len(forest.trees) or any(
            a >= b for a, b, _ in blocks) or any(blocks[i][1] != blocks[i + 1][0] for i in range(len(blocks) - 1)):
        raise ValueError("blocks must be nonempty contiguous ranges covering all trees in order")
    sources, stats = {}, []
    for i, (start, stop, spec) in enumerate(blocks):
        source, block_stats = generate_block(subforest(forest, start, stop), spec, calibration, f"block{i}", True)
        sources[f"block{i}"] = source
        stats.append({"start": start, "stop": stop, "spec": spec, **block_stats})
    main = ["#include <stdint.h>\n"]
    main += [f"float block{i}(const float *row, float acc);\n" for i in range(len(blocks))]
    main.append("\n__attribute__((visibility(\"default\")))\n"
                "void predict_row(const float *row, float *raw_margin) {\n"
                f"  float acc = {hexf(forest.base_margin)};\n")
    main += [f"  acc = block{i}(row, acc);\n" for i in range(len(blocks))]
    main.append("  *raw_margin = acc;\n}\n")
    sources["main"] = "".join(main)
    return build(sources, forest, model_path, output_dir, lowering="blockmix", cc=cc, optimization=optimization,
                 started=started, extra={"exact_accumulation_order": True, "blocks": stats})


def compile_spec(model_path: str | Path, output_dir: str | Path, spec: str, *,
                 calibration: np.ndarray | None = None, start: int = 0, stop: int | None = None,
                 cc: str = "clang", optimization: str = "O3") -> Path:
    """Standalone library for one strategy over trees [start, stop) (whole model by default).

    A partial range is a sub-model used for per-block tuning; its output is that
    range's sum starting from the model's base margin.
    """
    started = time.perf_counter()
    forest = Forest.load(model_path)
    stop = len(forest.trees) if stop is None else stop
    part = subforest(forest, start, stop)
    source, stats = generate_block(part, spec, calibration, "predict_row", False)
    return build(source, part, model_path, output_dir, lowering=parse_spec(spec)[0], cc=cc,
                 optimization=optimization, started=started,
                 extra={"exact_accumulation_order": True, "spec": spec, "tree_range": [start, stop], **stats})
