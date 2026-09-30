"""Shared helpers for lowerings that emit C and compile it with Clang."""

import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

from .model import Forest, Tree

HEADER = ('#pragma once\n#ifdef __cplusplus\nextern "C" {\n#endif\n'
          '/* Dense float32, NaN missing; caller provides one output float. */\n'
          'void predict_row(const float *features, float *raw_margin);\n'
          '#ifdef __cplusplus\n}\n#endif\n')


def hexf(value: float) -> str:
    """Exact C float literal (NaN becomes a quiet NaN)."""
    value = float(np.float32(value))
    if math.isnan(value):
        return "__builtin_nanf(\"\")"
    if math.isinf(value):
        return "__builtin_inff()" if value > 0 else "(-__builtin_inff())"
    return value.hex() + "f"


def array(ctype: str, name: str, values, fmt=str, align: int = 64) -> str:
    body = ",".join(fmt(v) for v in values) or "0"
    return f"static const {ctype} {name}[] __attribute__((aligned({align}))) = {{{body}}};\n"


def signature(entry: str = "predict_row", chained: bool = False) -> str:
    """Standalone: ``void predict_row(row, raw_margin)``. Chained block:
    ``float ENTRY(row, acc)`` adds its trees to the running sum and returns it."""
    if chained:
        return f"\n__attribute__((visibility(\"hidden\")))\nfloat {entry}(const float *row, float acc) {{\n"
    return f"\n__attribute__((visibility(\"default\")))\nvoid {entry}(const float *row, float *raw_margin) {{\n"


def sequential_sum(forest: Forest, values: str, chained: bool = False) -> str:
    """Exact accumulation contract: base margin, then each tree in original order."""
    loop = f"  for (uint32_t t = 0; t < {len(forest.trees)}u; ++t) acc += {values}[t];\n"
    if chained:
        return loop + "  return acc;\n"
    return f"  float acc = {hexf(forest.base_margin)};\n" + loop + "  *raw_margin = acc;\n"


def subforest(forest: Forest, start: int, stop: int) -> Forest:
    return Forest(forest.trees[start:stop], forest.num_feature, forest.base_margin, forest.objective, forest.version)


def reach_counts(tree: Tree, counts: dict[int, tuple[int, int]] | None) -> list[float]:
    """Calibration rows reaching each node; unobserved nodes get 0 (placed last, still exact)."""
    reach = [0.0] * len(tree.left)
    reach[0] = 1.0 if counts is None else float(sum(counts.get(0, (1, 0))))
    order = [0]
    for node in order:
        if tree.left[node] == -1:
            continue
        if counts is None:
            nl = nr = reach[node] / 2
        else:
            nl, nr = counts[node]
        reach[tree.left[node]], reach[tree.right[node]] = float(nl), float(nr)
        order.extend([tree.left[node], tree.right[node]])
    return reach


def build(source: str, forest: Forest, model_path, output_dir, *, lowering: str, cc: str = "clang",
          optimization: str = "O3", extra: dict | None = None, started: float | None = None) -> Path:
    """Write model.c/.o/.s/.so, model.h and metadata.json; return the shared library path."""
    import time
    if optimization not in {"O3", "O2", "Os"}:
        raise ValueError("optimization must be O3, O2 or Os")
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Several translation units keep each block's static tables private.
    sources = {"model": source} if isinstance(source, str) else dict(source)
    native = "-mcpu=native" if platform.machine() in ("arm64", "aarch64") else "-march=native"
    flags = ["-" + optimization, native, "-fPIC", "-fno-fast-math", "-ffp-contract=off", "-std=c11"]
    lib = out / ("model.dylib" if sys.platform == "darwin" else "model.so")
    objects = []
    for name, text in sources.items():
        (out / f"{name}.c").write_text(text)
        subprocess.run([cc, *flags, "-c", str(out / f"{name}.c"), "-o", str(out / f"{name}.o")],
                       check=True, capture_output=True)
        subprocess.run([cc, *flags, "-S", str(out / f"{name}.c"), "-o", str(out / f"{name}.s")],
                       check=True, capture_output=True)
        objects.append(str(out / f"{name}.o"))
    if len(objects) > 1:  # Keep a single model.o artifact like the other lowerings.
        subprocess.run(["ld", "-r", *objects, "-o", str(out / "model.o")], check=True, capture_output=True)
    subprocess.run([cc, "-dynamiclib" if sys.platform == "darwin" else "-shared",
                    str(out / "model.o"), "-o", str(lib)], check=True, capture_output=True)
    (out / "model.h").write_text(HEADER)
    metadata = {"format_version": 1, "lowering": lowering, "num_feature": forest.num_feature,
                "num_trees": len(forest.trees), "objective": forest.objective, "output": "raw_margin",
                "base_margin": forest.base_margin,
                "source_sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
                "compiler": subprocess.check_output([cc, "--version"], text=True).splitlines()[0],
                "flags": flags, **(extra or {}),
                "compile_seconds": None if started is None else time.perf_counter() - started,
                "library_bytes": lib.stat().st_size}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return lib
