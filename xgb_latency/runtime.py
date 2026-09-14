"""Convenience wrapper; latency measurements must use the native harness."""

import ctypes
import json
from pathlib import Path

import numpy as np


class Predictor:
    def __init__(self, library: str | Path):
        library = Path(library).resolve()
        self.metadata = json.loads((library.parent / "metadata.json").read_text())
        self._lib = ctypes.CDLL(str(library))
        self._fn = self._lib.predict_row
        self._ptr = ctypes.POINTER(ctypes.c_float)
        self._fn.argtypes = [self._ptr, self._ptr]
        self._fn.restype = None

    def predict(self, rows: np.ndarray) -> np.ndarray:
        rows = np.require(rows, dtype=np.float32, requirements=["C", "A"])
        if rows.ndim != 2 or rows.shape[1] != self.metadata["num_feature"]:
            raise ValueError("Expected shape (rows, num_feature)")
        out = np.empty(len(rows), dtype=np.float32)
        for i, row in enumerate(rows):
            self._fn(row.ctypes.data_as(self._ptr),
                     ctypes.cast(out.ctypes.data + i * 4, self._ptr))
        return out
