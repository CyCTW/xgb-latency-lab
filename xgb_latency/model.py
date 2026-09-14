"""A deliberately small, validated frontend for XGBoost save_model JSON."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Tree:
    left: tuple[int, ...]
    right: tuple[int, ...]
    feature: tuple[int, ...]
    value: tuple[float, ...]
    default_left: tuple[bool, ...]
    height: tuple[int, ...]


@dataclass(frozen=True)
class Forest:
    trees: tuple[Tree, ...]
    num_feature: int
    base_margin: float
    objective: str
    version: tuple[int, ...]

    @classmethod
    def load(cls, path: str | Path) -> "Forest":
        doc = json.loads(Path(path).read_text())
        if not isinstance(doc, dict) or "learner" not in doc:
            raise ValueError("Use booster.save_model('model.json'), not dump_model JSON")
        learner = doc["learner"]
        params = learner["learner_model_param"]
        gb = learner["gradient_booster"]
        if gb["name"] != "gbtree":
            raise ValueError("Only gbtree is supported (no DART or gblinear)")
        if int(params.get("num_target", 1)) != 1 or int(params["num_class"]) != 0:
            raise ValueError("Only single-output regression and binary models are supported")
        objective = learner["objective"]["name"]
        supported = {"reg:squarederror", "reg:absoluteerror", "binary:logistic", "reg:logistic"}
        if objective not in supported:
            raise ValueError(f"Unsupported objective: {objective}")
        score = json.loads(str(params["base_score"]))
        if isinstance(score, list):
            if len(score) != 1:
                raise ValueError("Expected scalar base_score")
            score = score[0]
        base = np.float32(score)
        if not np.isfinite(base):
            raise ValueError("base_score must be finite")
        if objective in {"binary:logistic", "reg:logistic"}:
            if not 0 < base < 1:
                raise ValueError("Logistic base_score must be in (0, 1)")
            # Match the float32 arithmetic of XGBoost's ProbToMargin.
            base = -np.log(np.float32(1) / base - np.float32(1))
        nf = int(params["num_feature"])
        if nf < 1:
            raise ValueError("Model must declare at least one feature")
        model = gb["model"]
        if any(v != 0 for v in model["tree_info"]):
            raise ValueError("Only scalar-output trees are supported")
        trees = []
        for raw in model["trees"]:
            p = raw["tree_param"]
            if int(p.get("size_leaf_vector", 1)) not in (0, 1):
                raise ValueError("Vector leaves are unsupported")
            if int(p.get("num_deleted", 0)):
                raise ValueError("Pruned models with deleted nodes are unsupported")
            if any(raw.get("split_type", [])) or raw.get("categories"):
                raise ValueError("Categorical splits are unsupported")
            n = int(p["num_nodes"])
            keys = ("left_children", "right_children", "split_indices", "split_conditions", "default_left")
            if n < 1 or any(len(raw[k]) != n for k in keys):
                raise ValueError("Invalid tree array lengths")
            left, right = tuple(raw[keys[0]]), tuple(raw[keys[1]])
            features = tuple(raw[keys[2]])
            values = tuple(float(np.float32(v)) for v in raw[keys[3]])
            if not all(np.isfinite(values)):
                raise ValueError("Nonfinite thresholds/leaves are unsupported")
            if any(v not in (0, 1, False, True) for v in raw[keys[4]]):
                raise ValueError("Invalid default_left flag")
            heights = [0] * n
            seen = set()
            stack = [(0, False, 0)]
            while stack:
                node, visited, depth = stack.pop()
                if visited:
                    if left[node] != -1:
                        heights[node] = 1 + max(heights[left[node]], heights[right[node]])
                    continue
                if node in seen or not 0 <= node < n:
                    raise ValueError("Invalid tree: cycle, shared child, or out-of-bounds node")
                if depth > 64:
                    raise ValueError("Prototype supports tree depths up to 64")
                seen.add(node)
                if left[node] == -1:
                    if right[node] != -1:
                        raise ValueError("Invalid leaf")
                else:
                    if not 0 <= features[node] < nf or right[node] < 0 or left[node] < 0:
                        raise ValueError("Invalid split")
                    stack.extend([(node, True, depth), (right[node], False, depth + 1),
                                  (left[node], False, depth + 1)])
            if len(seen) != n:
                raise ValueError("Unreachable nodes in tree")
            trees.append(Tree(left, right, features, values,
                              tuple(bool(v) for v in raw[keys[4]]), tuple(heights)))
        if len(model["tree_info"]) != len(trees):
            raise ValueError("tree_info length mismatch")
        return cls(tuple(trees), nf, float(base), objective, tuple(doc["version"]))

    def branch_counts(self, rows: np.ndarray) -> list[dict[int, tuple[int, int]]]:
        rows = np.asarray(rows, dtype=np.float32)
        if rows.ndim != 2 or rows.shape[1] != self.num_feature or len(rows) == 0:
            raise ValueError("Calibration requires a nonempty (rows, num_feature) array")
        result = []
        for tree in self.trees:
            counts = {}
            stack = [(0, np.arange(len(rows)))]
            while stack:
                node, indices = stack.pop()
                if tree.left[node] == -1:
                    continue
                x = rows[indices, tree.feature[node]]
                go_left = np.where(np.isnan(x), tree.default_left[node], x < tree.value[node])
                nl = int(go_left.sum())
                counts[node] = (nl, len(indices) - nl)
                stack.extend([(tree.left[node], indices[go_left]),
                              (tree.right[node], indices[~go_left])])
            result.append(counts)
        return result
