import pytest

from xgb_latency import compile_model
from xgb_latency.compiler import _cost_select_nodes
from xgb_latency.model import Tree


def test_cost_policy_distinguishes_predictability_and_depth():
    tree = Tree((1, 3, 5, -1, -1, -1, -1), (2, 4, 6, -1, -1, -1, -1),
                (0,) * 7, (0.,) * 7, (False,) * 7, (2, 1, 1, 0, 0, 0, 0))
    balanced = {0: (50, 50), 1: (25, 25), 2: (25, 25)}
    predictable = {0: (100, 0), 1: (100, 0), 2: (0, 0)}
    assert not _cost_select_nodes(tree, balanced, 2, 0.)
    assert _cost_select_nodes(tree, balanced, 2, 4.) == {0, 1, 2}
    assert _cost_select_nodes(tree, balanced, 1, 4.) == {1, 2}
    assert not _cost_select_nodes(tree, predictable, 2, 4.)
    assert not _cost_select_nodes(tree, balanced, 0, 4.)


@pytest.mark.parametrize("penalty", [-1, float("nan"), float("inf"), "4"])
def test_invalid_penalty(tmp_path, penalty):
    with pytest.raises(ValueError, match="select_branch_penalty"):
        compile_model(tmp_path / "unused.json", tmp_path, select_branch_penalty=penalty)


def test_cost_requires_calibration(tmp_path):
    with pytest.raises(ValueError, match="calibration"):
        compile_model(tmp_path / "unused.json", tmp_path, select_policy="cost")
