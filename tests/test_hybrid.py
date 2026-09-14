from xgb_latency.model import Forest,Tree
from xgb_latency.hybrid import plan_hybrid
import pytest


def test_unobserved_subtree_retained_and_root_probability():
    # root -> two height-two subtrees; calibration only visits the left half.
    tree=Tree((1,3,5,7,-1,9,-1,-1,-1,-1,-1),
              (2,4,6,8,-1,10,-1,-1,-1,-1,-1),
              (0,)*11,(0.,)*11,(True,)*11,(3,2,2,1,0,1,0,0,0,0,0))
    forest=Forest((tree,),1,0.,'reg:squarederror',())
    counts=[{0:(100,0),1:(50,50),2:(0,0),3:(25,25),5:(0,0)}]
    plan=plan_hybrid(forest,counts,3,.01)
    assert plan.roots=={(0,2):0}
    assert plan.ancestors=={(0,0)}
    assert len(plan.records)==5  # The whole unobserved subtree, including leaves.
    assert sum(r[0]==-1 for r in plan.records)==3
    assert not plan_hybrid(forest,counts,0,.01).roots
    assert plan_hybrid(forest,None,64,1.).roots=={(0,0):0}


def test_stumps_and_constant_forest_need_no_interpreter():
    tree=Tree((-1,),(-1,),(0,),(2.,),(False,),(0,))
    forest=Forest((tree,),1,0.,'reg:squarederror',())
    assert not plan_hybrid(forest,[{}],64,0.).roots
    assert not plan_hybrid(Forest((),1,0.,'reg:squarederror',()),None,64,1.).records


def test_compact_rejects_unrepresentable_child_offset():
    from llvmlite import ir
    from xgb_latency.hybrid import HybridPlan,emit_hybrid
    plan=HybridPlan([(2*(1<<28),0.,1,2),(-1,1.,-1,-1),(-1,2.,-1,-1)],{(0,0):0},set())
    with pytest.raises(ValueError,match='cannot encode'):
        emit_hybrid(ir.Module(),plan,'compact')
