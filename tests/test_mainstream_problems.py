import numpy as np
from sinkhorn_lmflow.problems.mainstream import make_mainstream,main_cases,point_pools
from sinkhorn_lmflow.io.problem_io import problem_fingerprint


def case(**kw):
    return dict(dict(case_id='small',family='M3',n=31,m=23,data_seed=42,epsilon=.003,alpha=None),**kw)


def test_fixed_pools_and_independent_marginals():
    p,a=make_mainstream(case())
    q,b=make_mainstream(case(epsilon=.01,alpha=.1))
    np.testing.assert_array_equal(p.M,q.M)
    np.testing.assert_array_equal(a['x'],b['x'])
    assert q.a.min()>0 and q.a.min()<1e-12 and q.metadata['marginal_floor'] is None
    np.testing.assert_array_equal(a['x'],point_pools('mixture',42)[0][:31])
    r,c=make_mainstream(case(family='M6'))
    assert problem_fingerprint(r)==problem_fingerprint(p)
    assert not np.array_equal(p.a,q.a)


def test_plan_and_random_cost():
    plan=main_cases()
    assert len(plan)==57 and len({x['case_id'] for x in plan})==57
    assert sum(x['family']=='M5' for x in plan)==20
    p,_=make_mainstream(case(family='M1'))
    assert p.M.max()==1 and p.M.min()>0
    assert np.linalg.matrix_rank(p.M)==23
