import numpy as np
from sinkhorn_lmflow.problems.mainstream import make_mainstream
from sinkhorn_lmflow.solvers.lbfgs_reduced import reduced_value_gradient,solve_lbfgs_reduced
from sinkhorn_lmflow.solvers.sinkhorn_fast import prepare_fast_warmup
from sinkhorn_lmflow.types import SolverConfig
from sinkhorn_lmflow.core.transport import transport_plan_from_potentials
from sinkhorn_lmflow.core.residuals import compute_stop_metrics


def test_reduced_gradient_and_true_stopping():
    p,_=make_mainstream(dict(case_id='lbfgs',family='M2',n=9,m=7,data_seed=42,epsilon=.1))
    x=np.linspace(-.3,.5,6);f,g,_=reduced_value_gradient(p,x)
    h=1e-5;fd=np.array([(reduced_value_gradient(p,x+h*d)[0]-reduced_value_gradient(p,x-h*d)[0])/(2*h) for d in np.eye(6)])
    np.testing.assert_allclose(g,fd,atol=1e-10)
    cfg=SolverConfig(max_iter=1000,tol=1e-8,sr1_warmup_iters=3,compute_spectrum_every=0,max_backtracks=5)
    _,snap=prepare_fast_warmup(p,cfg);r=solve_lbfgs_reduced(p,cfg,snap)
    rho=compute_stop_metrics(transport_plan_from_potentials(p.M,p.epsilon,r.mu,r.nu),p.a,p.b)['residual_l1']
    assert r.converged==(rho<=cfg.tol)
    assert r.summary['objective_gradient_count']>0 and r.summary['bregman_count']==0
    assert r.history[:4]==snap.history
