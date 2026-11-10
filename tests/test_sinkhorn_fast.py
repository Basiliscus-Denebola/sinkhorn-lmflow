from dataclasses import replace
import numpy as np
import pytest
from sinkhorn_lmflow.types import SolverConfig
from sinkhorn_lmflow.problems.mainstream import make_mainstream
from sinkhorn_lmflow.problems.weak_coupling import weak_coupling_problem,initial_columns
from sinkhorn_lmflow.core.semi_dual import ReducedWork,evaluate_reduced
from sinkhorn_lmflow.solvers.sinkhorn_fast import LogEvaluator,solve_sk_log_fast,solve_sk_scaling_fast,prepare_fast_warmup
from sinkhorn_lmflow.solvers.mirror_sr1 import base_diagonal,solve_mirror_sr1,solve_sinkhorn_reduced


def problem(epsilon=.1):
    return make_mainstream(dict(case_id='test',family='M3',n=17,m=23,data_seed=42,epsilon=epsilon,alpha=1.))[0]


def config(**kw):
    return SolverConfig(**dict(dict(tol=1e-8,max_iter=1000,compute_spectrum_every=0,max_backtracks=5),**kw))


def test_light_multistep_endpoints_and_cache():
    p=problem();work=ReducedWork();ev=LogEvaluator(p,work)
    z=np.linspace(-1,1,23);light=ev(z);full=evaluate_reduced(p,z)
    for _ in range(25):
        np.testing.assert_allclose(light.c,full.c,rtol=1e-12,atol=2e-15)
        np.testing.assert_allclose(light.z_col,full.z_col,atol=2e-12)
        light=ev(light.z_col-base_diagonal(light.c,p.b,light.log_c)[1])
        full=evaluate_reduced(p,full.z_col-base_diagonal(full.c,p.b,full.log_c)[1])
    assert work.plan_build_count==work.bregman_count==0 and work.state_eval_count==26
    assert work.logsumexp_count==52 and work.log_kernel_build_count==1


def test_scaling_equivalence_and_underflow():
    p=problem();cfg=config(max_iter=20,tol=1e-14)
    a=solve_sk_log_fast(p,cfg);b=solve_sk_scaling_fast(p,cfg)
    np.testing.assert_allclose(a.nu,b.nu,atol=2e-13)
    np.testing.assert_allclose([r.residual_l1 for r in a.history],[r.residual_l1 for r in b.history],atol=2e-14)
    r=solve_sk_scaling_fast(problem(.001),cfg)
    assert r.stop_reason=='not_applicable_kernel_underflow' and not r.converged
    assert r.summary['matrix_vector_count']==0


def test_shared_warmup_and_performance_diagnostics_identical():
    p=problem(.03);cfg=config(max_iter=100,sr1_warmup_iters=5)
    _,snap=prepare_fast_warmup(p,cfg)
    perf=solve_mirror_sr1(p,cfg,warmup=snap,performance=True)
    diag=solve_mirror_sr1(p,cfg,warmup=snap,performance=True,diagnostics=True)
    assert [r.state_fingerprint for r in perf.history]==[r.state_fingerprint for r in diag.history]
    assert [r.step_size for r in perf.history]==[r.step_size for r in diag.history]
    assert [r.accept_reason for r in perf.history]==[r.accept_reason for r in diag.history]
    assert snap.fingerprint==snap.digest()
    bad=snap.clone();bad.state.z_col[0]+=1
    with pytest.raises(ValueError):solve_sk_log_fast(p,cfg,warmup=bad)
    assert perf.summary['state_eval_count']==perf.n_iter+2+perf.summary['sr1_reject_count']


def test_zero_warmup_and_early_convergence():
    p=weak_coupling_problem(16,2,.01)
    cfg=config(sr1_warmup_iters=0,initial_nu=-initial_columns(16,2,42),max_iter=2)
    _,snap=prepare_fast_warmup(p,cfg)
    assert snap.previous is None and snap.iteration==0
    r=solve_mirror_sr1(p,cfg,warmup=snap,performance=True)
    assert r.history[1].sr1_reason=='no_history' and r.history[1].phase=='sk_fallback'
    cfg=config();_,snap=prepare_fast_warmup(p,cfg)
    assert snap.iteration==0
    r=solve_mirror_sr1(p,cfg,warmup=snap,performance=True)
    assert r.stop_reason=='warmup_converged' and r.summary['plan_build_count']==0


def test_cache_preserves_v0_updates_and_amplitude_definition():
    p=problem(.05);cfg=config(sr1_warmup_iters=3,max_iter=30)
    old=solve_mirror_sr1(p,cfg)
    perf=solve_mirror_sr1(p,cfg,performance=True)
    np.testing.assert_array_equal(old.nu,perf.nu)
    assert [r.step_size for r in old.history]==[r.step_size for r in perf.history]
    assert perf.summary['objective_diff_count']<old.summary['objective_diff_count']
    amp=solve_mirror_sr1(p,cfg,performance=True,amplification=True)
    assert amp.solver_name=='mirror_amp_ablation'
    assert all(r.sr1_cos==1 for r in amp.history if r.phase=='sr1')
