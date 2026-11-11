from collections import defaultdict
from dataclasses import replace
from time import perf_counter
import numpy as np
import pytest
from sinkhorn_lmflow.types import SolverConfig
from sinkhorn_lmflow.problems.mainstream import make_mainstream
from sinkhorn_lmflow.problems.instance import OTProblem
from sinkhorn_lmflow.core.semi_dual import center, ReducedWork, evaluate_reduced, stable_objective_difference
from sinkhorn_lmflow.core.cost_profile_distance import generic_distance
from sinkhorn_lmflow.solvers.sinkhorn_fast import LogEvaluator, prepare_fast_warmup, solve_sk_log_fast, solve_sk_scaling_fast
from sinkhorn_lmflow.solvers.energy_distance import (solve_energy_distance, stable_light_difference,
    energy_terms, frozen_gamma, EqualMassEnergyProblem)


def problem(eps=.05):
    return make_mainstream(dict(case_id='small',family='M3',n=17,m=23,data_seed=42,epsilon=eps,alpha=.4))


def config(**kwargs):
    return SolverConfig(**dict(dict(tol=1e-8,max_iter=1000,sr1_warmup_iters=3,max_backtracks=12),**kwargs))


@pytest.mark.parametrize('scale',[1e-9,1e-5,.2,3.])
def test_blocked_difference_reference(scale):
    p,_=problem();gen=np.random.default_rng(8);z=gen.normal(size=23);d=center(gen.normal(size=23))*scale
    work=ReducedWork();ev=LogEvaluator(p,work);light=ev(z);full=evaluate_reduced(p,z)
    got=stable_light_difference(p,light,d,ev.log_kernel,work,defaultdict(int),block_rows=5)
    expected=stable_objective_difference(p,full,d)
    assert got==pytest.approx(expected,rel=1e-11,abs=1e-25 if scale<1e-8 else 1e-13)
    # Long double direct partition ratio supplies an independent reference.
    lp=np.asarray(full.log_conditional,dtype=np.longdouble)
    prob=np.exp(lp);prob/=prob.sum(1)[:,None]
    dd=np.asarray(d,dtype=np.longdouble)
    reference=np.sum(p.a.astype(np.longdouble)*np.log1p(np.sum(prob*np.expm1(dd),axis=1)))-p.b.astype(np.longdouble)@dd
    assert got==pytest.approx(float(reference),rel=2e-6,abs=2e-15*scale)


def test_unit_sk_zero_and_full_convergence_and_counters():
    p,arrays=problem();cfg=config(max_iter=4,tol=1e-13)
    _,warm=prepare_fast_warmup(p,cfg)
    baseline=solve_sk_log_fast(p,cfg,warm)
    zero=solve_energy_distance(p,cfg,warm,reconstruction=arrays,variant='energy_zero')
    assert zero.history[-1].step_size==1.
    np.testing.assert_allclose(zero.nu,baseline.nu,atol=2e-15)
    assert zero.summary['distance_bytes']==0 and zero.summary['energy_matvec_count']==0
    full=solve_energy_distance(p,config(),reconstruction=arrays)
    assert full.converged and full.summary['last_residual']<=1e-8
    assert full.summary['accepted_updates']>0 and full.summary['objective_diff_count']>0
    assert full.summary['conditional_block_matvec_count']>=full.summary['objective_diff_count']
    assert full.summary['distance_backend']=='exact_geometric'
    parts=sum(full.summary[k] for k in ('warmup_sec','distance_build_sec','gamma_setup_sec','bridge_setup_sec','iteration_sec'))
    assert parts==pytest.approx(full.elapsed_sec,abs=1e-12)


def test_direction_identity_sign_permutation_and_gamma_degenerate():
    p,_=problem();D=generic_distance(p.M,p.a,p.epsilon);ev=LogEvaluator(p,ReducedWork())
    z=np.linspace(-1,1,23);state=ev(z);e=state.c-p.b;psi=center(state.log_c-np.log(p.b))
    u,G=energy_terms(D,e);v=psi+.7*u
    assert e@v==pytest.approx(e@psi+1.4*G,rel=1e-12)
    e_pair=np.zeros(23);e_pair[0]=1;e_pair[1]=-1
    up,_=energy_terms(D,e_pair)
    assert up[0]>up[1]
    perm=np.random.default_rng(1).permutation(23)
    pp=OTProblem(p.a,p.b[perm],p.M[:,perm],p.epsilon)
    sp=LogEvaluator(pp,ReducedWork())(z[perm]);Dp=generic_distance(pp.M,pp.a,pp.epsilon)
    vp=center(sp.log_c-np.log(pp.b))+.7*energy_terms(Dp,sp.c-pp.b)[0]
    np.testing.assert_allclose(vp,v[perm],atol=2e-13)
    new=ev(z-.1*v);newp=LogEvaluator(pp,ReducedWork())(z[perm]-.1*vp)
    np.testing.assert_allclose(newp.c,new.c[perm],atol=1e-14)
    stats=defaultdict(int);gamma,_,_=frozen_gamma(np.zeros_like(D),e,psi,1.,stats)
    assert gamma==0 and stats['geometry_degenerate']


def test_fallback_deadline_underflow_warmup_and_general_mass():
    p,arrays=problem(.001)
    assert solve_sk_scaling_fast(p,config()).stop_reason=='not_applicable_kernel_underflow'
    cfg=config(max_iter=4,max_backtracks=0)
    result=solve_energy_distance(p,cfg,reconstruction=arrays)
    assert result.summary['fallback_count']==1
    _,warm=prepare_fast_warmup(p,cfg)
    stopped=solve_energy_distance(p,cfg,warm,deadline=perf_counter()-1,reconstruction=arrays)
    assert stopped.stop_reason=='timeout' and stopped.n_iter==warm.iteration
    p0=OTProblem(np.ones(3)/3,np.ones(4)/4,np.zeros((3,4)),.1)
    early=solve_energy_distance(p0,config())
    assert early.stop_reason=='warmup_converged' and early.summary['distance_bytes']==0
    p,arrays=problem()
    general=EqualMassEnergyProblem(3*p.a,3*p.b,p.M,p.epsilon)
    r=solve_energy_distance(general,config())
    assert r.converged and r.summary['last_residual']<=1e-8
    P=np.exp(-(general.M+r.mu[:,None]+r.nu[None,:])/general.epsilon)
    assert np.abs(P.sum(0)-general.b).sum()/3<=1e-8


def test_amplitude_same_frozen_gamma_independent_trajectory():
    p,a=problem();cfg=config(max_iter=15)
    _,warm=prepare_fast_warmup(p,cfg)
    full=solve_energy_distance(p,cfg,warm,reconstruction=a)
    amp=solve_energy_distance(p,cfg,warm,reconstruction=a,variant='energy_amp',gamma_override=full.summary['gamma'])
    assert amp.summary['gamma']==full.summary['gamma']
    assert amp.summary['distance_bytes']>0 and amp.summary['energy_matvec_count']>0
    assert any(x.state_fingerprint!=y.state_fingerprint for x,y in zip(full.history,amp.history))


def test_near_solution_difference_with_decimal_reference():
    from decimal import Decimal, localcontext
    p,_=problem(.1)
    solved=solve_sk_log_fast(p,config(max_iter=3000,tol=1e-14))
    ev=LogEvaluator(p,ReducedWork());state=ev(-solved.nu/p.epsilon)
    displacement=center(np.linspace(-1,1,len(p.b)))*1e-7
    value=stable_light_difference(p,state,displacement,ev.log_kernel,ReducedWork(),defaultdict(int))
    with localcontext() as ctx:
        ctx.prec=60
        decimal=lambda v: Decimal.from_float(float(v))
        reference=Decimal(0)
        for i in range(len(p.a)):
            weights=[(decimal(ev.log_kernel[i,j])+decimal(state.z_col[j])).exp() for j in range(len(p.b))]
            den=sum(weights)
            num=sum(w*decimal(displacement[j]).exp() for j,w in enumerate(weights))
            reference+=decimal(p.a[i])*(num/den).ln()
        reference-=sum(decimal(b)*decimal(d) for b,d in zip(p.b,displacement))
    assert value>0
    assert value==pytest.approx(float(reference),abs=2e-21,rel=1e-5)


def test_nonfinite_direction_falls_back_without_parameter_clipping(monkeypatch):
    import sinkhorn_lmflow.solvers.energy_distance as module
    original=module.frozen_gamma
    def nonfinite(*args):
        _,u,e=original(*args)
        return float('inf'),u,e
    monkeypatch.setattr(module,'frozen_gamma',nonfinite)
    p,a=problem();cfg=config(max_iter=4)
    result=solve_energy_distance(p,cfg,reconstruction=a)
    assert result.summary['fallback_reasons']=={'nonfinite_direction':1}
    assert np.isinf(result.summary['gamma'])
    assert np.isfinite(result.nu).all()
