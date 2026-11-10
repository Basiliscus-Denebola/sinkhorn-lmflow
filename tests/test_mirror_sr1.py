from dataclasses import replace
import numpy as np
import pytest
from sinkhorn_lmflow.types import SolverConfig
from sinkhorn_lmflow.problems.weak_coupling import weak_coupling_problem, initial_columns
from sinkhorn_lmflow.core.semi_dual import evaluate_reduced, center
import sinkhorn_lmflow.solvers.mirror_sr1 as mod


def cfg(**kwargs):
    return SolverConfig(compute_spectrum_every=0, max_backtracks=5, **kwargs)


def test_base_and_secant_dissipation():
    b = np.array([.2, .3, .5])
    c = np.array([.2, .3*(1+1e-12), .5-.3e-12])
    beta, psi = mod.base_diagonal(c, b, np.log(c))
    np.testing.assert_allclose(beta*(c-b), psi, atol=1e-28)
    assert beta[0] == 1/b[0]
    # A very small column must use finite log_c rather than log1p(-1).
    beta, psi = mod.base_diagonal(np.array([0., 1.]), np.array([.5,.5]), np.array([-1000.,0.]))
    assert np.isfinite(beta).all() and np.isfinite(psi).all()
    p = weak_coupling_problem(16, 2, .01)
    prev = evaluate_reduced(p, initial_columns(16,2,42))
    s = evaluate_reduced(p, prev.z_col-prev.log_c+np.log(p.b))
    d = mod.sr1_direction(s.z_col, s.c, p.b, s.log_c, (prev.z_col, prev.c))
    assert d.valid
    y, secant = s.c-prev.c, s.z_col-prev.z_col
    np.testing.assert_allclose(d.beta*y+d.q*(d.q@y)/d.d, secant, atol=1e-13)
    e = s.c-p.b
    assert e@d.V == pytest.approx(d.J+(d.q@e)**2/d.d, rel=1e-12)
    assert e@d.V >= d.J
    shifted = mod.sr1_direction(s.z_col+12, s.c, p.b, s.log_c, (prev.z_col-4, prev.c))
    np.testing.assert_allclose(shifted.V, d.V, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('kind', ['zero_history','zero_q','nonpositive_denominator','small_denominator','nonfinite'])
def test_guards(kind):
    b = np.array([.5,.5]); c = b.copy(); z = np.zeros(2)
    y = np.array([.01,-.01]); beta = 2.
    if kind == 'zero_history':
        previous = (z,c)
    elif kind == 'zero_q':
        # Binary-exact differences make q exactly zero.
        y = np.array([.125,-.125]); previous = (-beta*y,c-y)
    elif kind == 'nonpositive_denominator':
        previous = (-y,c-y)
    elif kind == 'small_denominator':
        q = np.array([1.,1.])+1e-10*np.array([1.,-1.])
        # Three coordinates allow centered q almost orthogonal to y.
        b = c = np.full(3,1/3); z = np.zeros(3)
        y = np.array([.01,-.01,0]); q = np.array([1.,1.,-2.])+1e-10*np.array([1.,-1.,0.])
        previous = (-(3*y+q), c-y)
    else:
        previous = (np.full(2,np.nan),c-y)
    d = mod.sr1_direction(z,c,b,np.log(c),previous)
    assert d.reason == kind


def test_disabled_correction_and_cache(monkeypatch):
    p = weak_coupling_problem(16,2,.01)
    c = cfg(max_iter=12, sr1_warmup_iters=20, initial_nu=-initial_columns(16,2,42))
    a = mod.solve_sinkhorn_reduced(p,c)
    b = mod.solve_mirror_sr1(p,c)
    np.testing.assert_array_equal(a.nu,b.nu)
    assert a.summary['state_eval_count'] == 13
    assert a.summary['plan_build_count'] == 13
    assert a.summary['objective_diff_count'] == 12
    # Cache entry must never call full-potential reconstruction.
    import sinkhorn_lmflow.solvers.base as base
    monkeypatch.setattr(base,'transport_plan_from_potentials',lambda *a: pytest.fail('cache missed'))
    mod.solve_mirror_sr1(p,c)


@pytest.mark.parametrize('rejects', [1, 6])
def test_backtrack_budget_and_accepted_history(monkeypatch, rejects):
    p = weak_coupling_problem(16,2,.01)
    c = cfg(max_iter=4,sr1_warmup_iters=2,initial_nu=-initial_columns(16,2,42),tol=1e-14)
    _, warm = mod.prepare_warmup(p,c)
    original_diff, original_dir = mod.stable_objective_difference, mod.sr1_direction
    calls, seen = [], []
    def diff(*args):
        calls.append(args[1].z_col.copy())
        return 1. if len(calls) <= rejects else original_diff(*args)
    def direction(z, cols, b, lc, previous, tol):
        seen.append((z.copy(),previous[0].copy()))
        return original_dir(z,cols,b,lc,previous,tol)
    monkeypatch.setattr(mod,'stable_objective_difference',diff)
    monkeypatch.setattr(mod,'sr1_direction',direction)
    result = mod.solve_mirror_sr1(p,c,warmup=warm)
    if rejects == 6:
        assert result.history[3].phase == 'sk_fallback'
        assert result.history[3].sr1_G is None
        assert result.summary['sr1_exhausted_count'] >= 1
    else:
        assert result.history[3].step_size <= .5
    for trial in calls[:rejects]:
        np.testing.assert_array_equal(trial,warm.state.z_col)
    np.testing.assert_array_equal(seen[1][1],warm.state.z_col)
    assert result.summary['state_eval_count'] == warm.work.state_eval_count+len(calls)
    assert warm.fingerprint == warm.digest()


def test_converged_before_direction(monkeypatch):
    p = weak_coupling_problem(16,2,.01)
    monkeypatch.setattr(mod,'sr1_direction',lambda *a: pytest.fail('zero residual direction'))
    r = mod.solve_mirror_sr1(p,cfg(sr1_warmup_iters=0))
    assert r.converged and r.n_iter == 0


def test_nonfinite_trials_and_failed_fallback_preserve_state(monkeypatch):
    p=weak_coupling_problem(16,2,.01)
    c=cfg(max_iter=4,sr1_warmup_iters=2,initial_nu=-initial_columns(16,2,42))
    _,warm=mod.prepare_warmup(p,c)
    calls=[]
    def fail(problem,z,work):
        work.state_eval_count+=1
        calls.append(z.copy())
        raise FloatingPointError('forced nonfinite evaluation')
    monkeypatch.setattr(mod,'evaluate_reduced',fail)
    r=mod.solve_mirror_sr1(p,c,warmup=warm)
    assert len(calls)==7 and r.n_iter==2 and r.stop_reason.startswith('fallback_failure')
    assert r.summary['state_eval_count']==warm.work.state_eval_count+7
    assert r.summary['sr1_reject_count']==6
    np.testing.assert_array_equal(r.nu,warm.state.nu)


def test_deadline_checked_between_trials(monkeypatch):
    p=weak_coupling_problem(16,2,.01)
    c=cfg(max_iter=4,sr1_warmup_iters=2,initial_nu=-initial_columns(16,2,42))
    _,warm=mod.prepare_warmup(p,c)
    now=[0.]
    monkeypatch.setattr(mod,'perf_counter',lambda:now[0])
    def reject(*args):
        now[0]=2.
        return 1.
    monkeypatch.setattr(mod,'stable_objective_difference',reject)
    r=mod.solve_mirror_sr1(p,c,warmup=warm,deadline=1.)
    assert r.stop_reason=='timeout' and r.summary['trial_count']==1 and r.n_iter==2
