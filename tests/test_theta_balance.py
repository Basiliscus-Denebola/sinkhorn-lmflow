import numpy as np
import pytest
from sinkhorn_lmflow.types import SolverConfig
from sinkhorn_lmflow.core.transport import transport_plan_from_potentials as plan
from sinkhorn_lmflow.solvers.sinkhorn import sinkhorn_log_sweep
from sinkhorn_lmflow.solvers.theta_balance import balance_direction, balance_trial, solve_theta_balance
from sinkhorn_lmflow.io.result_io import save_result, load_result


@pytest.mark.parametrize("theta_max", [.5, 1., 10.])
def test_balance_math_and_exact_column(problem, theta_max):
    mu, nu = sinkhorn_log_sweep(problem, np.zeros_like(problem.a), np.zeros_like(problem.b))
    P = plan(problem.M, problem.epsilon, mu, nu)
    ps, qs, info = balance_direction(P, problem.a, problem.b, theta_max)
    theta = info["balance_theta"]
    np.testing.assert_allclose(P.T @ (theta*ps)+problem.b*(theta*qs), 0, atol=1e-15)
    ra = P.sum(1)-problem.a
    s = ra+P @ qs
    loss = np.linalg.norm(theta*s-ra)**2
    endpoints = [np.linalg.norm(ra)**2, np.linalg.norm(theta_max*s-ra)**2]
    if theta_max >= 1:
        endpoints.append(np.linalg.norm(s-ra)**2)
    assert loss <= min(endpoints)+1e-16
    for h in (.1, .5):
        x, y = balance_trial(problem, mu, theta*ps, h)
        Q = plan(problem.M, problem.epsilon, x, y)
        np.testing.assert_allclose(Q.sum(0), problem.b, rtol=2e-14, atol=1e-15)
    x, y = balance_trial(problem, mu, ps, 1.)
    sx, sy = sinkhorn_log_sweep(problem, mu, nu)
    np.testing.assert_allclose(plan(problem.M, problem.epsilon, x, y),
                               plan(problem.M, problem.epsilon, sx, sy), rtol=2e-14)


def test_solver_roundtrip_and_explicit_failure(problem, tmp_path):
    result = solve_theta_balance(problem, SolverConfig(tol=1e-8, max_iter=1000))
    assert result.converged
    assert all(r.theta is None for r in result.history)
    save_result(result, tmp_path)
    loaded = load_result(tmp_path)
    assert loaded.history[-1].balance_theta == result.history[-1].balance_theta
    failed = solve_theta_balance(problem, SolverConfig(step_size=1e10, max_backtracks=0))
    assert not failed.converged
    assert "positivity" in failed.stop_reason


def test_degenerate_and_invalid_bound():
    _, _, info = balance_direction(np.diag([.6, .4]), np.array([.5, .5]), np.array([.6, .4]), 10)
    assert info["balance_reason"] == "small_s"
    with pytest.raises(ValueError):
        SolverConfig(balance_theta_max=0)


def test_degenerate_fallback_is_explicit():
    from sinkhorn_lmflow.types import OTProblem
    problem = OTProblem(np.array([.5, .5]), np.array([.6, .4]),
                        np.array([[0., 10000.], [10000., 0.]]), 1.)
    result = solve_theta_balance(problem, SolverConfig(max_iter=1))
    assert not result.converged
    assert result.history[-1].phase == "sinkhorn_fallback"
    assert result.summary["balance_fallback_count"] == 1
    assert result.summary["degeneration_events"][0]["reason"] == "small_s"
    assert result.summary["logsumexp_count"] == 3  # initial column plus fallback sweep


def test_fixed_skips_adaptive_matvecs_and_none_unchanged(problem):
    class NoMatvec(np.ndarray):
        def __matmul__(self, other):
            raise AssertionError('fixed theta must not multiply P')
    mu, nu = sinkhorn_log_sweep(problem, np.zeros_like(problem.a), np.zeros_like(problem.b))
    P = plan(problem.M, problem.epsilon, mu, nu)
    ps, qs, info = balance_direction(P.view(NoMatvec), problem.a, problem.b, 10, fixed=1.5)
    assert qs is None and info['balance_theta_free'] is None and info['balance_eta'] is None
    old = balance_direction(P, problem.a, problem.b, 10)
    new = balance_direction(P, problem.a, problem.b, 10, fixed=None)
    np.testing.assert_array_equal(old[0], new[0])
    np.testing.assert_array_equal(old[1], new[1])
    assert old[2] == new[2]
    result = solve_theta_balance(problem, SolverConfig(balance_theta_fixed=1.5, max_iter=3))
    assert result.summary['p_matvec_count'] == 0
    assert all(x.balance_predicted_ratio_l2 is None for x in result.history)


def test_fixed_one_full_step_and_damped_step(problem):
    from dataclasses import replace
    mu, nu = sinkhorn_log_sweep(problem, np.zeros_like(problem.a), np.zeros_like(problem.b))
    cfg = SolverConfig(balance_theta_fixed=1, initial_mu=mu, initial_nu=nu,
                       max_iter=1, compute_spectrum_every=0, line_search=False)
    full = solve_theta_balance(problem, cfg)
    x, y = sinkhorn_log_sweep(problem, mu, nu)
    expected = plan(problem.M, problem.epsilon, x, y)
    actual = plan(problem.M, problem.epsilon, full.mu, full.nu)
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=1e-16)
    np.testing.assert_allclose(actual.sum(0), problem.b, atol=1e-15)
    damped = solve_theta_balance(problem, replace(cfg, step_size=.5))
    assert np.linalg.norm(plan(problem.M, problem.epsilon, damped.mu, damped.nu)-expected) > 1e-8
    np.testing.assert_allclose(full.history[1].pre_residual_l1, full.history[0].residual_l1)
    np.testing.assert_allclose(full.history[1].balance_actual_ratio_l2,
        np.linalg.norm(actual.sum(1)-problem.a)/np.linalg.norm(plan(problem.M, problem.epsilon, mu, nu).sum(1)-problem.a))


def study_module():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    import run_theta_balance_pilot
    return run_theta_balance_pilot


def test_study_inputs_matrix_and_initial_fingerprints():
    from dataclasses import replace
    runner = study_module()
    config = runner.load_config('configs/theta_balance_study.yaml')
    specs = runner.study_cases(config)
    # 120 main + 6 high-precision + 4 pilot cases; BC overlap is merged.
    assert len(specs) == 130
    assert sum(len(c['methods']) for c in specs) == 440
    spec = dict(cost='random_dense', n=100, epsilon=.01, seed=42, marginal='uniform')
    base = runner.study_problem(spec, config)
    eps = runner.study_problem(dict(spec, epsilon=.001), config)
    marginal = runner.study_problem(dict(spec, marginal='spiky'), config)
    for key in ['a', 'b', 'M']:
        np.testing.assert_array_equal(getattr(base, key), getattr(eps, key))
    np.testing.assert_array_equal(base.M, marginal.M)
    assert np.all(marginal.a > 0) and np.all(marginal.b > 0)
    assert np.isclose(np.sort(marginal.a)[-2:].sum(), .9)
    assert runner.problem_fingerprint(base) != runner.problem_fingerprint(eps)
    mu, nu = sinkhorn_log_sweep(base, np.zeros(100), np.zeros(100))
    case = dict(tol=1e-8)
    fingerprints = []
    for method in ['sinkhorn', 'adaptive', 'fixed_1']:
        cfg = replace(runner.study_solver_config(config, case, method), initial_mu=mu, initial_nu=nu)
        from sinkhorn_lmflow.solvers.base import initial_state
        fingerprints.append(runner.digest_arrays(*initial_state(base, cfg)))
    assert len(set(fingerprints)) == 1


def test_timeout_status_and_old_results(problem, tmp_path):
    import csv
    from sinkhorn_lmflow.solvers.sinkhorn import solve_sinkhorn
    runner = study_module()
    for solver in [solve_sinkhorn, solve_theta_balance]:
        result = solver(problem, SolverConfig(wall_time_limit_sec=1e-12))
        assert not result.converged and result.stop_reason == 'timeout'
        assert runner.run_status(result) == 'timeout'
    assert runner.unexecuted_status(dict(skip_reason=None)) == 'pending'
    assert runner.unexecuted_status(dict(skip_reason='after timeout')) == 'skipped_repeat'
    result = solve_theta_balance(problem, SolverConfig(max_iter=1))
    assert runner.run_status(result) == 'max_iter'
    failed = solve_theta_balance(problem, SolverConfig(step_size=1e10, max_backtracks=0))
    assert runner.run_status(failed) == 'failed'
    save_result(result, tmp_path)
    # A legacy CSV does not contain newly optional columns.
    path = tmp_path/'history.csv'
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = [k for k in reader.fieldnames if k not in ['pre_residual_l1', 'row_residual_l2',
            'balance_predicted_ratio_l2', 'balance_actual_ratio_l2']]
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    assert load_result(tmp_path).history[-1].pre_residual_l1 is None


def test_study_budget_preserves_pending_and_resume_protocol(tmp_path, monkeypatch):
    import json
    runner = study_module()
    config = runner.load_config('configs/theta_balance_study.yaml')
    config['budget_sec'] = 0
    root = tmp_path/'study'
    root.mkdir()
    monkeypatch.setattr(runner, 'create_run_dir', lambda *args: root)
    monkeypatch.setattr(runner, 'analyze_study', lambda *args: None)
    monkeypatch.setattr(runner, 'runtime_metadata', lambda *args: {})
    runner.run_study(config)
    manifest = json.loads((root/'manifest.json').read_text())
    assert manifest['sessions'][-1]['status'] == 'budget_exhausted'
    assert all(not task['runs'] and runner.unexecuted_status(task) == 'pending'
        for case in manifest['cases'] for task in case['methods'].values())
    runner.run_study(config, root)
    assert len(json.loads((root/'manifest.json').read_text())['sessions']) == 2
    changed = dict(config, repeats=4)
    with pytest.raises(ValueError, match='protocol'):
        runner.run_study(changed, root)
