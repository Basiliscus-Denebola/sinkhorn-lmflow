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
