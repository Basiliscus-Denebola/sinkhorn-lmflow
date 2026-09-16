import numpy as np
import pytest
from sinkhorn_lmflow.core.transport import kernel
from sinkhorn_lmflow.solvers.sinkhorn import sinkhorn_scaling_sweep, solve_sinkhorn
from sinkhorn_lmflow.solvers.lmflow import gs_scaling_endpoint, solve_lmflow
from sinkhorn_lmflow.types import SolverConfig
from sinkhorn_lmflow.problems.instance import OTProblem


def test_exact_sinkhorn_endpoint(problem):
    rng = np.random.default_rng(12)
    u, v = rng.uniform(.5, 2, 4), rng.uniform(.5, 2, 3)
    K = kernel(problem.M, problem.epsilon)
    us, vs = sinkhorn_scaling_sweep(K, problem.a, problem.b, u, v)
    mu, nu = gs_scaling_endpoint(problem, -problem.epsilon*np.log(u), -problem.epsilon*np.log(v))
    np.testing.assert_allclose(np.exp(-mu/problem.epsilon), us, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(np.exp(-nu/problem.epsilon), vs, rtol=1e-12, atol=1e-14)
    flow = solve_lmflow(problem, SolverConfig(theta=0, gauss_seidel=True, max_iter=1,
                        initial_mu=-problem.epsilon*np.log(u), initial_nu=-problem.epsilon*np.log(v),
                        gauge_normalization=False, line_search=False))
    np.testing.assert_allclose(flow.u, us, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(flow.v, vs, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("mode", ["primal_scaling", "log_stabilized"])
def test_sinkhorn_converges(problem, mode):
    result = solve_sinkhorn(problem, SolverConfig(numerical_mode=mode, tol=1e-11))
    assert result.converged
    assert result.summary["residual_l1"] <= 1e-11


def test_log_underflow_stress():
    p = OTProblem([.5, .5], [.5, .5], [[100., 101.], [101., 100.]], .001)
    log = solve_sinkhorn(p, SolverConfig(numerical_mode="log_stabilized", tol=1e-9))
    raw = solve_sinkhorn(p, SolverConfig(numerical_mode="primal_scaling"))
    assert log.converged
    assert not raw.converged
    assert "numerical_failure" in raw.stop_reason


def test_jacobi_endpoint_is_not_gs(problem):
    cfg = SolverConfig(theta=0, max_iter=1, line_search=False, gauge_normalization=False)
    jacobi = solve_lmflow(problem, cfg)
    gs = solve_lmflow(problem, SolverConfig(theta=0, gauss_seidel=True, max_iter=1,
                                          line_search=False, gauge_normalization=False))
    assert not np.allclose(jacobi.nu, gs.nu)
