import numpy as np
from sinkhorn_lmflow.core.dual import dual_hessian
from sinkhorn_lmflow.core.linalg import lm_matrix, lm_direction, augmented_solve
from sinkhorn_lmflow.core.gauge import gauge_vector
from sinkhorn_lmflow.core.transport import transport_plan_from_potentials
from sinkhorn_lmflow.solvers.newton import solve_newton
from sinkhorn_lmflow.solvers.lmflow import solve_lmflow
from sinkhorn_lmflow.types import SolverConfig


def test_newton_endpoint(problem):
    P = transport_plan_from_potentials(problem.M, problem.epsilon, np.zeros(4), np.zeros(3))
    H = dual_hessian(P, problem.epsilon)
    np.testing.assert_array_equal(lm_matrix(P, problem.epsilon, 1), H)
    rhs = np.r_[P.sum(1)-problem.a, P.sum(0)-problem.b]
    d = lm_direction(P, problem.epsilon, problem.a, problem.b, 1)
    ref = augmented_solve(H, rhs, 4, 3)
    np.testing.assert_allclose(d, ref, atol=1e-14)
    np.testing.assert_allclose(H @ d, rhs, atol=1e-13)
    assert abs(gauge_vector(4, 3) @ d) < 1e-13
    cfg = SolverConfig(theta=1, discretization="additive", tol=1e-11)
    a, b = solve_newton(problem, cfg), solve_lmflow(problem, cfg)
    assert a.converged and b.converged
    np.testing.assert_allclose(a.mu, b.mu, atol=1e-14)
    np.testing.assert_allclose(a.nu, b.nu, atol=1e-14)
