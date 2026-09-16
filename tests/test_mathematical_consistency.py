import numpy as np
import pytest
from sinkhorn_lmflow.core.linalg import lm_direction
from sinkhorn_lmflow.core.transport import kernel, transport_plan_from_potentials
from sinkhorn_lmflow.core.stability import scaling_coordinate_update
from sinkhorn_lmflow.core.dual import dual_objective, transport_objective
from sinkhorn_lmflow.types import SolverConfig
from sinkhorn_lmflow.solvers.sinkhorn import solve_sinkhorn
from sinkhorn_lmflow.solvers.newton import solve_newton
from sinkhorn_lmflow.solvers.lmflow import solve_lmflow


@pytest.mark.parametrize("theta", [0, .5, 1])
def test_scaling_flow_system(problem, theta):
    mu = np.array([.05, -.02, .01, .03])
    nu = np.array([-.03, .01, .02])
    eps = problem.epsilon
    u, v = np.exp(-mu/eps), np.exp(-nu/eps)
    K = kernel(problem.M, eps)
    P = transport_plan_from_potentials(problem.M, eps, mu, nu)
    d = lm_direction(P, eps, problem.a, problem.b, theta)
    velocity = -np.r_[u, v]*d/eps
    A = np.block([[np.diag(K@v), theta*u[:, None]*K],
                  [theta*v[:, None]*K.T, np.diag(K.T@u)]])
    rhs = np.r_[problem.a-P.sum(1), problem.b-P.sum(0)]
    np.testing.assert_allclose(A@velocity, rhs, rtol=1e-12, atol=1e-13)
    h = .01
    new = scaling_coordinate_update(np.r_[mu, nu], d, eps, h)
    np.testing.assert_allclose(np.exp(-new/eps), np.r_[u,v]+h*velocity, rtol=1e-13)


def test_shared_optimum_and_dual_sign(problem):
    cfg = SolverConfig(tol=1e-12, theta=.75)
    results = [solver(problem, cfg) for solver in (solve_sinkhorn, solve_newton, solve_lmflow)]
    plans = []
    for result in results:
        assert result.converged
        P = transport_plan_from_potentials(problem.M, problem.epsilon, result.mu, result.nu)
        plans.append(P)
        assert abs(dual_objective(problem, result.mu, result.nu, P)+transport_objective(problem, P)) < 1e-11
    for P in plans[1:]:
        np.testing.assert_allclose(P, plans[0], rtol=1e-9, atol=1e-11)
