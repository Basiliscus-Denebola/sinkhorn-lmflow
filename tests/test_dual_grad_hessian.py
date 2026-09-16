import numpy as np
from sinkhorn_lmflow.core.dual import dual_objective, dual_gradient, dual_hessian
from sinkhorn_lmflow.core.transport import transport_plan_from_potentials


def test_gradient_finite_difference(problem):
    rng = np.random.default_rng(31)
    z = rng.normal(0, .03, 7)
    h = 1e-6
    fd = np.zeros(7)
    for i in range(7):
        e = np.eye(7)[i] * h
        fd[i] = (dual_objective(problem, (z+e)[:4], (z+e)[4:])
                 - dual_objective(problem, (z-e)[:4], (z-e)[4:]))/(2*h)
    g = dual_gradient(problem, z[:4], z[4:])
    assert np.linalg.norm(g-fd)/np.linalg.norm(g) < 1e-6


def test_hessian_vector_finite_difference(problem):
    rng = np.random.default_rng(21)
    z = rng.normal(0, .03, 7)
    v = rng.normal(size=7)
    h = 1e-6
    P = transport_plan_from_potentials(problem.M, problem.epsilon, z[:4], z[4:])
    fd = (dual_gradient(problem, (z+h*v)[:4], (z+h*v)[4:])
          - dual_gradient(problem, (z-h*v)[:4], (z-h*v)[4:]))/(2*h)
    hv = dual_hessian(P, problem.epsilon) @ v
    assert np.linalg.norm(hv-fd)/np.linalg.norm(hv) < 1e-6
