import numpy as np
from sinkhorn_lmflow.core.transport import transport_plan_from_potentials, transport_plan_from_scalings


def test_reconstruction(problem):
    mu = np.array([.1, -.2, .03, .04])
    nu = np.array([.05, .02, -.1])
    P = transport_plan_from_potentials(problem.M, problem.epsilon, mu, nu)
    Q = transport_plan_from_scalings(problem.M, problem.epsilon,
                                     np.exp(-mu/problem.epsilon), np.exp(-nu/problem.epsilon))
    np.testing.assert_allclose(P, Q, rtol=1e-14)
