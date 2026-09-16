import numpy as np
from sinkhorn_lmflow.core.dual import dual_objective
from sinkhorn_lmflow.core.transport import transport_plan_from_potentials
from sinkhorn_lmflow.core.residuals import compute_stop_metrics
from sinkhorn_lmflow.core.gauge import normalize_gauge


def test_gauge_invariance(problem):
    mu = np.array([.1, -.1, .02, .05])
    nu = np.array([.02, .03, -.02])
    P = transport_plan_from_potentials(problem.M, problem.epsilon, mu, nu)
    for shift in (-1.3, .71, 5):
        u, v = mu+shift, nu-shift
        Q = transport_plan_from_potentials(problem.M, problem.epsilon, u, v)
        np.testing.assert_allclose(P, Q, rtol=1e-13)
        np.testing.assert_allclose(dual_objective(problem, mu, nu), dual_objective(problem, u, v), rtol=1e-13)
        np.testing.assert_allclose(list(compute_stop_metrics(P, problem.a, problem.b).values()),
                                   list(compute_stop_metrics(Q, problem.a, problem.b).values()), rtol=1e-13)
        x, y = normalize_gauge(u, v)
        assert abs(x.mean()) < 1e-14
        np.testing.assert_allclose(x[:, None]+y, mu[:, None]+nu, atol=1e-14)
