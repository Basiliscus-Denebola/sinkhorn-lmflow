from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from ..types import OTProblem, SolverConfig, SolverResult
from ..core.dual import dual_objective, dual_gradient
from ..core.transport import transport_plan_from_potentials
from ..core.gauge import normalize_gauge
from .base import Monitor, initial_state


class _ToleranceReached(Exception):
    pass


def solve_lbfgs(problem: OTProblem, config: SolverConfig) -> SolverResult:
    """L-BFGS-B on the exact dual with nu[-1]=0 removing the gauge.

    SciPy stopping is not called convergence unless the shared marginal
    criterion passes. No unreported polishing steps are performed.
    """
    monitor = Monitor(problem, config, "lbfgs")
    mu, nu = initial_state(problem, config)
    n = len(mu)
    z0 = np.r_[mu+nu[-1], nu[:-1]-nu[-1]]
    latest = [mu, nu]
    count = 0
    evaluations = 0

    def unpack(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return z[:n], np.r_[z[n:], 0.0]

    def fun(z: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal evaluations
        evaluations += 1
        x, y = unpack(z)
        try:
            P = transport_plan_from_potentials(problem.M, problem.epsilon, x, y)
            value = dual_objective(problem, x, y, P)
            g = dual_gradient(problem, x, y, P)
            if not np.isfinite(value) or not np.isfinite(g).all():
                raise FloatingPointError("Nonfinite objective/gradient")
            return value, g[:-1]
        except FloatingPointError:
            # Reject overflowing trial points; never replace an accepted state.
            return np.inf, np.zeros_like(z)

    def callback(z: np.ndarray) -> None:
        nonlocal count
        count += 1
        x, y = unpack(z)
        if config.gauge_normalization:
            x, y = normalize_gauge(x, y)
        latest[:] = [x, y]
        if monitor.observe(count, x, y, phase="lbfgs") <= config.tol:
            raise _ToleranceReached

    reason = "max_iter"
    try:
        residual = monitor.observe(0, mu, nu, phase="lbfgs")
        if residual > config.tol and config.max_iter > 0:
            opt = minimize(fun, z0, jac=True, method="L-BFGS-B", callback=callback,
                           options={"maxiter": config.max_iter, "ftol": 0.0,
                                    "gtol": 0.0, "maxls": 50, "maxcor": 20})
            reason = f"scipy_stop: {opt.message}"
    except _ToleranceReached:
        reason = "tolerance"
    except (FloatingPointError, np.linalg.LinAlgError) as exc:
        reason = f"numerical_failure: {exc}"
    return monitor.finish(*latest, reason, {"objective_gradient_evaluations": evaluations})
