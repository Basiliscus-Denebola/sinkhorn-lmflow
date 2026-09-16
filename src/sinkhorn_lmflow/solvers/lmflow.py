from __future__ import annotations

from time import perf_counter
import numpy as np
from scipy.linalg import LinAlgWarning
from ..types import OTProblem, SolverConfig, SolverResult, SolverState
from ..core.dual import dual_gradient
from ..core.gauge import normalize_gauge
from ..core.linalg import lm_direction
from ..core.transport import transport_plan_from_potentials
from ..core.stability import scaling_coordinate_update
from ..policies.theta import ConstantTheta, ThetaPolicy
from ..policies.stepsize import choose_step, StepFailure
from .base import Monitor, initial_state
from .sinkhorn import sinkhorn_log_sweep


def gs_scaling_endpoint(problem: OTProblem, mu: np.ndarray, nu: np.ndarray,
                        h: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """GS forward Euler for theta=0 in scaling coordinates.

    At h=1 evaluate the algebraically identical Sinkhorn ratios in log space;
    this avoids catastrophic cancellation in 1-(r-a)/r.
    """
    if h == 1:
        return sinkhorn_log_sweep(problem, mu, nu)
    P = transport_plan_from_potentials(problem.M, problem.epsilon, mu, nu)
    r = P.sum(1)
    if np.any(r <= 0):
        raise FloatingPointError("Zero row marginal")
    x = scaling_coordinate_update(mu, problem.epsilon*(r-problem.a)/r, problem.epsilon, h)
    Q = transport_plan_from_potentials(problem.M, problem.epsilon, x, nu)
    c = Q.sum(0)
    if np.any(c <= 0):
        raise FloatingPointError("Zero column marginal")
    y = scaling_coordinate_update(nu, problem.epsilon*(c-problem.b)/c, problem.epsilon, h)
    return x, y


def flow_step(problem: OTProblem, config: SolverConfig, monitor: Monitor,
              mu: np.ndarray, nu: np.ndarray, theta: float) -> tuple[np.ndarray, np.ndarray, float]:
    """One direction solve and safeguarded step, shared with exact Newton."""
    if config.gauss_seidel and (theta != 0 or config.discretization != "scaling_coordinate"):
        raise ValueError("GS is defined only for theta=0 scaling-coordinate endpoint")
    t = perf_counter()
    d = lm_direction(monitor.last_P, problem.epsilon, problem.a, problem.b, theta)
    dt = perf_counter()-t
    monitor.last_linear_sec = dt
    monitor.linear_time += dt
    monitor.linear_count += 1
    if not np.isfinite(d).all():
        raise FloatingPointError("nonfinite direction")
    g = dual_gradient(problem, mu, nu, monitor.last_P)
    slope = float(g @ d)
    if slope >= 0:
        raise FloatingPointError("non_descent_direction")
    n = len(mu)

    def update(h: float) -> tuple[np.ndarray, np.ndarray]:
        if config.gauss_seidel:
            return gs_scaling_endpoint(problem, mu, nu, h)
        if config.discretization == "additive":
            return mu + h*d[:n], nu + h*d[n:]
        z = scaling_coordinate_update(np.r_[mu, nu], d, problem.epsilon, h)
        return z[:n], z[n:]

    try:
        x, y, h, safeguards, backtracks = choose_step(problem, config, mu, nu,
            monitor.last_objective, slope, monitor.last_metrics["residual_l1"], update)
    except StepFailure as exc:
        monitor.safeguards += exc.safeguards
        monitor.backtracks += exc.backtracks
        raise
    monitor.safeguards += safeguards
    monitor.backtracks += backtracks
    if config.gauge_normalization:
        x, y = normalize_gauge(x, y)
    return x, y, h


def solve_lmflow(problem: OTProblem, config: SolverConfig,
                 theta_policy: ThetaPolicy | None = None, *, name: str = "lmflow") -> SolverResult:
    policy = theta_policy or ConstantTheta(config.theta)
    if config.gauss_seidel and (config.theta != 0 or config.discretization != "scaling_coordinate"):
        raise ValueError("GS requires theta=0 and scaling_coordinate")
    monitor = Monitor(problem, config, name)
    mu, nu = initial_state(problem, config)
    reason = "max_iter"
    try:
        residual = monitor.observe(0, mu, nu, config.theta, phase=name)
        for k in range(1, config.max_iter+1):
            if residual <= config.tol:
                break
            theta = float(policy(SolverState(k-1, residual, mu, nu)))
            if not 0 <= theta <= 1:
                raise ValueError("Theta policy returned value outside [0,1]")
            mu, nu, h = flow_step(problem, config, monitor, mu, nu, theta)
            residual = monitor.observe(k, mu, nu, theta, h, name)
    except (FloatingPointError, np.linalg.LinAlgError, LinAlgWarning) as exc:
        reason = f"numerical_failure: {exc}"
    return monitor.finish(mu, nu, reason)
