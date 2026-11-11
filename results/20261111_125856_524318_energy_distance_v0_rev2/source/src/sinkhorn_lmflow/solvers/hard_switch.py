from __future__ import annotations

from dataclasses import replace
import numpy as np
from scipy.linalg import LinAlgWarning
from ..types import OTProblem, SolverConfig, SolverResult
from ..core.gauge import normalize_gauge
from .base import Monitor, initial_state
from .sinkhorn import sinkhorn_log_sweep
from .lmflow import flow_step


def solve_hard_switch(problem: OTProblem, config: SolverConfig) -> SolverResult:
    """Stabilized Sinkhorn until R<=switch_tol, then exact Newton permanently."""
    cfg = replace(config, theta=1, discretization="additive", gauss_seidel=False, line_search=True)
    monitor = Monitor(problem, config, "hard_switch")
    mu, nu = initial_state(problem, config)
    reason, phase, switched_at = "max_iter", "sinkhorn", None
    try:
        residual = monitor.observe(0, mu, nu, 0, phase=phase)
        for k in range(1, config.max_iter+1):
            if residual <= config.tol:
                break
            if phase == "sinkhorn" and residual <= config.switch_tol:
                phase, switched_at = "newton", k-1
            if phase == "sinkhorn":
                mu, nu = sinkhorn_log_sweep(problem, mu, nu)
                if config.gauge_normalization:
                    mu, nu = normalize_gauge(mu, nu)
                h, theta = 1.0, 0.0
                monitor.last_linear_sec = 0.0
            else:
                mu, nu, h = flow_step(problem, cfg, monitor, mu, nu, 1.0)
                theta = 1.0
            residual = monitor.observe(k, mu, nu, theta, h, phase)
    except (FloatingPointError, np.linalg.LinAlgError, LinAlgWarning) as exc:
        reason = f"numerical_failure: {exc}"
    return monitor.finish(mu, nu, reason, {"switch_iteration": switched_at,
                                          "switch_tol": config.switch_tol})
