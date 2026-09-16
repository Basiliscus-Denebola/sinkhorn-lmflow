from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from ..types import OTProblem, SolverConfig, SolverResult
from ..core.transport import kernel
from ..core.gauge import normalize_gauge
from .base import Monitor, initial_state


def sinkhorn_scaling_sweep(K: np.ndarray, a: np.ndarray, b: np.ndarray,
                           u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One exact GS sweep: u+=a/(Kv), then v+=b/(K.T u+)."""
    with np.errstate(divide="raise", over="raise", invalid="raise"):
        unew = a / (K @ v)
        vnew = b / (K.T @ unew)
    if not np.isfinite(unew).all() or not np.isfinite(vnew).all() or np.any(unew <= 0) or np.any(vnew <= 0):
        raise FloatingPointError("primal scaling underflow/overflow")
    return unew, vnew


def sinkhorn_log_sweep(problem: OTProblem, mu: np.ndarray,
                       nu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Same GS Sinkhorn sweep in potentials, with logsumexp stabilization."""
    eps = problem.epsilon
    x = eps * (logsumexp(-(problem.M + nu[None, :])/eps, axis=1) - np.log(problem.a))
    y = eps * (logsumexp(-(problem.M + x[:, None])/eps, axis=0) - np.log(problem.b))
    return x, y


def solve_sinkhorn(problem: OTProblem, config: SolverConfig) -> SolverResult:
    monitor = Monitor(problem, config, "sinkhorn")
    mu, nu = initial_state(problem, config)
    reason = "max_iter"
    try:
        residual = monitor.observe(0, mu, nu, theta=0, phase="sinkhorn")
        K = kernel(problem.M, problem.epsilon) if config.numerical_mode == "primal_scaling" else None
        for k in range(1, config.max_iter+1):
            if residual <= config.tol:
                break
            if K is None:
                x, y = sinkhorn_log_sweep(problem, mu, nu)
            else:
                with np.errstate(over="raise", under="ignore"):
                    u, v = np.exp(-mu/problem.epsilon), np.exp(-nu/problem.epsilon)
                u, v = sinkhorn_scaling_sweep(K, problem.a, problem.b, u, v)
                x, y = -problem.epsilon*np.log(u), -problem.epsilon*np.log(v)
                monitor.matvec_count += 2
            mu, nu = normalize_gauge(x, y) if config.gauge_normalization else (x, y)
            residual = monitor.observe(k, mu, nu, theta=0, step=1, phase="sinkhorn")
    except (FloatingPointError, np.linalg.LinAlgError) as exc:
        reason = f"numerical_failure: {exc}"
    return monitor.finish(mu, nu, reason, {"numerical_mode": config.numerical_mode})
