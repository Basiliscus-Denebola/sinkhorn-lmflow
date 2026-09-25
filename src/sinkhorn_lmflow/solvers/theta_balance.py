"""One-dimensional residual balance followed by exact Sinkhorn columns."""
from __future__ import annotations

import numpy as np
from ..types import OTProblem, SolverConfig, SolverResult
from ..core.gauge import normalize_gauge
from ..policies.stepsize import choose_step, StepFailure
from .base import Monitor, initial_state
from .sinkhorn import sinkhorn_log_column, sinkhorn_log_sweep


def balance_direction(P, a, b, theta_max, *, fixed=None, direction_data=None):
    m = P.sum(axis=1)
    if np.any(m <= 0):
        raise FloatingPointError("nonpositive row mass")
    ra = m-a
    ps = ra/m
    if fixed is not None:
        return ps, None, dict(balance_theta_free=None, balance_theta=float(fixed),
            balance_eta=None, balance_reason="",
            direction_residual_l1=float(np.linalg.norm(ra, 1)))
    qs = -(P.T @ ps)/b
    s = ra + P @ qs
    if direction_data is not None:
        direction_data.update(r=ra, s=s)
    nr, ns = np.linalg.norm(ra), np.linalg.norm(s)
    reason = ""
    free = theta = eta = None
    if not np.isfinite([nr, ns]).all():
        reason = "nonfinite_direction"
    elif ns <= 64*np.finfo(float).eps*nr or ns == 0:
        reason = "small_s"
    else:
        # Normalize by ||r|| to avoid squaring tiny residuals.
        rr, ss = ra/nr, s/nr
        free = float((ss @ rr)/(ss @ ss))
        if not np.isfinite(free):
            free = None
            reason = "nonfinite_theta"
        else:
            theta = float(np.clip(free, 0, theta_max))
            eta = float(np.linalg.norm(theta*ss-rr))
            if theta <= 0 or not np.any(theta*ps):
                reason = "ineffective_theta"
    return ps, qs, dict(balance_theta_free=free, balance_theta=theta,
                        balance_eta=eta, balance_reason=reason,
                        direction_residual_l1=float(np.linalg.norm(ra, 1)))


def balance_trial(problem, mu, p, h):
    hp = h*p
    if not np.isfinite(hp).all() or np.any(1-hp <= 0):
        raise FloatingPointError("scaling positivity violated")
    x = mu-problem.epsilon*np.log1p(-hp)
    return x, sinkhorn_log_column(problem, x)


def solve_theta_balance(problem: OTProblem, config: SolverConfig) -> SolverResult:
    monitor = Monitor(problem, config, "theta_balance")
    mu, nu = initial_state(problem, config)
    reason = "max_iter"
    clips = degenerates = fallbacks = 0
    events = []
    try:
        residual = monitor.observe(0, mu, nu, phase="initial")
        monitor.work["reduction_count"] += 1
        column_error = np.linalg.norm(monitor.last_P.sum(axis=0)-problem.b, 1)
        if column_error > 256*np.finfo(float).eps*max(problem.M.shape):
            # Public solver supports zero initialization; this cost is explicit.
            nu = sinkhorn_log_column(problem, mu)
            monitor.work["logsumexp_count"] += 1
            monitor.work["column_update_count"] += 1
            residual = monitor.observe(0, mu, nu, phase="initial_column_normalization")
        for k in range(1, config.max_iter+1):
            if residual <= config.tol:
                break
            if monitor.timed_out():
                reason = "timeout"
                break
            direction_data = {}
            ps, qs, info = balance_direction(monitor.last_P, problem.a, problem.b,
                config.balance_theta_max, fixed=config.balance_theta_fixed,
                direction_data=direction_data)
            monitor.work["reduction_count"] += 1
            monitor.work["p_matvec_count"] += 2 if config.balance_theta_fixed is None else 0
            free, theta = info["balance_theta_free"], info["balance_theta"]
            clips += int(free is not None and (free < 0 or free > config.balance_theta_max))
            if info["balance_reason"]:
                degenerates += 1
                events.append(dict(iteration=k, reason=info["balance_reason"]))
                x, y = sinkhorn_log_sweep(problem, mu, nu)
                monitor.work["logsumexp_count"] += 2
                monitor.work["column_update_count"] += 1
                fallbacks += 1
                h, phase = 1., "sinkhorn_fallback"
                info["balance_effective"] = None
            else:
                p = theta*ps

                def update(h):
                    x, y = balance_trial(problem, mu, p, h)
                    monitor.work["logsumexp_count"] += 1
                    monitor.work["column_update_count"] += 1
                    return x, y

                ra = ps*monitor.last_P.sum(axis=1)
                monitor.work["reduction_count"] += 1
                slope = -problem.epsilon*float(ra @ p)
                x, y, h, guards, backs = choose_step(problem, config, mu, nu,
                    monitor.last_objective, slope, residual, update, work=monitor.work)
                monitor.safeguards += guards
                monitor.backtracks += backs
                info["balance_effective"] = h*theta
                phase = "theta_balance"
            info["balance_predicted_ratio_l2"] = None
            if phase != "sinkhorn_fallback" and direction_data:
                r, s = direction_data["r"], direction_data["s"]
                nr = np.linalg.norm(r)
                if nr:
                    info["balance_predicted_ratio_l2"] = float(np.linalg.norm(r-h*theta*s)/nr)
            monitor.balance = dict(info, balance_clip_count=clips,
                balance_degenerate_count=degenerates, balance_fallback_count=fallbacks)
            mu, nu = normalize_gauge(x, y) if config.gauge_normalization else (x, y)
            residual = monitor.observe(k, mu, nu, step=h, phase=phase)
    except StepFailure as exc:
        monitor.safeguards += exc.safeguards
        monitor.backtracks += exc.backtracks
        reason = str(exc)
        events.append(dict(iteration=k, reason=reason, failed_trial=True))
    except (FloatingPointError, np.linalg.LinAlgError) as exc:
        reason = f"numerical_failure: {exc}"
    return monitor.finish(mu, nu, reason, dict(balance_theta_max=config.balance_theta_max,
                                              balance_theta_fixed=config.balance_theta_fixed,
                                              degeneration_events=events))
