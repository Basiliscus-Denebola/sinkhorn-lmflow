"""Row-normalized, dimensionless column coordinates (z = -nu/epsilon).

Work counters count executed kernels, including rejected trials. dense_pass_proxy
is an explicit array-pass proxy, not FLOPs: evaluation 8, Bregman small 9 / large
11. Logsumexp and reduction counts are separately available. No Hessian is built.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
from scipy.special import logsumexp
from ..problems.instance import OTProblem
from .residuals import residual_norms
from .gauge import normalize_gauge


@dataclass
class ReducedWork:
    state_eval_count: int = 0
    objective_diff_count: int = 0
    bregman_count: int = 0
    plan_build_count: int = 0
    logsumexp_count: int = 0
    reduction_count: int = 0
    dense_pass_proxy: int = 0
    probability_normalization_count: int = 0
    probability_mass_error_max: float = 0.0
    probability_mean_error_max: float = 0.0

    def to_dict(self):
        return asdict(self)


@dataclass
class ReducedState:
    z_col: np.ndarray
    mu: np.ndarray
    nu: np.ndarray
    P: np.ndarray
    r: np.ndarray
    c: np.ndarray
    log_c: np.ndarray
    log_conditional: np.ndarray
    f: float
    full_objective: float
    metrics: dict


def center(z):
    z = np.asarray(z, dtype=np.float64)
    return z - z.mean()


def evaluate_reduced(problem: OTProblem, z_col, work: ReducedWork | None = None):
    if work is not None:
        work.state_eval_count += 1
    z = center(z_col)
    if z.shape != problem.b.shape or not np.isfinite(z).all():
        raise FloatingPointError("invalid reduced coordinates")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        logits = -problem.M/problem.epsilon + z[None, :]
        ell = logsumexp(logits, axis=1)
        logp = logits-ell[:, None]
        if work is not None:
            work.logsumexp_count += 1
            work.plan_build_count += 1
            work.reduction_count += 2
            work.dense_pass_proxy += 8
        P = problem.a[:, None]*np.exp(logp)
        r, c = P.sum(axis=1), P.sum(axis=0)
        # Use log-domain column masses even if a column sum underflows.
        log_c = logsumexp(logp+np.log(problem.a)[:, None], axis=0)
        if work is not None:
            work.logsumexp_count += 1
        f = float(problem.a@ell-problem.b@z)
        mu, nu = normalize_gauge(problem.epsilon*(ell-np.log(problem.a)),
                                  -problem.epsilon*z)
        full = float(problem.epsilon*P.sum()+mu@problem.a+nu@problem.b)
        if work is not None:
            work.reduction_count += 1
        metrics = residual_norms(r-problem.a, c-problem.b)
    if not np.isfinite([f, full, *metrics.values()]).all() or not np.isfinite(log_c).all():
        raise FloatingPointError("nonfinite reduced state")
    return ReducedState(z, mu, nu, P, r, c, log_c, logp, f, full, metrics)


def expm1_minus_x(x):
    """Avoid cancellation in exp(x)-1-x (Taylor begins at degree two)."""
    x = np.asarray(x)
    out = np.empty_like(x)
    small = np.abs(x) < .01
    t = x[small]
    out[small] = t*t*(.5+t*(1/6+t*(1/24+t*(1/120+t*(1/720+t*(1/5040+t/40320))))))
    out[~small] = np.expm1(x[~small])-x[~small]
    return out


def bregman_phi(problem, state, displacement, work=None):
    """D_Phi(z+u,z), with evaluation-only row probability renormalization.

    P/c/residuals are never altered. Check row masses and centered expectations;
    the normalized conditional law defines the stable log-partition difference.
    """
    if work is not None:
        work.bregman_count += 1
        work.probability_normalization_count += 1
        work.reduction_count += 4
        work.dense_pass_proxy += 9
    u = center(displacement)
    p = state.P/problem.a[:, None]
    mass = p.sum(axis=1)
    mass_error = float(np.max(np.abs(mass-1)))
    if not np.isfinite(mass).all() or mass_error > 1e-11:
        raise FloatingPointError("conditional probability normalization error")
    p = p/mass[:, None]
    mean = p@u
    v = u[None, :]-mean[:, None]
    mean_error = float(np.max(np.abs(np.sum(p*v, axis=1))))
    if work is not None:
        work.probability_mass_error_max = max(work.probability_mass_error_max, mass_error)
        work.probability_mean_error_max = max(work.probability_mean_error_max, mean_error)
    if mean_error > 1e-12*max(1., float(np.max(np.abs(u)))):
        raise FloatingPointError("conditional centered mean error")
    if np.max(np.abs(v)) <= .5:
        values = np.log1p(np.sum(p*expm1_minus_x(v), axis=1))
    else:
        values = logsumexp(state.log_conditional-np.log(mass)[:, None]+v, axis=1)
        if work is not None:
            work.logsumexp_count += 1
            work.dense_pass_proxy += 2
    value = float(problem.a@values)
    if not np.isfinite(value):
        raise FloatingPointError("nonfinite Bregman value")
    return value


def stable_objective_difference(problem, state, displacement, work=None):
    if work is not None:
        work.objective_diff_count += 1
    u = center(displacement)
    return float((state.c-problem.b)@u+bregman_phi(problem, state, u, work))
