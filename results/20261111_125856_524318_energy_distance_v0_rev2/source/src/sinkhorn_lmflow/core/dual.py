from __future__ import annotations

import numpy as np
from scipy.special import xlogy
from .transport import transport_plan_from_potentials, row_col_sums
from ..problems.instance import OTProblem


def dual_objective(problem: OTProblem, mu: np.ndarray, nu: np.ndarray,
                   P: np.ndarray | None = None) -> float:
    """Convex dual Phi=eps*sum(P)+mu.a+nu.b (minimum = -primal optimum)."""
    if P is None:
        P = transport_plan_from_potentials(problem.M, problem.epsilon, mu, nu)
    return float(problem.epsilon * P.sum() + mu @ problem.a + nu @ problem.b)


def dual_gradient(problem: OTProblem, mu: np.ndarray, nu: np.ndarray,
                  P: np.ndarray | None = None) -> np.ndarray:
    """Gradient of Phi = [a-r, b-c]."""
    if P is None:
        P = transport_plan_from_potentials(problem.M, problem.epsilon, mu, nu)
    r, c = row_col_sums(P)
    return np.concatenate((problem.a - r, problem.b - c))


def dual_hessian(P: np.ndarray, eps: float) -> np.ndarray:
    """Full Hessian H=(1/eps)[Diag(r),P;P.T,Diag(c)], singular in gauge."""
    r, c = row_col_sums(P)
    return np.block([[np.diag(r), P], [P.T, np.diag(c)]]) / eps


def transport_objective(problem: OTProblem, P: np.ndarray) -> float:
    """<P,M>+eps*sum(P log P-P); infeasible iterates are not primal bounds."""
    return float(np.sum(P * problem.M) + problem.epsilon * np.sum(xlogy(P, P) - P))
