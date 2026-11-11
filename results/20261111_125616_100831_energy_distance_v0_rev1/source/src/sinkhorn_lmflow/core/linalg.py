from __future__ import annotations

import warnings
import numpy as np
from scipy.linalg import solve, LinAlgWarning
from .dual import dual_hessian
from .gauge import gauge_vector
from .transport import row_col_sums


def lm_matrix(P: np.ndarray, eps: float, theta: float) -> np.ndarray:
    """G_theta=(1/eps)[Dr,theta*P;theta*P.T,Dc]."""
    if not 0 <= theta <= 1:
        raise ValueError("theta must be in [0, 1]")
    H = dual_hessian(P, eps)
    n = P.shape[0]
    H[:n, n:] *= theta
    H[n:, :n] *= theta
    return H


def augmented_solve(H: np.ndarray, rhs: np.ndarray, n: int, m: int) -> np.ndarray:
    """Solve [H,q;q.T,0][d,alpha]=[rhs,0], never solve singular H directly."""
    q = gauge_vector(n, m)
    A = np.block([[H, q[:, None]], [q[None, :], np.zeros((1, 1))]])
    with warnings.catch_warnings():
        warnings.simplefilter("error", LinAlgWarning)
        return solve(A, np.r_[rhs, 0.0], assume_a="sym")[:-1]


def lm_direction(P: np.ndarray, eps: float, a: np.ndarray, b: np.ndarray,
                 theta: float) -> np.ndarray:
    """Solve G_theta d=[r-a,c-b], using a gauge constraint at theta=1."""
    r, c = row_col_sums(P)
    if np.any(r <= 0) or np.any(c <= 0):
        raise FloatingPointError("Zero marginal: metric is not invertible")
    rhs = np.r_[r-a, c-b]
    if theta == 0:
        return eps * rhs / np.r_[r, c]
    G = lm_matrix(P, eps, theta)
    if theta == 1:
        return augmented_solve(G, rhs, len(a), len(b))
    with warnings.catch_warnings():
        warnings.simplefilter("error", LinAlgWarning)
        return solve(G, rhs, assume_a="pos")
