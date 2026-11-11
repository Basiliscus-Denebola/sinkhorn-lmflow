from __future__ import annotations

import numpy as np


def kernel(M: np.ndarray, eps: float) -> np.ndarray:
    """K = exp(-M / eps); underflow is allowed, overflow is an error."""
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        return np.exp(-M / eps)


def transport_plan_from_scalings(M: np.ndarray, eps: float, u: np.ndarray,
                                 v: np.ndarray) -> np.ndarray:
    """P = Diag(u) K Diag(v)."""
    with np.errstate(over="raise", invalid="raise"):
        return u[:, None] * kernel(M, eps) * v[None, :]


def transport_plan_from_potentials(M: np.ndarray, eps: float, mu: np.ndarray,
                                  nu: np.ndarray) -> np.ndarray:
    """P_ij = exp(-(mu_i + nu_j + M_ij) / eps), without forming K."""
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        return np.exp(-(mu[:, None] + nu[None, :] + M) / eps)


def row_col_sums(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return r=P1 and c=P^T1."""
    return P.sum(axis=1), P.sum(axis=0)
