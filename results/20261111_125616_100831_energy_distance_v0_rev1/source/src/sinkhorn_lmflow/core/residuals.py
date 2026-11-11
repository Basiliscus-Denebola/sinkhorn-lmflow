from __future__ import annotations

import numpy as np
from .transport import row_col_sums


def marginal_residual(P: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return marginal defects r-a and c-b = negative dual gradient blocks."""
    r, c = row_col_sums(P)
    return r - a, c - b


def residual_norms(ra: np.ndarray, rb: np.ndarray) -> dict[str, float]:
    """Shared L1 stopping metric, Linf defect and L2 dual gradient norm."""
    return {
        "residual_l1": float(max(np.linalg.norm(ra, 1), np.linalg.norm(rb, 1))),
        "residual_linf": float(max(np.max(np.abs(ra)), np.max(np.abs(rb)))),
        "grad_norm_l2": float(np.linalg.norm(np.concatenate((ra, rb)))),
    }


def compute_stop_metrics(P: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """R=max(||P1-a||_1, ||P^T1-b||_1), for every solver."""
    return residual_norms(*marginal_residual(P, a, b))
