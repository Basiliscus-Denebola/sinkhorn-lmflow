from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def geometric_cost(x: np.ndarray, y: np.ndarray, p: float = 2) -> np.ndarray:
    """M_ij=||x_i-y_j||_2**p, only a generator utility."""
    return cdist(np.atleast_2d(x), np.atleast_2d(y)) ** p


def normalize_cost(M: np.ndarray) -> np.ndarray:
    """Scale a generated nonnegative cost by its maximum (not user inputs)."""
    return M / max(float(M.max()), np.finfo(float).tiny)
