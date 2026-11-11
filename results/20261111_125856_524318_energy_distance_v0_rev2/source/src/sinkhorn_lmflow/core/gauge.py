from __future__ import annotations

import numpy as np


def normalize_gauge(mu: np.ndarray, nu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Set mean(mu)=0 via (mu-s,nu+s), preserving P and Phi."""
    s = float(np.mean(mu))
    return mu - s, nu + s


def gauge_vector(n: int, m: int) -> np.ndarray:
    """Null vector q=[1_n,-1_m] for H."""
    return np.concatenate((np.ones(n), -np.ones(m)))
