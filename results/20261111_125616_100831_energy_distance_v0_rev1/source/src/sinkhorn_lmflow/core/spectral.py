from __future__ import annotations

import numpy as np
from scipy.linalg import svdvals
from .transport import row_col_sums


def standardized_B(P: np.ndarray, r: np.ndarray | None = None,
                   c: np.ndarray | None = None) -> np.ndarray:
    """B=Dr^(-1/2) P Dc^(-1/2); zero marginals invalidate diagnostics."""
    if r is None or c is None:
        r, c = row_col_sums(P)
    if np.any(r <= 0) or np.any(c <= 0):
        raise FloatingPointError("Zero marginal in standardized B")
    return P / np.sqrt(r)[:, None] / np.sqrt(c)[None, :]


def nontrivial_sigma_max(B: np.ndarray) -> float:
    """Remove exactly one mass/gauge singular mode; keep other unit modes."""
    values = svdvals(B)
    return float(values[1]) if len(values) > 1 else 0.0


def spectral_diagnostics(P: np.ndarray, theta: float | None) -> dict[str, float | None]:
    """Report sigma1, sigma2, and delta=1-theta*sigma2 (no clipping)."""
    values = svdvals(standardized_B(P))
    sigma = float(values[1]) if len(values) > 1 else 0.0
    return {"sigma1": float(values[0]), "sigma_nontrivial": sigma,
            "spectral_gap": None if theta is None else 1 - theta * sigma}
