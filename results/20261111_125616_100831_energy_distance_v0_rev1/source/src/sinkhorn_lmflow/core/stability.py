from __future__ import annotations

import numpy as np


def scaling_coordinate_update(z: np.ndarray, d: np.ndarray, eps: float, h: float) -> np.ndarray:
    """z_new=z-eps*log(1-h*d/eps); reject nonpositive factors, never clip."""
    t = -h * d / eps
    if np.any(t <= -1) or not np.isfinite(t).all():
        raise FloatingPointError("scaling positivity safeguard")
    return z - eps * np.log1p(t)


def finite_scalings(mu: np.ndarray, nu: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """Export u,v; overflow may occur even for valid finite log-domain states.

    Preserve inf/zero explicitly; potentials remain the authoritative state.
    """
    with np.errstate(over="ignore", under="ignore"):
        return np.exp(-mu / eps), np.exp(-nu / eps)
