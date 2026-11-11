"""Validation of unit-mass, strictly positive dense EOT inputs."""
from __future__ import annotations

import numpy as np


def validate_problem(a: np.ndarray, b: np.ndarray, M: np.ndarray, epsilon: float) -> None:
    """Require finite M, positive marginals of mass one, and epsilon > 0.

    Zero-mass support reduction is deliberately not performed implicitly.
    Negative costs are allowed.
    """
    if a.ndim != 1 or b.ndim != 1 or M.ndim != 2:
        raise ValueError("a, b must be vectors and M must be a matrix")
    if not a.size or not b.size or M.shape != (a.size, b.size):
        raise ValueError("M.shape must equal (len(a), len(b)), both nonempty")
    if not all(np.isfinite(x).all() for x in (a, b, M)):
        raise ValueError("Inputs must be finite")
    if np.any(a <= 0) or np.any(b <= 0):
        raise ValueError("Marginals must be strictly positive")
    if not (np.isclose(a.sum(), 1, rtol=0, atol=1e-12)
            and np.isclose(b.sum(), 1, rtol=0, atol=1e-12)):
        raise ValueError("Both marginals must have unit mass")
    if not np.isscalar(epsilon) or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
