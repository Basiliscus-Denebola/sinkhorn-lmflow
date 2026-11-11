"""Full Euclidean distances of weighted, centered cost columns (51,4-7)."""
from __future__ import annotations

import numpy as np


def _distance_from_gram(gram):
    diag = np.diag(gram).copy()
    scale = max(float(np.max(np.abs(gram))), np.finfo(float).tiny)
    gram *= -2
    gram += diag[:, None]
    gram += diag[None, :]
    tolerance = 128 * np.finfo(float).eps * scale
    if not np.isfinite(gram).all() or float(gram.min()) < -tolerance:
        raise FloatingPointError('negative/nonfinite squared cost-profile distance')
    np.maximum(gram, 0, out=gram)
    np.sqrt(gram, out=gram)
    # Rowwise symmetrization avoids an additional m*m temporary.
    for j in range(len(gram)):
        row = .5 * (gram[j, j:] + gram[j:, j])
        gram[j, j:] = row
        gram[j:, j] = row
    np.fill_diagonal(gram, 0)
    return gram


def generic_distance(M, a, epsilon):
    w = np.asarray(a, dtype=float) / np.sum(a)
    # Subtract a common column first: exact feature translation, removing row
    # offsets before Gram formation and greatly reducing cancellation.
    X = np.asarray(M, dtype=float) - np.asarray(M)[:, :1]
    X -= w @ X
    X *= np.sqrt(w)[:, None] / epsilon
    return _distance_from_gram(X.T @ X)


def geometric_distance(x, y, a, epsilon, h):
    x, y = np.asarray(x), np.asarray(y)
    w = np.asarray(a) / np.sum(a)
    xc = x - w @ x
    covariance = xc.T @ (w[:, None] * xc)
    yc = y - y[:1]
    gram = (yc @ covariance) @ yc.T
    gram *= 4 / (epsilon * h) ** 2
    return _distance_from_gram(gram)


def build_distance(problem, reconstruction=None):
    """Use verified generator metadata only; otherwise use the general formula."""
    arrays = reconstruction or {}
    meta = problem.metadata
    reason = None
    if meta.get('family') in ('M2', 'M3', 'M4', 'M5', 'M6', 'M7') and 'x' in arrays and 'y' in arrays:
        x, y = arrays['x'], arrays['y']
        h = meta.get('raw_cost_max')
        if (np.shape(x) == (len(problem.a), 2) and np.shape(y) == (len(problem.b), 2)
                and h is not None and np.isfinite(h) and h > 0):
            # Verify the metadata against every cost, in small blocks, outside
            # any extra full-matrix allocation. Verification is charged setup.
            valid = True
            for lo in range(0, len(x), 64):
                expected = ((x[lo:lo+64, None, :] - y[None, :, :]) ** 2).sum(2) / h
                if not np.allclose(expected, problem.M[lo:lo+64], rtol=2e-12, atol=2e-14):
                    valid = False
                    break
            if valid:
                return geometric_distance(x, y, problem.a, problem.epsilon, h), 'exact_geometric', None
            reason = 'geometric metadata does not reproduce M'
        else:
            reason = 'invalid geometric metadata'
    elif meta.get('family') != 'M1':
        reason = 'geometric metadata unavailable'
    return generic_distance(problem.M, problem.a, problem.epsilon), 'generic_gram', reason
