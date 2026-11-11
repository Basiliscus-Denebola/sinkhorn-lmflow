"""Controlled positive doubly stochastic kernels; no cost normalization."""
from __future__ import annotations

import hashlib
import numpy as np
from .instance import OTProblem


def weak_coupling_problem(n=256, blocks=2, delta=.01):
    if not isinstance(n, int) or not isinstance(blocks, int) or blocks < 2 or n % blocks or n < blocks:
        raise ValueError("n must be divisible by blocks >= 2")
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be positive")
    ids = np.arange(blocks)
    gamma = 1./(1+np.abs(ids[:, None]-ids[None, :]))
    np.fill_diagonal(gamma, 0)
    block_transition = np.eye(blocks)-delta*(np.diag(gamma.sum(axis=1))-gamma)
    if np.any(block_transition <= 0):
        raise ValueError("block transition must be strictly positive")
    m = n//blocks
    K = np.repeat(np.repeat(block_transition, m, axis=0), m, axis=1)/m
    if not (np.allclose(K.sum(0), 1, atol=1e-14, rtol=0) and
            np.allclose(K.sum(1), 1, atol=1e-14, rtol=0)):
        raise ValueError("kernel is not doubly stochastic")
    return OTProblem(np.full(n, 1/n), np.full(n, 1/n), -np.log(K), 1.,
                     "weak_coupling", {"n": n, "blocks": blocks, "delta": delta,
                                       "cost_scale": "none", "known_reduced_optimum": 0.})


def initial_columns(n, blocks, seed):
    if blocks < 2 or n % blocks:
        raise ValueError("n must be divisible by blocks >= 2")
    rng = np.random.default_rng(seed)
    z = np.repeat(rng.normal(size=blocks), n//blocks)+.1*rng.normal(size=n)
    return z-z.mean()


def initial_fingerprint(z):
    a = np.asarray(z, dtype="<f8")
    return hashlib.sha256(str(a.shape).encode("ascii")+a.tobytes()).hexdigest()
