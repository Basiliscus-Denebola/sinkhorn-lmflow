from __future__ import annotations

import numpy as np
from .instance import OTProblem
from .costs import geometric_cost, normalize_cost


def generate_problem(generator: str = "random_dense", n: int = 20, m: int = 20,
                     epsilon: float = .05, seed: int = 42,
                     marginal_alpha: float | None = None, alpha_a: float | None = None,
                     alpha_b: float | None = None, marginal_floor: float = 1e-12,
                     clustered: bool = False, p: float = 2) -> OTProblem:
    """Generate dense general/geometric costs and safeguarded Dirichlet marginals.

    The floor applies only during generation, before normalization, and is
    recorded. Solvers never modify marginals or costs.
    """
    if n < 1 or m < 1 or not np.isfinite(marginal_floor) or marginal_floor <= 0:
        raise ValueError("Positive dimensions and marginal_floor required")
    alpha = marginal_alpha if marginal_alpha is not None else (.1 if generator == "spiky_marginal" else 1.)
    aa = alpha if alpha_a is None else alpha_a
    ab = alpha if alpha_b is None else alpha_b
    if not np.isfinite([aa, ab]).all() or min(aa, ab) <= 0:
        raise ValueError("Dirichlet alpha must be finite and positive")
    rng = np.random.default_rng(seed)
    a = np.maximum(rng.dirichlet(np.full(n, aa)), marginal_floor)
    b = np.maximum(rng.dirichlet(np.full(m, ab)), marginal_floor)
    a, b = a/a.sum(), b/b.sum()
    if generator in ("random_dense", "spiky_marginal"):
        M = rng.uniform(0, 1, (n, m))
    elif generator in ("1d_squared", "1d_geometric"):
        x, y = rng.uniform(0, 1, (n, 1)), rng.uniform(0, 1, (m, 1))
        M = geometric_cost(x, y, 2 if generator == "1d_squared" else p)
    elif generator == "2d_pointcloud":
        if clustered:
            centers = np.array([[.2, .2], [.8, .8], [.2, .8]])
            x = centers[rng.integers(0, 3, n)] + rng.normal(0, .05, (n, 2))
            y = centers[rng.integers(0, 3, m)] + rng.normal(0, .05, (m, 2))
        else:
            x, y = rng.uniform(0, 1, (n, 2)), rng.uniform(0, 1, (m, 2))
        M = geometric_cost(x, y)
    else:
        raise ValueError(f"Unknown generator: {generator}")
    return OTProblem(a, b, normalize_cost(M), epsilon, name=generator,
                     metadata={"generator": generator, "seed": seed, "n": n, "m": m,
                               "epsilon": epsilon, "alpha_a": aa, "alpha_b": ab,
                               "marginal_floor": marginal_floor, "cost_scale": "max",
                               "clustered": clustered, "p": p,
                               "raw_cost_max": float(np.max(M))})
