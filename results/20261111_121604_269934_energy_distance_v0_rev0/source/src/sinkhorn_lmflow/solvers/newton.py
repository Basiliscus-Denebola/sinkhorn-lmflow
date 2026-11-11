from __future__ import annotations

from dataclasses import replace
from ..types import OTProblem, SolverConfig, SolverResult
from .lmflow import solve_lmflow


def solve_newton(problem: OTProblem, config: SolverConfig) -> SolverResult:
    """Exact full-Hessian, gauge-augmented Newton with additive Armijo steps."""
    cfg = replace(config, theta=1, discretization="additive", gauss_seidel=False, line_search=True)
    return solve_lmflow(problem, cfg, name="newton")
