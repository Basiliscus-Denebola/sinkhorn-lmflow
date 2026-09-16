"""Future RegOT Tier B adapter: external optimized C++ implementation.

Planned methods: sinkhorn_bcd, sinkhorn_lbfgs_dual, sinkhorn_newton,
sinkhorn_sparse_newton, sinkhorn_ssns, optional sinkhorn_splr.
TODO: verify objective/epsilon conventions and shared stopping metrics before
using external results. No automatic clone, installation, or speed claims.
"""
from __future__ import annotations

from ..types import OTProblem, SolverConfig, SolverResult


def solve_regot(problem: OTProblem, config: SolverConfig, method: str = "sinkhorn_ssns") -> SolverResult:
    raise NotImplementedError("Phase two: validated RegOT adapter required")
