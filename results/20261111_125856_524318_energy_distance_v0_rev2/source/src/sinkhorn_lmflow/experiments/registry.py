from __future__ import annotations

from collections.abc import Callable
from ..types import OTProblem, SolverConfig, SolverResult
from ..solvers.sinkhorn import solve_sinkhorn
from ..solvers.newton import solve_newton
from ..solvers.lbfgs import solve_lbfgs
from ..solvers.hard_switch import solve_hard_switch
from ..solvers.lmflow import solve_lmflow
from ..solvers.theta_balance import solve_theta_balance
from ..solvers.mirror_sr1 import solve_mirror_sr1, solve_sinkhorn_reduced
from ..solvers.sinkhorn_fast import solve_sk_log_fast, solve_sk_scaling_fast
from ..solvers.lbfgs_reduced import solve_lbfgs_reduced


def get_solver(name: str) -> Callable[[OTProblem, SolverConfig], SolverResult]:
    solvers = {"sinkhorn": solve_sinkhorn, "newton": solve_newton,
               "lbfgs": solve_lbfgs, "hard_switch": solve_hard_switch, "lmflow": solve_lmflow,
               "theta_balance": solve_theta_balance, "mirror_sr1": solve_mirror_sr1,
               "sinkhorn_reduced": solve_sinkhorn_reduced, "sk_log_fast": solve_sk_log_fast,
               "sk_scaling_fast": solve_sk_scaling_fast, "lbfgs_reduced": solve_lbfgs_reduced}
    if name not in solvers:
        raise ValueError(f"Unknown solver {name!r}; choose from {tuple(solvers)}")
    return solvers[name]
