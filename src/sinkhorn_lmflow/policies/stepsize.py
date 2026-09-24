from __future__ import annotations

from collections.abc import Callable
import numpy as np
from ..types import SolverConfig, OTProblem
from ..core.transport import transport_plan_from_potentials
from ..core.dual import dual_objective
from ..core.residuals import compute_stop_metrics


class StepFailure(FloatingPointError):
    def __init__(self, reason: str, safeguards: int, backtracks: int):
        super().__init__(reason)
        self.safeguards = safeguards
        self.backtracks = backtracks


def choose_step(problem: OTProblem, config: SolverConfig, mu: np.ndarray,
                nu: np.ndarray, objective: float, slope: float, residual: float,
                update: Callable[[float], tuple[np.ndarray, np.ndarray]],
                work: dict | None = None) -> tuple:
    """Backtrack along the chosen discretization; positivity is always enforced.

    Fixed-step mode disables Armijo only. Near roundoff, accept nonincreasing
    objective (to numerical precision) only with strictly decreasing residual.
    """
    h = config.step_size
    safeguards = 0
    last_reason = "line_search_failed"
    for j in range(config.max_backtracks + 1):
        if work is not None:
            work["trial_count"] += 1
        try:
            x, y = update(h)
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise FloatingPointError("Nonfinite potential")
            Q = transport_plan_from_potentials(problem.M, problem.epsilon, x, y)
            if work is not None:
                work["plan_build_count"] += 1
                work["reduction_count"] += 1
            value = dual_objective(problem, x, y, Q)
            if not np.isfinite(value):
                raise FloatingPointError("Nonfinite objective")
        except FloatingPointError as exc:
            if "positivity" in str(exc):
                safeguards += 1
                last_reason = "positivity_line_search_failed"
            else:
                last_reason = "nonfinite_trial"
            h *= config.backtrack_factor
            continue
        accept = not config.line_search or value <= objective + config.armijo*h*slope
        if not accept and abs(value-objective) <= 16*np.finfo(float).eps*max(1, abs(objective)):
            if work is not None:
                work["reduction_count"] += 2
            accept = compute_stop_metrics(Q, problem.a, problem.b)["residual_l1"] < residual
        if accept:
            return x, y, h, safeguards, j
        last_reason = "armijo_line_search_failed"
        h *= config.backtrack_factor
    raise StepFailure(last_reason, safeguards, config.max_backtracks)
