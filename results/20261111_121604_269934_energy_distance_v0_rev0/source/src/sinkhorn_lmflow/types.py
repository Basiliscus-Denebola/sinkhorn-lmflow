from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np

from .problems.instance import OTProblem


@dataclass
class SolverConfig:
    tol: float = 1e-9
    max_iter: int = 1000
    theta: float = 0.5
    discretization: str = "scaling_coordinate"
    gauss_seidel: bool = False
    step_size: float = 1.0
    line_search: bool = True
    backtrack_factor: float = 0.5
    armijo: float = 1e-4
    max_backtracks: int = 60
    numerical_mode: str = "log_stabilized"
    gauge_normalization: bool = True
    switch_tol: float = 1e-4
    record_every: int = 1
    compute_spectrum_every: int = 10
    initial_mu: np.ndarray | None = None
    initial_nu: np.ndarray | None = None
    balance_theta_max: float = 10.0
    balance_theta_fixed: float | None = None
    wall_time_limit_sec: float | None = None
    sr1_warmup_iters: int = 20
    sr1_secant_tol: float = 1e-8

    def __post_init__(self) -> None:
        if not isinstance(self.sr1_warmup_iters, int) or self.sr1_warmup_iters < 0:
            raise ValueError("Invalid sr1_warmup_iters")
        if not np.isfinite(self.sr1_secant_tol) or self.sr1_secant_tol <= 0:
            raise ValueError("Invalid sr1_secant_tol")
        for name in ("balance_theta_fixed", "wall_time_limit_sec"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive or None")
        if not np.isfinite(self.balance_theta_max) or self.balance_theta_max <= 0:
            raise ValueError("balance_theta_max must be finite and positive")
        if not np.isfinite(self.tol) or self.tol <= 0:
            raise ValueError("tol must be finite and positive")
        for name in ("max_iter", "max_backtracks", "record_every", "compute_spectrum_every"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < (1 if name == "record_every" else 0):
                raise ValueError(f"Invalid {name}")
        if not np.isfinite(self.theta) or not 0 <= self.theta <= 1:
            raise ValueError("theta must be in [0, 1]")
        if not np.isfinite(self.step_size) or self.step_size <= 0:
            raise ValueError("step_size must be positive")
        if not 0 < self.backtrack_factor < 1 or not 0 < self.armijo < 1:
            raise ValueError("Invalid line-search constants")
        if not np.isfinite(self.switch_tol) or self.switch_tol <= 0:
            raise ValueError("switch_tol must be positive")
        if self.discretization not in ("additive", "scaling_coordinate"):
            raise ValueError("Unknown discretization")
        if self.numerical_mode not in ("primal_scaling", "log_stabilized"):
            raise ValueError("Unknown Sinkhorn numerical mode")


@dataclass
class IterationRecord:
    iteration: int
    elapsed_sec: float
    residual_l1: float
    residual_linf: float
    grad_norm_l2: float
    dual_objective: float | None
    theta: float | None
    step_size: float | None
    sigma_nontrivial: float | None
    spectral_gap: float | None
    linear_solve_iters: int | None
    safeguard_count: int
    sigma1: float | None = None
    linear_solve_sec: float = 0.0
    linear_solve_count: int = 0
    matrix_vector_count: int = 0
    backtrack_count: int = 0
    phase: str = ""
    diagnostic_sec: float = 0.0
    balance_theta_free: float | None = None
    balance_theta: float | None = None
    balance_effective: float | None = None
    balance_eta: float | None = None
    direction_residual_l1: float | None = None
    residual_sum_l1: float | None = None
    pre_residual_l1: float | None = None
    row_residual_l2: float | None = None
    balance_predicted_ratio_l2: float | None = None
    balance_actual_ratio_l2: float | None = None
    balance_clip_count: int = 0
    balance_degenerate_count: int = 0
    balance_fallback_count: int = 0
    balance_reason: str = ""
    plan_build_count: int = 0
    reduction_count: int = 0
    logsumexp_count: int = 0
    p_matvec_count: int = 0
    column_update_count: int = 0
    trial_count: int = 0
    dense_pass_proxy: int = 0
    reduced_f: float | None = None
    energy_gap: float | None = None
    row_residual_l1: float | None = None
    col_residual_l1: float | None = None
    state_eval_count: int = 0
    objective_diff_count: int = 0
    bregman_count: int = 0
    sr1_attempt_count: int = 0
    sr1_valid_count: int = 0
    sr1_accept_count: int = 0
    sr1_fallback_count: int = 0
    sr1_exhausted_count: int = 0
    sr1_reject_count: int = 0
    sr1_skip_count: int = 0
    sr1_d: float | None = None
    sr1_relative_d: float | None = None
    sr1_J: float | None = None
    sr1_e_dot_V: float | None = None
    sr1_G: float | None = None
    sr1_Q: float | None = None
    sr1_cos: float | None = None
    sr1_actual_decrease: float | None = None
    sr1_reason: str = ""
    sr1_ratio_reason: str = ""
    accept_reason: str = ""
    from_iteration: int | None = None
    post_elapsed_sec: float = 0.0
    objective_gradient_count: int = 0
    log_kernel_build_count: int = 0
    kernel_build_count: int = 0
    state_fingerprint: str = ""


@dataclass
class SolverResult:
    solver_name: str
    converged: bool
    stop_reason: str
    n_iter: int
    elapsed_sec: float
    mu: np.ndarray
    nu: np.ndarray
    u: np.ndarray
    v: np.ndarray
    history: list[IterationRecord]
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class SolverState:
    iteration: int
    residual_l1: float
    mu: np.ndarray
    nu: np.ndarray
