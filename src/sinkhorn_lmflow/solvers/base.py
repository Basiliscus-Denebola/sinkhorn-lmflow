from __future__ import annotations

from time import perf_counter
from typing import Protocol
import numpy as np
from ..types import OTProblem, SolverConfig, IterationRecord, SolverResult
from ..core.transport import transport_plan_from_potentials
from ..core.dual import dual_objective, transport_objective
from ..core.gauge import normalize_gauge
from ..core.residuals import compute_stop_metrics
from ..core.spectral import spectral_diagnostics
from ..core.stability import finite_scalings


class OTSolver(Protocol):
    name: str

    def solve(self, problem: OTProblem, config: SolverConfig) -> SolverResult:
        ...


def initial_state(problem: OTProblem, config: SolverConfig) -> tuple[np.ndarray, np.ndarray]:
    mu = np.zeros_like(problem.a) if config.initial_mu is None else np.array(config.initial_mu, dtype=float, copy=True)
    nu = np.zeros_like(problem.b) if config.initial_nu is None else np.array(config.initial_nu, dtype=float, copy=True)
    if mu.shape != problem.a.shape or nu.shape != problem.b.shape:
        raise ValueError("Initial potential shapes must match marginals")
    if not np.isfinite(mu).all() or not np.isfinite(nu).all():
        raise ValueError("Initial potentials must be finite")
    return normalize_gauge(mu, nu) if config.gauge_normalization else (mu, nu)


class Monitor:
    """Shared metrics, timing, counters and result construction.

    Wall time includes diagnostics; diagnostic time is also reported separately.
    Matrix-vector count counts explicit K@v/K.T@u operations, not dense BLAS
    operations hidden inside factorizations or logsumexp reductions.
    """
    def __init__(self, problem: OTProblem, config: SolverConfig, name: str):
        self.problem, self.config, self.name = problem, config, name
        self.start = perf_counter()
        self.history: list[IterationRecord] = []
        self.safeguards = self.backtracks = self.linear_count = self.matvec_count = 0
        self.linear_time = self.diagnostic_time = 0.0
        self.last_linear_sec = 0.0
        self.last_P: np.ndarray | None = None
        self.last_metrics: dict[str, float] = {}
        self.last_objective: float | None = None
        self.iteration = 0
        self.last_theta: float | None = None
        self.last_step: float | None = None
        self.last_phase = "initial"
        self.diagnostic_errors: list[str] = []
        self.work = dict(plan_build_count=0, reduction_count=0, logsumexp_count=0,
                         p_matvec_count=0, column_update_count=0, trial_count=0)
        self.balance: dict = {}
        self.last_row_residual_l2: float | None = None
        self.pre_residual_l1: float | None = None
        self.pre_row_residual_l2: float | None = None

    def timed_out(self) -> bool:
        limit = self.config.wall_time_limit_sec
        return limit is not None and perf_counter()-self.start >= limit

    def observe(self, k: int, mu: np.ndarray, nu: np.ndarray, theta: float | None = None,
                step: float | None = None, phase: str = "", force: bool = False) -> float:
        p, cfg = self.problem, self.config
        if k != self.iteration:
            self.pre_residual_l1 = self.last_metrics.get("residual_l1")
            self.pre_row_residual_l2 = self.last_row_residual_l2
        self.iteration, self.last_theta, self.last_step, self.last_phase = k, theta, step, phase
        self.last_P = transport_plan_from_potentials(p.M, p.epsilon, mu, nu)
        self.last_metrics = compute_stop_metrics(self.last_P, p.a, p.b)
        self.last_objective = dual_objective(p, mu, nu, self.last_P)
        self.work["plan_build_count"] += 1
        self.work["reduction_count"] += 5  # metrics (2), objective (1), residual sum (2)
        row_residual = self.last_P.sum(axis=1)-p.a
        row_l2 = float(np.linalg.norm(row_residual))
        actual_ratio = (row_l2/self.pre_row_residual_l2
                        if k > 0 and self.pre_row_residual_l2 else None)
        residual_sum = float(np.linalg.norm(row_residual, 1)
                             + np.linalg.norm(self.last_P.sum(axis=0)-p.b, 1))
        self.last_row_residual_l2 = row_l2
        if not np.isfinite(list(self.last_metrics.values())).all() or not np.isfinite(self.last_objective):
            raise FloatingPointError("nonfinite metrics")
        spectrum_due = cfg.compute_spectrum_every > 0 and k % cfg.compute_spectrum_every == 0
        if force or k % cfg.record_every == 0 or spectrum_due or self.last_metrics["residual_l1"] <= cfg.tol:
            spec = {"sigma1": None, "sigma_nontrivial": None, "spectral_gap": None}
            if spectrum_due:
                t = perf_counter()
                try:
                    spec = spectral_diagnostics(self.last_P, theta)
                except (FloatingPointError, np.linalg.LinAlgError) as exc:
                    self.diagnostic_errors.append(f"iteration {k}: {exc}")
                self.diagnostic_time += perf_counter() - t
            record = IterationRecord(k, perf_counter()-self.start, **self.last_metrics,
                dual_objective=self.last_objective, theta=theta, step_size=step,
                linear_solve_iters=1 if self.last_linear_sec else None,
                safeguard_count=self.safeguards, linear_solve_sec=self.last_linear_sec,
                linear_solve_count=self.linear_count, matrix_vector_count=self.matvec_count,
                backtrack_count=self.backtracks, phase=phase,
                diagnostic_sec=self.diagnostic_time, residual_sum_l1=residual_sum,
                pre_residual_l1=self.pre_residual_l1, row_residual_l2=row_l2,
                balance_actual_ratio_l2=actual_ratio,
                dense_pass_proxy=sum(self.work[key] for key in
                    ("plan_build_count", "reduction_count", "logsumexp_count", "p_matvec_count")),
                **self.work, **self.balance, **spec)
            if self.history and self.history[-1].iteration == k:
                self.history[-1] = record
            else:
                self.history.append(record)
        return self.last_metrics["residual_l1"]

    def finish(self, mu: np.ndarray, nu: np.ndarray, reason: str,
               extra: dict | None = None) -> SolverResult:
        # Preserve the final accepted iterate, including failures and thinned logs.
        if not self.history or self.history[-1].iteration != self.iteration:
            try:
                self.observe(self.iteration, mu, nu, self.last_theta, self.last_step, self.last_phase, force=True)
            except FloatingPointError:
                pass
        if self.history:
            self.history[-1].safeguard_count = self.safeguards
            self.history[-1].backtrack_count = self.backtracks
        u, v = finite_scalings(mu, nu, self.problem.epsilon)
        converged = bool(self.last_metrics and self.last_metrics["residual_l1"] <= self.config.tol)
        summary = dict(self.last_metrics)
        summary.update(dual_objective=self.last_objective,
            transport_objective=None if self.last_P is None else transport_objective(self.problem, self.last_P),
            safeguard_count=self.safeguards, backtrack_count=self.backtracks,
            linear_solve_count=self.linear_count, linear_solve_sec=self.linear_time,
            matrix_vector_count=self.matvec_count, diagnostic_sec=self.diagnostic_time,
            diagnostics_errors=self.diagnostic_errors,
            scalings_representable=bool(np.isfinite(u).all() and np.isfinite(v).all()
                                        and np.all(u > 0) and np.all(v > 0)),
            timing_includes_diagnostics=True)
        summary.update(extra or {})
        summary.update(self.work)
        summary.update(self.balance)
        elapsed = perf_counter()-self.start
        return SolverResult(self.name, converged, "tolerance" if converged else reason,
                            self.iteration, elapsed, mu.copy(), nu.copy(), u, v,
                            self.history, summary)
