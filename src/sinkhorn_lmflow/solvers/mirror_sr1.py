"""Single-pair positive inverse SR1 over a shared row-normalized Sinkhorn flow.

Step diagnostics on row k refer to (from_iteration=k-1, to_iteration=k).
Only accepted states enter the secant history. No dense inverse/Hessian exists.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
from time import perf_counter
import numpy as np
from ..core.semi_dual import (ReducedState, ReducedWork, center, evaluate_reduced,
                             stable_objective_difference, bregman_phi)
from ..io.problem_io import problem_fingerprint
from ..problems.weak_coupling import initial_fingerprint
from ..types import SolverConfig
from .base import Monitor, deadline_reached


def base_diagonal(c, b, log_c):
    e = c-b
    t = e/b
    psi = np.asarray(log_c)-np.log(b)
    beta = np.empty_like(c)
    small = np.abs(t) < 1e-4
    x = t[small]
    beta[small] = (1-x/2+x*x/3-x**3/4+x**4/5)/b[small]
    # This also makes near-equilibrium psi consistent with the computed c.
    psi[small] = np.log1p(t[small])
    beta[~small] = psi[~small]/e[~small]
    return beta, psi


@dataclass
class SR1Direction:
    V: np.ndarray
    psi: np.ndarray
    beta: np.ndarray
    q: np.ndarray | None = None
    d: float | None = None
    relative_d: float | None = None
    J: float | None = None
    e_dot_V: float | None = None
    G: float | None = None
    cosine: float | None = None
    reason: str = "no_history"
    ratio_reason: str = ""

    @property
    def valid(self):
        return self.reason == "valid"


def sr1_direction(z, c, b, log_c, previous=None, secant_tol=1e-8):
    beta, psi = base_diagonal(c, b, log_c)
    e = c-b
    out = SR1Direction(psi.copy(), psi, beta)
    if not np.isfinite(np.r_[beta, psi, z, c]).all():
        out.reason = "nonfinite"
        return out
    out.J = float(e@psi)
    if previous is None:
        return out
    s, y = center(z)-center(previous[0]), c-previous[1]
    q = s-beta*y
    out.q = q
    if not np.isfinite(np.r_[s, y, q]).all():
        out.reason = "nonfinite"
        return out
    nq, ny = np.linalg.norm(q), np.linalg.norm(y)
    if np.linalg.norm(s) == 0 or ny == 0:
        out.reason = "zero_history"
        return out
    if nq == 0:
        out.reason = "zero_q"
        return out
    d = float(q@y)
    scale = nq*ny
    if not np.isfinite([d, scale]).all():
        out.reason = "nonfinite"
        return out
    out.d = d
    out.relative_d = d/scale if scale > 0 else None
    if d <= 0:
        out.reason = "nonpositive_denominator"
        return out
    if d < secant_tol*scale:
        out.reason = "small_denominator"
        return out
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        V = psi+q*(float(q@e)/d)
        edv = float(e@V)
    if not np.isfinite(V).all() or not np.isfinite(edv) or edv <= 0:
        out.reason = "nonfinite_correction" if not np.isfinite(edv) or not np.isfinite(V).all() else "nonpositive_descent"
        return out
    out.V, out.e_dot_V, out.reason = V, edv, "valid"
    if out.J > np.finfo(float).tiny and np.isfinite(edv/out.J):
        out.G = edv/out.J
    else:
        out.ratio_reason = "unreliable_J"
    v, p = center(V), center(psi)
    normprod = np.linalg.norm(v)*np.linalg.norm(p)
    if normprod > np.finfo(float).tiny and np.isfinite(normprod):
        out.cosine = float(v@p/normprod)
    return out


@dataclass
class WarmupSnapshot:
    state: ReducedState
    previous: tuple | None
    history: list
    work: ReducedWork
    elapsed_sec: float
    iteration: int
    problem_sha256: str
    initial_sha256: str
    protocol: str
    fingerprint: str = ""
    diagnostic_reference: ReducedState | None = None

    def digest(self):
        h = hashlib.sha256()
        for value in (self.problem_sha256, self.initial_sha256, self.protocol, str(self.iteration)):
            h.update(value.encode())
        for value in (self.state.z_col, self.state.c, self.state.P, self.state.mu, self.state.nu):
            h.update(np.asarray(value, dtype="<f8").tobytes())
        if self.previous is not None:
            for value in self.previous:
                h.update(np.asarray(value, dtype="<f8").tobytes())
        return h.hexdigest()

    def clone(self):
        return deepcopy(self)


def _validate(config):
    if config.compute_spectrum_every != 0 or config.record_every != 1:
        raise ValueError("reduced SR1 requires record_every=1, compute_spectrum_every=0")
    if config.step_size != 1 or config.backtrack_factor != .5 or config.armijo != 1e-4:
        raise ValueError("v0 fixes h=1, contraction=.5, Armijo=1e-4")
    if config.max_backtracks not in (5, 60):
        raise ValueError("v0 supports six trials (max_backtracks=5; legacy default 60 maps to 5)")
    if not config.line_search or config.gauss_seidel or config.numerical_mode != "log_stabilized":
        raise ValueError("v0 requires log stabilization and its fixed globalization")


def warmup_protocol(config):
    return f"row_normalized_sk:{config.sr1_warmup_iters}:tol={config.tol:.17g}"


def _solve(problem, config, use_sr1, warmup=None, deadline=None, diagnostic_reference_z=None,
           return_snapshot=False, performance=False, diagnostics=False, amplification=False):
    _validate(config)
    diagnostic_mode = diagnostics
    started = perf_counter()
    monitor = Monitor(problem, config, "mirror_amp_ablation" if amplification else "mirror_sr1" if use_sr1 else "sinkhorn_reduced")
    supplied_z = np.zeros_like(problem.b) if config.initial_nu is None else -np.asarray(config.initial_nu)/problem.epsilon
    initial_z = center(supplied_z)
    counters = dict(sr1_attempt_count=0, sr1_valid_count=0, sr1_accept_count=0,
                    sr1_fallback_count=0, sr1_exhausted_count=0, sr1_reject_count=0,
                    sr1_skip_count=0, trial_count=0)
    skips = Counter()
    work = ReducedWork()
    previous, reference = None, None
    warm_seconds, fork_iteration = 0., 0
    light_warmup = bool(warmup is not None and getattr(warmup, 'lightweight', False))
    log_kernel = warmup.log_kernel if light_warmup else None
    if performance and log_kernel is None and not light_warmup:
        log_kernel = -problem.M/problem.epsilon
        work.log_kernel_build_count += 1
    setup_sec = perf_counter()-started

    def evaluate(z):
        if performance:
            return evaluate_reduced(problem,z,work,log_kernel=log_kernel)
        return evaluate_reduced(problem,z,work)

    if warmup is not None:
        if light_warmup:
            if not performance:
                raise ValueError('lightweight warmup requires performance mode')
            warmup.validate(problem,config)
        elif (warmup.problem_sha256 != problem_fingerprint(problem) or
                warmup.initial_sha256 != initial_fingerprint(supplied_z) or
                warmup.protocol != warmup_protocol(config) or warmup.fingerprint != warmup.digest()):
            raise ValueError("warmup fingerprint/protocol mismatch")
        snap = warmup.clone()
        state, previous, work, reference = snap.state, snap.previous, snap.work, snap.diagnostic_reference
        if light_warmup:
            # Construct P only when SR1 actually needs it, charged to this method.
            if log_kernel is None:
                log_kernel=-problem.M/problem.epsilon
                work.log_kernel_build_count+=1
            state = evaluate(snap.state.z_col)
        monitor.history = snap.history
        warm_seconds, fork_iteration = snap.elapsed_sec, snap.iteration
        monitor.iteration = snap.iteration
        monitor.start = started-warm_seconds
        monitor.last_P, monitor.last_metrics = state.P, dict(state.metrics)
        monitor.last_objective = state.full_objective
        monitor.last_row_residual_l2 = float(np.linalg.norm(state.r-problem.a))
    else:
        state = evaluate(initial_z)
        if diagnostic_reference_z is not None:
            reference = evaluate_reduced(problem, diagnostic_reference_z, work)
    if performance:
        setup_sec=perf_counter()-started

    def expired():
        now = perf_counter()
        return (deadline_reached(deadline, now) or
                (config.wall_time_limit_sec is not None and now-started+(warm_seconds if performance else 0.) >= config.wall_time_limit_sec))

    def record(k, current, step=None, phase="initial", diagnostics=None):
        old_residual = monitor.last_metrics.get("residual_l1", np.inf)
        checkpoint = k in (0, config.sr1_warmup_iters, config.max_iter) or any(
            old_residual > t >= current.metrics["residual_l1"] for t in (1e-4, 1e-6, 1e-8))
        energy = None
        if reference is not None and checkpoint:
            energy = bregman_phi(problem, reference, current.z_col-reference.z_col, work)
        for key in monitor.work:
            if hasattr(work, key):
                monitor.work[key] = getattr(work, key)
        monitor.work["trial_count"] = counters["trial_count"]
        monitor.observe(k, current.mu, current.nu, step=step, phase=phase, cached_state=current)
        row = monitor.history[-1]
        for key, value in counters.items():
            setattr(row, key, value)
        for key in ("state_eval_count", "objective_diff_count", "bregman_count", "dense_pass_proxy"):
            setattr(row, key, getattr(work, key))
        row.log_kernel_build_count = work.log_kernel_build_count
        if performance:
            row.state_fingerprint = hashlib.sha256(current.z_col.tobytes()+current.c.tobytes()).hexdigest()
        row.reduced_f, row.energy_gap = current.f, energy
        row.row_residual_l1 = float(np.linalg.norm(current.r-problem.a, 1))
        row.col_residual_l1 = float(np.linalg.norm(current.c-problem.b, 1))
        row.post_elapsed_sec = perf_counter()-started if warmup is not None else 0.
        for key, value in (diagnostics or {}).items():
            setattr(row, key, value)

    if warmup is None:
        record(0, state)
    reason = "max_iter"
    for k in range(monitor.iteration+1, config.max_iter+1):
        if state.metrics["residual_l1"] <= config.tol:
            break
        if expired():
            reason = "timeout"
            break
        direction = None
        trial = None
        diagnostics = {"from_iteration": k-1}
        phase = "warmup" if k <= config.sr1_warmup_iters else "sinkhorn"
        step, delta = 1., None
        if use_sr1 and k > config.sr1_warmup_iters:
            counters["sr1_attempt_count"] += 1
            direction = sr1_direction(state.z_col, state.c, problem.b, state.log_c, previous, config.sr1_secant_tol)
            if amplification and direction.valid:
                if direction.G is None or not np.isfinite(direction.G) or direction.G <= 0:
                    direction.reason = 'unreliable_amplification_ratio'
                else:
                    direction.V = direction.G*direction.psi
                    direction.e_dot_V = float((state.c-problem.b)@direction.V)
                    direction.cosine = 1.
            diagnostics.update(sr1_reason=direction.reason, sr1_d=direction.d,
                               sr1_relative_d=direction.relative_d, sr1_ratio_reason=direction.ratio_reason)
            if direction.valid:
                counters["sr1_valid_count"] += 1
                for trial_index in range(6):
                    if expired():
                        reason = "timeout"
                        break
                    step = .5**trial_index
                    u = -step*center(direction.V)
                    counters["trial_count"] += 1
                    try:
                        candidate = evaluate(state.z_col+u)
                        delta = stable_objective_difference(problem, state, u, work)
                        if delta <= -config.armijo*step*direction.e_dot_V:
                            trial = candidate
                            phase = "sr1"
                            counters["sr1_accept_count"] += 1
                            diagnostics.update(accept_reason="armijo", sr1_J=direction.J,
                                sr1_e_dot_V=direction.e_dot_V, sr1_G=direction.G, sr1_cos=direction.cosine,
                                sr1_Q=-delta/(step*direction.e_dot_V))
                            break
                    except (FloatingPointError, OverflowError):
                        pass
                    counters["sr1_reject_count"] += 1
                    monitor.backtracks += 1
                if reason == "timeout":
                    break
                if trial is None:
                    counters["sr1_exhausted_count"] += 1
                    diagnostics["sr1_reason"] = "backtracks_exhausted"
            else:
                counters["sr1_skip_count"] += 1
                skips[direction.reason] += 1
            if trial is None:
                phase = "sk_fallback"
                counters["sr1_fallback_count"] += 1
        if trial is None:
            if expired():
                reason = "timeout"
                break
            _, psi = base_diagonal(state.c, problem.b, state.log_c)
            u = -center(psi)
            step = 1.
            try:
                trial = evaluate(state.z_col+u)
                delta = stable_objective_difference(problem, state, u, work) if not performance or diagnostic_mode else None
            except (FloatingPointError, OverflowError) as exc:
                reason = f"fallback_failure: {exc}"
                break
            diagnostics["accept_reason"] = "sinkhorn"
        diagnostics["sr1_actual_decrease"] = -delta if delta is not None else None
        previous = (state.z_col.copy(), state.c.copy())
        state = trial
        record(k, state, step, phase, diagnostics)
    for key in monitor.work:
        if hasattr(work, key):
            monitor.work[key] = getattr(work, key)
    monitor.work["trial_count"] = counters["trial_count"]
    extra = dict(work.to_dict(), **{k: v for k, v in counters.items() if k != "trial_count"},
                 sr1_skip_reasons=dict(skips), reduced_f=state.f,
                 warmup_elapsed_sec=warm_seconds, fork_iteration=fork_iteration,
                 post_elapsed_sec=perf_counter()-started,
                 sr1_trial_limit=6, diagnostic_reference_evaluations=int(reference is not None),
                 performance_mode=performance, diagnostic_mode=bool(diagnostic_mode),
                 setup_sec=setup_sec, bridge_state_evaluations=int(light_warmup),
                 warmup_sha256=warmup.fingerprint if warmup is not None else None,
                 energy_checkpoint_policy="initial,warmup,threshold crossings,max_iter",
                 probability_renormalization="Bregman evaluation only; P and marginals unchanged")
    result = monitor.finish(state.mu, state.nu, reason, extra,include_transport_objective=not performance)
    if warmup is not None and state.metrics["residual_l1"] <= config.tol and monitor.iteration == fork_iteration:
        result.stop_reason = "warmup_converged"
    snapshot = None
    if return_snapshot:
        snapshot = WarmupSnapshot(state, previous, deepcopy(result.history), deepcopy(work),
            result.elapsed_sec, result.n_iter, problem_fingerprint(problem), initial_fingerprint(supplied_z),
            warmup_protocol(config), diagnostic_reference=reference)
        snapshot.fingerprint = snapshot.digest()
    result.elapsed_sec = perf_counter()-monitor.start
    result.summary['post_elapsed_sec'] = perf_counter()-started
    if snapshot is not None:
        snapshot.elapsed_sec = result.elapsed_sec
    return result, snapshot


def prepare_warmup(problem, config, deadline=None, diagnostic_reference_z=None):
    cfg = replace(config, max_iter=min(config.max_iter, config.sr1_warmup_iters))
    return _solve(problem, cfg, False, deadline=deadline, diagnostic_reference_z=diagnostic_reference_z,
                  return_snapshot=True)


def solve_mirror_sr1(problem, config, warmup=None, deadline=None, *, performance=False, diagnostics=False, amplification=False):
    if warmup is not None and getattr(warmup,'lightweight',False) and warmup.state.metrics['residual_l1']<=config.tol:
        return warmup.finished_result(problem,config,'mirror_amp_ablation' if amplification else 'mirror_sr1')
    return _solve(problem, config, True, warmup=warmup, deadline=deadline,performance=performance,
                  diagnostics=diagnostics,amplification=amplification)[0]


def solve_sinkhorn_reduced(problem, config, warmup=None, deadline=None):
    return _solve(problem, config, False, warmup=warmup, deadline=deadline)[0]
