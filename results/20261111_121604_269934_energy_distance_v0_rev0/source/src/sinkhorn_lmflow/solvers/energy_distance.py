"""Fixed full-cost energy-distance driver, with stable Armijo and unit SK fallback."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from time import perf_counter
import numpy as np
from scipy.special import logsumexp
from ..core.cost_profile_distance import build_distance
from ..core.semi_dual import center, expm1_minus_x
from ..problems.instance import OTProblem
from .sinkhorn_fast import LogEvaluator, prepare_fast_warmup, finish_light, light_record


class DeadlineExpired(Exception):
    pass


class EqualMassEnergyProblem(OTProblem):
    """Opt-in general equal-mass problem; legacy OTProblem stays unit-mass."""
    def __post_init__(self):
        for key in ('a','b','M'):
            value = np.array(getattr(self,key),dtype=float,copy=True)
            value.flags.writeable = False
            object.__setattr__(self,key,value)
        if (self.a.ndim!=1 or self.b.ndim!=1 or not len(self.a) or not len(self.b)
                or self.M.shape!=(len(self.a),len(self.b))
                or not all(np.isfinite(x).all() for x in (self.a,self.b,self.M))
                or np.any(self.a<=0) or np.any(self.b<=0)
                or not np.isclose(self.a.sum(),self.b.sum(),rtol=1e-12,atol=0)
                or not np.isfinite(self.epsilon) or self.epsilon<=0):
            raise ValueError('finite cost, positive equal marginals and epsilon required')


def check_deadline(deadline):
    if deadline is not None and (bool(deadline()) if callable(deadline) else perf_counter() >= deadline):
        raise DeadlineExpired()


def stable_light_difference(problem, state, displacement, log_kernel, work, stats,
                            deadline=None, block_rows=64):
    """Same centered conditional-law Bregman formula as semi_dual, in blocks.

    P is never retained. Each block conditional multiply is explicitly counted;
    one logical plan construction per invocation, including rejected trials.
    """
    work.objective_diff_count += 1
    work.bregman_count += 1
    work.plan_build_count += 1
    work.probability_normalization_count += 1
    u = center(displacement)
    value = float((state.c - problem.b) @ u)
    for lo in range(0, len(problem.a), block_rows):
        check_deadline(deadline)
        hi = min(lo + block_rows, len(problem.a))
        lp = log_kernel[lo:hi] + state.z_col[None, :] - state.ell[lo:hi, None]
        p = np.exp(lp)
        mass = p.sum(1)
        error = float(np.max(np.abs(mass - 1)))
        work.probability_mass_error_max = max(work.probability_mass_error_max, error)
        if not np.isfinite(mass).all() or error > 1e-11:
            raise FloatingPointError('conditional probability normalization error')
        p /= mass[:, None]
        mean = p @ u
        stats['conditional_block_matvec_count'] += 1
        v = u[None, :] - mean[:, None]
        mean_error = float(np.max(np.abs(np.sum(p*v, axis=1))))
        work.probability_mean_error_max = max(work.probability_mean_error_max, mean_error)
        if mean_error > 1e-12*max(1., float(np.max(np.abs(u)))):
            raise FloatingPointError('conditional centered mean error')
        if np.max(np.abs(v)) <= .5:
            values = np.log1p(np.sum(p * expm1_minus_x(v), axis=1))
        else:
            values = logsumexp(lp - np.log(mass)[:, None] + v, axis=1)
            work.logsumexp_count += 1
        value += float(problem.a[lo:hi] @ values)
        work.reduction_count += 4
    if not np.isfinite(value):
        raise FloatingPointError('nonfinite stable objective difference')
    return value


def energy_terms(D, e, stats=None):
    e0 = center(e)
    start = perf_counter()
    product = D @ e0
    if stats is not None:
        stats['energy_matvec_count'] += 1
        stats['energy_matvec_sec'] += perf_counter() - start
    return center(-.5 * product), float(-.25 * (e0 @ product))


def frozen_gamma(D, e, psi, kappa, stats):
    u, energy = energy_terms(D, e, stats)
    norm_u, norm_psi = float(np.linalg.norm(u)), float(np.linalg.norm(center(psi)))
    scale = 100*np.finfo(float).eps*max(1., float(np.linalg.norm(D, np.inf)))*np.linalg.norm(center(e))
    degenerate = bool(norm_u <= scale)
    gamma = 0. if degenerate else kappa*norm_psi/norm_u
    stats.update(gamma_psi_norm=norm_psi, gamma_u_norm=norm_u,
                 gamma_unresolvable_threshold=float(scale), geometry_degenerate=degenerate,
                 geometry_degenerate_reason='u below numerical resolution' if degenerate else None)
    return float(gamma), u, energy


def solve_energy_distance(problem, config, warmup=None, deadline=None, reconstruction=None,
                          kappa=1., variant='energy_distance', gamma_override=None,
                          diagnostics=False):
    if variant not in ('energy_distance', 'energy_zero', 'energy_amp'):
        raise ValueError('unknown energy variant')
    mass = float(problem.a.sum())
    relative_tol = config.tol
    # Legacy light baselines stop in absolute L1; use an absolute equivalent
    # for an internally prepared general-mass warmup and its protocol hash.
    if not np.isclose(mass, 1., rtol=0, atol=1e-12):
        config = replace(config, tol=relative_tol*mass)
    if warmup is None:
        _, warmup = prepare_fast_warmup(problem, config, deadline)
    start = perf_counter()
    warmup.validate(problem, config)
    snap = warmup.clone()
    state, work, history = snap.state, snap.work, snap.history
    warm_history_length = len(history)
    stats = dict(kappa=kappa, gamma=0., gamma_psi_norm=None, gamma_u_norm=None,
        gamma_unresolvable_threshold=None, geometry_degenerate=False,
        distance_backend='not_built', distance_build_sec=0., distance_bytes=0,
        energy_matvec_count=0, energy_matvec_sec=0., gamma_setup_sec=0.,
        line_search_trials=0, rejected_trials=0, fallback_count=0,
        conditional_block_matvec_count=0, objective_diff_sec=0., trial_state_sec=0.,
        warmup_sec=snap.elapsed_sec, warmup_converged=False, gamma_initialized=False,
        direction_diagnostics=[], accepted_energy_trace=[])
    fallback_reasons = Counter()
    evaluator = LogEvaluator(problem, work, snap.log_kernel)
    D = None
    reason = 'max_iter'
    setup_end = start
    setup_completed = False
    initial_terms = None
    gamma = 0.
    try:
        if state.metrics['residual_l1']/mass <= relative_tol:
            reason = 'warmup_converged'
            stats['warmup_converged'] = True
        else:
            check_deadline(deadline)
            if variant != 'energy_zero':
                t = perf_counter()
                D, backend, fallback = build_distance(problem, reconstruction)
                stats.update(distance_build_sec=perf_counter()-t, distance_backend=backend,
                             distance_bytes=D.nbytes, distance_fallback_reason=fallback)
                check_deadline(deadline)
                t = perf_counter()
                gamma, initial_u, initial_energy = frozen_gamma(D, state.c-problem.b,
                    state.log_c-np.log(problem.b), kappa, stats)
                initial_terms = (initial_u, initial_energy)
                if gamma_override is not None:
                    if not np.isclose(gamma, gamma_override, rtol=1e-12, atol=0):
                        raise ValueError('amplitude gamma differs from full method at shared warmup')
                    gamma = gamma_override
                stats['gamma_setup_sec'] = perf_counter()-t
                stats['gamma_initialized'] = True
            stats['gamma'] = gamma
        setup_end = perf_counter()
        setup_completed = True
        for k in range(snap.iteration+1, config.max_iter+1):
            if state.metrics['residual_l1']/mass <= relative_tol:
                break
            check_deadline(deadline)
            psi = center(state.log_c - np.log(problem.b))
            e = state.c - problem.b
            if initial_terms is not None:
                u, energy = initial_terms
                initial_terms = None
            else:
                u, energy = (np.zeros_like(e), 0.) if D is None else energy_terms(D, e, stats)
            v = psi + gamma*u
            fallback_reason = None
            if variant == 'energy_amp':
                denom = float(e @ psi)
                scale = 100*np.finfo(float).eps*np.linalg.norm(e)*np.linalg.norm(psi)
                if denom <= scale or not np.isfinite(denom):
                    fallback_reason = 'amplitude_denominator_unresolvable'
                else:
                    v = (float(e @ v)/denom)*psi
            dot = float(e @ v) if np.isfinite(v).all() else float('nan')
            if not np.isfinite(v).all() or not np.isfinite(dot):
                fallback_reason = 'nonfinite_direction'
            elif dot <= 0:
                fallback_reason = 'non_descent'
            if diagnostics:
                pn, vn = np.linalg.norm(psi), np.linalg.norm(v)
                stats['direction_diagnostics'].append(dict(iteration=k-1, energy=energy,
                    cos=float(psi@v/(pn*vn)) if pn*vn > 0 and np.isfinite(vn) else None,
                    q=float(gamma*np.linalg.norm(u)/pn) if pn > 0 and np.isfinite(gamma) else None))
            new = None
            eta = 1.
            if fallback_reason is None:
                for trial in range(config.max_backtracks):
                    check_deadline(deadline)
                    stats['line_search_trials'] += 1
                    t = perf_counter()
                    try:
                        proposal = center(state.z_col - eta*v)
                        delta = stable_light_difference(problem, state, proposal-state.z_col, evaluator.log_kernel,
                            work, stats, deadline)
                    except FloatingPointError:
                        delta = float('inf')
                    finally:
                        stats['objective_diff_sec'] += perf_counter()-t
                    if delta <= -config.armijo*eta*dot:
                        check_deadline(deadline)
                        t = perf_counter()
                        try:
                            new = evaluator(proposal)
                        except FloatingPointError:
                            new = None
                        finally:
                            stats['trial_state_sec'] += perf_counter()-t
                        if new is not None:
                            break
                    stats['rejected_trials'] += 1
                    eta *= config.backtrack_factor
                if new is None:
                    fallback_reason = 'armijo_exhausted'
            if fallback_reason is not None:
                check_deadline(deadline)
                stats['fallback_count'] += 1
                fallback_reasons[fallback_reason] += 1
                eta = 1.
                new = evaluator(state.z_col - psi)
            state = new
            row = light_record(k, state, snap.elapsed_sec+perf_counter()-start, work,
                               phase=variant, step=eta, post=perf_counter()-start)
            row.accept_reason = 'fallback:'+fallback_reason if fallback_reason else 'armijo'
            row.trial_count = stats['line_search_trials']
            row.backtrack_count = stats['rejected_trials']
            row.safeguard_count = stats['fallback_count']
            row.p_matvec_count = stats['conditional_block_matvec_count']
            # Keep energy work separate from K/P work in the supplemental trace.
            history.append(row)
            stats['accepted_energy_trace'].append(dict(iteration=k,
                energy_matvec_count=stats['energy_matvec_count'], energy_matvec_sec=stats['energy_matvec_sec'],
                conditional_block_matvec_count=stats['conditional_block_matvec_count']))
    except DeadlineExpired:
        reason = 'timeout'
    except (FloatingPointError, OverflowError) as exc:
        reason = 'numerical_failure: '+str(exc)
    if not setup_completed:
        setup_end = perf_counter()
    # Relative stopping for general equal positive mass, while baseline files
    # retain their established (unit-mass) metric conventions.
    result = finish_light(problem, config, variant, state, history, work, start, reason,
                          snap.elapsed_sec, snap.setup_sec, setup_end-start, snap.fingerprint)
    result.converged = state.metrics['residual_l1']/mass <= relative_tol
    if result.converged:
        result.stop_reason = 'warmup_converged' if stats['warmup_converged'] else 'tolerance'
    stats.update(fallback_reasons=dict(fallback_reasons),
        accepted_updates=len(history)-warm_history_length,
        iteration_sec=max(0., result.elapsed_sec-snap.elapsed_sec-(setup_end-start)),
        bridge_setup_sec=max(0., setup_end-start-stats['distance_build_sec']-stats['gamma_setup_sec']),
        kernel_matrix_vector_count=work.matrix_vector_count,
        total_sec=result.elapsed_sec, last_residual=state.metrics['residual_l1']/mass)
    result.summary.update(stats)
    return result
