"""SciPy L-BFGS-B on the same row-eliminated dual, last column coordinate fixed."""
from __future__ import annotations

from copy import deepcopy
from time import perf_counter
import numpy as np
from scipy.optimize import minimize
from ..core.semi_dual import ReducedWork, evaluate_reduced
from ..core.stability import finite_scalings
from ..types import SolverResult
from .sinkhorn_fast import light_record
from .base import deadline_reached


class _Stop(Exception):
    pass


def reduced_value_gradient(problem,x,work=None,log_kernel=None):
    state=evaluate_reduced(problem,np.r_[x,0.],work,log_kernel=log_kernel)
    if work is not None:work.objective_gradient_count+=1
    return state.f,(state.c-problem.b)[:-1],state


def solve_lbfgs_reduced(problem,config,warmup=None,deadline=None):
    start=perf_counter();work=ReducedWork();warm_seconds=0.;history=[];k=0
    if warmup is not None:
        if warmup.state.metrics['residual_l1']<=config.tol:
            return warmup.finished_result(problem,config,'lbfgs_reduced')
        warmup.validate(problem,config);snap=warmup.clone()
        work,history,k=snap.work,snap.history,snap.iteration
        warm_seconds=snap.elapsed_sec;z=snap.state.z_col;log_kernel=snap.log_kernel
        if log_kernel is None:
            log_kernel=-problem.M/problem.epsilon;work.log_kernel_build_count+=1
    else:
        z=np.zeros_like(problem.b) if config.initial_nu is None else -np.asarray(config.initial_nu)/problem.epsilon
        log_kernel=-problem.M/problem.epsilon;work.log_kernel_build_count+=1
    x0=z[:-1]-z[-1];setup=perf_counter()-start
    cache_x=None;cache=None;latest=None;reason='max_iter';library={}

    def expired():
        now=perf_counter()
        return deadline_reached(deadline,now) or (config.wall_time_limit_sec is not None and now-start+warm_seconds>=config.wall_time_limit_sec)

    def fun(x):
        nonlocal cache_x,cache,reason
        if expired():reason='timeout';raise _Stop
        if cache_x is None or not np.array_equal(cache_x,x):
            cache=reduced_value_gradient(problem,x,work,log_kernel)
            cache_x=x.copy()
        return cache[0],cache[1]

    def record(state):
        row=light_record(k,state,warm_seconds+perf_counter()-start,work,phase='lbfgs_reduced',
                         post=perf_counter()-start)
        row.reduced_f=state.f;row.dual_objective=state.full_objective
        row.row_residual_l1=float(np.linalg.norm(state.r-problem.a,1))
        row.col_residual_l1=float(np.linalg.norm(state.c-problem.b,1))
        row.accept_reason='scipy_accepted'
        history.append(row)

    def callback(x):
        nonlocal latest,k,reason
        fun(x);latest=cache[2];k+=1;record(latest)
        if latest.metrics['residual_l1']<=config.tol:reason='tolerance';raise _Stop

    try:
        # This necessary bridge evaluation is timed and counted.
        value,gradient,latest=reduced_value_gradient(problem,x0,work,log_kernel)
        cache_x=x0.copy();cache=(value,gradient,latest)
        setup=perf_counter()-start
        if not history:record(latest)
        if latest.metrics['residual_l1']>config.tol and k<config.max_iter:
            opt=minimize(fun,x0,method='L-BFGS-B',jac=True,callback=callback,
                options=dict(maxcor=20,maxls=50,ftol=0.,gtol=0.,maxiter=config.max_iter-k,maxfun=5000000))
            library=dict(success=bool(opt.success),message=str(opt.message),status=int(opt.status),nit=int(opt.nit),nfev=int(opt.nfev))
            reason='library_stop: '+str(opt.message)
    except _Stop:
        pass
    except (FloatingPointError,np.linalg.LinAlgError) as exc:
        reason='numerical_failure: '+str(exc)
    if latest is None:
        # Invalid initial setup has no accepted valid endpoint for the dispatcher.
        raise FloatingPointError(reason)
    mu,nu=latest.mu,latest.nu;u,v=finite_scalings(mu,nu,problem.epsilon)
    success=latest.metrics['residual_l1']<=config.tol
    elapsed=warm_seconds+perf_counter()-start
    summary=dict(latest.metrics,**work.to_dict(),reduced_f=latest.f,setup_sec=setup,
        warmup_elapsed_sec=warm_seconds,post_elapsed_sec=elapsed-warm_seconds,
        warmup_sha256=warmup.fingerprint if warmup else None,library_result=library,
        lbfgs_options=dict(maxcor=20,maxls=50,ftol=0.,gtol=0.,maxfun=5000000),
        initial_residual_l1=history[0].residual_l1,bridge_state_evaluations=int(warmup is not None),
        dense_pass_proxy_comparable=False)
    return SolverResult('lbfgs_reduced',bool(success),'tolerance' if success else reason,k,elapsed,
                         mu.copy(),nu.copy(),u,v,history,summary)
