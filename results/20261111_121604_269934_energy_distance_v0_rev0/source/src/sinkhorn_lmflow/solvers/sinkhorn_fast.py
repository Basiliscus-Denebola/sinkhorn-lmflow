"""Lightweight Sinkhorn baselines with row-normalized observation endpoints.

Log mode eliminates row masses analytically, without materializing P. Floating
point row defects are checked independently from exported potentials after timing.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from time import perf_counter
import hashlib
import numpy as np
from scipy.special import logsumexp
from ..core.semi_dual import center, ReducedWork
from ..core.residuals import residual_norms
from ..core.gauge import normalize_gauge
from ..core.stability import finite_scalings
from ..io.problem_io import problem_fingerprint
from ..problems.weak_coupling import initial_fingerprint
from ..types import IterationRecord, SolverResult
from .base import deadline_reached


@dataclass
class LightState:
    z_col: np.ndarray
    ell: np.ndarray
    c: np.ndarray
    log_c: np.ndarray
    metrics: dict


class LogEvaluator:
    def __init__(self, problem, work, log_kernel=None):
        self.problem,self.work=problem,work
        if log_kernel is None:
            self.log_kernel=-problem.M/problem.epsilon
            self.log_kernel.flags.writeable=False
            work.log_kernel_build_count+=1
        else:
            self.log_kernel=log_kernel
        self.log_a=np.log(problem.a)

    def __call__(self,z):
        p,w=self.problem,self.work
        w.state_eval_count+=1
        z=center(z)
        if z.shape!=p.b.shape or not np.isfinite(z).all():raise FloatingPointError('invalid columns')
        logits=self.log_kernel+z[None,:]
        ell=logsumexp(logits,axis=1);w.logsumexp_count+=1
        logits-=ell[:,None]
        logits+=self.log_a[:,None]
        log_c=logsumexp(logits,axis=0);w.logsumexp_count+=1
        c=np.exp(log_c)
        if not np.isfinite(log_c).all() or not np.isfinite(c).all():raise FloatingPointError('nonfinite light state')
        return LightState(z,ell,c,log_c,residual_norms(np.zeros_like(p.a),c-p.b))


def export_potentials(problem,state):
    return normalize_gauge(problem.epsilon*(state.ell-np.log(problem.a)),-problem.epsilon*state.z_col)


def light_record(k,state,elapsed,work,phase='sinkhorn',step=1.,post=0.):
    row=IterationRecord(k,elapsed,**state.metrics,dual_objective=None,theta=None,
        step_size=step if k else None,sigma_nontrivial=None,spectral_gap=None,
        linear_solve_iters=None,safeguard_count=0,phase=phase,
        row_residual_l1=0.,col_residual_l1=state.metrics['residual_l1'],post_elapsed_sec=post,
        from_iteration=k-1 if k else None,accept_reason='sinkhorn' if k else '')
    for key in ('state_eval_count','logsumexp_count','plan_build_count','matrix_vector_count',
                'objective_diff_count','bregman_count','objective_gradient_count','log_kernel_build_count','kernel_build_count'):
        setattr(row,key,getattr(work,key))
    row.state_fingerprint=hashlib.sha256(state.z_col.tobytes()+state.c.tobytes()).hexdigest()
    return row


@dataclass
class FastSnapshot:
    state: LightState
    previous: tuple | None
    history: list
    work: ReducedWork
    elapsed_sec: float
    setup_sec: float
    iteration: int
    problem_sha256: str
    initial_sha256: str
    protocol: str
    log_kernel: np.ndarray
    problem_identity: int
    fingerprint: str = ''
    lightweight: bool = True
    diagnostic_reference: object = None

    def digest(self):
        h=hashlib.sha256((self.problem_sha256+self.initial_sha256+self.protocol+str(self.iteration)).encode())
        for x in (self.state.z_col,self.state.c,self.state.log_c,self.state.ell):h.update(x.tobytes())
        if self.previous is not None:
            for x in self.previous:h.update(x.tobytes())
        return h.hexdigest()

    def clone(self):
        # Read-only log kernel is shared; all mutable vector/history state is copied.
        return replace(self,state=deepcopy(self.state),previous=deepcopy(self.previous),
                       history=deepcopy(self.history),work=deepcopy(self.work))

    def validate(self,problem,config):
        from .mirror_sr1 import warmup_protocol
        supplied=np.zeros_like(problem.b) if config.initial_nu is None else -np.asarray(config.initial_nu)/problem.epsilon
        if (self.problem_identity!=id(problem) and self.problem_sha256!=problem_fingerprint(problem)) or self.fingerprint!=self.digest():
            raise ValueError('invalid fast warmup problem/state fingerprint')
        if self.initial_sha256!=initial_fingerprint(supplied) or self.protocol!=warmup_protocol(config):
            raise ValueError('invalid fast warmup initial/protocol fingerprint')

    def finished_result(self,problem,config,name):
        start=perf_counter();self.validate(problem,config);snap=self.clone()
        return finish_light(problem,config,name,snap.state,snap.history,snap.work,start,
                            'warmup_converged',self.elapsed_sec,self.setup_sec,0.,self.fingerprint)


def finish_light(problem,config,name,state,history,work,start,reason,warm_seconds=0.,setup_sec=0.,post_setup=0.,warm_hash=None):
    mu,nu=export_potentials(problem,state);u,v=finite_scalings(mu,nu,problem.epsilon)
    success=state.metrics['residual_l1']<=config.tol
    elapsed=warm_seconds+perf_counter()-start
    summary=dict(state.metrics,**work.to_dict(),setup_sec=setup_sec,post_setup_sec=post_setup,
        warmup_elapsed_sec=warm_seconds,post_elapsed_sec=elapsed-warm_seconds,
        warmup_sha256=warm_hash,initial_residual_l1=history[0].residual_l1 if history else None,
        row_residual_policy='analytically eliminated; independent floating-point endpoint audit',
        dense_pass_proxy_comparable=False)
    return SolverResult(name,bool(success),'warmup_converged' if success and reason=='warmup_converged' else 'tolerance' if success else reason,
                        history[-1].iteration if history else 0,elapsed,mu,nu,u,v,history,summary)


def solve_sk_log_fast(problem,config,warmup=None,deadline=None,return_snapshot=False):
    from .mirror_sr1 import base_diagonal,warmup_protocol
    if warmup is not None and warmup.state.metrics['residual_l1']<=config.tol and not return_snapshot:
        return warmup.finished_result(problem,config,'sk_log_fast')
    start=perf_counter();warm_seconds=0.;work=ReducedWork();previous=None;history=[]
    supplied=np.zeros_like(problem.b) if config.initial_nu is None else -np.asarray(config.initial_nu)/problem.epsilon
    if warmup is not None:
        warmup.validate(problem,config);snap=warmup.clone()
        state,work,history,previous=snap.state,snap.work,snap.history,snap.previous
        warm_seconds=snap.elapsed_sec;evaluator=LogEvaluator(problem,work,snap.log_kernel)
        setup=perf_counter()-start;first_k=snap.iteration
    else:
        evaluator=LogEvaluator(problem,work);setup=perf_counter()-start
        state=evaluator(supplied);first_k=0
        setup=perf_counter()-start
        history.append(light_record(0,state,perf_counter()-start,work,phase='initial',step=None))
    reason='max_iter'
    for k in range(first_k+1,config.max_iter+1):
        if state.metrics['residual_l1']<=config.tol:break
        now=perf_counter()
        if deadline_reached(deadline,now) or (config.wall_time_limit_sec is not None and now-start+warm_seconds>=config.wall_time_limit_sec):
            reason='timeout';break
        try:
            _,psi=base_diagonal(state.c,problem.b,state.log_c)
            new=evaluator(state.z_col-center(psi))
        except FloatingPointError as exc:
            reason='numerical_failure: '+str(exc);break
        previous=(state.z_col.copy(),state.c.copy());state=new
        history.append(light_record(k,state,warm_seconds+perf_counter()-start,work,
            phase='warmup' if k<=config.sr1_warmup_iters else 'sinkhorn',
            post=perf_counter()-start if warmup is not None else 0.))
    result=finish_light(problem,config,'sk_log_fast',state,history,work,start,reason,warm_seconds,setup,
                        warm_hash=warmup.fingerprint if warmup else None)
    snapshot=None
    if return_snapshot:
        snapshot=FastSnapshot(state,previous,deepcopy(history),deepcopy(work),result.elapsed_sec,setup,result.n_iter,
            problem_fingerprint(problem),initial_fingerprint(supplied),warmup_protocol(config),evaluator.log_kernel,id(problem))
        snapshot.fingerprint=snapshot.digest()
        snapshot.elapsed_sec=perf_counter()-start
        result.elapsed_sec=snapshot.elapsed_sec
    return (result,snapshot) if return_snapshot else result


def prepare_fast_warmup(problem,config,deadline=None):
    return solve_sk_log_fast(problem,replace(config,max_iter=min(config.max_iter,config.sr1_warmup_iters)),
                             deadline=deadline,return_snapshot=True)


def solve_sk_scaling_fast(problem,config,deadline=None):
    start=perf_counter();work=ReducedWork();work.kernel_build_count=1
    with np.errstate(under='ignore',over='ignore'):
        K=np.exp(-problem.M/problem.epsilon)
    setup=perf_counter()-start
    if not np.isfinite(K).all() or np.any(K<=0):
        # Not a solver failure; no altered zero-containing kernel is iterated.
        state=LightState(np.zeros_like(problem.b),np.zeros_like(problem.a),problem.b.copy(),np.log(problem.b),
                         dict(residual_l1=float('inf'),residual_linf=float('inf'),grad_norm_l2=float('inf')))
        r=finish_light(problem,config,'sk_scaling_fast',state,[],work,start,'not_applicable_kernel_underflow',setup_sec=setup)
        r.summary['endpoint_valid']=False
        return r
    z=center(np.zeros_like(problem.b) if config.initial_nu is None else -np.asarray(config.initial_nu)/problem.epsilon)
    history=[];reason='max_iter';state=None
    try:
        with np.errstate(over='raise',divide='raise',invalid='raise',under='ignore'):
            v=np.exp(z)
            for k in range(config.max_iter+1):
                if k:
                    now=perf_counter()
                    if deadline_reached(deadline,now) or (config.wall_time_limit_sec is not None and now-start>=config.wall_time_limit_sec):
                        reason='timeout';break
                    v*=problem.b/state.c
                    logs=np.log(v);shift=logs.mean();v*=np.exp(-shift)
                kv=K@v;work.matrix_vector_count+=1
                u=problem.a/kv
                cols=v*(K.T@u);work.matrix_vector_count+=1
                if not np.isfinite(np.r_[u,v,cols]).all() or min(u.min(),v.min(),cols.min())<=0:
                    raise FloatingPointError('nonpositive/nonfinite scaling iterate')
                work.state_eval_count+=1
                z=np.log(v);z=center(z)
                # v already has geometric mean one; ell has matching gauge.
                state=LightState(z,np.log(kv)-np.log(v).mean(),cols,np.log(cols),
                    residual_norms(u*kv-problem.a,cols-problem.b))
                history.append(light_record(k,state,perf_counter()-start,work,
                    phase='scaling',post=perf_counter()-start))
                history[-1].row_residual_l1=float(np.linalg.norm(u*kv-problem.a,1))
                if state.metrics['residual_l1']<=config.tol:break
    except FloatingPointError as exc:
        reason='numerical_failure: '+str(exc)
    if state is None:
        state=LightState(z,np.zeros_like(problem.a),problem.b.copy(),np.log(problem.b),
                         dict(residual_l1=float('inf'),residual_linf=float('inf'),grad_norm_l2=float('inf')))
    result=finish_light(problem,config,'sk_scaling_fast',state,history,work,start,reason,setup_sec=setup)
    result.summary['endpoint_valid']=bool(history)
    return result
