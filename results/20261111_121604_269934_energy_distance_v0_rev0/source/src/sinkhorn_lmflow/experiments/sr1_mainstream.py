"""Frozen v1 plan, serial budgeted benchmarks, durable attempts and independent audit."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import threading
from time import perf_counter, time, sleep
import numpy as np
import psutil
from scipy.special import logsumexp
from ..types import SolverConfig
from ..problems.mainstream import main_cases,make_mainstream,array_hash
from ..problems.mnist import ensure_mnist,choose_pairs,image_problem
from ..problems.weak_coupling import weak_coupling_problem,initial_columns,initial_fingerprint
from ..core.semi_dual import ReducedWork
from ..core.residuals import residual_norms
from ..solvers.sinkhorn_fast import prepare_fast_warmup,solve_sk_log_fast,FastSnapshot,LightState
from ..solvers.sinkhorn_fast import solve_sk_scaling_fast
from ..solvers.mirror_sr1 import solve_mirror_sr1
from ..solvers.lbfgs_reduced import solve_lbfgs_reduced
from ..io.config import save_json,save_config,load_config,json_safe
from ..io.result_io import save_result,load_result
from ..io.problem_io import problem_fingerprint
from ..io.paths import create_run_dir
from .reproducibility import runtime_metadata

ROOT=Path(__file__).resolve().parents[3]
TERMINAL={'success','timeout','max_iter','numerical_failure','library_stop','skipped_resource',
          'not_applicable_kernel_underflow'}


def implementation_fingerprint():
    files=['core/semi_dual.py','core/residuals.py','core/gauge.py','core/stability.py',
           'solvers/mirror_sr1.py','solvers/sinkhorn_fast.py','solvers/lbfgs_reduced.py','solvers/base.py',
           'problems/mainstream.py','problems/mnist.py','problems/weak_coupling.py','problems/instance.py','types.py']
    hashes={f:hashlib.sha256((ROOT/'src/sinkhorn_lmflow'/f).read_bytes()).hexdigest() for f in files}
    return hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),hashes


def config_fingerprint(config):
    effective={k:v for k,v in config.items() if k not in ('selection',)}
    return hashlib.sha256(json.dumps(json_safe(effective),sort_keys=True).encode()).hexdigest()


def build_plan(config):
    repeats=int(config.get('repeats',3));cases=config.get('cases') or main_cases()
    cases=deepcopy(cases);jobs=[]
    def add(case,method,repeat,warm=20,group='main',diagnostic=False):
        jobs.append(dict(job_id=f'{case["case_id"]}__{method}__w{warm}__r{repeat}'+('__diag' if diagnostic else ''),
            case_id=case['case_id'],method=method,repeat=repeat,warmup_iters=warm,group=group,
            mode='diagnostic' if diagnostic else 'performance',status='pending',attempts=[]))
    for repeat in range(repeats):
        for i,case in enumerate(cases):
            methods=['sk_log_fast','mirror_sr1']
            if (i+repeat)%2:methods.reverse()
            for method in methods:add(case,method,repeat,config.get('warmup_iters',20))
    for repeat in range(repeats):
        for case in sorted(cases,key=lambda c:-max(c['n'],c['m'])):
            add(case,'sk_scaling_fast',repeat,group='scaling')
    selected=[c for c in cases if (c['family'] in ('M1','M2','M3') and c['data_seed']==42 and c['epsilon'] in (.01,.001))
              or (c['family']=='M5' and c['pair_name'] in ('same_0','cross_0_1'))]
    if config.get('smoke'):selected=cases
    for repeat in range(repeats):
        for case in selected:add(case,'lbfgs_reduced',repeat,group='lbfgs')
    if not config.get('smoke') and not config.get('cases'):
        toys=[dict(case_id=f'T{b}_N256_d0.001_s42',family='toy',n=256,m=256,blocks=b,delta=.001,
                   initial_seed=42,data_seed=None,epsilon=1.,alpha=None) for b in (2,4)]
        cases+=toys
        for repeat in range(repeats):
            for case in toys:
                methods=['sk_log_fast','mirror_sr1','mirror_amp_ablation']
                methods=methods[repeat%3:]+methods[:repeat%3]
                for method in methods:add(case,method,repeat,group='amplitude')
        mixture=next(c for c in cases if c['case_id']=='M3_N1024_s42_e0.003')
        mnist=next(c for c in cases if c['case_id']=='M5_cross_0_1_e0.001')
        for repeat in range(repeats):
            add(mixture,'mirror_amp_ablation',repeat,group='amplitude')
            for case in (mixture,mnist):
                for warm in (0,5,50):
                    for method in (['sk_log_fast','mirror_sr1'] if repeat%2==0 else ['mirror_sr1','sk_log_fast']):
                        add(case,method,repeat,warm,group='warmup')
        diagnostic=[c for c in cases if
            (c['family'] in ('M1','M2','M3') and c['data_seed']==42 and c['epsilon']==.003) or
            (c['family']=='M4' and c['data_seed']==42 and c['alpha']==.1) or
            (c['family']=='M5' and c['pair_name'] in ('same_0','cross_0_1') and c['epsilon']==.001) or
            (c['family']=='toy' and c['blocks']==4)]
        for case in diagnostic:add(case,'mirror_sr1',0,group='diagnostic',diagnostic=True)
    return cases,jobs


def atomic_json(data,path):
    path=Path(path);temp=path.with_name(path.name+'.next')
    # Windows readers/indexers can briefly deny replacement of an open file.
    # Keep the previous checkpoint intact, retry boundedly, and surface lasting
    # permission failures instead of deleting the destination as a workaround.
    for attempt in range(100):
        try:
            save_json(data,temp);temp.replace(path);return
        except PermissionError:
            if attempt==99:raise
            sleep(.05)


class Budget:
    def __init__(self,path,performance_sec,diagnostic_sec):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        if self.path.exists():self.data=json.loads(self.path.read_text(encoding='utf-8'))
        else:self.data=dict(performance_budget_sec=float(performance_sec),diagnostic_budget_sec=float(diagnostic_sec),
                            performance_used_sec=0.,diagnostic_used_sec=0.,active_run_dir=None)
        if (self.data['performance_budget_sec']!=float(performance_sec) or
            self.data['diagnostic_budget_sec']!=float(diagnostic_sec)):
            raise ValueError('budget ledger limits differ; no implicit budget reset is permitted')
        self.interrupted=False;self.save()

    def save(self):atomic_json(self.data,self.path)
    def remaining(self,mode):return max(0.,self.data[mode+'_budget_sec']-self.data[mode+'_used_sec'])
    def charge(self,seconds,mode):
        self.data[mode+'_used_sec']+=max(0.,seconds);self.save()


@contextmanager
def ledger_lock(path):
    lock=Path(str(path)+'.lock');lock.parent.mkdir(parents=True,exist_ok=True)
    if lock.exists():
        owner=int(lock.read_text())
        if psutil.pid_exists(owner):raise RuntimeError('another serial benchmark owns the budget ledger')
        lock.unlink()
    fd=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:yield
    finally:lock.unlink(missing_ok=True)


class MemoryWatch:
    """Sample process RSS; a sample peak is a lower bound on instantaneous peak."""
    def __init__(self,limit):
        self.limit=limit;self.peak=psutil.Process().memory_info().rss
        self.exceeded=False;self.done=threading.Event();self.samples=0
    def __enter__(self):
        def monitor():
            process=psutil.Process()
            while not self.done.is_set():
                rss=process.memory_info().rss;self.peak=max(self.peak,rss);self.samples+=1
                if rss>self.limit:self.exceeded=True
                self.done.wait(.05)
        self.thread=threading.Thread(target=monitor,daemon=True);self.thread.start();return self
    def __exit__(self,*args):
        self.peak=max(self.peak,psutil.Process().memory_info().rss)
        self.done.set();self.thread.join()


def estimated_peak(problem,method):
    # Include the resident public problem and a conservative method workspace.
    matrices=14 if method in ('mirror_sr1','mirror_amp_ablation') else 9 if method=='lbfgs_reduced' else 5 if method=='sk_log_fast' else 2
    return psutil.Process().memory_info().rss+matrices*problem.M.nbytes


def make_input(case,dataset):
    if case['family']=='M5':
        if dataset is None:raise FileNotFoundError('blocked_data: MNIST test data unavailable')
        images,labels,provenance,pairs=dataset
        pair=next(p for p in pairs['pairs'] if p['name']==case['pair_name'])
        problem,arrays=image_problem(images,pair,case['epsilon'],case['case_id'],provenance)
        z=np.zeros_like(problem.b)
    elif case['family']=='toy':
        problem=weak_coupling_problem(case['n'],case['blocks'],case['delta'])
        problem.metadata.update(case_id=case['case_id'],family='toy',n=case['n'],m=case['m'],epsilon=1.,
            data_seed=None,weight_seed_a=None,weight_seed_b=None,marginal_type='uniform',alpha=None,
            raw_cost_max=float(problem.M.max()),normalization_policy='none')
        z=initial_columns(case['n'],case['blocks'],case['initial_seed']);arrays=dict(a=problem.a,b=problem.b)
    else:
        problem,arrays=make_mainstream(case);z=np.zeros_like(problem.b)
    arrays['initial_z']=z
    return problem,arrays,z


def audit_endpoint(problem,result,block_rows=128):
    start=perf_counter()
    if result.summary.get('endpoint_valid',True) is False:
        return dict(passed=None,reason='no applicable endpoint',elapsed_sec=perf_counter()-start)
    c=np.zeros_like(problem.b);row_error=0.
    for lo in range(0,len(problem.a),block_rows):
        hi=min(len(problem.a),lo+block_rows)
        with np.errstate(over='raise',invalid='raise'):
            P=np.exp(-(problem.M[lo:hi]+result.mu[lo:hi,None]+result.nu[None,:])/problem.epsilon)
        row_error+=float(np.abs(P.sum(1)-problem.a[lo:hi]).sum());c+=P.sum(0)
    rho=max(row_error,float(np.abs(c-problem.b).sum()))
    claimed=result.summary.get('residual_l1')
    close=claimed is not None and np.isfinite(rho) and abs(rho-claimed)<=1e-10*max(1.,rho)
    passed=bool(close and (not result.converged or rho<=1e-8))
    return dict(passed=passed,residual_l1=rho,row_residual_l1=row_error,claimed_residual_l1=claimed,
                elapsed_sec=perf_counter()-start,method='independent full-potential row blocks',block_rows=block_rows)


def save_warmup(root,key,result,snapshot,config):
    path=root/'warmups'/key
    save_result(result,path,dict(asdict(config),max_iter=min(config.max_iter,config.sr1_warmup_iters)))
    np.savez_compressed(path/'snapshot.npz',z=snapshot.state.z_col,ell=snapshot.state.ell,c=snapshot.state.c,
        log_c=snapshot.state.log_c,previous_z=np.array([]) if snapshot.previous is None else snapshot.previous[0],
        previous_c=np.array([]) if snapshot.previous is None else snapshot.previous[1])
    save_json(dict(work=snapshot.work.to_dict(),elapsed_sec=snapshot.elapsed_sec,setup_sec=snapshot.setup_sec,
        iteration=snapshot.iteration,problem_sha256=snapshot.problem_sha256,initial_sha256=snapshot.initial_sha256,
        protocol=snapshot.protocol,fingerprint=snapshot.fingerprint,metrics=snapshot.state.metrics),path/'snapshot.json')


def load_warmup(root,key,problem):
    path=root/'warmups'/key;meta=json.loads((path/'snapshot.json').read_text(encoding='utf-8'))
    result=load_result(path)
    with np.load(path/'snapshot.npz',allow_pickle=False) as d:
        state=LightState(d['z'],d['ell'],d['c'],d['log_c'],meta['metrics'])
        previous=(d['previous_z'],d['previous_c']) if d['previous_z'].size else None
    snap=FastSnapshot(state,previous,result.history,ReducedWork(**meta['work']),meta['elapsed_sec'],meta['setup_sec'],
        meta['iteration'],meta['problem_sha256'],meta['initial_sha256'],meta['protocol'],None,id(problem),meta['fingerprint'])
    if snap.digest()!=snap.fingerprint or snap.problem_sha256!=problem_fingerprint(problem):raise ValueError('saved warmup fingerprint mismatch')
    return snap


def artifact_valid(root,job,implementation,problem_sha):
    if not job['attempts']:return False
    attempt=job['attempts'][-1]
    if attempt['status'] not in TERMINAL:return False
    if attempt.get('implementation_sha256')!=implementation or attempt.get('problem_sha256')!=problem_sha:return False
    if 'run_path' not in attempt:return attempt['status']=='skipped_resource' or (attempt['status']=='numerical_failure' and 'error' in attempt)
    try:
        result=load_result(root/attempt['run_path'])
        return (result.summary.get('implementation_sha256')==implementation and
                result.summary.get('problem_sha256')==problem_sha and
                result.summary.get('initial_sha256')==attempt.get('initial_sha256') and
                result.solver_name==job['method'] and
                (bool(result.history) or result.summary.get('endpoint_valid') is False) and
                result.summary.get('audit',{}).get('passed') is not False)
    except (OSError,ValueError,KeyError,TypeError):return False


def selected(job,case,selection):
    if selection=='all':return True
    if selection=='mnist':return case['family']=='M5'
    if selection=='scale':return case['family'] in ('M6','M7') or case['case_id']=='M3_N1024_s42_e0.003'
    if selection=='baselines':return job['group'] in ('scaling','lbfgs')
    if selection=='ablation':return job['group'] in ('amplitude','warmup','diagnostic')
    raise ValueError('unknown selection')


def apply_operator_policy(manifest,policy):
    """Record explicit user exclusions without erasing already observed results."""
    manifest['operator_policy']=deepcopy(policy)
    exclusions=policy.get('excluded_cases',{})
    for job in manifest['jobs']:
        if job['case_id'] not in exclusions:continue
        reason=exclusions[job['case_id']]['reason']
        job['comparison_excluded_reason']=reason
        if job['status'] in ('pending','running','interrupted','blocked_data','skipped_by_user'):
            job['status']='skipped_by_user';job['skip_reason']=reason


def run_mainstream(config,output_root='results',resume=None,new_revision=False,retry_failures=False,
                   stop_after_job=None,continue_paused=False):
    ledger_path=Path(config.get('budget_ledger','results/sr1_v1_budget.json'))
    with ledger_lock(ledger_path):
        return _run_locked(config,output_root,resume,new_revision,retry_failures,ledger_path,stop_after_job,continue_paused)


def _run_locked(config,output_root,resume,new_revision,retry_failures,ledger_path,stop_after_job=None,continue_paused=False):
    last_ledger_write=ledger_path.stat().st_mtime if ledger_path.exists() else None
    budget=Budget(ledger_path,config.get('performance_budget_sec',21600),config.get('diagnostic_budget_sec',3600))
    impl,sources=implementation_fingerprint();cfgsha=config_fingerprint(config)
    auto=budget.data.get('active_run_dir')
    resume=resume or (auto if auto and Path(auto).exists() else None)
    if resume and not new_revision:
        root=Path(resume);manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        if manifest['implementation_sha256']!=impl or manifest['config_sha256']!=cfgsha:
            raise ValueError('implementation/config mismatch: use --new-revision; budget ledger remains cumulative')
    else:
        root=create_run_dir(output_root,'sr1_v1_smoke' if config.get('smoke') else 'sr1_v1_mainstream')
        cases,jobs=build_plan(config)
        manifest=dict(schema=1,config_sha256=cfgsha,implementation_sha256=impl,source_hashes=sources,
            cases=cases,jobs=jobs,invocations=[],previous_revision=str(resume) if resume else None,
            budget_ledger=str(ledger_path.resolve()),scale_alias=dict(N1024='M3_N1024_s42_e0.003'))
        save_config(config,root/'study_config.yaml')
    pause_file=root/'pause.json'
    if pause_file.exists() and json.loads(pause_file.read_text(encoding='utf-8')).get('paused') and not continue_paused:
        print('Paused by user request. Resume explicitly with --continue-paused.',flush=True)
        return root
    stop_index=None
    if stop_after_job is not None:
        stop_index=next((i for i,j in enumerate(manifest['jobs']) if j['job_id']==stop_after_job),None)
        if stop_index is None:raise ValueError('stop-after-job is not in the frozen plan')
    if continue_paused and pause_file.exists():
        atomic_json(dict(paused=False,resumed_explicitly=True),pause_file)
    if continue_paused:
        (root/'pause_request.json').unlink(missing_ok=True)
    # A killed process cannot flush its finally block. Preserve the old attempt
    # and charge a conservative wall-time bound since the last durable ledger
    # update, capped at the full per-track allowance plus one-kernel tolerance.
    stale=[j for j in manifest['jobs'] if j['status']=='running']
    for job in stale:
        case=next(c for c in manifest['cases'] if c['case_id']==job['case_id'])
        limit=config.get('track_limit_large_sec',1200) if max(case.get('actual_n',case['n']),case.get('actual_m',case['m']))>2048 else config.get('track_limit_sec',300)
        charge=min(float(limit)+1.,max(0.,time()-last_ledger_write)) if last_ledger_write else float(limit)+1.
        budget.charge(charge,job['mode'])
        job['status']='interrupted'
        job['attempts'][-1].update(status='interrupted',endpoint_saved=False,
            recovered_machine_charge_sec=charge,
            recovery_charge_policy='conservative wall-time bound since last ledger write, capped at track limit + 1s',
            reason='process ended before durable endpoint checkpoint; retry keeps the old attempt')
    if stale:
        for previous in manifest['invocations']:
            if previous['status']=='running':previous['status']='interrupted_recovered'
    policy_path=root/'operator_policy.json'
    if policy_path.exists():
        apply_operator_policy(manifest,json.loads(policy_path.read_text(encoding='utf-8')))
    budget.data['active_run_dir']=str(root.resolve());budget.save()
    environment=runtime_metadata(None);available=psutil.virtual_memory().available
    memory_limit=min(int(config.get('memory_cap_gib',8)*2**30),int(.6*available))
    environment.update(available_memory_at_start_bytes=available,memory_limit_bytes=memory_limit,
                       psutil_version=psutil.__version__,implementation_sha256=impl)
    save_json(environment,root/'environment.json')
    invocation=dict(status='running',selection=config.get('selection','all'),environment=environment,
                    excluded_data_sec=0.,excluded_io_sec=0.,excluded_audit_sec=0.)
    manifest['invocations'].append(invocation)
    def checkpoint():
        manifest['budget_snapshot']=deepcopy(budget.data)
        atomic_json(manifest,root/'manifest.json')
    checkpoint()
    print('Study: '+str(root),flush=True)
    dataset=None
    try:
        images,labels,provenance=ensure_mnist(config.get('mnist_cache','data/mnist'),download=config.get('download_mnist',True))
        pairs=choose_pairs(labels)
        if (root/'mnist_pairs.json').exists():
            if json.loads((root/'mnist_pairs.json').read_text(encoding='utf-8'))!=pairs:raise ValueError('MNIST pair list changed')
        else:save_json(pairs,root/'mnist_pairs.json')
        dataset=(images,labels,provenance,pairs)
        save_json(provenance,root/'mnist_provenance.json')
    except (OSError,ValueError) as exc:
        invocation['mnist_data_error']=str(exc)
    # Freeze every descriptor and actual input hash before any formal solver.
    for case in manifest['cases']:
        if case.get('problem_sha256'):continue
        started=perf_counter()
        try:
            p,arrays,z=make_input(case,dataset)
            directory=root/'problems'/case['case_id'];directory.mkdir(parents=True,exist_ok=True)
            psha=problem_fingerprint(p);isha=initial_fingerprint(z)
            case.update(problem_sha256=psha,initial_sha256=isha,actual_n=len(p.a),actual_m=len(p.b),
                        input_arrays_sha256=array_hash(*[arrays[k] for k in sorted(arrays)]),data_status='ready')
            save_json(dict(case=case,metadata=p.metadata,problem_sha256=psha,initial_sha256=isha),directory/'descriptor.json')
            np.savez_compressed(directory/'reconstruction.npz',**arrays)
            del p,arrays;gc.collect()
        except (OSError,ValueError,FloatingPointError,MemoryError) as exc:
            case.update(data_status='blocked_data' if case['family']=='M5' else 'data_generation_failure',data_error=str(exc))
        invocation['excluded_data_sec']+=perf_counter()-started
    checkpoint()
    # Import/warm-call overhead is not part of the performance ledger.
    probe,_=make_mainstream(dict(case_id='preflight',family='M2',n=12,m=13,data_seed=42,epsilon=.1))
    probe_cfg=SolverConfig(max_iter=3,sr1_warmup_iters=1,compute_spectrum_every=0,max_backtracks=5)
    _,probe_warm=prepare_fast_warmup(probe,probe_cfg)
    solve_sk_log_fast(probe,probe_cfg,probe_warm);solve_mirror_sr1(probe,probe_cfg,probe_warm,performance=True)
    solve_sk_scaling_fast(probe,probe_cfg);solve_lbfgs_reduced(probe,probe_cfg,probe_warm)
    del probe,probe_warm;gc.collect()
    old_handler=signal.signal(signal.SIGINT,lambda *_:setattr(budget,'interrupted',True))
    cached_case=None;problem=None;z=None;warm_key=None;snapshot=None;boundary_paused=False
    try:
        cases={c['case_id']:c for c in manifest['cases']}
        for index,job in enumerate(manifest['jobs']):
            request_path=root/'pause_request.json'
            if request_path.exists():
                request=json.loads(request_path.read_text(encoding='utf-8'))
                requested=request['after_job_id']
                requested_index=next((i for i,j in enumerate(manifest['jobs']) if j['job_id']==requested),None)
                if requested_index is None:raise ValueError('pause request is not in the frozen plan')
                stop_after_job,stop_index=requested,requested_index
            if stop_index is not None and index>stop_index:
                boundary_paused=True;break
            case=cases[job['case_id']]
            if not selected(job,case,config.get('selection','all')):continue
            if budget.interrupted:break
            if job.get('comparison_excluded_reason'):continue
            mode=job['mode']
            if budget.remaining(mode)<=0:continue
            if case.get('data_status')!='ready':
                job['status']=case.get('data_status','blocked_data');checkpoint();continue
            if artifact_valid(root,job,impl,case['problem_sha256']) and not (retry_failures and job['status']!='success'):
                continue
            if cached_case!=case['case_id']:
                problem=None;snapshot=None;gc.collect();started=perf_counter()
                problem,arrays,z=make_input(case,dataset)
                if problem_fingerprint(problem)!=case['problem_sha256'] or initial_fingerprint(z)!=case['initial_sha256']:
                    raise ValueError('regenerated input fingerprint mismatch')
                del arrays;cached_case=case['case_id'];warm_key=None
                invocation['excluded_data_sec']+=perf_counter()-started
            track_limit=config.get('track_limit_large_sec',1200) if max(len(problem.a),len(problem.b))>2048 else config.get('track_limit_sec',300)
            cfg=SolverConfig(tol=1e-8,max_iter=config.get('max_iter',50000),sr1_warmup_iters=job['warmup_iters'],
                sr1_secant_tol=1e-8,compute_spectrum_every=0,record_every=1,max_backtracks=5,
                initial_nu=-problem.epsilon*z,wall_time_limit_sec=track_limit)
            attempt=dict(attempt=len(job['attempts'])+1,status='running',implementation_sha256=impl,
                         problem_sha256=case['problem_sha256'],initial_sha256=case['initial_sha256'])
            job['attempts'].append(attempt);job['status']='running';checkpoint()
            estimate=estimated_peak(problem,job['method'])
            attempt.update(estimated_peak_rss_bytes=estimate,memory_limit_bytes=memory_limit)
            print(f"[{index+1}/{len(manifest['jobs'])}] {job['job_id']}",flush=True)
            if estimate>memory_limit:
                attempt['status']=job['status']='skipped_resource';attempt['reason']='conservative workspace estimate exceeds startup limit'
                checkpoint();print('  skipped_resource',flush=True);continue
            end=perf_counter()+budget.remaining(mode)
            with MemoryWatch(memory_limit) as memory:
                cancel=lambda:budget.interrupted or memory.exceeded or perf_counter()>=end
                if job['method']!='sk_scaling_fast':
                    key=f"{case['case_id']}_w{job['warmup_iters']}_r{job['repeat']}"
                    if key!=warm_key:
                        snapshot=None
                        if (root/'warmups'/key/'snapshot.json').exists():
                            started=perf_counter();snapshot=load_warmup(root,key,problem)
                            invocation['excluded_io_sec']+=perf_counter()-started
                        else:
                            started=perf_counter();wr,snapshot=prepare_fast_warmup(problem,cfg,deadline=cancel)
                            budget.charge(perf_counter()-started,mode)
                            started=perf_counter();save_warmup(root,key,wr,snapshot,cfg)
                            invocation['excluded_io_sec']+=perf_counter()-started
                        warm_key=key
                    attempt.update(warmup_sha256=snapshot.fingerprint,warmup_path=str((root/'warmups'/key).relative_to(root)))
                end=perf_counter()+budget.remaining(mode)
                started=perf_counter();result=None
                try:
                    if job['method']=='sk_log_fast':result=solve_sk_log_fast(problem,cfg,snapshot,cancel)
                    elif job['method']=='sk_scaling_fast':result=solve_sk_scaling_fast(problem,cfg,cancel)
                    elif job['method']=='lbfgs_reduced':result=solve_lbfgs_reduced(problem,cfg,snapshot,cancel)
                    else:result=solve_mirror_sr1(problem,cfg,snapshot,cancel,performance=True,
                        diagnostics=mode=='diagnostic',amplification=job['method']=='mirror_amp_ablation')
                except (FloatingPointError,ValueError,np.linalg.LinAlgError,MemoryError) as exc:
                    attempt.update(status='numerical_failure',error=f'{type(exc).__name__}: {exc}')
                finally:budget.charge(perf_counter()-started,mode)
            attempt.update(peak_sampled_rss_bytes=memory.peak,rss_sample_count=memory.samples,
                           rss_measurement='process RSS sampled every 50ms; lower bound on instantaneous peak')
            if result is not None:
                if budget.interrupted:result.stop_reason='interrupted'
                elif memory.exceeded:result.stop_reason='skipped_resource'
                status='success' if result.converged else result.stop_reason.split(':')[0]
                if status=='tolerance':status='success'
                audit=audit_endpoint(problem,result);invocation['excluded_audit_sec']+=audit['elapsed_sec']
                if audit['passed'] is False:status='audit_failure'
                result.summary.update(case_id=case['case_id'],family=case['family'],n=len(problem.a),m=len(problem.b),
                    epsilon=problem.epsilon,repeat=job['repeat'],attempt=attempt['attempt'],mode=mode,group=job['group'],
                    warmup_iters=job['warmup_iters'],tol=cfg.tol,status=status,problem_sha256=case['problem_sha256'],
                    initial_sha256=case['initial_sha256'],implementation_sha256=impl,config_sha256=cfgsha,
                    peak_sampled_rss_bytes=memory.peak,memory_limit_bytes=memory_limit,estimated_peak_rss_bytes=estimate,
                    audit=audit,track_time_limit_sec=track_limit,kernel_overrun_policy='checked between numerical kernels; one kernel may overrun',
                    initial_residual_l1=result.history[0].residual_l1 if result.history else None,
                    setup_is_subset_of_post=True)
                if mode=='diagnostic':
                    reference=next((j for j in manifest['jobs'] if j['case_id']==case['case_id'] and
                        j['method']=='mirror_sr1' and j['repeat']==0 and j['warmup_iters']==20 and j['mode']=='performance'),None)
                    if reference and reference['attempts'] and reference['attempts'][-1].get('run_path'):
                        prior=load_result(root/reference['attempts'][-1]['run_path'])
                        pairs=list(zip(prior.history,result.history))
                        result.summary['diagnostic_replay_check']=dict(reference_job=reference['job_id'],
                            matching_prefix=all(a.state_fingerprint==b.state_fingerprint and a.step_size==b.step_size and a.accept_reason==b.accept_reason for a,b in pairs),
                            same_length=len(prior.history)==len(result.history),compared_rows=len(pairs))
                path=root/'runs'/f"j{index:04d}_a{attempt['attempt']}"
                started=perf_counter();save_result(result,path,asdict(cfg),dict(format='reconstructible_mainstream_v1',
                    descriptor=os.path.relpath(root/'problems'/case['case_id']/'descriptor.json',path),sha256=case['problem_sha256']),environment)
                invocation['excluded_io_sec']+=perf_counter()-started
                attempt.update(status=status,run_path=str(path.relative_to(root)),audit=audit)
                print(f'  {status}; k={result.n_iter}, res={result.summary.get("residual_l1")}, total={result.elapsed_sec:.3f}s',flush=True)
                del result
            job['status']=attempt['status'];checkpoint();gc.collect()
    except BaseException as exc:
        invocation['dispatch_error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        signal.signal(signal.SIGINT,old_handler)
        if stop_index is not None and not budget.interrupted and manifest['jobs'][stop_index]['status'] in TERMINAL:
            boundary_paused=True
        invocation['status']='failed_dispatch' if invocation.get('dispatch_error') else 'paused_by_user' if boundary_paused else 'interrupted' if budget.interrupted else 'finished_dispatch'
        if boundary_paused:
            atomic_json(dict(paused=True,reason='user requested pause after current experiment section',
                             stop_after_job=stop_after_job,resume_flag='--continue-paused'),pause_file)
        invocation['pending_count']=sum(j['status'] in ('pending','running','interrupted') for j in manifest['jobs'])
        checkpoint()
    print('Results: '+str(root),flush=True)
    print(f'Remaining machine budgets: performance={budget.remaining("performance"):.1f}s diagnostic={budget.remaining("diagnostic"):.1f}s',flush=True)
    return root
