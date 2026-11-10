"""Paired SR1 v0 study, durable per-track checkpoints and fingerprinted resume."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import itertools
import json
import os
from pathlib import Path
from time import perf_counter
import numpy as np
from ..types import SolverConfig
from ..problems.weak_coupling import weak_coupling_problem, initial_columns, initial_fingerprint
from ..problems.generators import generate_problem
from ..core.semi_dual import evaluate_reduced, center
from ..solvers.mirror_sr1 import prepare_warmup, solve_sinkhorn_reduced, solve_mirror_sr1, warmup_protocol
from ..io.config import save_config, save_json, json_safe
from ..io.paths import create_run_dir
from ..io.problem_io import save_problem, problem_fingerprint
from ..io.result_io import save_result, load_result
from .reproducibility import runtime_metadata


def config_fingerprint(config):
    return hashlib.sha256(json.dumps(json_safe(config), sort_keys=True).encode()).hexdigest()


def planned_cases(config):
    if config['stage'] == 'transfer':
        return [dict(case_id=f"pointcloud_eps{eps:g}", kind='transfer', n=128, epsilon=eps,
                     initial_seed=None, data_seed=42) for eps in config['epsilons']]
    return [dict(case_id=f"N{config['n']}_B{b}_d{d:g}_s{s}", kind='controlled',
                 n=config['n'], blocks=b, delta=d, initial_seed=s, data_seed=None)
            for b,d,s in itertools.product(config['blocks'],config['deltas'],config['initial_seeds'])]


def case_inputs(case):
    if case['kind'] == 'controlled':
        p = weak_coupling_problem(case['n'],case['blocks'],case['delta'])
        z = initial_columns(case['n'],case['blocks'],case['initial_seed'])
    else:
        p = generate_problem(generator='2d_pointcloud',n=128,m=128,epsilon=case['epsilon'],seed=42,clustered=True)
        z = np.zeros_like(p.b)
    return p, center(z)


def _checkpoint(manifest, root):
    path = root/'manifest.next.json'
    save_json(manifest,path)
    path.replace(root/'manifest.json')


def _complete_artifacts(root, attempt, problem_sha, initial_sha):
    if set(attempt.get('runs',{})) != {'sinkhorn_reduced','mirror_sr1'}:
        return False
    try:
        for path in attempt['runs'].values():
            result = load_result(root/path)
            if not result.converged or not result.history:
                return False
            if (result.summary.get('problem_sha256') != problem_sha or
                    result.summary.get('initial_sha256') != initial_sha or
                    result.summary.get('warmup_sha256') != attempt.get('warmup_sha256')):
                return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return True


def run_study(config, output_root='results', resume=None):
    command_start = perf_counter()
    budget = float(config.get('total_budget_sec',3600))
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError('total_budget_sec must be positive')
    deadline = command_start+budget
    fingerprint = config_fingerprint(config)
    if resume:
        root = Path(resume)
        manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        if manifest['config_sha256'] != fingerprint:
            raise ValueError('resume configuration fingerprint mismatch')
    else:
        root = create_run_dir(output_root,'sr1_'+config['stage'])
        save_config(config,root/'study_config.yaml')
        plan = []
        for case in planned_cases(config):
            for repeat in range(config['repeats']):
                plan.append(dict(case, repeat=repeat, pair_id=f"{case['case_id']}_r{repeat}",
                                 status='pending', attempts=[]))
        manifest = dict(schema=1,config_sha256=fingerprint,stage=config['stage'],pairs=plan,
                        invocations=[], resume_command=f'.\\scripts\\run_ot.cmd scripts/run_sr1_study.py --config "{root}/study_config.yaml" --resume "{root}"')
    meta = runtime_metadata(None)
    invocation = dict(start_metadata=meta,status='running',budget_sec=budget)
    manifest['invocations'].append(invocation)
    _checkpoint(manifest,root)
    # Untimed small pre-run; excluded from result tables, included in command budget.
    pre = weak_coupling_problem(16,2,.01)
    pcfg = SolverConfig(max_iter=3,sr1_warmup_iters=1,compute_spectrum_every=0,max_backtracks=5,
                        initial_nu=-initial_columns(16,2,42))
    solve_sinkhorn_reduced(pre,pcfg)
    solve_mirror_sr1(pre,pcfg)
    solver_names = ['sinkhorn_reduced','mirror_sr1']
    functions = dict(sinkhorn_reduced=solve_sinkhorn_reduced,mirror_sr1=solve_mirror_sr1)
    try:
        for pair_index, pair in enumerate(manifest['pairs']):
            if perf_counter() >= deadline:
                break
            p,z = case_inputs(pair)
            psha, isha = problem_fingerprint(p), initial_fingerprint(z)
            if pair.get('problem_sha256',psha) != psha or pair.get('initial_sha256',isha) != isha:
                raise ValueError('resume input fingerprint mismatch')
            if pair['status'] == 'completed':
                # A complete pair must still have two readable terminal artifacts.
                last = pair['attempts'][-1]
                if _complete_artifacts(root,last,psha,isha):
                    continue
            pair.update(problem_sha256=psha,initial_sha256=isha)
            problem_path = root/'problems'/psha/'problem.npz'
            if not problem_path.exists():
                save_problem(p,problem_path)
            # Short directory prefixes avoid legacy Windows MAX_PATH; full hashes
            # remain authoritative in metadata and are checked before reuse.
            initial_dir = root/'initial_states'/isha[:16]/psha[:16]
            initial_dir.mkdir(parents=True,exist_ok=True)
            if (initial_dir/'metadata.json').exists():
                existing = json.loads((initial_dir/'metadata.json').read_text(encoding='utf-8'))
                if existing['initial_sha256'] != isha or existing['problem_sha256'] != psha:
                    raise ValueError('initial directory hash-prefix collision')
            if not (initial_dir/'initial_state.npz').exists():
                initial = evaluate_reduced(p,z)
                np.savez_compressed(initial_dir/'initial_state.npz',z_col=z,mu=initial.mu,nu=initial.nu)
                save_json(dict(initial_sha256=isha,problem_sha256=psha,initial_seed=pair['initial_seed'],
                               distribution='group normal + .1 independent normal' if pair['kind']=='controlled' else 'zero columns'),initial_dir/'metadata.json')
            params = dict(config['solver'])
            params['initial_nu'] = -p.epsilon*z
            params['wall_time_limit_sec'] = config.get('track_budget_sec',120.)
            cfg = SolverConfig(**params)
            attempt_id = len(pair['attempts'])+1
            aid = f"{pair['pair_id']}_a{attempt_id}"
            attempt = dict(attempt=attempt_id,status='running',runs={},order=solver_names[::1 if pair_index % 2 == 0 else -1])
            pair['attempts'].append(attempt)
            pair['status'] = 'running'
            _checkpoint(manifest,root)
            print(f"[{pair_index+1}/{len(manifest['pairs'])}] {aid}",flush=True)
            warm_result,snapshot = prepare_warmup(p,cfg,deadline=deadline,
                diagnostic_reference_z=np.zeros_like(p.b) if pair['kind']=='controlled' else None)
            wdir = root/'warmups'/aid
            warm_config = dict(asdict(cfg), max_iter=min(cfg.max_iter,cfg.sr1_warmup_iters))
            save_result(warm_result,wdir,warm_config,{'sha256':psha},meta)
            np.savez_compressed(wdir/'snapshot.npz',z_col=snapshot.state.z_col,c=snapshot.state.c,
                previous_z=np.array([]) if snapshot.previous is None else snapshot.previous[0],
                previous_c=np.array([]) if snapshot.previous is None else snapshot.previous[1])
            save_json(dict(fingerprint=snapshot.fingerprint,protocol=snapshot.protocol,
                           initial_sha256=isha,problem_sha256=psha,work=snapshot.work.to_dict(),
                           elapsed_sec=snapshot.elapsed_sec,iteration=snapshot.iteration),wdir/'snapshot.json')
            attempt.update(warmup=str(wdir.relative_to(root)),warmup_sha256=snapshot.fingerprint)
            _checkpoint(manifest,root)
            if not warm_result.converged and snapshot.iteration < min(cfg.max_iter,cfg.sr1_warmup_iters):
                attempt['status'] = pair['status'] = 'partial'
                attempt['failure'] = warm_result.stop_reason
                _checkpoint(manifest,root)
                continue
            for name in attempt['order']:
                if perf_counter() >= deadline:
                    break
                result = functions[name](p,cfg,warmup=snapshot,deadline=deadline)
                result.summary.update(problem_sha256=psha,initial_sha256=isha,
                    warmup_sha256=snapshot.fingerprint,warmup_protocol=warmup_protocol(cfg),
                    case_id=pair['case_id'],pair_id=pair['pair_id'],repeat=pair['repeat'],attempt=attempt_id,
                    initial_seed=pair['initial_seed'],data_seed=pair['data_seed'],tol=cfg.tol,
                    diagnostic_setup_policy='reference evaluation included in shared warmup; serialization-only initial export excluded')
                rdir = root/'runs'/aid/name
                save_result(result,rdir,asdict(cfg),dict(sha256=psha,path=os.path.relpath(problem_path,rdir),
                    initial_sha256=isha,initial_path=os.path.relpath(initial_dir/'initial_state.npz',rdir)),meta)
                attempt['runs'][name] = str(rdir.relative_to(root))
                _checkpoint(manifest,root)
                print(f"  {name}: {result.stop_reason}, k={result.n_iter}, eval={result.summary['state_eval_count']}, {result.elapsed_sec:.3f}s",flush=True)
            complete = len(attempt['runs']) == 2
            if complete:
                complete = all(json.loads((root/path/'summary.json').read_text(encoding='utf-8'))['converged']
                               for path in attempt['runs'].values())
            attempt['status'] = pair['status'] = 'completed' if complete else 'partial'
            _checkpoint(manifest,root)
    except KeyboardInterrupt:
        invocation['status'] = 'interrupted'
    except Exception as exc:
        invocation.update(status='error',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if invocation['status'] == 'running':
            invocation['status'] = 'completed' if all(p['status']=='completed' for p in manifest['pairs']) else 'partial'
        invocation['elapsed_sec'] = perf_counter()-command_start
        for pair in manifest['pairs']:
            if pair['status'] == 'running':
                pair['status'] = 'partial'
                pair['attempts'][-1]['status'] = 'partial'
        _checkpoint(manifest,root)
    print(f'Results: {root}',flush=True)
    if invocation['status'] != 'completed':
        print('Resume: '+manifest['resume_command'],flush=True)
    return root
