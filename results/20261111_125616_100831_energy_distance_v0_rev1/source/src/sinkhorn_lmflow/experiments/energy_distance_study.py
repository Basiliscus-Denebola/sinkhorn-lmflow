"""262-entry frozen study, independent cumulative ledger and resumable artifacts."""
from __future__ import annotations

from collections import Counter, deque
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
from time import perf_counter
import numpy as np
import psutil
from ..types import SolverConfig
from ..io.config import load_config, save_config, json_safe
from ..io.result_io import save_result, load_result
from ..io.problem_io import problem_fingerprint
from ..problems.mnist import ensure_mnist, choose_pairs
from ..solvers.sinkhorn_fast import prepare_fast_warmup, solve_sk_log_fast, solve_sk_scaling_fast
from ..solvers.energy_distance import solve_energy_distance
from .reproducibility import runtime_metadata
from .sr1_mainstream import (atomic_json, ledger_lock, MemoryWatch, make_input,
                             save_warmup, load_warmup, audit_endpoint)

ROOT = Path(__file__).resolve().parents[3]
METHODS = ['sk_log_fast', 'sk_scaling_fast', 'energy_distance']
REPRESENTATIVES = ['M1_N1024_s42_e0.001', 'M2_N1024_s42_e0.003',
    'M3_N1024_s42_e0.003', 'M4_N1024_s42_a0.1_e0.003',
    'M5_same_0_e0.001', 'M5_cross_0_1_e0.001']
TERMINAL = {'success', 'timeout', 'max_iter', 'numerical_failure', 'audit_failure',
    'skipped_resource', 'not_applicable_kernel_underflow', 'blocked_data', 'dependency_missing'}


def formal_cases():
    cases = []
    for family in ('M1', 'M2', 'M3'):
        for seed in (42, 43):
            for eps in (.003, .001):
                cases.append(dict(case_id=f'{family}_N1024_s{seed}_e{eps:g}', family=family,
                    n=1024, m=1024, data_seed=seed, epsilon=eps, alpha=None))
    for seed in (42, 43):
        for alpha in (1., .1):
            cases.append(dict(case_id=f'M4_N1024_s{seed}_a{alpha:g}_e0.003', family='M4',
                n=1024, m=1024, data_seed=seed, epsilon=.003, alpha=alpha))
    for pair in ('same_0', 'same_2', 'same_4', 'same_6', 'same_8',
                 'cross_0_1', 'cross_2_3', 'cross_4_5', 'cross_6_7', 'cross_8_9'):
        for eps in (.01, .001):
            cases.append(dict(case_id=f'M5_{pair}_e{eps:g}', family='M5', pair_name=pair,
                n=784, m=784, data_seed=20261110, epsilon=eps, alpha=None))
    for n in (512, 2048):
        cases.append(dict(case_id=f'M6_N{n}_s42_e0.003', family='M6', n=n, m=n,
                          data_seed=42, epsilon=.003, alpha=None))
    cases.append(dict(case_id='M7_1024x2048_s42_e0.003', family='M7', n=1024, m=2048,
                      data_seed=42, epsilon=.003, alpha=None))
    assert len(cases) == 39 and max(max(c['n'], c['m']) for c in cases) == 2048
    return cases


def calibration_cases():
    return [dict(case_id='cal_'+f, family=f, n=512, m=512, data_seed=41,
                 epsilon=.003, alpha=.1 if f == 'M4' else None) for f in ('M1','M2','M3','M4')]


def round_robin(cases):
    groups = {f: deque(c for c in cases if c['family'] == f) for f in sorted({c['family'] for c in cases})}
    ordered = []
    while any(groups.values()):
        for group in groups.values():
            if group:
                ordered.append(group.popleft())
    return ordered


def build_plan(config):
    # Overrides exist solely for small development tests, never silently scale
    # the formal plan or change the production config.
    cases = config.get('test_cases') or formal_cases()
    cal = config.get('test_calibration_cases') or calibration_cases()
    reps = config.get('test_representatives', REPRESENTATIVES)
    jobs = []
    def add(case, method, repeat, stage, kappa=None):
        key = f'{stage}__{case["case_id"]}__{method}__r{repeat}'
        if kappa is not None:
            key += f'__k{kappa:g}'
        jobs.append(dict(job_id=key, case_id=case['case_id'], method=method,
                         repeat=repeat, stage=stage, kappa=kappa, status='pending_budget'))
    for case in cal:
        add(case, 'sk_log_fast', 0, 'calibration')
        for k in config['kappa_grid']:
            add(case, 'energy_distance', 0, 'calibration', k)
    for repeat in range(config['repeats']):
        for i, case in enumerate(round_robin(cases)):
            shift = (i+repeat) % 3
            for method in METHODS[shift:]+METHODS[:shift]:
                add(case, method, repeat, 'main')
    for cid in reps:
        case = next(c for c in cases if c['case_id'] == cid)
        for method in ('energy_zero', 'energy_amp'):
            add(case, method, 0, 'ablation')
    if not config.get('test_cases'):
        assert Counter(j['stage'] for j in jobs) == dict(calibration=16, main=234, ablation=12)
        assert len(jobs) == 262
    return cal+cases, jobs


def fingerprint(config):
    files = [p for p in (ROOT/'src/sinkhorn_lmflow').rglob('*.py')]
    files += [ROOT/'scripts/run_energy_distance_study.py', ROOT/'scripts/report_energy_distance.py']
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    cfg = hashlib.sha256(json.dumps(json_safe(config), sort_keys=True).encode()).hexdigest()
    return digest, cfg, hashes


class BudgetLedger:
    """Checkpoint live machine time every second; never charge closed downtime.

    A crashed session reserves one additional heartbeat interval on recovery.
    Stage usage and total usage include input preparation, all I/O and reports.
    """
    def __init__(self, path, config):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.total = float(config['total_budget_sec'])
        self.limits = dict(calibration=float(config['calibration_budget_sec']),
            main=float(config['calibration_budget_sec']+config['main_budget_sec']),
            ablation=self.total-float(config['report_reserve_sec']), report=self.total)
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding='utf-8'))
            if self.data['total_budget_sec'] != self.total or self.data['stage_boundaries'] != self.limits:
                raise ValueError('budget limits changed; ledger must not be reset')
            if self.data.get('active'):
                self.data['used_sec'] += 1.
                old = self.data['stage']
                self.data['stage_used_sec'][old] += 1.
                self.data['crash_reserve_sec'] = self.data.get('crash_reserve_sec', 0.)+1.
        else:
            self.data = dict(total_budget_sec=self.total, stage_boundaries=self.limits,
                used_sec=0., stage_used_sec={s: 0. for s in self.limits}, active_run_dir=None,
                revision=0, sessions=[], active=False)
        self.guard = threading.RLock()
        self.last = perf_counter()
        self.done = threading.Event()
        self.data.update(active=True, stage='calibration', pid=os.getpid())
        self.data['sessions'].append(dict(start_utc=datetime.now(timezone.utc).isoformat(),
                                         start_used_sec=self.data['used_sec']))
        self.checkpoint()
        self.worker = threading.Thread(target=self._heartbeat, daemon=True)
        self.worker.start()

    def _heartbeat(self):
        while not self.done.wait(1.):
            self.checkpoint()

    def checkpoint(self):
        with self.guard:
            now = perf_counter()
            delta = now-self.last
            self.data['used_sec'] += delta
            self.data['stage_used_sec'][self.data['stage']] += delta
            self.last = now
            self.data['overrun_sec'] = max(0., self.data['used_sec']-self.total)
            atomic_json(self.data, self.path)

    def used(self):
        with self.guard:
            return self.data['used_sec']+perf_counter()-self.last

    def remaining(self, stage):
        return max(0., self.limits[stage]-self.used())

    def stage(self, name):
        with self.guard:
            self.checkpoint()
            self.data['stage'] = name

    def close(self):
        self.done.set()
        self.worker.join()
        self.checkpoint()
        self.data['active'] = False
        self.data['sessions'][-1]['end_used_sec'] = self.data['used_sec']
        atomic_json(self.data, self.path)


def choose_kappa(rows, config):
    grid = config['kappa_grid']
    expected = 4*(len(config.get('test_calibration_cases') or calibration_cases()))
    if len(rows) != expected or any(r['status'] == 'pending_budget' for r in rows):
        return dict(kappa=config['kappa_if_calibration_incomplete'], reason='calibration_incomplete')
    if all(r.get('warmup_converged') for r in rows):
        return dict(kappa=1., reason='warmup_no_discrimination')
    counts = {k: sum(r['status']=='success' for r in rows if r['method']=='energy_distance' and r['kappa']==k) for k in grid}
    candidates = [k for k in grid if counts[k] == max(counts.values())]
    baseline = {r['case_id']: r for r in rows if r['method']=='sk_log_fast' and r['status']=='success'}
    common = set(baseline)
    for k in candidates:
        common &= {r['case_id'] for r in rows if r['method']=='energy_distance' and r['kappa']==k
                   and r['status']=='success' and not r.get('warmup_converged')}
    scores = {}
    for k in candidates:
        ratios = [r['total_sec']/baseline[r['case_id']]['total_sec'] for r in rows
                  if r['method']=='energy_distance' and r['kappa']==k and r['case_id'] in common]
        scores[k] = float(np.exp(np.mean(np.log(ratios)))) if ratios else None
    best = min(candidates)
    if common:
        best_score = min(scores.values())
        best = min(k for k in candidates if np.isclose(scores[k], best_score, rtol=1e-10, atol=1e-12))
    return dict(kappa=best, reason='success_count_then_common_geomean_then_smaller',
                success_counts=counts, common_cases=sorted(common), time_ratio_scores=scores)


def estimated_workspace(n, m, method):
    """Conservative live-array bound, bytes, not SR1's 14-matrix model.

    M + immutable logK + light evaluator/logsumexp scratch <= 8 n*m;
    energy adds Gram/D plus transient X and 12 block*m stable-difference arrays.
    The bound also includes up to 50001 vector-sized trace records (32 MB).
    """
    base = 8*(8*n*m+32*(n+m)) + 32*1024**2
    if method.startswith('energy') and method != 'energy_zero':
        base += 8*(m*m+n*m+12*64*m)
    return base


def artifact_hashes(path):
    return {name: hashlib.sha256((path/name).read_bytes()).hexdigest()
            for name in ('summary.json', 'history.csv', 'final_state.npz')}


def run_study(config, new_revision=False):
    ledger_path = ROOT/config['budget_ledger']
    with ledger_lock(ledger_path):
        ledger = BudgetLedger(ledger_path, config)
        try:
            return _run(config, ledger, new_revision)
        finally:
            ledger.close()


def _run(config, ledger, new_revision):
    source_sha, config_sha, hashes = fingerprint(config)
    root = Path(ledger.data['active_run_dir']) if ledger.data['active_run_dir'] else None
    inherited = None
    if root is not None:
        manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        mismatch = manifest['source_sha256'] != source_sha or manifest['config_sha256'] != config_sha
        if mismatch and not new_revision:
            raise ValueError('implementation/config changed; use --new-revision, retaining the same budget ledger')
        if mismatch:
            previous_hashes = json.loads((root/'source_fingerprints.json').read_text(encoding='utf-8'))
            changed = {name for name in set(hashes)|set(previous_hashes) if hashes.get(name)!=previous_hashes.get(name)}
            # Reporting-only revisions may inherit a completely finished plan.
            # No solver/input/kernel code or configuration can differ; pending
            # plans always start a new measured revision under the same ledger.
            report_only = changed <= {'src/sinkhorn_lmflow/plotting/energy_distance.py',
                                     'src/sinkhorn_lmflow/experiments/energy_distance_study.py'}
            if report_only and config_sha==manifest['config_sha256'] and all(j['status'] in TERMINAL for j in manifest['jobs']):
                inherited = (root, manifest)
            root = None
            ledger.data['revision'] += 1
    if root is None:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        root = ROOT/'results'/f'{stamp}_energy_distance_v0_rev{ledger.data["revision"]}'
        if inherited is not None:
            previous, manifest = inherited
            shutil.copytree(previous, root)
            manifest.update(inherited_measurements_from=str(previous),
                measurement_source_sha256=manifest.get('measurement_source_sha256',manifest['source_sha256']),
                source_sha256=source_sha, revision=ledger.data['revision'],
                revision_kind='reporting_only; all completed measurements inherited without rerunning solvers')
        else:
            root.mkdir(parents=True)
            cases, jobs = build_plan(config)
            manifest = dict(source_sha256=source_sha, config_sha256=config_sha, cases=cases,
                            jobs=jobs, revision=ledger.data['revision'])
        atomic_json(manifest, root/'manifest.json')
        save_config(config, root/'config_resolved.yaml')
        atomic_json(hashes, root/'source_fingerprints.json')
        for name in hashes:
            target = root/'source'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, target)
        atomic_json(runtime_metadata(None), root/('report_environment.json' if inherited is not None else 'environment.json'))
        ledger.data['active_run_dir'] = str(root)
        ledger.checkpoint()
    print('RUN_DIR='+str(root), flush=True)
    cases = {c['case_id']: c for c in manifest['cases']}
    jobs = manifest['jobs']
    for job in jobs:
        if job.get('run_path') and job['status'] in TERMINAL:
            path = root/job['run_path']
            if artifact_hashes(path) != job['artifact_sha256']:
                raise ValueError('saved artifact changed: '+job['job_id'])
        elif job['status'] == 'running':
            # A crash cannot turn an unverified partial artifact into success.
            job['status'] = 'pending_budget'
    dataset = None
    dataset_attempted = False
    cached_key, cached_problem, cached_arrays, cached_warm = None, None, None, None
    selected_path = root/'selected_kappa.json'
    selected = json.loads(selected_path.read_text(encoding='utf-8')) if selected_path.exists() else None
    cfg = SolverConfig(tol=float(config['tol']), max_iter=config['max_iter'],
        sr1_warmup_iters=config['warmup_iters'], armijo=config['armijo_sigma'],
        backtrack_factor=config['backtrack_factor'], max_backtracks=config['max_trial_steps'])
    for stage in ('calibration', 'main', 'ablation'):
        ledger.stage(stage)
        # Once selected, calibration is frozen, including its unstarted entries.
        if stage == 'calibration' and selected is not None:
            continue
        if stage == 'main' and selected is None:
            rows = []
            for j in jobs:
                if j['stage'] == 'calibration':
                    row = dict(j)
                    if j.get('run_path'):
                        row.update(json.loads((root/j['run_path']/'summary.json').read_text(encoding='utf-8')))
                    rows.append(row)
            selected = choose_kappa(rows, config)
            selected.update(source_sha256=source_sha, config_sha256=config_sha,
                            frozen_at_machine_sec=ledger.used())
            atomic_json(selected, selected_path)
            print('FROZEN_KAPPA='+str(selected), flush=True)
        for job in (j for j in jobs if j['stage'] == stage):
            if job['status'] in TERMINAL:
                continue
            if ledger.remaining(stage) <= 0:
                break
            case = cases[job['case_id']]
            full = None
            if stage == 'ablation':
                full_job = next(j for j in jobs if j['stage']=='main' and j['case_id']==job['case_id']
                                and j['method']=='energy_distance' and j['repeat']==0)
                if not full_job.get('run_path'):
                    job['status'] = 'dependency_missing'
                    atomic_json(manifest, root/'manifest.json')
                    continue
                full = json.loads((root/full_job['run_path']/'summary.json').read_text(encoding='utf-8'))
                if full.get('gamma') is None or not (full.get('gamma_initialized') or full.get('warmup_converged')):
                    job['status'] = 'dependency_missing'
                    continue
            t_job = perf_counter()
            job['status'] = 'running'
            atomic_json(manifest, root/'manifest.json')
            key = f'{case["case_id"]}__r{job["repeat"]}'
            limit = config['track_limit_sec'] if max(case['n'],case['m']) <= 1024 else config['track_limit_large_sec']
            deadline = perf_counter()+min(float(limit), ledger.remaining(stage))
            try:
                estimate = estimated_workspace(case['n'], case['m'], job['method'])
                available = psutil.virtual_memory().available
                rss = psutil.Process().memory_info().rss
                cap = min(config['memory_cap_gib']*1024**3, rss+config['memory_available_fraction']*available)
                if rss+estimate > cap:
                    job.update(status='skipped_resource', estimated_workspace_bytes=estimate,
                               available_bytes=available, resource_cap_bytes=cap)
                    continue
                if cached_key != key:
                    cached_problem, cached_arrays, cached_warm = None, None, None
                    gc.collect()
                    if case['family']=='M5' and not dataset_attempted:
                        dataset_attempted = True
                        images, labels, provenance = ensure_mnist(ROOT/config['mnist_cache'], config['download_mnist'])
                        pairs = choose_pairs(labels, seed=20261110)
                        dataset = images, labels, provenance, pairs
                        atomic_json(provenance, root/'mnist_provenance.json')
                        atomic_json(pairs, root/'mnist_pairs.json')
                    cached_problem, cached_arrays, _ = make_input(case, dataset)
                    cached_key = key
                    input_path = root/'inputs'/case['case_id']
                    input_path.mkdir(parents=True, exist_ok=True)
                    problem_sha = problem_fingerprint(cached_problem)
                    meta_path = input_path/'metadata.json'
                    if meta_path.exists():
                        old = json.loads(meta_path.read_text(encoding='utf-8'))
                        if old['input_sha256'] != problem_sha:
                            raise ValueError('input fingerprint changed')
                    else:
                        atomic_json(dict(cached_problem.metadata, input_sha256=problem_sha), meta_path)
                        np.savez_compressed(input_path/'reconstruction.npz', **cached_arrays)
                problem, arrays = cached_problem, cached_arrays
                problem_sha = problem_fingerprint(problem)
                if perf_counter() >= deadline:
                    job['status'] = 'pending_budget'
                    continue
                with MemoryWatch(cap) as memory:
                    cancelled = lambda: memory.exceeded or perf_counter() >= deadline
                    if job['method'] != 'sk_scaling_fast' and cached_warm is None:
                        if (root/'warmups'/key/'snapshot.json').exists():
                            cached_warm = load_warmup(root, key, problem)
                            cached_warm.log_kernel = -problem.M/problem.epsilon
                            cached_warm.log_kernel.flags.writeable = False
                        else:
                            wr, cached_warm = prepare_fast_warmup(problem, cfg, cancelled)
                            if cached_warm.iteration < config['warmup_iters'] and not wr.converged:
                                cached_warm = None
                                job['status'] = 'pending_budget'
                                continue
                            save_warmup(root, key, wr, cached_warm, cfg)
                    if perf_counter() >= deadline:
                        job['status'] = 'pending_budget'
                        continue
                    if job['method'] == 'sk_log_fast':
                        result = solve_sk_log_fast(problem, cfg, cached_warm, cancelled)
                    elif job['method'] == 'sk_scaling_fast':
                        result = solve_sk_scaling_fast(problem, cfg, cancelled)
                    else:
                        result = solve_energy_distance(problem, cfg, cached_warm, cancelled, arrays,
                            kappa=job['kappa'] if stage=='calibration' else selected['kappa'],
                            variant=job['method'], gamma_override=full['gamma'] if job['method']=='energy_amp' else None,
                            diagnostics=case['case_id'] in REPRESENTATIVES)
                audit = audit_endpoint(problem, result)
                status = 'success' if result.converged and audit['passed'] else result.stop_reason.split(':')[0]
                if audit['passed'] is False:
                    status = 'audit_failure'
                if memory.exceeded:
                    status = 'skipped_resource'
                result.summary.update(case_id=case['case_id'], family=case['family'], n=len(problem.a),
                    m=len(problem.b), epsilon=problem.epsilon, alpha=case.get('alpha'), data_seed=case['data_seed'],
                    repeat=job['repeat'], stage=stage, status=status, endpoint_audit=audit,
                    process_peak_rss=memory.peak, estimated_workspace_bytes=estimate,
                    resource_cap_bytes=cap, source_sha256=source_sha, config_sha256=config_sha,
                    input_sha256=problem_sha, total_sec=result.elapsed_sec,
                    kernel_matrix_vector_count=result.summary['matrix_vector_count'],
                    deadline_overrun_sec=max(0.,perf_counter()-deadline),
                    last_residual=result.summary['residual_l1'],
                    timing_environment='OT Python; serial; BLAS threads 1; shared fresh warmup per case/repeat')
                result.summary.setdefault('warmup_converged', result.stop_reason=='warmup_converged')
                result.summary.setdefault('warmup_sec', result.summary.get('warmup_elapsed_sec',0.))
                run_path = Path('tracks')/job['job_id']
                save_result(result, root/run_path, config, dict(input_sha256=problem_sha))
                job.update(status=status, run_path=str(run_path), artifact_sha256=artifact_hashes(root/run_path),
                           input_sha256=problem_sha)
                print(f'{stage} {job["job_id"]}: {status} rho={result.summary["residual_l1"]:.3g} '
                      f'total={result.elapsed_sec:.3f}s machine={ledger.used():.1f}s', flush=True)
            except (FileNotFoundError, OSError) as exc:
                # I/O failure is visible and is never counted as convergence.
                job.update(status='blocked_data', error=str(exc))
            except (FloatingPointError, MemoryError) as exc:
                job.update(status='numerical_failure' if isinstance(exc,FloatingPointError) else 'skipped_resource', error=str(exc))
            finally:
                job['machine_elapsed_sec'] = job.get('machine_elapsed_sec',0.)+perf_counter()-t_job
                atomic_json(manifest, root/'manifest.json')
                ledger.checkpoint()
        atomic_json(manifest, root/'manifest.json')
    ledger.stage('report')
    cached_problem, cached_arrays, cached_warm, dataset = None, None, None, None
    gc.collect()
    from ..plotting.energy_distance import generate_report
    generate_report(root, ROOT/'docs/energy_distance_v0_report.md', audit=False, ledger=ledger)
    ledger.checkpoint()
    atomic_json(ledger.data, root/'budget_snapshot.json')
    return root
