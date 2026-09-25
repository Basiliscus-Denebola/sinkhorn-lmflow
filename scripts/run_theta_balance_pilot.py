"""Small paired pilot; saved runs are the sole source of plots and tables."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from time import perf_counter, sleep
import os
import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sinkhorn_lmflow.types import SolverConfig, OTProblem
from sinkhorn_lmflow.problems.generators import generate_problem
from sinkhorn_lmflow.solvers.sinkhorn import sinkhorn_log_sweep
from sinkhorn_lmflow.experiments.registry import get_solver
from sinkhorn_lmflow.experiments.reproducibility import runtime_metadata
from sinkhorn_lmflow.io.config import load_config, save_config, save_json
from sinkhorn_lmflow.io.paths import create_run_dir
from sinkhorn_lmflow.io.problem_io import save_problem, load_problem, problem_fingerprint
from sinkhorn_lmflow.io.result_io import save_result, load_result
from sinkhorn_lmflow.plotting.convergence import save_figure


def stats(values):
    return f"{np.median(values):.6g} [{min(values):.6g}, {max(values):.6g}]"


def analyze(root, manifest):
    runtime = manifest["runtime"]
    lines = ["# Theta balance pilot", "",
        f"Results: [{root.name}]({root.resolve().as_posix()})", "",
        f"Base commit: `{runtime['git_commit']}`; working tree dirty (existing staged changes plus this implementation).",
        f"Environment: `{runtime['python_executable']}`, Python {runtime['python'].split()[0]}, "
        f"NumPy {runtime['numpy']}, SciPy {runtime['scipy']}; BLAS threads=1.", "",
        "Command: `scripts\\run_ot.cmd scripts/run_theta_balance_pilot.py`.", "",
        "Seed=42, tol=1e-8, max_iter=3000, balance_theta_max=10, initial h=1, "
        "Armijo=1e-4, contraction=.5, max_backtracks=60. Full resolved configs and runtime metadata are saved.", "",
        "Both methods receive identical potentials from one stabilized GS sweep starting at zero. "
        "This common initialization is saved and EXCLUDED from both solver clocks and work counts; its time is recorded below. "
        "Each solver has one discarded timing warm-up per case, then three saved repetitions; execution order alternates.", "",
        "Wall time includes initialization inside the solver, all monitoring and line search, but excludes I/O and plotting. "
        "Record every iteration; expensive spectral SVD is disabled for both. Values are median [min, max].", "",
        "## Work definition", "",
        "Dense-pass proxy = plan_build_count + reduction_count + logsumexp_count + p_matvec_count. "
        "One full P construction, one sum reduction over all entries (row/column/total), one full-matrix logsumexp, "
        "or one P-vector multiplication each counts as one unit. Logsumexp is a composite operation and can cost much more "
        "than a sum: this is an unweighted algorithmic proxy, NOT FLOPs, equivalent matvecs, or a runtime prediction. "
        "The two compared log-domain solvers count direction work, trial evaluation, roundoff residual checks and common monitoring. "
        "Vector-only arithmetic, allocations, gauge operations, the optional spectrum and final transport-objective export are excluded "
        "from this proxy (but included in solver wall time where executed). Work at a residual threshold is cumulative through that observation. "
        "Column updates and all trial attempts are also recorded separately; do not add them again. "
        "Other solver families and primal Sinkhorn do not have complete proxy accounting. Old matrix_vector_count is not used.", "",
        "balance_theta/eta/h in row k describe the step from k-1 to k; direction_residual_l1 stores the pre-step residual. "
        "Old theta remains empty. eta is a linearized defect, not the actual nonlinear residual ratio.", ""]
    rows = []
    for case in manifest["cases"]:
        label = case["label"]
        lines += [f"## {label}", "", f"Problem SHA256: `{case['sha256']}`; "
                  f"n=m={case['n']}, epsilon={case['epsilon']}. Common sweep: {case['warm_start_sec']:.6g} s, 2 logsumexp passes.", "",
                  "| Method | Converged | Iterations | Total seconds | Final R |",
                  "|---|---|---:|---|---:|"]
        results = {}
        histories = {}
        for name in ("sinkhorn", "theta_balance"):
            paths = [root/p for p in case["runs"][name]]
            results[name] = [load_result(p) for p in paths]
            histories[name] = [pd.read_csv(p/"history.csv") for p in paths]
            rr = results[name]
            lines.append(f"| {name} | {sum(r.converged for r in rr)}/{len(rr)} | "
                         f"{rr[0].n_iter} | {stats([r.elapsed_sec for r in rr])} | {rr[0].summary['residual_l1']:.3g} |")
        lines += ["", "| Method | Target R | Time seconds | Dense-pass proxy |",
                  "|---|---:|---|---|"]
        for name, hh in histories.items():
            for target in (1e-4, 1e-6, 1e-8):
                hits = [h[h.residual_l1 <= target] for h in hh]
                reached = all(not h.empty for h in hits)
                times = [float(h.iloc[0].elapsed_sec) for h in hits] if reached else []
                work = [int(h.iloc[0].dense_pass_proxy) for h in hits] if reached else []
                lines.append(f"| {name} | {target:g} | {stats(times) if reached else '—'} | {stats(work) if reached else '—'} |")
                rows.append(dict(case=label, solver=name, target=target,
                                 seconds_median=np.median(times) if times else None,
                                 seconds_min=min(times) if times else None, seconds_max=max(times) if times else None,
                                 dense_pass_proxy=np.median(work) if work else None))
        speed = np.median([r.elapsed_sec for r in results['sinkhorn']])/np.median([r.elapsed_sec for r in results['theta_balance']])
        h = histories['theta_balance'][0]
        valid = h[h.balance_theta.notna()]
        lines += ["", f"Total-time ratio Sinkhorn/theta_balance: {speed:.3f}x (values above 1 favor theta_balance)."]
        for key in ("balance_theta_free", "balance_theta", "balance_effective", "balance_eta", "step_size"):
            v = valid[key].dropna()
            lines.append(f"- {key}: min={v.min():.6g}, max={v.max():.6g}, first={v.iloc[0]:.6g}, last={v.iloc[-1]:.6g}." if len(v) else f"- {key}: no successful balance steps.")
        top = (valid.balance_theta_free > manifest['config']['solver']['balance_theta_max']).mean()
        lines += [f"- Upper clipping fraction: {top:.2%}; cumulative clips/degeneracies/fallbacks/backtracks: "
                  f"{int(h.iloc[-1].balance_clip_count)}/{int(h.iloc[-1].balance_degenerate_count)}/"
                  f"{int(h.iloc[-1].balance_fallback_count)}/{int(h.iloc[-1].backtrack_count)}.",
                  "- Stop reasons: " + "; ".join(f"{name}: {[r.stop_reason for r in rr]}" for name, rr in results.items()), "",
                  "| Method (repeat 1) | P builds | Reductions | LSE | P matvec | Column updates | Trial attempts |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for name, rr in results.items():
            lines.append("| " + name + " | " + " | ".join(str(rr[0].summary[k]) for k in
                ("plan_build_count", "reduction_count", "logsumexp_count", "p_matvec_count", "column_update_count", "trial_count")) + " |")
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        for name, hh in histories.items():
            for rep, hist in enumerate(hh):
                for ax, x in zip(axes[0, :2], ("elapsed_sec", "dense_pass_proxy")):
                    ax.semilogy(hist[x], hist.residual_l1.clip(lower=1e-300),
                                label=name if rep == 0 else None, color='C0' if name == 'sinkhorn' else 'C1',
                                alpha=1 if rep == 0 else .25)
                    if not results[name][rep].converged:
                        ax.scatter(hist[x].iloc[-1], hist.residual_l1.iloc[-1], marker='x', c='red')
        axes[0, 0].set(xlabel="Wall seconds", ylabel="R")
        axes[0, 1].set(xlabel="Dense-pass proxy (unweighted)", ylabel="R")
        for ax, x in ((axes[0, 2], 'iteration'), (axes[1, 0], 'direction_residual_l1')):
            v = valid if x == 'iteration' else valid[valid[x] > 0]
            ax.plot(v[x], v.balance_theta_free, label='theta_free', alpha=.7)
            ax.plot(v[x], v.balance_theta, label='theta')
            clipped = v[v.balance_clip_count.diff().fillna(v.balance_clip_count) > 0]
            ax.scatter(clipped[x], clipped.balance_theta, marker='x', color='red', label='clip')
            ax.set(xlabel=x, ylabel='Balance coefficient')
            if x != 'iteration':
                ax.set_xscale('log')
        axes[1, 1].plot(valid.iteration, valid.balance_eta, label='eta')
        axes[1, 1].set(xlabel='Iteration', ylabel='Linear relative defect')
        axes[1, 2].plot(valid.iteration, valid.step_size, label='h')
        axes[1, 2].plot(valid.iteration, valid.balance_effective, label='h * theta')
        axes[1, 2].set(xlabel='Iteration', ylabel='Step / effective coefficient')
        fallback = h[h.phase == 'sinkhorn_fallback']
        for _, row in fallback.iterrows():
            axes[0, 0].scatter(row.elapsed_sec, row.residual_l1, marker='s', c='black')
            axes[0, 1].scatter(row.dense_pass_proxy, row.residual_l1, marker='s', c='black')
        for ax in axes.flat:
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        fig.suptitle(f"{label}: repeats (faint), first-repeat theta diagnostics; red x=clip/failure, square=fallback")
        fig.tight_layout()
        save_figure(fig, root/'figures'/label)
        lines += ["", f"[Curves]({(root/'figures'/f'{label}.png').resolve().as_posix()})", ""]
    lines += ["## Limits", "", "One seed and small dense problems; three repeats characterize only this local run. "
              "No parameter tuning or universal speedup claim. Compare threshold times, total time and the unweighted work proxy separately. "
              "All three raw trajectories and failures are retained. Generated data/figures are not committed.", ""]
    pd.DataFrame(rows).to_csv(root/'thresholds.csv', index=False)
    (root/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='YAML configuration (default: configs/theta_balance_pilot.yaml)')
    parser.add_argument('--analyze', type=Path, help='Only read saved runs and regenerate plots/tables')
    parser.add_argument('--resume', type=Path, help='Resume one systematic study using its saved configuration')
    args = parser.parse_args()
    if args.analyze:
        import json
        root = args.analyze
        manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        (analyze_study if manifest.get('mode') == 'systematic' else analyze)(root, manifest)
        print(root)
        return
    if args.resume:
        run_study(load_config(args.config) if args.config else None, args.resume)
        return
    config = load_config(args.config or 'configs/theta_balance_pilot.yaml')
    if config.get('mode') == 'systematic':
        run_study(config)
        return
    root = create_run_dir('results', 'theta_balance_pilot')
    save_config(config, root/'pilot_config.yaml')
    runtime = runtime_metadata(42)
    manifest = dict(config=config, runtime=runtime, cases=[])
    for spec in config['cases']:
        spec = dict(spec)
        label = spec.pop('label')
        problem = generate_problem(**spec)
        problem_path = save_problem(problem, root/label/'input')
        t = perf_counter()
        mu, nu = sinkhorn_log_sweep(problem, np.zeros_like(problem.a), np.zeros_like(problem.b))
        warm_sec = perf_counter()-t
        np.savez_compressed(root/label/'initial_state.npz', mu=mu, nu=nu)
        cfg = SolverConfig(**config['solver'], initial_mu=mu, initial_nu=nu)
        names = ('sinkhorn', 'theta_balance')
        case = dict(label=label, **spec, sha256=problem_fingerprint(problem),
                    warm_start_sec=warm_sec, runs={name: [] for name in names})
        for name in names:
            get_solver(name)(problem, cfg)  # discarded timing warm-up
        for rep in range(config['repeats']):
            for name in names if rep % 2 == 0 else names[::-1]:
                result = get_solver(name)(problem, cfg)
                path = root/label/f'{name}_{rep+1}'
                reference = dict(path=os.path.relpath(problem_path, path), sha256=case['sha256'])
                save_result(result, path, dict(solver=dict(name=name, **asdict(cfg)), problem=spec,
                    common_warm_start_included=False), reference, runtime)
                case['runs'][name].append(str(path.relative_to(root)))
                print(label, name, rep+1, result.converged, result.n_iter, result.elapsed_sec, result.stop_reason, flush=True)
                if label == 'smoke' and not result.converged:
                    raise RuntimeError(f'Smoke failed; saved at {path}')
        manifest['cases'].append(case)
        save_json(manifest, root/'manifest.json')
    analyze(root, manifest)
    print(root.resolve(), flush=True)


def study_problem(spec, config):
    """Reuse costs exactly; an independent RNG supplies the study marginals."""
    if spec.get('regression'):
        return generate_problem(**spec['generator_args'])
    n = spec['n']
    base = generate_problem(generator=spec['cost'], n=n, m=n,
                            seed=spec['seed'], epsilon=spec['epsilon'])
    rng = np.random.default_rng(spec['seed']+config['marginal_seed_offset'])

    def marginal():
        if spec['marginal'] == 'uniform':
            return np.full(n, 1/n)
        if spec['marginal'] == 'random_positive':
            x = rng.uniform(.1, 1, n)
            return x/x.sum()
        count = max(1, int(np.ceil(config['spiky_fraction']*n)))
        if count >= n:
            raise ValueError('Spiky marginals require at least one non-spike')
        x = np.full(n, (1-config['spiky_mass'])/(n-count))
        x[rng.choice(n, count, replace=False)] = config['spiky_mass']/count
        return x

    return OTProblem(marginal(), marginal(), base.M, base.epsilon, base.name,
        dict(base.metadata, study_marginal=spec['marginal'],
             normalization=config['normalization'], marginal_seed=spec['seed']+config['marginal_seed_offset']))


def study_cases(config):
    cases = {}

    def add(cost, n, eps, marginal, seed, block, methods, tol=None):
        key = f'{cost}_n{n}_e{eps:g}_{marginal}_s{seed}'
        label = key + ('_highprecision' if tol is not None else '')
        if label not in cases:
            cases[label] = dict(label=label, input_key=key, cost=cost, n=n,
                epsilon=eps, marginal=marginal, seed=seed, tol=tol or config['solver']['tol'],
                blocks=[], methods={}, regression=False)
        case = cases[label]
        if block not in case['blocks']:
            case['blocks'].append(block)
        for method in methods:
            case['methods'].setdefault(method, dict(runs=[], warmup=None, skip_reason=None))

    pilot = load_config(config['regression_config'])
    for original in pilot['cases']:
        spec = dict(original)
        label = 'regression_'+spec.pop('label')
        cases[label] = dict(label=label, input_key=label, cost=spec['generator'], n=spec['n'],
            epsilon=spec['epsilon'], seed=spec['seed'], marginal='pilot_dirichlet',
            tol=pilot['solver']['tol'], blocks=['smoke'], regression=True, generator_args=spec,
            methods={m: dict(runs=[], warmup=None, skip_reason=None) for m in ['sinkhorn', 'adaptive']})
    methods = ['sinkhorn', 'adaptive']
    a, bc, d = [config['blocks'][b] for b in ['A', 'BC', 'D']]
    for large in (False, True):
        for cost in config['costs']:
            for n in a['sizes']:
                if (n >= 1000) != large:
                    continue
                for eps in a['epsilons']:
                    for seed in config['seeds']:
                        add(cost, n, eps, a['marginal'], seed, 'A_large' if large else 'A_small', methods)
        if not large:
            for cost in config['costs']:
                for eps in bc['epsilons']:
                    for marginal in config['marginals']:
                        for seed in config['seeds']:
                            add(cost, bc['size'], eps, marginal, seed, 'BC',
                                methods+[f'fixed_{x:g}' for x in config['fixed_thetas']])
    for cost in config['costs']:
        for marginal in config['marginals']:
            add(cost, d['size'], d['epsilon'], marginal, d['seed'], 'D', methods, d['tol'])
    return list(cases.values())


def digest_arrays(*arrays):
    h = hashlib.sha256()
    for value in arrays:
        a = np.asarray(value, dtype='<f8')
        h.update(str(a.shape).encode('ascii'))
        h.update(a.tobytes())
    return h.hexdigest()


def solver_source_hash():
    h = hashlib.sha256()
    for path in ['types.py', 'solvers/theta_balance.py', 'solvers/sinkhorn.py',
                 'solvers/base.py', 'policies/stepsize.py']:
        h.update((Path('src/sinkhorn_lmflow')/path).read_bytes())
    return h.hexdigest()


def study_solver_config(config, case, method):
    return SolverConfig(**dict(config['solver'], tol=case['tol'],
        balance_theta_fixed=float(method.split('_')[1]) if method.startswith('fixed_') else None))


def run_status(result):
    return 'converged' if result.converged else ('timeout' if result.stop_reason == 'timeout'
        else 'max_iter' if result.stop_reason == 'max_iter' else 'failed')


def unexecuted_status(task):
    return 'skipped_repeat' if task['skip_reason'] else 'pending'


def checkpoint(root, manifest):
    # Atomic index replacement; an interrupted completed run is recovered below.
    temporary = root/'manifest.tmp'
    save_json(manifest, temporary)
    # Windows scanners/readers can briefly lock the destination. Keep the
    # fully written temporary index and retry only this sharing failure.
    for attempt in range(8):
        try:
            temporary.replace(root/'manifest.json')
            break
        except PermissionError:
            if attempt == 7:
                raise
            sleep(.05 * (attempt + 1))


def run_study(config, root=None):
    start = perf_counter()
    warmed = set()
    if root is None:
        root = create_run_dir('results', 'theta_balance_study')
        (root/'scratch').mkdir()
        save_config(config, root/'study_config.yaml')
        manifest = dict(mode='systematic', config=config, runtime=runtime_metadata(42),
            source_hash=solver_source_hash(), cases=study_cases(config), sessions=[], io_sec=0.)
        checkpoint(root, manifest)
    else:
        root = Path(root)
        manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        if manifest['source_hash'] != solver_source_hash():
            raise ValueError('Solver source changed; refusing to mix timings on resume')
        if config is not None:
            saved = manifest['config']
            if {k: v for k, v in config.items() if k != 'budget_sec'} != {k: v for k, v in saved.items() if k != 'budget_sec'}:
                raise ValueError('Resume only permits changing budget_sec, not the experimental protocol')
            manifest['config']['budget_sec'] = config['budget_sec']
            save_config(manifest['config'], root/'study_config.yaml')
        config = manifest['config']
        for case in manifest['cases']:
            for method, task in case['methods'].items():
                for entry in task['runs']:
                    cfg = study_solver_config(config, case, method)
                    expected = hashlib.sha256(json.dumps(dict(config=asdict(cfg), problem=case['sha256'],
                        initial=case['initial_sha256'], source=manifest['source_hash']), sort_keys=True).encode()).hexdigest()
                    path = root/entry['path']
                    if entry['config_sha256'] != expected or not all((path/f).exists() for f in
                        ['history.csv', 'summary.json', 'final_state.npz', 'study_run.json']):
                        raise ValueError(f'Invalid completed run on resume: {path}')
    print('STUDY', root.resolve(), flush=True)
    max_n = max(c['n'] for c in manifest['cases'])
    manifest['estimated_peak_dense_bytes'] = 12*8*max_n*max_n
    print(f'Conservative dense-array estimate: {12*8*max_n*max_n/2**20:.1f} MiB; serial execution', flush=True)
    deadline = start+config['budget_sec']
    session = dict(runtime=runtime_metadata(42), budget_sec=config['budget_sec'],
                   solver_sec=0., warmup_sec=0., status='running')
    manifest['sessions'].append(session)
    checkpoint(root, manifest)

    def budget_available():
        # Reserve one complete run limit so a run never gets a different limit.
        return perf_counter()+config['solver']['wall_time_limit_sec'] < deadline

    stopped = False
    # Explicit stage loop keeps BC fixed-theta work after ALL A_small work.
    for stage in ['smoke', 'A_small', 'BC', 'A_large', 'D']:
        print('STAGE', stage, flush=True)
        for case in manifest['cases']:
            if stage not in case['blocks']:
                continue
            methods = list(case['methods'])
            if stage == 'A_small':
                methods = [m for m in methods if m in ('sinkhorn', 'adaptive')]
            if all(len(case['methods'][m]['runs']) >= config['repeats'] or case['methods'][m]['skip_reason'] for m in methods):
                continue
            if not budget_available():
                stopped = True
                break
            input_dir = root/'inputs'/case['input_key']
            t = perf_counter()
            new_input = not (input_dir/'problem.npz').exists()
            if not new_input:
                problem = load_problem(input_dir)
                with np.load(input_dir/'initial_state.npz', allow_pickle=False) as state:
                    mu, nu = state['mu'], state['nu']
                initial_meta = json.loads((input_dir/'initial_meta.json').read_text())
            else:
                problem = study_problem(case, config)
                save_problem(problem, input_dir)
                ts = perf_counter()
                mu, nu = sinkhorn_log_sweep(problem, np.zeros_like(problem.a), np.zeros_like(problem.b))
                initial_meta = dict(warm_start_sec=perf_counter()-ts, initial_sha256=digest_arrays(mu, nu))
                np.savez_compressed(input_dir/'initial_state.npz', mu=mu, nu=nu)
                save_json(initial_meta, input_dir/'initial_meta.json')
            manifest['io_sec'] += perf_counter()-t-(initial_meta['warm_start_sec'] if new_input else 0.)
            case.update(sha256=problem_fingerprint(problem), **initial_meta,
                cost_sha256=digest_arrays(problem.M), marginal_sha256=digest_arrays(problem.a, problem.b),
                raw_cost_max=problem.metadata['raw_cost_max'])
            if digest_arrays(mu, nu) != case['initial_sha256']:
                raise ValueError('Saved initial potentials fingerprint mismatch')
            for rep in range(config['repeats']):
                order = methods if rep % 2 == 0 else methods[::-1]
                for method in order:
                    task = case['methods'][method]
                    if len(task['runs']) > rep or task['skip_reason']:
                        continue
                    if not budget_available():
                        stopped = True
                        break
                    cfg = study_solver_config(config, case, method)
                    cfg_hash = hashlib.sha256(json.dumps(dict(config=asdict(cfg), problem=case['sha256'],
                        initial=case['initial_sha256'], source=manifest['source_hash']), sort_keys=True).encode()).hexdigest()
                    cfg = replace(cfg, initial_mu=mu, initial_nu=nu)
                    solver = get_solver('sinkhorn' if method == 'sinkhorn' else 'theta_balance')
                    path = root/'runs'/case['label']/f'{method}_{rep+1}'
                    marker = path/'study_run.json'
                    if marker.exists():
                        entry = json.loads(marker.read_text())
                        if entry['config_sha256'] != cfg_hash:
                            raise ValueError(f'Configuration fingerprint mismatch: {path}')
                        result = load_result(path)
                    else:
                        # Warm each method again in a resumed process, never include it in statistics.
                        warmkey = (case['label'], method)
                        if warmkey not in warmed:
                            warm = solver(problem, replace(cfg, max_iter=config['warmup_max_iter'],
                                wall_time_limit_sec=config['warmup_wall_sec']))
                            task['warmup'] = dict(n_iter=warm.n_iter, elapsed_sec=warm.elapsed_sec,
                                                 stop_reason=warm.stop_reason)
                            session['warmup_sec'] += warm.elapsed_sec
                            warmed.add(warmkey)
                        result = solver(problem, cfg)
                        session['solver_sec'] += result.elapsed_sec
                        ts = perf_counter()
                        reference = dict(path=os.path.relpath(input_dir/'problem.npz', path),
                            sha256=case['sha256'], initial_sha256=case['initial_sha256'])
                        save_result(result, path, dict(solver=asdict(cfg), study_method=method,
                            config_sha256=cfg_hash, problem_seed=case['seed'], common_warm_start_included=False),
                            reference, dict(session['runtime'], seed=case['seed']))
                        entry = dict(path=str(path.relative_to(root)), config_sha256=cfg_hash,
                            status=run_status(result), repeat=rep+1, stop_reason=result.stop_reason)
                        save_json(entry, marker)
                        manifest['io_sec'] += perf_counter()-ts
                    task['runs'].append(entry)
                    if not result.converged:
                        task['skip_reason'] = 'remaining repeats not run after '+result.stop_reason
                    checkpoint(root, manifest)
                    print(case['label'], method, rep+1, entry['status'], result.n_iter,
                          f'{result.elapsed_sec:.3f}s', flush=True)
                    if stage == 'smoke' and not result.converged:
                        stopped = True
                        session['status'] = 'regression_failed'
                        break
                if stopped:
                    break
            if stopped:
                break
        checkpoint(root, manifest)
        if stopped:
            break
    session['elapsed_sec'] = perf_counter()-start
    if session['status'] == 'running':
        session['status'] = 'budget_exhausted' if stopped else 'complete'
    checkpoint(root, manifest)
    analyze_study(root, manifest)
    print('FINISHED', session['status'], root.resolve(), flush=True)


def markdown_table(frame):
    def fmt(x):
        if pd.isna(x):
            return '—'
        return f'{x:.4g}' if isinstance(x, (float, np.floating)) else str(x).replace('|', '/')
    return '\n'.join(['| '+' | '.join(map(str, frame.columns))+' |',
        '| '+' | '.join(['---']*len(frame.columns))+' |']+
        ['| '+' | '.join(fmt(x) for x in row)+' |' for row in frame.itertuples(index=False, name=None)])


def median_or_nan(values):
    values = values.dropna()
    return float(values.median()) if len(values) else np.nan


def analyze_study(root, manifest):
    """Single report and six aggregate figures, entirely from saved artifacts."""
    plot_start = perf_counter()
    config = manifest['config']
    rows, thresholds, diagnostic_rows, representatives = [], [], [], {}
    representative_status = {}
    bands = {'early_R>1e-2': (1e-2, np.inf), 'middle_1e-6<R<=1e-2': (1e-6, 1e-2),
             'late_R<=1e-6': (-np.inf, 1e-6)}
    for case in manifest['cases']:
        for method, task in case['methods'].items():
            for rep in range(1, config['repeats']+1):
                row = {k: case[k] for k in ['label', 'cost', 'n', 'epsilon', 'marginal', 'seed', 'tol', 'regression']}
                row.update(method=method, repeat=rep, blocks=','.join(case['blocks']),
                    problem_sha256=case.get('sha256'), initial_sha256=case.get('initial_sha256'),
                    cost_sha256=case.get('cost_sha256'), marginal_sha256=case.get('marginal_sha256'),
                    raw_cost_max=case.get('raw_cost_max'), common_initial_sec=case.get('warm_start_sec'))
                entry = next((e for e in task['runs'] if e['repeat'] == rep), None)
                history = pd.DataFrame()
                if entry is None:
                    row.update(status=unexecuted_status(task),
                        stop_reason=task['skip_reason'] or 'not executed', converged=False)
                else:
                    path = root/entry['path']
                    summary = json.loads((path/'summary.json').read_text(encoding='utf-8'))
                    history = pd.read_csv(path/'history.csv')
                    row.update({k: v for k, v in summary.items() if not isinstance(v, (list, dict))})
                    row.update(status=entry['status'], path=entry['path'], config_sha256=entry['config_sha256'])
                    row['dense_pass_proxy'] = sum(summary[k] for k in
                        ['plan_build_count', 'reduction_count', 'logsumexp_count', 'p_matvec_count'])
                    row['failure_events'] = json.dumps(summary.get('degeneration_events', []))
                    steps = history[history.iteration > 0]
                    for band, (lo, hi) in bands.items():
                        selected = steps[(steps.pre_residual_l1 > lo) & (steps.pre_residual_l1 <= hi)]
                        diag = dict(label=case['label'], method=method, cost=case['cost'], n=case['n'],
                            marginal=case['marginal'], epsilon=case['epsilon'], band=band, steps=len(selected))
                        for key in ['balance_theta_free', 'balance_theta', 'balance_effective', 'step_size',
                                    'balance_eta', 'balance_predicted_ratio_l2', 'balance_actual_ratio_l2']:
                            values = selected[key].dropna()
                            for q, suffix in [(0, 'min'), (.1, 'q10'), (.5, 'median'), (.9, 'q90'), (1, 'max')]:
                                value = values.quantile(q) if len(values) else np.nan
                                row[f'{band}_{key}_{suffix}'] = value
                                diag[f'{key}_{suffix}'] = value
                        for key, values in [('upper_clip', selected.balance_theta_free > config['solver']['balance_theta_max']),
                                            ('lower_clip', selected.balance_theta_free < 0),
                                            ('backtracked', selected.step_size < 1),
                                            ('fallback', selected.phase == 'sinkhorn_fallback')]:
                            value = values.mean() if len(selected) else np.nan
                            if key.endswith('clip') and method != 'adaptive':
                                value = np.nan
                            row[f'{band}_{key}_rate'] = value
                            diag[key+'_rate'] = value
                        if rep == 1 and not case['regression'] and case['tol'] == config['solver']['tol']:
                            diagnostic_rows.append(diag)
                    representative_size = case['n'] == 500 or (case['n'] == 100 and
                        case['cost'] == 'random_dense' and case['epsilon'] == .001)
                    if rep == 1 and case['seed'] == 42 and representative_size and case['tol'] == 1e-8:
                        if case['marginal'] in ['uniform', 'spiky']:
                            representatives[(case['label'], method)] = history
                            representative_status[(case['label'], method)] = summary['stop_reason']
                rows.append(row)
                for target in config['thresholds']+([case['tol']] if case['tol'] < min(config['thresholds']) else []):
                    hit = history[history.residual_l1 <= target] if len(history) else history
                    th = {k: row[k] for k in ['label', 'cost', 'n', 'epsilon', 'marginal', 'seed', 'tol', 'method', 'repeat', 'status', 'regression']}
                    th.update(target=target, reached=bool(len(hit)), seconds=np.nan, iteration=np.nan, dense_pass_proxy=np.nan)
                    if len(hit):
                        th.update(seconds=hit.iloc[0].elapsed_sec, iteration=hit.iloc[0].iteration,
                                  dense_pass_proxy=hit.iloc[0].dense_pass_proxy)
                    thresholds.append(th)
    raw = pd.DataFrame(rows)
    th = pd.DataFrame(thresholds)
    raw.to_csv(root/'summary.csv', index=False)
    th.to_csv(root/'thresholds.csv', index=False)
    records = []
    for (label, method), group in raw.groupby(['label', 'method'], sort=False):
        observed = group[~group.status.isin(['pending', 'skipped_repeat'])]
        item = group.iloc[0][['label', 'cost', 'n', 'epsilon', 'marginal', 'seed', 'tol', 'regression', 'blocks']].to_dict()
        item.update(method=method, completed=len(observed), successes=int(observed.converged.sum()),
                    success=bool(len(observed) == config['repeats'] and observed.converged.all()),
                    status=','.join(sorted(group.status.unique())))
        for field in ['elapsed_sec', 'n_iter', 'dense_pass_proxy', 'residual_l1', 'backtrack_count',
                      'balance_fallback_count', 'safeguard_count']:
            item[field] = median_or_nan(observed[field]) if field in observed else np.nan
        item['seconds_min'] = observed.elapsed_sec.min() if len(observed) else np.nan
        item['seconds_max'] = observed.elapsed_sec.max() if len(observed) else np.nan
        records.append(item)
    instances = pd.DataFrame(records)
    main = instances[(~instances.regression) & (instances.tol == config['solver']['tol'])]
    baseline = main[main.method == 'sinkhorn'].set_index('label')
    adaptive = main[main.method == 'adaptive'].set_index('label')
    pairs = baseline[['cost', 'n', 'epsilon', 'marginal', 'seed', 'success', 'completed', 'elapsed_sec', 'n_iter', 'dense_pass_proxy']].join(
        adaptive[['success', 'completed', 'elapsed_sec', 'n_iter', 'dense_pass_proxy']], rsuffix='_adaptive')
    pairs['ratio'] = np.where(pairs.success & pairs.success_adaptive,
                              pairs.elapsed_sec/pairs.elapsed_sec_adaptive, np.nan)
    pairs['win'] = pairs.ratio > 1
    completed_pairs = pairs[pairs.ratio.notna()]
    repeated = instances[instances.success].copy()
    spread = repeated.seconds_max/repeated.seconds_min
    coverage = raw.groupby(['status']).size().rename('正式重复次数').reset_index()
    coverage_blocks = []
    for label, block, fixed in [('smoke', 'smoke', False), ('A_small', 'A_small', False),
            ('B', 'BC', False), ('C', 'BC', True), ('A_large', 'A_large', False), ('D', 'D', False)]:
        select = instances.blocks.str.split(',').apply(lambda values: block in values)
        select &= instances.method.str.startswith('fixed_') if fixed else instances.method.isin(['sinkhorn', 'adaptive'])
        group = instances[select]
        coverage_blocks.append(dict(block=label, 计划组合=len(group), 已启动=int((group.completed > 0).sum()),
            三次收敛=int(group.success.sum()), 未收敛=int(group.status.str.contains('timeout|max_iter|failed').sum()),
            计时未齐=int(((group.completed > 0) & group.status.str.contains('pending')).sum()),
            尚未启动=int((group.completed == 0).sum())))
    runtime = manifest['runtime']
    relroot = Path(os.path.relpath(root, Path('docs'))).as_posix()
    lines = ['# θ 系统实验与固定 θ 消融', '',
        '本报告只使用已保存的本轮实测数据；三次计时先在实例内取中位数，再跨实例汇总。', '',
        f'结果目录：[{root.name}]({relroot}/)；[总体 CSV]({relroot}/summary.csv)、'
        f'[阈值 CSV]({relroot}/thresholds.csv)、[共享索引]({relroot}/manifest.json)。', '',
        '## 实际覆盖与结论', '', markdown_table(coverage), '', markdown_table(pd.DataFrame(coverage_blocks)), '',
        '块覆盖按问题/方法组合计数。A/B 的重叠行引用同一批实测结果，不重复执行或重复计入总体样本数。', '',
        f'计划 {len(instances)} 个问题/方法组合（含原 pilot 回归、高精度）；已执行 '
        f'{int((instances.completed > 0).sum())} 个，三次均收敛 {int(instances.success.sum())} 个。'
        '未执行不计为失败；确定性失败、max_iter 或 timeout 后不机械补齐其余重复。', '',
        f'主实验 Sinkhorn/自适应计划 {len(pairs)} 个独立问题；主对照已执行 '
        f'{int(((pairs.completed > 0) | (pairs.completed_adaptive > 0)).sum())} 个，其中双方三次均收敛的 '
        f'{len(completed_pairs)} 个问题中，自适应实际总时间获胜 {int(completed_pairs.win.sum())} 个。'
        + (f'这些配对时间比的中位数为 {completed_pairs.ratio.median():.3f}×，范围 '
           f'[{completed_pairs.ratio.min():.3f}, {completed_pairs.ratio.max():.3f}]×。' if len(completed_pairs) else ''), '',
        '以下精确时间比仅包含双方完整收敛的配对，成功数另列，不能据此掩盖失败或未执行。', '']
    if len(spread):
        lines += [f'三次完整收敛的组合中，有 {int((spread > 1.2).sum())}/{len(spread)} 个的最大/最小耗时比超过1.2，'
            f'跨度最大为 {spread.max():.3g}。原始三次时间保留在 CSV；接近1的速度比需结合这一波动解读。', '']
    if 'elapsed_sec' in raw:
        solver_seconds = raw.elapsed_sec.sum()
        fixed_seconds = raw.loc[raw.method.str.startswith('fixed_'), 'elapsed_sec'].sum()
        if solver_seconds:
            lines += [f'已记录的正式求解累计 {solver_seconds:.1f}s，其中固定 θ 占 {fixed_seconds:.1f}s '
                f'（{fixed_seconds/solver_seconds:.1%}）；这些时间不含预热、公共初始化和文件读写。', '']
    review = manifest.get('review', {})
    if review.get('completed_runs') == int((~raw.status.isin(['pending', 'skipped_repeat'])).sum()):
        lines += review.get('paragraphs', []) + ['']
    group_rows = []
    for key, group in pairs.groupby(['cost', 'n', 'epsilon']):
        # A is uniform only; marginal slices are reported separately below.
        g = group[group.marginal == 'uniform']
        if len(g):
            group_rows.append(dict(cost=key[0], n=key[1], epsilon=key[2], problems=len(g),
                状态='未执行' if not (g.completed > 0).any() else '已执行',
                Sinkhorn成功=int(g.success.sum()), 自适应成功=int(g.success_adaptive.sum()),
                配对数=int(g.ratio.notna().sum()), 时间比中位数=median_or_nan(g.ratio),
                S秒=median_or_nan(g.loc[g.success, 'elapsed_sec']), A秒=median_or_nan(g.loc[g.success_adaptive, 'elapsed_sec_adaptive']),
                S迭代=median_or_nan(g.loc[g.success, 'n_iter']), A迭代=median_or_nan(g.loc[g.success_adaptive, 'n_iter_adaptive']),
                Sproxy=median_or_nan(g.loc[g.success, 'dense_pass_proxy']), Aproxy=median_or_nan(g.loc[g.success_adaptive, 'dense_pass_proxy_adaptive'])))
    scale = pd.DataFrame(group_rows)
    lines += ['## 规模与 ε（A，均匀边际）', '', markdown_table(scale), '',
              '## 边际与固定 θ（B/C）', '']
    bc = main[(main.n == config['blocks']['BC']['size']) & main.epsilon.isin(config['blocks']['BC']['epsilons'])]
    fixed_rows = []
    for method, g in bc.groupby('method', sort=False):
        matched = g.set_index('label').join(baseline[['success', 'elapsed_sec']], rsuffix='_base')
        ok = matched.success & matched.success_base
        ratios = matched.loc[ok, 'elapsed_sec_base']/matched.loc[ok, 'elapsed_sec']
        fixed_rows.append(dict(method=method, 计划问题=len(g), 已执行=int((g.completed > 0).sum()),
            三次收敛=int(g.success.sum()), 有效配对=len(ratios),
            时间比中位数=median_or_nan(ratios), 获胜数=int((ratios > 1).sum()),
            成功耗时秒=median_or_nan(g.loc[g.success, 'elapsed_sec']), 成功迭代中位数=median_or_nan(g.loc[g.success, 'n_iter']),
            成功proxy中位数=median_or_nan(g.loc[g.success, 'dense_pass_proxy'])))
    fixed_table = pd.DataFrame(fixed_rows)
    lines += [markdown_table(fixed_table), '',
        '固定 θ=1 是 GS 整步核对及实现开销参照，不作为独立算法优势。每个固定值都保留成绩；'
        '固定的是提议 θ，Armijo 仍可改变 h，因此 θ=2/4/8 的 hθ 可能相同。没有逐题挑最优后包装成新方法。'
        '各方法的有效配对子集可能不同，须结合已执行数、完整收敛数判断稳定性。', '']
    vs_adaptive = []
    for method, group in bc[bc.method.str.startswith('fixed_')].groupby('method', sort=False):
        matched = group.set_index('label').join(adaptive[['success', 'elapsed_sec']], rsuffix='_adaptive')
        ok = matched.success & matched.success_adaptive
        ratio = matched.loc[ok, 'elapsed_sec_adaptive']/matched.loc[ok, 'elapsed_sec']
        vs_adaptive.append(dict(method=method, 完整配对数=len(ratio), 固定值更快的题数=int((ratio > 1).sum()),
            自适应除以固定的时间比=median_or_nan(ratio), 最小比=ratio.min(), 最大比=ratio.max()))
    lines += ['额外选 θ 的代价是否值得，直接看自适应与各个全局固定值的配对：大于1表示固定值更快。', '',
              markdown_table(pd.DataFrame(vs_adaptive)), '']
    marginal_rows = []
    for key, g in pairs[pairs.n == 500].groupby(['cost', 'epsilon', 'marginal']):
        marginal_rows.append(dict(cost=key[0], epsilon=key[1], marginal=key[2],
            状态='未执行' if not (g.completed > 0).any() else '已执行',
            计划=len(g), S成功=int(g.success.sum()), A成功=int(g.success_adaptive.sum()),
            配对=int(g.ratio.notna().sum()), 时间比=median_or_nan(g.ratio)))
    lines += [markdown_table(pd.DataFrame(marginal_rows)), '', '## 精度与时间口径', '',
        '1e-4/1e-6/1e-8 为完整轨迹首次命中时间，不是独立求解总时间。'
        'D 的协议设定为从同一公共初值重新完整运行至 1e-10（不是把后半段耗时当总时间）；共享输入/初值，单独标记 highprecision。实际执行状态如下。', '']
    threshold_instances = th[(~th.regression) & (th.tol == 1e-8)].groupby(
        ['label', 'method', 'target']).agg(hits=('reached', 'sum'), seconds=('seconds', 'median')).reset_index()
    tolerance_rows = []
    for target, g in threshold_instances.groupby('target'):
        pp = g.pivot(index='label', columns='method', values='seconds')
        hh = g.pivot(index='label', columns='method', values='hits')
        if 'sinkhorn' in pp and 'adaptive' in pp:
            ok = (hh.sinkhorn == config['repeats']) & (hh.adaptive == config['repeats'])
            rr = pp.loc[ok, 'sinkhorn']/pp.loc[ok, 'adaptive']
            tolerance_rows.append(dict(target=target, S至少一次命中=int((hh.sinkhorn > 0).sum()),
                A至少一次命中=int((hh.adaptive > 0).sum()), S三次命中=int((hh.sinkhorn == config['repeats']).sum()),
                A三次命中=int((hh.adaptive == config['repeats']).sum()), 配对数=len(rr), 时间比中位数=median_or_nan(rr)))
    lines += [markdown_table(pd.DataFrame(tolerance_rows)), '',
        markdown_table(instances[instances.tol < 1e-8][['cost', 'marginal', 'method', 'status', 'elapsed_sec', 'n_iter', 'residual_l1']]), '',
        '## θ 与保护机制', '',
        '每步 θ/h 与更新前 R 对齐；残差分段为 R>1e-2、1e-6<R≤1e-2、R≤1e-6。'
        'summary.csv 保存每条轨迹各段的 min/q10/median/q90/max 和截断/回溯/回退比例。'
        '下表先逐实例计算再取中位数，每个实例只用首条正式轨迹，避免把计时重复当独立样本。', '']
    diags = pd.DataFrame(diagnostic_rows)
    if len(diags):
        ad = diags[(diags.method == 'adaptive') & (diags.steps > 0)]
        cols = ['balance_theta_free_q10', 'balance_theta_free_median', 'balance_theta_free_q90',
                'balance_theta_median', 'balance_effective_median', 'upper_clip_rate',
                'lower_clip_rate', 'backtracked_rate', 'fallback_rate']
        band_table = ad.groupby('band')[cols].median().reindex(list(bands))
        band_table.insert(0, '实例数', ad.groupby('band').size())
        band_table = band_table.rename(columns={
            'balance_theta_free_q10': 'free q10',
            'balance_theta_free_median': 'free q50',
            'balance_theta_free_q90': 'free q90',
            'balance_theta_median': 'θ中位数', 'balance_effective_median': 'hθ中位数',
            'upper_clip_rate': '上截断率', 'lower_clip_rate': '下截断率',
            'backtracked_rate': 'h<1比例', 'fallback_rate': '回退率'})
        lines += [markdown_table(band_table.reset_index()), '',
            '比例以0～1表示。各残差段参与实例数不同，尚未到达后期的实例不进入后期统计。', '']
        if len(ad):
            lines += [f'所有已记录主实验方向的 θ_free 范围为 '
                f'[{ad.balance_theta_free_min.min():.6g}, {ad.balance_theta_free_max.max():.6g}]，'
                f'截断 θ 范围 [{ad.balance_theta_min.min():.6g}, {ad.balance_theta_max.max():.6g}]，'
                f'非回退 hθ 范围 [{ad.balance_effective_min.min():.6g}, {ad.balance_effective_max.max():.6g}]。', '']
    lines += ['预测值是 ||r−hθs||₂/||r||₂，真实值是 ||r_next||₂/||r||₂；η 对应 h=1 的线性预测。'
        '这些量不能直接解释 L1 停止指标。固定分支不计算 qS、s、θ_free、η 或预测比，空值表示不适用。'
        '回退的 hθ 为空；负 θ_free 原样保留；失败试步原因保存在 summary 的事件记录。', '',
        '## 失败、超时、最慢与未执行', '']
    failed = raw[(~raw.status.isin(['converged', 'pending', 'skipped_repeat']))]
    lines += [markdown_table(failed[['label', 'method', 'repeat', 'status', 'stop_reason', 'elapsed_sec', 'residual_l1']])
              if len(failed) else '目前无已执行运行失败或超时。', '',
        '耗时最长的十个已执行问题/方法组合（实例内中位数）：', '',
        markdown_table(instances[instances.completed > 0].nlargest(10, 'elapsed_sec')[
            ['label', 'method', 'status', 'elapsed_sec', 'n_iter', 'residual_l1']]), '',
        f'尚未执行正式重复数：{int((raw.status == "pending").sum())}；具体标签、方法和重复号见 summary.csv 的 pending 行。', '',
        '## 配置、环境与复现', '',
        f'环境 `{runtime["python_executable"]}`；Python {runtime["python"].split()[0]}，NumPy {runtime["numpy"]}，SciPy {runtime["scipy"]}。'
        f'CPU float64，BLAS 单线程；基础 commit `{runtime["git_commit"]}`，源码指纹 `{manifest["source_hash"]}`。', '',
        f'最大尺寸 {max(c["n"] for c in manifest["cases"])}，按 12 个稠密 float64 数组保守估计 '
        f'{manifest["estimated_peak_dense_bytes"]/2**20:.1f} MiB（非峰值实测，不含解释器/绘图）。串行运行，不保存每步 P。', '',
        '成本复用原生成器并除以原最大值，原尺度、成本/边际/问题/初值指纹保存于 CSV/manifest。'
        'uniform、独立 Uniform(.1,1) 归一化、2% 位置分配90%质量的 spiky 为本轮边际；'
        '与旧 pilot 的 Dirichlet 边际不同，回归块单列。seeds=42/123/2026。', '',
        f'主容差 {config["solver"]["tol"]:g}、max_iter={config["solver"]["max_iter"]}、'
        f'每次 {config["solver"]["wall_time_limit_sec"]} 秒迭代边界上限，本会话预算 {config["budget_sec"]} 秒；'
        '预算尾部不足一次完整上限时不启动新运行。θ_max=10，h初值1，Armijo=1e-4，收缩.5，最多60次回溯。'
        '每个方法20步/2秒有界预热，正式三次，方法顺序交替；重启进程重新预热。双方同一公共 GS 初值，初始化单列。', '',
        '总时间含内部初始化、方向、全部试步、正性/Armijo、列更新、回退、共同监控、最终诊断；不含公共 GS、保存和绘图。'
        '两边逐步记录且关闭谱/SVD。成绩是当前统一监控框架内的性能。'
        'diagnostic_sec 继承旧格式，只累计谱诊断，本轮为0不代表共同监控没有成本。'
        'dense-pass proxy 为 P构造+全矩阵归约+LSE+P向量乘的未加权计数，不是 FLOPs 或等价 matvec；'
        '最终 transport objective、向量算术和分配不计入 proxy，但计入总时间。', '',
        f'累计输入处理及逐次结果写入时间约 {manifest["io_sec"]:.3f}s（不含索引检查点写入，另列）；本次分析/绘图耗时见 manifest.analysis_sec。'
        '全体运行保留配置和运行环境；同问题只存一份输入。断点续跑按问题、初值、solver配置和源码指纹校验。', '',
        '主运行命令：', '', '```powershell',
        '.\\scripts\\run_ot.cmd scripts/run_theta_balance_pilot.py --config configs/theta_balance_study.yaml', '```', '',
        '同一目录续跑 / 仅分析：', '', '```powershell',
        f'.\\scripts\\run_ot.cmd scripts/run_theta_balance_pilot.py --resume {root.as_posix()}',
        f'.\\scripts\\run_ot.cmd scripts/run_theta_balance_pilot.py --analyze {root.as_posix()}', '```', '',
        '续跑默认使用索引中的原配置。需要更改本次预算时，只修改同一 YAML 的 budget_sec，'
        '并在 --resume 命令后添加 --config configs/theta_balance_study.yaml；数值协议改动会被拒绝。', '',
        '没有矩形、真实数据、GPU 或额外文献基线。本轮不改变算法及保护规则，不把相关性写成普遍定理。', '']
    if manifest.get('audit'):
        audit = manifest['audit']
        lines += ['## 保存结果复核', '',
            f'已复核 {audit["runs"]} 份正式结果、{audit["distinct_saved_inputs"]} 份共享输入；'
            f'{audit["fixed_runs"]} 份固定 θ 运行的 p_matvec_count 全为0，选参诊断全为空。'
            f'首轮正式结果最终列 L1 残差最大 {audit["max_final_column_l1_first_repeats"]:.3g}。'
            '前后残差对齐、成本/边际切片不变性和原 pilot 迭代数均通过检查。', '',
            f'元数据修正：{audit["seed_metadata_corrected"]} 份 runtime_metadata 最初继承了总控 seed=42，'
            '现已按共享问题的实际 seed 更正；问题数据、轨迹、状态和计时均未改动。', '']
    if review.get('completed_runs') == int((~raw.status.isin(['pending', 'skipped_repeat'])).sum()):
        lines += review.get('validation_and_changes', []) + ['']
    lines += ['## 总体图', '',
        '代表曲线为 seed=42 的首条正式轨迹，展示随机成本 n=100/ε=.001 与 n=500/ε=.01，'
        '一维成本 n=500/ε=.01 与 .001，均为 uniform。此处按已观测的快速、缓慢及未收敛情形选图，'
        '总体统计仍保留全部实例。未执行的方法不补曲线。时间统计先取实例内三次中位数。'
        '保护率柱图先按实例计算比例再求均值，正文分段表取实例中位数。', '']
    # Six aggregate figures, never a per-case gallery.
    figdir = root/'figures'
    figdir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, cost in zip(axes, config['costs']):
        g = pairs[(pairs.cost == cost) & (pairs.marginal == 'uniform')]
        matrix = g.groupby(['n', 'epsilon']).ratio.median().unstack().reindex(index=config['blocks']['A']['sizes'], columns=config['blocks']['A']['epsilons'])
        ax.imshow(np.log2(matrix.clip(lower=.1, upper=10)), cmap='RdYlGn', vmin=-2, vmax=2, aspect='auto')
        for i, n in enumerate(matrix.index):
            for j, eps in enumerate(matrix.columns):
                cell = g[(g.n == n) & (g.epsilon == eps)]
                value = matrix.loc[n, eps]
                label = f'{value:.2f}x' if pd.notna(value) else 'no pair'
                counts = (f'S {cell.success.sum()}/{(cell.completed > 0).sum()}  '
                          f'A {cell.success_adaptive.sum()}/{(cell.completed_adaptive > 0).sum()}')
                ax.text(j, i, label+'\n'+counts+f'\nplanned {len(cell)}', ha='center', va='center', fontsize=8)
        ax.set(xticks=range(len(matrix.columns)), xticklabels=matrix.columns,
               yticks=range(len(matrix.index)), yticklabels=matrix.index, xlabel='epsilon', ylabel='n', title=cost)
    fig.suptitle('Paired total-time ratio S/adaptive; successes/executed instances, planned counts shown')
    fig.tight_layout()
    save_figure(fig, figdir/'01_scale_epsilon')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for cost in config['costs']:
        for marginal in config['marginals']:
            g = pairs[(pairs.cost == cost) & (pairs.marginal == marginal) & (pairs.n == 500)]
            med = g.groupby('epsilon').ratio.median().dropna()
            if len(med):
                axes[0].plot(med.index, med.values, 'o-', label=f'{cost}/{marginal}')
    tt = pd.DataFrame(tolerance_rows)
    if len(tt):
        axes[1].plot(tt.target, tt['时间比中位数'], 'o-')
        for _, rr in tt.iterrows():
            axes[1].annotate(f'n={int(rr["配对数"])}', (rr.target, rr['时间比中位数']))
    for ax in axes:
        ax.set_xscale('log')
        ax.axhline(1, color='gray', ls='--')
        ax.set_ylabel('Median paired S/adaptive time ratio')
        ax.grid(alpha=.2)
    axes[0].set_xlabel('epsilon (n=500)')
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel('First-hit residual target (main trajectories)')
    fig.tight_layout()
    save_figure(fig, figdir/'02_marginal_precision')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    order = ['sinkhorn', 'adaptive']+[f'fixed_{x:g}' for x in config['fixed_thetas']]
    ft = fixed_table.set_index('method').reindex(order)
    axes[0].bar(order, ft['时间比中位数'])
    axes[0].axhline(1, color='gray', ls='--')
    for i, method in enumerate(order):
        value = ft.loc[method, '时间比中位数']
        axes[0].text(i, value+.04 if pd.notna(value) else .04,
            f'n={int(ft.loc[method, "有效配对"])}' if pd.notna(value) else 'no pair',
            ha='center', fontsize=8)
    axes[0].set_ylim(0, max(1., ft['时间比中位数'].max())*1.17)
    axes[0].set_ylabel('Median paired S/method total-time ratio')
    axes[1].bar(order, ft['计划问题'], color='lightgray', label='planned')
    axes[1].bar(order, ft['已执行'], color='orange', label='executed')
    axes[1].bar(order, ft['三次收敛'], color='seagreen', label='3/3 converged')
    axes[1].legend()
    for ax in axes:
        ax.set(xticks=range(len(order)), xticklabels=order, xlim=(-.6, len(order)-.4))
        ax.tick_params(axis='x', rotation=35)
    fig.suptitle('All prespecified fixed proposals; B/C only, no per-case selection')
    fig.tight_layout()
    save_figure(fig, figdir/'03_fixed_theta')

    # Explicit display choices, including an iteration-limit case; all runs remain in tables.
    chosen = ['random_dense_n100_e0.001_uniform_s42', 'random_dense_n500_e0.01_uniform_s42',
              '1d_squared_n500_e0.01_uniform_s42', '1d_squared_n500_e0.001_uniform_s42']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, label in zip(axes.flat, chosen):
        plotted = False
        for method in order:
            h = representatives.get((label, method))
            if h is not None:
                plotted = True
                state = representative_status[(label, method)]
                legend = method if state == 'tolerance' else f'{method} [{state}]'
                ax.loglog(h.elapsed_sec, h.residual_l1, label=legend, linewidth=1)
                if h.iloc[-1].residual_l1 > 1e-8:
                    ax.plot(h.iloc[-1].elapsed_sec, h.iloc[-1].residual_l1, 'rx')
        ax.set(title=label.replace('_s42', ''), xlabel='Solver seconds (log)', ylabel='R (L1 max)')
        ax.axhline(1e-8, color='gray', ls='--')
        ax.grid(alpha=.2)
        if plotted:
            ax.legend(fontsize=8)
        else:
            ax.text(.5, .5, 'not executed', transform=ax.transAxes, ha='center')
    fig.tight_layout()
    save_figure(fig, figdir/'04_residuals')

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, label in zip(axes.flat, chosen):
        h = representatives.get((label, 'adaptive'))
        if h is not None:
            h = h[h.iteration > 0]
            for key, legend in [('balance_theta_free', 'free'), ('balance_theta', 'used'), ('balance_effective', 'h*theta')]:
                ax.plot(h.pre_residual_l1, h[key], '.-', lw=.6, ms=2, label=legend)
            ax.axhline(config['solver']['balance_theta_max'], color='gray', ls='--', lw=.7, label='upper cap')
            ax.text(.02, .96, f'fallbacks: {int((h.phase == "sinkhorn_fallback").sum())}',
                    transform=ax.transAxes, va='top', fontsize=8)
            ax.set_xscale('log')
            ax.invert_xaxis()
            ax.set_yscale('symlog', linthresh=1)
            ax.legend(fontsize=8)
        else:
            ax.text(.5, .5, 'not executed', transform=ax.transAxes, ha='center')
        ax.set(title=label.replace('_s42', ''), xlabel='Pre-step R', ylabel='theta (symlog, negatives retained)')
        ax.grid(alpha=.2)
    fig.tight_layout()
    save_figure(fig, figdir/'05_theta')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    if len(diags):
        ad = diags[(diags.method == 'adaptive') & (diags.steps > 0)]
        if len(ad):
            ad.groupby('band')[['upper_clip_rate', 'lower_clip_rate', 'backtracked_rate', 'fallback_rate']].mean().reindex(list(bands)).plot.bar(ax=axes[0])
    for label in chosen:
        h = representatives.get((label, 'adaptive'))
        if h is not None:
            valid = h.dropna(subset=['balance_predicted_ratio_l2', 'balance_actual_ratio_l2'])
            axes[1].scatter(valid.balance_predicted_ratio_l2, valid.balance_actual_ratio_l2, s=4, alpha=.25, label=label.replace('_n500', '').replace('_s42', ''))
    axes[1].plot([0, 1], [0, 1], 'k--', lw=.7)
    axes[1].set(xlabel='Predicted row L2 ratio (accepted h)', ylabel='Actual next row L2 ratio')
    axes[1].legend(fontsize=6)
    axes[0].set_ylabel('Mean per-instance event fraction')
    axes[0].tick_params(axis='x', rotation=15)
    fig.tight_layout()
    save_figure(fig, figdir/'06_safeguards_prediction')
    for name in ['01_scale_epsilon', '02_marginal_precision', '03_fixed_theta', '04_residuals', '05_theta', '06_safeguards_prediction']:
        lines += [f'![{name}]({relroot}/figures/{name}.png)', '']
    Path('docs/theta_balance_study.md').write_text('\n'.join(lines), encoding='utf-8')
    manifest['analysis_sec'] = perf_counter()-plot_start
    checkpoint(root, manifest)
    plt.close('all')
    return raw, instances


if __name__ == '__main__':
    main()
