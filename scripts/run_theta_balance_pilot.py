"""Small paired pilot; saved runs are the sole source of plots and tables."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import os
import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sinkhorn_lmflow.types import SolverConfig
from sinkhorn_lmflow.problems.generators import generate_problem
from sinkhorn_lmflow.solvers.sinkhorn import sinkhorn_log_sweep
from sinkhorn_lmflow.experiments.registry import get_solver
from sinkhorn_lmflow.experiments.reproducibility import runtime_metadata
from sinkhorn_lmflow.io.config import load_config, save_config, save_json
from sinkhorn_lmflow.io.paths import create_run_dir
from sinkhorn_lmflow.io.problem_io import save_problem, problem_fingerprint
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
    parser.add_argument('--config', default='configs/theta_balance_pilot.yaml')
    parser.add_argument('--analyze', type=Path, help='Only read saved runs and regenerate plots/tables')
    args = parser.parse_args()
    if args.analyze:
        import json
        root = args.analyze
        analyze(root, json.loads((root/'manifest.json').read_text(encoding='utf-8')))
        print(root)
        return
    config = load_config(args.config)
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


if __name__ == '__main__':
    main()
