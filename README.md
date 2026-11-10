# sinkhorn-lmflow

Research question: does continuously opening the Hessian coupling between the
Sinkhorn-like block metric and full Newton offer useful numerical behavior for
discrete two-marginal entropic optimal transport?

Minimize `<P,M> + epsilon*sum(P log P-P)` subject to `P1=a`, `P.T1=b`.
Costs may be arbitrary finite dense matrices. The implemented convex dual has
gradient `[a-r,b-c]`, with `P=exp(-(mu+nu+M)/epsilon)`. Directions solve
`G_theta d=[r-a,c-b]`, where `G_theta=(1/epsilon)[[Dr,theta P],[theta P.T,Dc]]`.
See [mathematics](docs/math_spec.md) for signs, gauge and discretizations.

## Structure

- `problems/`: immutable validated inputs, independent deterministic generators.
- `core/`: shared transport, objectives, derivatives, residuals, gauge and spectra.
- `solvers/`: Sinkhorn, exact Newton, fixed-theta LM, L-BFGS and hard switch.
- `policies/`: ConstantTheta protocol and safeguarded steps.
- `experiments/`, `io/`: configs, reproducible sweeps, NPZ/CSV/JSON serialization.
- `plotting/`: plots from saved runs only. `external/`: phase-two placeholders.
- `configs/`, `scripts/`, `tests/`, `docs/`: runnable configurations and validation.

## Environment and installation

Use the existing **OT** Conda environment (`D:\miniconda_envs\OT`). The user
approved Python >=3.9 compatibility instead of the original prompt's >=3.11.
Dependencies: NumPy, SciPy, pandas, matplotlib, PyYAML; pytest for tests.

```powershell
conda activate OT
python -m pip install -e ".[test]"
python -m pytest -q
```

Source-tree commands work without editable installation. On this Windows host,
the helper selects the OT executable and native DLL paths, and fixes BLAS threads
to one (no change to system execution policy):

```powershell
.\scripts\run_ot.cmd -m pytest -q
.\scripts\run_ot.cmd scripts/run_experiment.py --config configs/smoke.yaml
```

## Generate one shared problem and run a solver

```powershell
.\scripts\run_ot.cmd scripts/generate_problem.py --config configs/smoke.yaml --output data/problem.npz
.\scripts\run_ot.cmd scripts/run_experiment.py --config configs/smoke.yaml --problem data/problem.npz --solver sinkhorn
```

Solver choices: `sinkhorn`, `lbfgs`, `newton`, `hard_switch`, `lmflow`.
Configuration separates problem generation, solver settings and diagnostics.
Set `line_search: false` for fixed requested step size; scaling positivity still
backtracks when required. `gauss_seidel: true` is supported for the theta=0
scaling-coordinate endpoint. Other GS combinations are rejected explicitly.

## Sweeps and saved plots

```powershell
.\scripts\run_ot.cmd scripts/run_sweep.py --config configs/theta_sweep.yaml
.\scripts\run_ot.cmd scripts/run_sweep.py --config configs/baseline_compare.yaml
.\scripts\run_ot.cmd scripts/run_sweep.py --config configs/epsilon_sweep.yaml
.\scripts\run_ot.cmd scripts/run_sweep.py --config configs/ablation.yaml
.\scripts\run_ot.cmd scripts/summarize_runs.py results/<sweep-directory>
.\scripts\run_ot.cmd scripts/plot_run.py results/<sweep-directory>
```

Run from the repository root. Substitute the actual printed directory. Each run
contains resolved YAML, problem reference, summary JSON, history CSV, final-state
NPZ and runtime metadata. Sweep problems are saved once and hashed. All failed
runs remain in summaries and figures. Numerical result files stay local via
`.gitignore`; PNG/PDF figures under `results/` and `.gitkeep` can be committed.

## Mirror-driven positive SR1 v0

The paired study compares `mirror_sr1` with `sinkhorn_reduced` (standard
Sinkhorn observed after row normalization). It uses one most-recent secant pair,
20 shared warmup updates, up to six Armijo trials and unit Sinkhorn fallback.
The existing `sinkhorn` observation convention is unchanged.

```powershell
.\scripts\run_ot.cmd scripts/run_sr1_study.py --config configs/sr1_smoke.yaml
.\scripts\run_ot.cmd scripts/run_sr1_study.py --config configs/sr1_controlled.yaml
.\scripts\run_ot.cmd scripts/run_sr1_study.py --config configs/sr1_transfer.yaml
.\scripts\run_ot.cmd scripts/report_sr1.py --run-dir results/<actual-study-directory>
```

Resume with the same config and `--resume results/<actual-study-directory>`.
Completed matching pairs are skipped; partial pairs get a new attempt. All
failures remain on disk. New solver entry points require `record_every=1` and
`compute_spectrum_every=0`; v0 fixes its line-search constants and six trials.
See the [task specification](docs/sr1_v0_task.md) and
[Chinese implementation and experiment report](docs/sr1_v0_report.md).

## Original MVP status and limits

First MVP implemented and exercised; see [MVP report](docs/mvp_report.md) for
the actual tree, file responsibilities, tests, tables, failure reasons and paths.
Read [experiment protocol](docs/experiment_protocol.md) before interpreting
timings and [baseline notes](docs/baseline_notes.md) for fairness rules.

- NumPy/SciPy dense float64 CPU; strictly positive unit-mass marginals.
- Full direct solves target small/medium sizes. Conservative ill-conditioning
  failures are recorded; the mathematical problem is never regularized silently.
- Acc-Sinkhorn and RegOT/SSNS adapters are **not implemented** (phase two).
- No closed-loop theta, PCG, GPU, sparse Hessian or large-scale benchmark claims.
- Current numerical observations are **experimental**. Multi-seed, repeated
  timings and larger size coverage are required before research conclusions.
