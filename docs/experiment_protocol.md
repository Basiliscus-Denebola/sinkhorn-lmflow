# Experiment protocol

## Inputs and reproducibility

- Use OT Conda environment, Python 3.9 compatibility explicitly authorized by
  the user, float64 NumPy/SciPy CPU. No GPU or old repository code.
- Windows source-tree commands use `scripts/run_ot.cmd`, which selects the OT
  executable/DLL directories and sets MKL/OpenBLAS/OMP thread counts to one.
  The optional PowerShell wrapper needs local script execution permission;
  the CMD wrapper works without changing PowerShell execution policy.
- Deterministic seed 42. One fingerprinted NPZ per distinct problem within a
  sweep; all methods share identical a,b,M,epsilon and zero initial potentials.
- Generators: random dense, squared 1D, squared 2D point cloud, spiky marginal.
  Dirichlet alpha defaults to 1, or .1 for spiky. Floor 1e-12 before normalization.
  Alpha 10,1,.1 and separate alpha_a/alpha_b are configurable.
- Paths in config are relative to the process working directory. Run CLIs from
  repository root. Problem references inside run folders are relative paths.
- Runs retain resolved dataclass defaults, generator metadata, problem SHA256,
  Python executable/version, NumPy/SciPy/BLAS information, CPU/OS, seed, UTC time,
  and git commit/dirty state if available. Git metadata is null in an uninitialized
  repository; there is no fabricated commit identifier.

## Staged experiments

1. Correctness tests before experiments; smoke n=m=20, epsilon=.05, tol=1e-8.
2. Fixed theta n=m=100, epsilon=.01, four problem types, theta
   `[0,.25,.5,.75,.9,.99,1]`, tol=1e-9. Scaling-coordinate, Armijo, Jacobi.
3. Epsilon sweep `.1,.05,.01,.005,.001`, theta `.75,.9,.99`. These three were
   retained from the first sweep for examining the near-Newton region; this
   choice is exploratory, not held-out validation.
4. Baselines on random dense n=100, epsilon=.01: stabilized Sinkhorn, L-BFGS,
   exact additive Newton, hard-switch thresholds 1e-2/1e-4/1e-6 and theta=.99
   scaling LM. Four tolerances 1e-4,1e-6,1e-8,1e-10. Theta .99 is the candidate
   interior value selected from the first sweep; no universal optimality claim.
5. Ablation: fixed h=1 Jacobi/GS at theta=0 and additive/scaling comparisons.

This MVP is deliberately not a full size/seed Cartesian product. Sizes 50/200,
multiple seeds, repetitions and warm-ups remain necessary before runtime claims.

## Timing and accounting

Wall time starts at solver entry, includes initialization and diagnostics,
and excludes serialization and plotting. Diagnostic time is separately recorded.
All methods within a comparison use the same diagnostic frequency. Different
iteration counts imply different numbers of diagnostics. No repeats/warm-up
are performed in the first MVP, so first-call overhead can be visible.

`linear_solve_sec` in history is the latest direction time; summary accumulates
direction time. `linear_solve_count` includes closed-form diagonal theta=0 solves.
`linear_solve_iters=1` denotes a direct direction solve, **not PCG**.
`matrix_vector_count` counts explicit K@v/K.T@u operations in primal Sinkhorn;
logsumexp reductions and dense factorization FLOPs are not represented by that
count. It must not be used alone to compare total work across methods. L-BFGS
also reports objective/gradient evaluations. Dense matrices are reconstructed
for common metrics; no large-scale performance claim is supported.

No peak memory sampler is included in this MVP. No dense P is stored in final
state; reconstruct it from potentials for numerically stable evaluation.

## Failures and presentation

Save every solver result, including max_iter, ill-conditioned solve, trial
overflow and SciPy premature termination. Do not reinterpret SciPy success as
the requested residual tolerance. CSV summaries retain all rows. Curves mark
nonconverged endpoints with a red cross and failure labels, including a failure
at iteration zero. Plots separate different problem fingerprints/tolerances.

Plots read existing CSV/JSON/YAML only and emit PNG/PDF. Changing plots never
reruns a solver. Sigma1~1 is expected; never label the trivial gauge mode a failure.
