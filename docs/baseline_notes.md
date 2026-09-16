# Baseline scope

## Tier A (implemented)

| Method | Exact problem / curvature | Numerical details |
|---|---|---|
| Sinkhorn | Same EOT | Primal scaling and log-stabilized GS modes |
| L-BFGS | Same exact convex dual | SciPy L-BFGS-B; nu[-1]=0 removes gauge |
| Exact Newton | Full H, no approximation | Augmented gauge solve, additive Armijo |
| Hard-switch | Sinkhorn then full H | Switch thresholds 1e-2,1e-4,1e-6 |
| LM flow | Full prescribed G_theta | Fixed theta; additive/scaling ablations |

The main research comparison is continuous Hessian coupling versus hard switch,
not merely whether a method eventually uses Newton. Fixed theta experiments
do not establish a closed-loop theta policy.

## Tier A+ and Tier B (not implemented)

Acc-Sinkhorn citation metadata is provided in `external/acc_sinkhorn.py`:
Zeyi Xu and Long Chen, *Accelerating Sinkhorn for Entropy-Regularized Optimal
Transport* (2026), [arXiv:2605.30267](https://arxiv.org/abs/2605.30267).
Only metadata was verified; this project does not infer an implementation from
the abstract or claim to have reproduced its results. The adapter raises
NotImplementedError. Algorithm 1/2 study, official implementation search,
independent tests and synthetic reproduction are prerequisites to inclusion.

RegOT/SSNS/sparse Newton belong to a future engineering frontier comparison.
`external/regot_adapter.py` is a placeholder and does not clone or install RegOT.
Its eventual figures must say **external optimized C++ implementation**. A raw
Python-versus-C++ timing comparison is not evidence of algorithmic superiority.
Future adapters must verify epsilon/objective/sign conventions and report the
same residual, dual objective, transport objective, iterations and wall time.

Closed-loop theta, Schur-PCG, low-rank preconditioners, SSNS and acceleration are
future work. Only ConstantTheta is currently implemented as a ThetaPolicy.
