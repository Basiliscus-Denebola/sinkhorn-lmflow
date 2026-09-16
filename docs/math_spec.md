# Mathematical specification

## Problem and sign conventions

For strictly positive unit-mass marginals `a,b`, arbitrary finite real dense cost
`M`, and `epsilon>0`, minimize

`J(P) = <P,M> + epsilon * sum(P log P - P)`

subject to `P1=a`, `P.T1=b`, `P>=0`. Zero marginal support is rejected explicitly;
the caller may reduce it before constructing the problem. Costs need not be
geometric or nonnegative. Generators normalize their own nonnegative costs, but
neither OTProblem nor a solver rescales user inputs.

Use `u=exp(-mu/epsilon)`, `v=exp(-nu/epsilon)`, `K=exp(-M/epsilon)` and
`P=diag(u) K diag(v)=exp(-(mu[:,None]+nu[None,:]+M)/epsilon)`.

The convex dual to **minimize** is

`Phi = epsilon*sum(P) + mu@a + nu@b`.

With `r=P1`, `c=P.T1`, its gradient is `[a-r,b-c]`. At the optimum, `J=-Phi`.
The transport objective at an infeasible iterate is recorded as an objective
evaluation, not as a feasible upper bound or primal-dual gap.

## Metric and spectrum

`H = (1/epsilon) [[diag(r),P],[P.T,diag(c)]]`.

`G_theta = (1/epsilon) [[diag(r),theta*P],[theta*P.T,diag(c)]]`.

Solve `G_theta d = [r-a,c-b] = -grad(Phi)`. No Hessian sparsification, diagonal
regularizer, marginal flooring, or hidden modification of this system is used.
For theta=0 the diagonal system is solved elementwise; for 0<theta<1 use a dense
SPD direct solve. For theta=1 use the full augmented symmetric system
`[[H,q],[q.T,0]] [d,alpha] = [-grad(Phi),0]`, `q=[1_n,-1_m]`.
An ill-conditioning warning is an explicit numerical failure, not a silent
least-squares fallback. This conservative policy can stop otherwise recoverable
instances; diagonal equilibration/Schur methods are future work.

Let `D=(1/epsilon) diag(r,c)` and `B=diag(r)^(-1/2) P diag(c)^(-1/2)`.
Then `D^(-1/2) G_theta D^(-1/2) = [[I,theta B],[theta B.T,I]]`.
The paired eigenvalues are `1 +/- theta*sigma_i(B)`; rectangular problems also
have `abs(n-m)` unit eigenvalues. Sigma1=1 is the mass/gauge mode. Remove exactly
one leading singular value; an additional singular value near one is **not**
removed. Record sigma1, sigma2 (0 if no nontrivial singular value exists), and
`delta=1-theta*sigma2`. This is the requested nontrivial spectral proxy, not the
condition number of the unscaled direct system. Diagnostics do not clip values.

## Discretizations and gauge

Additive: `z_new=z+h*d`.

Scaling-coordinate: `z_new=z-epsilon*log1p(-h*d/epsilon)`.
Every factor `1-h*d/epsilon` must be strictly positive. Reject and shrink the
step otherwise; never clip. Positivity checks remain active when Armijo is off.
The scaling ODE satisfies `udot=-u*d_mu/epsilon`, `vdot=-v*d_nu/epsilon`, hence

`[[D_Kv,theta D_u K],[theta D_v K.T,D_KTu]] [udot,vdot]=[a-r,b-c]`.

For theta=0, the Jacobi update changes both blocks using the old state. The GS
endpoint updates u first and recomputes the v block. At h=1 this is exactly
`u_new=a/(K v)`, `v_new=b/(K.T u_new)`. The implementation evaluates this endpoint
using logsumexp to avoid cancellation, and verifies it against those ratios in
float64. GS is only specified for theta=0 scaling-coordinate; other GS requests
raise ValueError rather than inventing a coupling discretization.

After an accepted update, optional normalization sets mean(mu)=0 by
`mu-=mean(mu)` and `nu+=mean(mu_old)`, leaving P invariant. The Newton direction
uses the q-orthogonal gauge; the final iterate is then normalized consistently.
Theta=1 matches exact Newton **directions**; its scaling-coordinate trajectory
need not match the additive exact Newton baseline.

## Globalization and stopping

Use Armijo on Phi with initial h=1, contraction 0.5, c=1e-4 and at most 60
backtracks by default. If objective differences reach float64 roundoff, allow
a roundoff-sized difference only when the shared residual strictly decreases.
Trial overflow triggers backtracking; no accepted state is replaced by clipping.

Every solver uses `R=max(norm(r-a,1),norm(c-b,1))`; default tolerance 1e-9.
L-BFGS minimizes a reduced exact dual with nu[-1]=0, using the shared objective
and gradient. SciPy's own termination is reported as failure if R exceeds tol.
Hard-switch uses stabilized Sinkhorn until R<=switch_tol, then stays in exact
additive Newton. Initial state is zero potentials unless explicitly supplied.

Finite potentials are authoritative. Exported u,v can underflow/overflow even
when log-domain P is valid; `scalings_representable` reports this without
declaring stabilized Sinkhorn a failure.
