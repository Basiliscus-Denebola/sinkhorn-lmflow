# 第一阶段 MVP 交付报告

已使用 OT / Python 3.9.24 完成 Phase 0–6。Python 版本兼容例外由用户明确授权。
共保存 87 次实验：81 次达到各自 residual tolerance，6 次未收敛。所有失败记录保留。

## A. 实际目录树

缓存目录省略；实验目录在 G 中展开说明。

```text
sinkhorn-lmflow/
├── configs/
│   ├── ablation.yaml
│   ├── baseline_compare.yaml
│   ├── epsilon_sweep.yaml
│   ├── smoke.yaml
│   ├── smoke_baselines.yaml
│   └── theta_sweep.yaml
├── data/
│   ├── README.md
│   ├── smoke_problem.json
│   └── smoke_problem.npz
├── docs/
│   ├── baseline_notes.md
│   ├── experiment_protocol.md
│   ├── math_spec.md
│   ├── mvp_report.md
│   └── test_report.txt
├── results/
│   [实验文件见 G；包含 .gitkeep、各 run 和 figures]
├── scripts/
│   ├── _bootstrap.py
│   ├── generate_problem.py
│   ├── plot_run.py
│   ├── report_mvp.py
│   ├── run_experiment.py
│   ├── run_ot.cmd
│   ├── run_ot.ps1
│   ├── run_sweep.py
│   └── summarize_runs.py
├── src/
│   └── sinkhorn_lmflow/
│       ├── core/
│       │   ├── __init__.py
│       │   ├── dual.py
│       │   ├── gauge.py
│       │   ├── linalg.py
│       │   ├── residuals.py
│       │   ├── spectral.py
│       │   ├── stability.py
│       │   └── transport.py
│       ├── experiments/
│       │   ├── __init__.py
│       │   ├── registry.py
│       │   ├── reproducibility.py
│       │   ├── runner.py
│       │   └── sweep.py
│       ├── external/
│       │   ├── __init__.py
│       │   ├── acc_sinkhorn.py
│       │   └── regot_adapter.py
│       ├── io/
│       │   ├── __init__.py
│       │   ├── config.py
│       │   ├── paths.py
│       │   ├── problem_io.py
│       │   └── result_io.py
│       ├── plotting/
│       │   ├── __init__.py
│       │   ├── comparisons.py
│       │   ├── convergence.py
│       │   └── spectral.py
│       ├── policies/
│       │   ├── __init__.py
│       │   ├── stepsize.py
│       │   └── theta.py
│       ├── problems/
│       │   ├── __init__.py
│       │   ├── costs.py
│       │   ├── generators.py
│       │   ├── instance.py
│       │   └── validation.py
│       ├── solvers/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── hard_switch.py
│       │   ├── lbfgs.py
│       │   ├── lmflow.py
│       │   ├── newton.py
│       │   └── sinkhorn.py
│       ├── __init__.py
│       └── types.py
├── tests/
│   ├── .gitkeep
│   ├── conftest.py
│   ├── test_dual_grad_hessian.py
│   ├── test_experiments.py
│   ├── test_gauge.py
│   ├── test_io_roundtrip.py
│   ├── test_lmflow_matrix.py
│   ├── test_mathematical_consistency.py
│   ├── test_newton_endpoint.py
│   ├── test_problem_validation.py
│   ├── test_sinkhorn_endpoint.py
│   ├── test_solvers.py
│   └── test_transport.py
├── .gitignore
├── AGENTS.md
├── README.md
└── pyproject.toml
```

## B. 已实现模块及职责

| 文件 | 职责 |
| --- | --- |
| `src/sinkhorn_lmflow/__init__.py` | 包入口，无迭代或实验副作用 |
| `src/sinkhorn_lmflow/core/__init__.py` | 包入口，无迭代或实验副作用 |
| `src/sinkhorn_lmflow/core/dual.py` | 共享 convex dual、梯度、完整 Hessian、transport objective |
| `src/sinkhorn_lmflow/core/gauge.py` | mean(mu)=0 归一化和 gauge null vector |
| `src/sinkhorn_lmflow/core/linalg.py` | G_theta、对角/SPD/augmented gauge 直接求解 |
| `src/sinkhorn_lmflow/core/residuals.py` | 共享 marginal residual 与所有 solver 停止指标 |
| `src/sinkhorn_lmflow/core/spectral.py` | 标准化 B、sigma1、最大非平凡奇异值与 gap 数学诊断 |
| `src/sinkhorn_lmflow/core/stability.py` | scaling positivity 检查及 scaling 导出 |
| `src/sinkhorn_lmflow/core/transport.py` | K/P 重构及行列和 |
| `src/sinkhorn_lmflow/experiments/__init__.py` | 包入口，无迭代或实验副作用 |
| `src/sinkhorn_lmflow/experiments/registry.py` | 显式 solver 名称映射 |
| `src/sinkhorn_lmflow/experiments/reproducibility.py` | Python/包/BLAS/CPU/OS/git/seed/UTC 元数据 |
| `src/sinkhorn_lmflow/experiments/runner.py` | 单实验编排、解析配置、保存问题和结果 |
| `src/sinkhorn_lmflow/experiments/sweep.py` | 有大小上限的参数扫描与共享 problem artifact |
| `src/sinkhorn_lmflow/external/__init__.py` | 包入口，无迭代或实验副作用 |
| `src/sinkhorn_lmflow/external/acc_sinkhorn.py` | 有论文元数据的 NotImplementedError 占位 |
| `src/sinkhorn_lmflow/external/regot_adapter.py` | 第二阶段 RegOT/SSNS 占位，不安装外部实现 |
| `src/sinkhorn_lmflow/io/__init__.py` | 包入口，无迭代或实验副作用 |
| `src/sinkhorn_lmflow/io/config.py` | YAML 与严格 JSON 序列化 |
| `src/sinkhorn_lmflow/io/paths.py` | 唯一 UTC run 目录命名 |
| `src/sinkhorn_lmflow/io/problem_io.py` | problem NPZ/JSON 保存、加载与 SHA256 |
| `src/sinkhorn_lmflow/io/result_io.py` | result CSV/JSON/NPZ roundtrip |
| `src/sinkhorn_lmflow/plotting/__init__.py` | 包入口，无迭代或实验副作用 |
| `src/sinkhorn_lmflow/plotting/comparisons.py` | 已保存结果发现、分组、失败标记与 CSV 汇总 |
| `src/sinkhorn_lmflow/plotting/convergence.py` | 读取历史绘制残差-迭代、残差-耗时、theta 比较 |
| `src/sinkhorn_lmflow/plotting/spectral.py` | 从历史 CSV 绘制 sigma 与 gap，不运行 solver |
| `src/sinkhorn_lmflow/policies/__init__.py` | 包入口，无迭代或实验副作用 |
| `src/sinkhorn_lmflow/policies/stepsize.py` | 固定请求步长/Armijo、positivity backtracking 与失败原因 |
| `src/sinkhorn_lmflow/policies/theta.py` | ThetaPolicy 接口与 ConstantTheta |
| `src/sinkhorn_lmflow/problems/__init__.py` | 包入口，无迭代或实验副作用 |
| `src/sinkhorn_lmflow/problems/costs.py` | 几何 cost 与生成器专用归一化 |
| `src/sinkhorn_lmflow/problems/generators.py` | 四类确定性问题与 Dirichlet 边际生成 |
| `src/sinkhorn_lmflow/problems/instance.py` | 不可变 float64 OTProblem 与初始化校验 |
| `src/sinkhorn_lmflow/problems/validation.py` | 维度、有限性、严格正边际、单位质量、epsilon 校验 |
| `src/sinkhorn_lmflow/solvers/__init__.py` | 包入口，无迭代或实验副作用 |
| `src/sinkhorn_lmflow/solvers/base.py` | Solver Protocol、初始化、统一监测与结果构造 |
| `src/sinkhorn_lmflow/solvers/hard_switch.py` | 残差阈值控制 Sinkhorn→Newton 永久切换 |
| `src/sinkhorn_lmflow/solvers/lbfgs.py` | SciPy reduced-dual L-BFGS-B wrapper |
| `src/sinkhorn_lmflow/solvers/lmflow.py` | fixed theta、两种离散格式与 GS endpoint |
| `src/sinkhorn_lmflow/solvers/newton.py` | 完整 Hessian additive Newton + Armijo |
| `src/sinkhorn_lmflow/solvers/sinkhorn.py` | primal/log 两种数值实现与单轮 GS 更新 |
| `src/sinkhorn_lmflow/types.py` | 统一问题导出、配置、迭代记录、结果和 policy state dataclass |

`scripts/` 分别提供问题生成、单次运行、扫描、绘图、汇总和本报告生成；
`run_ot.cmd`/`run_ot.ps1` 指定 OT 环境和单线程 BLAS；`_bootstrap.py` 支持源码直接运行。
`configs/` 提供 smoke、smoke baselines、theta、epsilon、baseline 和 ablation 六组配置。
`tests/` 覆盖验证、数学导数、gauge、两端点、谱、solver、坐标一致性及 I/O。

## C. 测试报告

```text
Command: .\scripts\run_ot.cmd -m pytest -q
Environment: D:\miniconda_envs\OT\python.exe
Python: 3.9.24
NumPy: 2.0.2
SciPy: 1.13.1
pytest: 8.4.2

............................................................             [100%]
60 passed in 1.09s

0 failed. Phase 0 intentionally collected 0 tests before implementation.
An early native NumPy SVD process crash was resolved by supplying the OT
Conda DLL directories on PATH; the final suite ran with run_ot.cmd.
```

## D. 数学核验

| 项目 | 验证方法与结果 |
| --- | --- |
| Gradient | 中心差分，相对误差 < 1e-6，通过 |
| Hessian-vector | 梯度方向差分，相对误差 < 1e-6，通过 |
| Gauge | P、Phi、残差不变，mean(mu)=0，通过 |
| theta=0 GS endpoint | 与显式 Sinkhorn 比值更新比较，rtol=1e-12/atol=1e-14，通过 |
| theta=1 Newton endpoint | G1=H，Hd=rhs，q.T d=0；additive 轨迹一致，通过 |
| Spectrum | 矩形标准化矩阵与 1±theta*sigma 特征值比较，通过 |
| Scaling ODE | 坐标变换后的块方程与 forward Euler 一致，通过 |
| Same optimum | Sinkhorn/Newton/LM 共享解，J+Phi≈0，通过 |

## E. Smoke 实验

n=m=20，epsilon=.05，seed=42，tol=1e-8。时间包含诊断。

| solver | converged | n_iter | elapsed_sec | residual_l1 | grad_norm_l2 |
| --- | --- | --- | --- | --- | --- |
| sinkhorn primal_scaling | True | 64 | 0.014417 | 8.05861e-09 | 3.30872e-09 |
| sinkhorn log_stabilized | True | 64 | 0.005581 | 8.05861e-09 | 3.30872e-09 |
| lbfgs | True | 92 | 0.012715 | 6.48761e-09 | 2.43206e-09 |
| newton | True | 10 | 0.0033681 | 6.27615e-09 | 8.73702e-09 |
| hard_switch (0.01) | True | 14 | 0.0019959 | 1.06859e-15 | 4.44531e-16 |
| hard_switch (0.0001) | True | 28 | 0.0034563 | 2.19443e-15 | 1.29206e-15 |
| hard_switch (1e-06) | True | 45 | 0.0040542 | 4.61009e-12 | 2.5384e-12 |
| lmflow theta=0.5 | True | 69 | 0.0122807 | 8.33203e-09 | 4.33131e-09 |

## F. Fixed-theta 实验

n=m=100，epsilon=.01，seed=42，tol=1e-9；scaling-coordinate、Jacobi、Armijo。
theta=1 此表是 scaling 离散格式，其方向等于 Newton，但不是 additive Newton 轨迹。

| problem | theta | n_iter | elapsed_sec | residual_l1 | safeguard_count |
| --- | --- | --- | --- | --- | --- |
| random_dense | 0 | 299 | 0.0770268 | 9.70225e-10 | 0 |
| random_dense | 0.25 | 227 | 0.164391 | 9.37994e-10 | 3 |
| random_dense | 0.5 | 157 | 0.120165 | 9.44508e-10 | 6 |
| random_dense | 0.75 | 89 | 0.06739 | 9.40421e-10 | 16 |
| random_dense | 0.9 | 49 | 0.0379011 | 8.89697e-10 | 23 |
| random_dense | 0.99 | 24 | 0.0209108 | 6.17543e-10 | 27 |
| random_dense | 1 | 22 | 0.0252884 | 3.07257e-15 | 33 |
| 1d_squared | 0 | 820 | 0.17934 | 9.891e-10 | 0 |
| 1d_squared | 0.25 | 618 | 0.307345 | 9.93445e-10 | 1 |
| 1d_squared | 0.5 | 417 | 0.207432 | 9.98966e-10 | 2 |
| 1d_squared | 0.75 | 218 | 0.11438 | 9.20092e-10 | 5 |
| 1d_squared | 0.9 | 97 | 0.0570394 | 9.35247e-10 | 9 |
| 1d_squared | 0.99 | 28 | 0.0150619 | 7.69397e-10 | 16 |
| 1d_squared | 1 | 18 | 0.0404294 | 7.87998e-16 | 15 |
| 2d_pointcloud | 0 | 374 | 0.0932373 | 9.63537e-10 | 0 |
| 2d_pointcloud | 0.25 | 285 | 0.14562 | 9.82395e-10 | 3 |
| 2d_pointcloud | 0.5 | 195 | 0.126524 | 9.70431e-10 | 3 |
| 2d_pointcloud | 0.75 | 105 | 0.0542589 | 8.92473e-10 | 3 |
| 2d_pointcloud | 0.9 | 51 | 0.038568 | 8.03799e-10 | 4 |
| 2d_pointcloud | 0.99 | 18 | 0.0150543 | 2.71296e-10 | 5 |
| 2d_pointcloud | 1 | 12 | 0.0152253 | 2.05179e-13 | 4 |
| spiky_marginal | 0 | 1987 | 0.45106 | 9.98582e-10 | 0 |
| spiky_marginal | 0.25 | 1520 | 0.775332 | 9.95696e-10 | 68 |
| spiky_marginal | 0.5 | 1041 | 0.547703 | 9.98002e-10 | 130 |
| spiky_marginal | 0.75 | 568 | 0.303908 | 9.78527e-10 | 218 |
| spiky_marginal | 0.9 | 292 | 0.160388 | 9.66155e-10 | 326 |
| spiky_marginal | 0.99 | 139 | 0.0760213 | 9.7506e-10 | 451 |
| spiky_marginal | 1 | 129 | 0.169691 | 1.45609e-13 | 463 |

这 28 个样本均收敛。外层迭代随 theta 增大减少，但 wall time 不单调；
时间为单次运行且包含诊断与首次调用开销，不能据此宣称普遍加速。

### 多 tolerance 基线

| method | tol | converged | n_iter | elapsed_sec | residual_l1 |
| --- | --- | --- | --- | --- | --- |
| sinkhorn | 0.0001 | True | 55 | 0.0258203 | 8.98858e-05 |
| sinkhorn | 1e-06 | True | 94 | 0.0232264 | 9.33335e-07 |
| sinkhorn | 1e-08 | True | 134 | 0.0352221 | 9.0963e-09 |
| sinkhorn | 1e-10 | True | 173 | 0.0439368 | 9.98976e-11 |
| lbfgs | 0.0001 | True | 117 | 0.0414124 | 9.87224e-05 |
| lbfgs | 1e-06 | True | 177 | 0.0468548 | 9.62291e-07 |
| lbfgs | 1e-08 | False | 215 | 0.0581491 | 2.06965e-08 |
| lbfgs | 1e-10 | False | 215 | 0.0560442 | 2.06965e-08 |
| newton | 0.0001 | True | 11 | 0.0146722 | 1.29973e-05 |
| newton | 1e-06 | True | 12 | 0.0141166 | 8.30832e-08 |
| newton | 1e-08 | True | 13 | 0.0157578 | 3.80349e-12 |
| newton | 1e-10 | True | 13 | 0.0147434 | 3.80349e-12 |
| hard_switch (0.01) | 0.0001 | True | 21 | 0.0077656 | 3.9967e-06 |
| hard_switch (0.01) | 1e-06 | True | 22 | 0.0090569 | 2.06945e-10 |
| hard_switch (0.01) | 1e-08 | True | 22 | 0.0087047 | 2.06945e-10 |
| hard_switch (0.01) | 1e-10 | True | 23 | 0.0103021 | 4.01805e-16 |
| hard_switch (0.0001) | 0.0001 | True | 55 | 0.0222536 | 8.98858e-05 |
| hard_switch (0.0001) | 1e-06 | True | 56 | 0.0156878 | 1.66104e-07 |
| hard_switch (0.0001) | 1e-08 | True | 57 | 0.0172383 | 1.56216e-13 |
| hard_switch (0.0001) | 1e-10 | True | 57 | 0.0184198 | 1.56216e-13 |
| hard_switch (1e-06) | 0.0001 | True | 55 | 0.0153479 | 8.98858e-05 |
| hard_switch (1e-06) | 1e-06 | True | 94 | 0.0246406 | 9.33335e-07 |
| hard_switch (1e-06) | 1e-08 | True | 95 | 0.0250708 | 1.86628e-11 |
| hard_switch (1e-06) | 1e-10 | True | 95 | 0.0265105 | 1.86628e-11 |
| lmflow | 0.0001 | True | 19 | 0.0109153 | 1.59539e-05 |
| lmflow | 1e-06 | True | 21 | 0.0127053 | 2.40179e-07 |
| lmflow | 1e-08 | True | 23 | 0.0205143 | 4.39945e-09 |
| lmflow | 1e-10 | True | 25 | 0.0145822 | 8.77365e-11 |

### Epsilon 扫描

| epsilon | theta | converged | n_iter | elapsed_sec | residual_l1 | safeguard_count |
| --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 0.75 | True | 16 | 0.0247727 | 2.02207e-10 | 0 |
| 0.1 | 0.9 | True | 13 | 0.0105163 | 2.07483e-10 | 0 |
| 0.1 | 0.99 | True | 10 | 0.008206 | 7.14776e-10 | 0 |
| 0.05 | 0.75 | True | 20 | 0.0133368 | 4.78975e-10 | 0 |
| 0.05 | 0.9 | True | 15 | 0.0086445 | 1.89113e-10 | 0 |
| 0.05 | 0.99 | True | 10 | 0.0058609 | 9.14416e-10 | 0 |
| 0.01 | 0.75 | True | 89 | 0.0469329 | 9.40421e-10 | 16 |
| 0.01 | 0.9 | True | 49 | 0.0259862 | 8.89697e-10 | 23 |
| 0.01 | 0.99 | True | 24 | 0.0183545 | 6.17543e-10 | 27 |
| 0.005 | 0.75 | True | 290 | 0.203357 | 9.46658e-10 | 78 |
| 0.005 | 0.9 | True | 142 | 0.111202 | 9.71545e-10 | 115 |
| 0.005 | 0.99 | True | 58 | 0.0476413 | 7.38632e-10 | 140 |
| 0.001 | 0.75 | False | 0 | 0.0033651 | 10.4169 | 0 |
| 0.001 | 0.9 | False | 0 | 0.002568 | 10.4169 | 0 |
| 0.001 | 0.99 | False | 0 | 0.0025434 | 10.4169 | 0 |

### Ablation

| theta | discretization | GS | Armijo | converged | n_iter | residual_l1 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | scaling_coordinate | False | False | False | 3000 | 104.124 |
| 0 | scaling_coordinate | True | False | True | 154 | 8.99956e-10 |
| 0 | additive | False | True | True | 300 | 9.45964e-10 |
| 0.5 | additive | False | True | True | 156 | 9.26713e-10 |
| 0.5 | scaling_coordinate | False | True | True | 157 | 9.44508e-10 |
| 0.99 | additive | False | True | True | 16 | 2.89045e-10 |
| 0.99 | scaling_coordinate | False | True | True | 24 | 6.17543e-10 |

### 全部未收敛记录

| solver_name | epsilon | theta | tol | n_iter | residual_l1 | stop_reason |
| --- | --- | --- | --- | --- | --- | --- |
| lbfgs | 0.01 | — | 1e-08 | 215 | 2.06965e-08 | scipy_stop: CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH |
| lbfgs | 0.01 | — | 1e-10 | 215 | 2.06965e-08 | scipy_stop: CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH |
| lmflow | 0.001 | 0.75 | 1e-09 | 0 | 10.4169 | numerical_failure: Ill-conditioned matrix (rcond=2.99936e-23): result may not be accurate. |
| lmflow | 0.001 | 0.9 | 1e-09 | 0 | 10.4169 | numerical_failure: Ill-conditioned matrix (rcond=2.76256e-23): result may not be accurate. |
| lmflow | 0.001 | 0.99 | 1e-09 | 0 | 10.4169 | numerical_failure: Ill-conditioned matrix (rcond=2.63762e-23): result may not be accurate. |
| lmflow | 0.01 | 0 | 1e-09 | 3000 | 104.124 | max_iter |

三个 epsilon=.001 初始求解失败来自未缩放的 dense G 的条件数警告。此时标准化谱
proxy 不能代表未缩放系统的条件数。没有加正则、隐藏 warm start 或改变问题。
L-BFGS 自身终止并不代表通过共享残差阈值。固定 h=1 的 Jacobi 端点不等于 GS Sinkhorn。

## G. 已生成文件

- **smoke_baselines**：[results/20260916_033533_912192_smoke_baselines_090d31](../results/20260916_033533_912192_smoke_baselines_090d31)；[comparison.csv](../results/20260916_033533_912192_smoke_baselines_090d31/comparison.csv)。
  PNG/PDF 各 3 个；[figures](../results/20260916_033533_912192_smoke_baselines_090d31/figures)。
- **theta_sweep**：[results/20260916_033536_030127_theta_sweep_5acc12](../results/20260916_033536_030127_theta_sweep_5acc12)；[comparison.csv](../results/20260916_033536_030127_theta_sweep_5acc12/comparison.csv)。
  PNG/PDF 各 12 个；[figures](../results/20260916_033536_030127_theta_sweep_5acc12/figures)。
- **baseline_compare**：[results/20260916_033634_099312_baseline_compare_e2079b](../results/20260916_033634_099312_baseline_compare_e2079b)；[comparison.csv](../results/20260916_033634_099312_baseline_compare_e2079b/comparison.csv)。
  PNG/PDF 各 12 个；[figures](../results/20260916_033634_099312_baseline_compare_e2079b/figures)。
- **epsilon_sweep**：[results/20260916_033638_234791_epsilon_sweep_fdc86a](../results/20260916_033638_234791_epsilon_sweep_fdc86a)；[comparison.csv](../results/20260916_033638_234791_epsilon_sweep_fdc86a/comparison.csv)。
  PNG/PDF 各 15 个；[figures](../results/20260916_033638_234791_epsilon_sweep_fdc86a/figures)。
- **ablation**：[results/20260916_033641_361397_ablation_86dfbf](../results/20260916_033641_361397_ablation_86dfbf)；[comparison.csv](../results/20260916_033641_361397_ablation_86dfbf/comparison.csv)。
  PNG/PDF 各 3 个；[figures](../results/20260916_033641_361397_ablation_86dfbf/figures)。
- **smoke**：[results/20260916_034108_813972_smoke_c8c226](../results/20260916_034108_813972_smoke_c8c226)；[comparison.csv](../results/20260916_034108_813972_smoke_c8c226/comparison.csv)。

每个 run 保存：`config_resolved.yaml`、`problem_ref.json`、`summary.json`、
`history.csv`、`final_state.npz`、`runtime_metadata.json`。扫描共享问题保存在 `problems/`。
[总表](../results/all_runs.csv) 包含全部实验（不排除失败）；
[配置](../configs/) 和 [可复用 smoke problem](../data/smoke_problem.npz) 已生成。

已目视检查 theta 收敛图和含失败标记的 baseline 耗时图；其余图由同一保存结果流程输出。

## 当前边界

Acc-Sinkhorn/RegOT/SSNS 仍为明确占位；closed-loop theta、Schur-PCG、稀疏 Hessian、GPU 未实现。
未运行 n=50/200 的完整矩阵或多 seed 重复；本轮仅 n=20 smoke 和 n=100 MVP。
矩阵向量计数目前只覆盖显式 primal K 乘法，不能独立用作跨方法复杂度结论。
完整计数和 timing 限制见 [实验协议](experiment_protocol.md)。
