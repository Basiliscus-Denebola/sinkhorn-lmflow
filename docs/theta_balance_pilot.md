# 一维最小残差 theta：最小实验报告

按第 2 版文档实现：`p=balance_theta*(m-a)/m`，先更新行缩放，再用更新后的行势执行精确 Sinkhorn 列子步。`q_aux` 仅用于选 theta，不更新列变量。`h*balance_theta=1` 与标准 GS Sinkhorn 整步一致。旧 `lmflow.theta` 语义不变。

本机固定 seed=42 的三个 100×100 样本中，新方法的总时间中位数均低于裸 stabilized Sinkhorn；这只是小样本实测，不是普遍加速结论。

## 复现与结果位置

- 基础 commit：`1c2f5ea9d0bf1bbd921310b0c943a3277f072209`，工作区 dirty；本次未提交，原有暂存修改保留。
- 环境：`D:\miniconda_envs\OT\python.exe`，Python 3.9.24、NumPy 2.0.2、SciPy 1.13.1，Windows，BLAS 单线程。
- 参数：`tol=1e-8`、`max_iter=3000`、`balance_theta_max=10`；Armijo 初始 h=1、收缩 .5、常数 1e-4、最多 60 次回溯。逐步记录，两边均关闭谱 SVD。
- 相同问题、相同初值：从零势做一次共同 GS sweep，保存同一份势给两种方法。公共初始化不计入双方时间或工作量；其耗时另存在 manifest。每方法先 warm-up 一次，再交替顺序正式运行三次。
- [最终结果目录](../results/20260924_143313_825935_theta_balance_pilot_ad26af/)、[详细统计与完整指纹](../results/20260924_143313_825935_theta_balance_pilot_ad26af/report.md)、[阈值时间 CSV（含范围）](../results/20260924_143313_825935_theta_balance_pilot_ad26af/thresholds.csv)、[曲线 PNG/PDF](../results/20260924_143313_825935_theta_balance_pilot_ad26af/figures/)。各次运行包含配置、环境、history、summary 和最终势。
- 较早的 `20260924_143053_115611_theta_balance_pilot_d65efe` 是计时修正前的开发试跑；本报告仅使用上述最终目录。

```powershell
.\scripts\run_ot.cmd -m pytest -q
.\scripts\run_ot.cmd scripts/run_theta_balance_pilot.py
# 仅从保存的数据重画，不重跑 solver：
.\scripts\run_ot.cmd scripts/run_theta_balance_pilot.py --analyze results/20260924_143313_825935_theta_balance_pilot_ad26af
```

问题指纹前缀（完整 SHA256 见各组 problem_ref.json）：smoke `a070c667`，random_dense `757b1f8f`，1d_squared `ed3288a1`，spiky_marginal `4b69a2b4`。Smoke 为 20×20、epsilon=.05，其余 100×100、epsilon=.01。

## 时间与工作量

总时间包含 solver 内初始化、方向、所有试步、监控及最终诊断，不含保存和画图。单位 ms，三次中位数 [最小, 最大]。每行双方均 3/3 收敛至 R≤1e-8。

| 样本 | Sinkhorn 迭代 | 新方法迭代 | Sinkhorn 总时间 | 新方法总时间 | 总时间比 S/new |
|---|---:|---:|---|---|---:|
| smoke | 63 | 35 | 5.343 [5.210, 5.523] | 4.817 [4.708, 5.154] | 1.11 |
| random_dense | 133 | 44 | 32.010 [31.347, 41.800] | 12.899 [12.835, 13.182] | 2.48 |
| 1d_squared | 373 | 88 | 93.041 [86.395, 97.371] | 25.696 [25.108, 28.248] | 3.62 |
| spiky_marginal | 914 | 249 | 220.363 [216.648, 225.060] | 74.310 [71.147, 79.860] | 2.97 |

下表每格为“首次达到阈值的时间中位数 ms / 累计 dense-pass proxy”；时间范围见上述 CSV 和详细统计。

| 样本 / 方法 | R≤1e-4 | R≤1e-6 | R≤1e-8 |
|---|---|---|---|
| smoke / S | 2.257 / 206 | 3.703 / 350 | 5.306 / 510 |
| smoke / new | 2.217 / 202 | 3.444 / 332 | 4.783 / 462 |
| random_dense / S | 13.216 / 438 | 22.253 / 750 | 31.928 / 1070 |
| random_dense / new | 8.062 / 358 | 11.167 / 501 | 12.821 / 579 |
| 1d_squared / S | 44.429 / 1358 | 68.561 / 2174 | 92.952 / 2990 |
| 1d_squared / new | 12.809 / 566 | 19.193 / 852 | 25.620 / 1153 |
| spiky_marginal / S | 111.514 / 3726 | 167.386 / 5526 | 220.276 / 7318 |
| spiky_marginal / new | 45.712 / 1957 | 60.608 / 2607 | 74.231 / 3250 |

Proxy 将一次完整 P 构造、一次全矩阵行/列/总和归约、一次 logsumexp、一次 P 向量乘各记 1，包含方向、回溯试算及共同监控。它是**未加权的算法工作量代理，不是 FLOPs 或等价 matvec**；LSE、指数与普通归约成本不同。向量运算、分配、gauge、可选谱计算和最终 transport-objective 导出不计入 proxy。计数覆盖本次两个 log-domain solver，不能用于其他 solver 的完整成本比较。`column_update_count` 和 `trial_count` 是另列事件数，不重复加进 proxy。旧 `matrix_vector_count` 不用于比较。

新方法每步增加两个 P 向量乘和试步评估，却减少了迭代及 LSE 次数。例如 random_dense 的 LSE 次数由 266 降为 44，代价是 88 次 P 向量乘。Smoke 的时间优势较小，低精度阈值处计时范围重叠。统一监控自身有成本，这里比较的是同一实验框架下、含监控的性能。

## theta、eta 与失败情况

下表为第一条原始轨迹；theta 范围包含退化时截断为 0 的记录，回退不算成功加速步。

| 样本 | theta_free 范围 | theta 范围 | 上界截断比例 | eta 范围 | h 范围 | 回退 / 回溯 |
|---|---|---|---:|---|---|---|
| smoke | 1.231～2.738 | 1.231～2.738 | 0% | .170～.637 | .5～1 | 0 / 3 |
| random_dense | 1.014～7.771 | 1.014～7.771 | 0% | .359～.806 | .5～1 | 0 / 5 |
| 1d_squared | .767～22.227 | .767～10 | 13.64% | .496～.971 | .5～1 | 0 / 2 |
| spiky_marginal | -2.763～51.183 | 0～10 | 18.07% | .187～1 | .125～1 | 6 / 76 |

theta 对迭代和残差均不单调，常有高低交替。random_dense 后期交替振幅增大；1d_squared 多次触顶。压力样本约第 30～60 步存在近零有效系数、残差停滞和局部上升，不能仅用最终收敛概括其过程。它在第 30、54、57、58、59、60 步因 `ineffective_theta` 显式回退，全部成本已经计入；共 45 次上界截断、6 次下界截断。

前两个主要样本的回溯集中在初期，之后 h 多为 1；压力样本有更多正性保护。eta 是线性化缺陷，不是非线性一步残差缩减率；它不会随 R 必然下降。图中另画 h*theta，并以更新前的 `direction_residual_l1` 对齐 theta 与 R，避免错位一轮。

无正式运行失败；压力样本的退化事件保存在 summary 和 history。只覆盖一个种子、两个尺寸及固定上界，三次计时仍有系统波动，没有扫描上界或专门调参，也未验证普遍规律。

## 验证与改动文件

66 项测试通过，包括原 Sinkhorn/LMFlow，以及第二行线性缺陷、区间最小残差、精确列归一化、GS 端点、显式失败/回退和保存读取。24 份正式结果重新读取，全部收敛；最终列边缘最大 L1 误差为 1.634e-15。旧历史结果仍可读取，曲线已查看。

新增：`solvers/theta_balance.py`、`tests/test_theta_balance.py`、`configs/theta_balance_pilot.yaml`、`scripts/run_theta_balance_pilot.py`、本报告。

小幅修改：`types.py`（独立配置与默认记录字段）、`solvers/sinkhorn.py`（提取稳定化列子步）、`solvers/base.py`（共同监控/计数，总计时覆盖最终诊断）、`policies/stepsize.py`（可选分项计数）、`experiments/registry.py`（独立 solver 注册）、`io/result_io.py`（新字段读取）。源码均位于 `src/sinkhorn_lmflow/` 下。未修改旧报告、旧算法 policy、README 或忽略规则；新数据和图未暂存、未提交。
