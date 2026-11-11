# 完整成本距离驱动 Sinkhorn：v0 实测报告

本轮实际计划 262 条，状态统计：{"success": 250, "not_applicable_kernel_underflow": 12}。正式输入覆盖 39/39；三方法两次均成功的输入 33/39。机器累计时间约 2266.5 秒（报告生成期间仍计费）；未完成项保留在 pending.csv。

相对 sk_log_fast：暂无双方验收成功的完成时间配对，不能判断加速。
相对 sk_scaling_fast：暂无双方验收成功的完成时间配对，不能判断加速。

开销观察：distance_build_sec 中位数 0.00251；energy_matvec_sec 中位数 0.00173；objective_diff_sec 中位数 0.141；trial_state_sec 中位数 0.0922；rejected_trials 中位数 0；fallback_count 中位数 0。接受步减少并不自动意味着总墙钟收益；下方按家族配对表包含本轮重新计时的两种 SK。目标差、D@e 是 iteration_sec 的子项，不能再次加到总时间。

方向是否改变及是否有用：见方向诊断和六例消融。cos 越接近 1，方向越接近原对数残差；幅度匹配与完整方向各走独立轨道。有限消融只有一次计时，不能据此宣称统计显著性。若消融未完成，机制问题仍未回答。

## 固定协议与参数

统一 kappa=3.0，选择原因 `success_count_then_common_geomean_then_smaller`。gamma：最小 0.08718，中位 0.42，最大 40.46（含预热收敛/退化的 0）。

使用 OT Python 3.9、串行、BLAS 线程 1；环境详见 environment.json。tol=1e-8，max_iter=50000，最多 20 步共享预热。log SK 和新方法各自总时间都含预热；原始缩放 SK 冷启动并计入核构造。M5 按真实非零灰度支持计算，尺寸见 inputs/*/metadata.json 与 runs.csv。没有运行 5000 或 10000 求解。

实现指纹：`3983cd0a5337589c26f93d4e624ef873d24d21d886f3b40b6316a9c7b9fb5d3e`。配置指纹：`7ffaaff2ae918dcf9911acf65a63ce89fe2d81d40c10c213291312d74ace1aeb`。起点提交由 environment.json 记录；任务开始时为 61af801ed0b89a0c2b6c23ac1a29382f02a3e886，用户已有 notes 改动保留。代码快照在 source/。

## 覆盖与失败

| stage | family | method | status | count |
| --- | --- | --- | --- | --- |
| ablation | M1 | energy_amp | success | 1 |
| ablation | M1 | energy_zero | success | 1 |
| ablation | M2 | energy_amp | success | 1 |
| ablation | M2 | energy_zero | success | 1 |
| ablation | M3 | energy_amp | success | 1 |
| ablation | M3 | energy_zero | success | 1 |
| ablation | M4 | energy_amp | success | 1 |
| ablation | M4 | energy_zero | success | 1 |
| ablation | M5 | energy_amp | success | 2 |
| ablation | M5 | energy_zero | success | 2 |
| calibration | M1 | energy_distance | success | 3 |
| calibration | M1 | sk_log_fast | success | 1 |
| calibration | M2 | energy_distance | success | 3 |
| calibration | M2 | sk_log_fast | success | 1 |
| calibration | M3 | energy_distance | success | 3 |
| calibration | M3 | sk_log_fast | success | 1 |
| calibration | M4 | energy_distance | success | 3 |
| calibration | M4 | sk_log_fast | success | 1 |
| main | M1 | energy_distance | success | 8 |
| main | M1 | sk_log_fast | success | 8 |
| main | M1 | sk_scaling_fast | not_applicable_kernel_underflow | 4 |
| main | M1 | sk_scaling_fast | success | 4 |
| main | M2 | energy_distance | success | 8 |
| main | M2 | sk_log_fast | success | 8 |
| main | M2 | sk_scaling_fast | not_applicable_kernel_underflow | 4 |
| main | M2 | sk_scaling_fast | success | 4 |
| main | M3 | energy_distance | success | 8 |
| main | M3 | sk_log_fast | success | 8 |
| main | M3 | sk_scaling_fast | not_applicable_kernel_underflow | 4 |
| main | M3 | sk_scaling_fast | success | 4 |
| main | M4 | energy_distance | success | 8 |
| main | M4 | sk_log_fast | success | 8 |
| main | M4 | sk_scaling_fast | success | 8 |
| main | M5 | energy_distance | success | 40 |
| main | M5 | sk_log_fast | success | 40 |
| main | M5 | sk_scaling_fast | success | 40 |
| main | M6 | energy_distance | success | 4 |
| main | M6 | sk_log_fast | success | 4 |
| main | M6 | sk_scaling_fast | success | 4 |
| main | M7 | energy_distance | success | 2 |
| main | M7 | sk_log_fast | success | 2 |
| main | M7 | sk_scaling_fast | success | 2 |

not_applicable_kernel_underflow 是原始缩放表示不适用，不当作收敛失败。timeout、资源跳过、预算未启动均保留在计划分母，不用截止时间伪装成功。终点依据保存的 mu、nu 独立分块重算；审计详见 endpoint_audit.json。

## 各家族、epsilon 的配对总时间比

暂无已完成且可比较的数据。

同一输入的两次计时不是独立数据样本；不报告由此生成的置信区间。预热内收敛的收益不归因于新驱动，配对表有 warmup_converged 标志。

## 最差反例

暂无已完成且可比较的数据。

## 有限消融与方向诊断

| case_id | method | status | total_sec | n_iter | gamma | fallback_count |
| --- | --- | --- | --- | --- | --- | --- |
| M1_N1024_s42_e0.001 | energy_zero | success | 16.19 | 233 | 0 | 0 |
| M1_N1024_s42_e0.001 | energy_amp | success | 15.8 | 153 | 13.91 | 0 |
| M2_N1024_s42_e0.003 | energy_zero | success | 27.66 | 577 | 0 | 0 |
| M2_N1024_s42_e0.003 | energy_amp | success | 14.68 | 303 | 0.4102 | 0 |
| M3_N1024_s42_e0.003 | energy_zero | success | 43.69 | 881 | 0 | 0 |
| M3_N1024_s42_e0.003 | energy_amp | success | 22.29 | 463 | 0.2918 | 0 |
| M4_N1024_s42_a0.1_e0.003 | energy_zero | success | 34.45 | 779 | 0 | 0 |
| M4_N1024_s42_a0.1_e0.003 | energy_amp | success | 16.97 | 413 | 0.3161 | 0 |
| M5_same_0_e0.001 | energy_zero | success | 1.255 | 983 | 0 | 0 |
| M5_same_0_e0.001 | energy_amp | success | 0.7631 | 520 | 0.7267 | 0 |
| M5_cross_0_1_e0.001 | energy_zero | success | 0.8392 | 733 | 0 | 0 |
| M5_cross_0_1_e0.001 | energy_amp | success | 0.5138 | 405 | 1.232 | 0 |

| case_id | method | median_cos | min_cos | median_q | fallbacks |
| --- | --- | --- | --- | --- | --- |
| M2_N1024_s42_e0.003 | energy_distance | 0.9235 | 0.8802 | 1.296 | 0 |
| M3_N1024_s42_e0.003 | energy_distance | 0.8217 | 0.8217 | 1.637 | 0 |
| M1_N1024_s42_e0.001 | energy_distance | 0.9117 | 0.899 | 4.339 | 0 |
| M4_N1024_s42_a0.1_e0.003 | energy_distance | 0.8213 | 0.8093 | 1.447 | 0 |
| M5_same_0_e0.001 | energy_distance | 0.8824 | 0.8139 | 2.688 | 0 |
| M5_cross_0_1_e0.001 | energy_distance | 0.8943 | 0.6733 | 1.154 | 0 |
| M1_N1024_s42_e0.001 | energy_zero | 1 | 1 | 0 | 0 |
| M1_N1024_s42_e0.001 | energy_amp | 1 | 1 | 3.148 | 0 |
| M2_N1024_s42_e0.003 | energy_zero | 1 | 1 | 0 | 0 |
| M2_N1024_s42_e0.003 | energy_amp | 1 | 1 | 1.784 | 0 |
| M3_N1024_s42_e0.003 | energy_zero | 1 | 1 | 0 | 0 |
| M3_N1024_s42_e0.003 | energy_amp | 1 | 1 | 1.757 | 0 |
| M4_N1024_s42_a0.1_e0.003 | energy_zero | 1 | 1 | 0 | 0 |
| M4_N1024_s42_a0.1_e0.003 | energy_amp | 1 | 1 | 1.026 | 0 |
| M5_same_0_e0.001 | energy_zero | 1 | 1 | 0 | 0 |
| M5_same_0_e0.001 | energy_amp | 1 | 1 | 1.875 | 0 |
| M5_cross_0_1_e0.001 | energy_zero | 1 | 1 | 0 | 0 |
| M5_cross_0_1_e0.001 | energy_amp | 1 | 1 | 2.64 | 0 |

## 理论边界

Energy distance 是已有构造。本方法是由成本几何构造的预条件梯度方法，保持 EOT 镜像几何与原平衡点，并提供连续下降方向；离散加速没有先验证明。固定成本距离不移除实际质量响应 L_P，也不保证跨越所有弱连接。使用距离而非平方距离，不构造 Hessian、谱分解或近似几何。结果仅覆盖至 2048，不能外推更大规模。

## 产物与恢复

结果目录：`D:\sinkhorn-lmflow\results\20261111_121604_269934_energy_distance_v0_rev0`。

CSV：calibration.csv、runs.csv、threshold_costs.csv、coverage.csv、failures.csv、pending.csv、paired_speedups.csv、direction_summary.csv。

恢复执行：`scripts\run_ot.cmd scripts/run_energy_distance_study.py --config configs/energy_distance_v0.yaml`。同一账本累计预算，匹配终态不重跑；实现变更必须使用 --new-revision，仍保留累计账本。预算已耗尽时恢复不会增加求解额度。

只重建报告：`scripts\run_ot.cmd scripts/report_energy_distance.py --run-dir "D:\sinkhorn-lmflow\results\20261111_121604_269934_energy_distance_v0_rev0" --audit --report docs/energy_distance_v0_report.md`。

单独执行报告命令的机器开销记录到 report_rebuilds.json，不作为新增正式求解时间。首次自动报告包含在实验账本；没有复用旧 SR1 结果或账本。

## 图表

![performance](D:/sinkhorn-lmflow/results/20261111_121604_269934_energy_distance_v0_rev0/figures/performance.png)

![convergence](D:/sinkhorn-lmflow/results/20261111_121604_269934_energy_distance_v0_rev0/figures/convergence.png)

![costs](D:/sinkhorn-lmflow/results/20261111_121604_269934_energy_distance_v0_rev0/figures/costs.png)

![mechanisms](D:/sinkhorn-lmflow/results/20261111_121604_269934_energy_distance_v0_rev0/figures/mechanisms.png)

![scale_memory](D:/sinkhorn-lmflow/results/20261111_121604_269934_energy_distance_v0_rev0/figures/scale_memory.png)

本次自动报告耗时 11.90 秒；报告完成时机器累计 2271.04 秒。
