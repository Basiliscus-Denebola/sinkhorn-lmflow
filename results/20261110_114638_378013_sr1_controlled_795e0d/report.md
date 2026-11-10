# SR1 v0 controlled 结果

计划 54 个 case × repeat 配对；状态：{'completed': 54, 'partial': 0, 'pending': 0}。所有 attempt、失败和 pending 保留在 all_runs.csv。

方法共用行归一化观测位置与 20 次 Sinkhorn 预热。受控种子仅改变初值；重复编号仅用于计时。
时间包含方向、试探和诊断，排除保存与绘图；双方各承担一次共享预热。state_eval_count 包含已知参考态的一次诊断评估。
稳定目标差每次单独计数；能量仅在初始、预热末、首次跨越阈值和 max_iter 检查点计算。稠密遍历代理不是 FLOPs。
Bregman 核仅在数值评估中重新归一化行条件分布；实际 P、边际和停止容差不变。未使用 roundoff_residual 接受。

## 最终阈值 1e-8

| case | 方法 | 成功/已运行 | 迭代中位数 | 评估中位数 | 目标差核中位数 | 总秒中位数 [最小,最大] | 后期秒中位数 |
|---|---|---:|---:|---:|---:|---:|---:|
| N256_B2_d0.001_s42 | mirror_sr1 | 3/3 | 24 | 26 | 24 | 0.07447 [0.06438, 0.121] | 0.01543 |
| N256_B2_d0.001_s42 | sinkhorn_reduced | 3/3 | 5833 | 5835 | 5833 | 18.91 [16.39, 19.91] | 18.85 |
| N256_B2_d0.001_s43 | mirror_sr1 | 3/3 | 23 | 25 | 23 | 0.08184 [0.06402, 0.08855] | 0.02017 |
| N256_B2_d0.001_s43 | sinkhorn_reduced | 3/3 | 5362 | 5364 | 5362 | 14.87 [14.07, 17.64] | 14.82 |
| N256_B2_d0.001_s44 | mirror_sr1 | 3/3 | 24 | 26 | 24 | 0.08763 [0.08299, 0.1354] | 0.01776 |
| N256_B2_d0.001_s44 | sinkhorn_reduced | 3/3 | 5833 | 5835 | 5833 | 20.57 [17.05, 23.54] | 20.45 |
| N256_B2_d0.01_s42 | mirror_sr1 | 3/3 | 24 | 26 | 24 | 0.1084 [0.08809, 0.1121] | 0.02461 |
| N256_B2_d0.01_s42 | sinkhorn_reduced | 3/3 | 695 | 697 | 695 | 2.733 [2.406, 3.248] | 2.639 |
| N256_B2_d0.01_s43 | mirror_sr1 | 3/3 | 23 | 25 | 23 | 0.08703 [0.07662, 0.1038] | 0.01446 |
| N256_B2_d0.01_s43 | sinkhorn_reduced | 3/3 | 649 | 651 | 649 | 2.208 [2.147, 2.478] | 2.12 |
| N256_B2_d0.01_s44 | mirror_sr1 | 3/3 | 24 | 26 | 24 | 0.09235 [0.07091, 0.1099] | 0.01793 |
| N256_B2_d0.01_s44 | sinkhorn_reduced | 3/3 | 695 | 697 | 695 | 2.461 [2.427, 2.544] | 2.368 |
| N256_B2_d0.1_s42 | mirror_sr1 | 3/3 | 22 | 24 | 22 | 0.09202 [0.08359, 0.09647] | 0.01236 |
| N256_B2_d0.1_s42 | sinkhorn_reduced | 3/3 | 77 | 79 | 77 | 0.2872 [0.2763, 0.2936] | 0.2095 |
| N256_B2_d0.1_s43 | mirror_sr1 | 3/3 | 21 | 23 | 21 | 0.1107 [0.08743, 0.141] | 0.007714 |
| N256_B2_d0.1_s43 | sinkhorn_reduced | 3/3 | 73 | 75 | 73 | 0.3189 [0.2643, 0.4144] | 0.216 |
| N256_B2_d0.1_s44 | mirror_sr1 | 3/3 | 22 | 24 | 22 | 0.09573 [0.08354, 0.1129] | 0.01192 |
| N256_B2_d0.1_s44 | sinkhorn_reduced | 3/3 | 77 | 79 | 77 | 0.3037 [0.2615, 0.3399] | 0.2198 |
| N256_B4_d0.001_s42 | mirror_sr1 | 3/3 | 1792 | 1794 | 1792 | 5.987 [5.6, 7.324] | 5.903 |
| N256_B4_d0.001_s42 | sinkhorn_reduced | 3/3 | 4380 | 4382 | 4380 | 13.82 [12.79, 15.98] | 13.75 |
| N256_B4_d0.001_s43 | mirror_sr1 | 3/3 | 1157 | 1159 | 1157 | 2.91 [2.659, 3.602] | 2.859 |
| N256_B4_d0.001_s43 | sinkhorn_reduced | 3/3 | 4496 | 4498 | 4496 | 12.43 [10.83, 13.79] | 12.38 |
| N256_B4_d0.001_s44 | mirror_sr1 | 3/3 | 827 | 829 | 827 | 2.317 [1.913, 2.661] | 2.268 |
| N256_B4_d0.001_s44 | sinkhorn_reduced | 3/3 | 3786 | 3788 | 3786 | 8.681 [8.343, 9.573] | 8.632 |
| N256_B4_d0.01_s42 | mirror_sr1 | 3/3 | 264 | 266 | 264 | 1.042 [1.021, 1.064] | 0.9784 |
| N256_B4_d0.01_s42 | sinkhorn_reduced | 3/3 | 525 | 527 | 525 | 1.786 [1.595, 1.931] | 1.73 |
| N256_B4_d0.01_s43 | mirror_sr1 | 3/3 | 232 | 234 | 232 | 0.895 [0.8816, 1.084] | 0.8315 |
| N256_B4_d0.01_s43 | sinkhorn_reduced | 3/3 | 536 | 538 | 536 | 2.047 [1.986, 2.345] | 1.984 |
| N256_B4_d0.01_s44 | mirror_sr1 | 3/3 | 225 | 227 | 225 | 0.9157 [0.9048, 0.9484] | 0.8523 |
| N256_B4_d0.01_s44 | sinkhorn_reduced | 3/3 | 464 | 466 | 464 | 1.895 [1.753, 2.232] | 1.783 |
| N256_B4_d0.1_s42 | mirror_sr1 | 3/3 | 28 | 30 | 28 | 0.09347 [0.09323, 0.1253] | 0.03712 |
| N256_B4_d0.1_s42 | sinkhorn_reduced | 3/3 | 58 | 60 | 58 | 0.2161 [0.1942, 0.2677] | 0.1612 |
| N256_B4_d0.1_s43 | mirror_sr1 | 3/3 | 27 | 29 | 27 | 0.1028 [0.1019, 0.1152] | 0.02786 |
| N256_B4_d0.1_s43 | sinkhorn_reduced | 3/3 | 59 | 61 | 59 | 0.2347 [0.2049, 0.2465] | 0.1579 |
| N256_B4_d0.1_s44 | mirror_sr1 | 3/3 | 27 | 29 | 27 | 0.09944 [0.08395, 0.111] | 0.03059 |
| N256_B4_d0.1_s44 | sinkhorn_reduced | 3/3 | 53 | 55 | 53 | 0.1918 [0.1585, 0.2214] | 0.124 |

有效配对的最终总时间比 T_SK/T_SR1：中位数、最小、最大 = ('3.126', '1.499', '268.7')。预热已达阈值或任一失败的配对不报告加速倍数。

## 诊断与限制

G、Q 和夹角对应真正接受的 SR1 步，记录在终点行，from_iteration 指向起点。回退行不复用已拒绝方向的预测值。
有效方向的同点耗散关系不是整条离散轨道加速定理；实际收益须同时看评估数、额外目标差核和时间。
所有阈值均取同一轨道首次接受状态；初始已达阈值记零时间。缺失值不是零成本。
图中红叉标记非收敛终点；能量图仅绘制有效正值，不裁剪非正能量。完整值在 history.csv。

[全部运行](tables/all_runs.csv) · [阈值成本](tables/threshold_costs.csv) · [中位数及范围](tables/threshold_summary.csv) · [配对比值](tables/paired_speedups.csv) · [保护统计](tables/sr1_events.csv)

## 数值诊断摘要

最新 attempt 的 SR1 事件汇总：`{'attempts': 13278, 'valid': 13278, 'accepted': 13278, 'fallbacks': 0, 'skips': 0, 'exhausted': 0, 'rejected': 0}`。

合法方向比例 100.00%，接受比例 100.00%，SK 回退比例 0.00%。

跳过原因合计：`{}`。

| case | 接受步 G 中位数的中位数 | 接受步 Q 中位数的中位数 | 最小接受步长 |
|---|---:|---:|---:|
| N256_B2_d0.001_s42 | 416.8 | 0.5563 | 1 |
| N256_B2_d0.001_s43 | 483.1 | 0.5171 | 1 |
| N256_B2_d0.001_s44 | 416.8 | 0.5563 | 1 |
| N256_B2_d0.01_s42 | 47.1 | 0.5301 | 1 |
| N256_B2_d0.01_s43 | 49.47 | 0.5078 | 1 |
| N256_B2_d0.01_s44 | 47.1 | 0.5301 | 1 |
| N256_B2_d0.1_s42 | 5.263 | 0.5 | 1 |
| N256_B2_d0.1_s43 | 5.263 | 0.5 | 1 |
| N256_B2_d0.1_s44 | 5.263 | 0.5 | 1 |
| N256_B4_d0.001_s42 | 3.74 | 0.6316 | 1 |
| N256_B4_d0.001_s43 | 7.374 | 0.59 | 1 |
| N256_B4_d0.001_s44 | 8.822 | 0.582 | 1 |
| N256_B4_d0.01_s42 | 2.614 | 0.6709 | 1 |
| N256_B4_d0.01_s43 | 3.485 | 0.6249 | 1 |
| N256_B4_d0.01_s44 | 2.582 | 0.6712 | 1 |
| N256_B4_d0.1_s42 | 2.089 | 0.673 | 1 |
| N256_B4_d0.1_s43 | 3.108 | 0.5045 | 1 |
| N256_B4_d0.1_s44 | 3.349 | 0.503 | 1 |

以上 G/Q 只统计接受的 SR1 步；跳过或耗尽后的 Sinkhorn 步不使用失败方向的预测量。
若 G 大而 Q 小、步长缩小，说明同点增益未完全转化为有限步下降；最终成本比判断额外工作是否抵消收益。

## 图表

### N256_B2_d0.1_s42
![收敛](figures/N256_B2_d0.1_s42_convergence.png)
![诊断](figures/N256_B2_d0.1_s42_diagnostics.png)

### N256_B2_d0.1_s43
![收敛](figures/N256_B2_d0.1_s43_convergence.png)
![诊断](figures/N256_B2_d0.1_s43_diagnostics.png)

### N256_B2_d0.1_s44
![收敛](figures/N256_B2_d0.1_s44_convergence.png)
![诊断](figures/N256_B2_d0.1_s44_diagnostics.png)

### N256_B2_d0.01_s42
![收敛](figures/N256_B2_d0.01_s42_convergence.png)
![诊断](figures/N256_B2_d0.01_s42_diagnostics.png)

### N256_B2_d0.01_s43
![收敛](figures/N256_B2_d0.01_s43_convergence.png)
![诊断](figures/N256_B2_d0.01_s43_diagnostics.png)

### N256_B2_d0.01_s44
![收敛](figures/N256_B2_d0.01_s44_convergence.png)
![诊断](figures/N256_B2_d0.01_s44_diagnostics.png)

### N256_B2_d0.001_s42
![收敛](figures/N256_B2_d0.001_s42_convergence.png)
![诊断](figures/N256_B2_d0.001_s42_diagnostics.png)

### N256_B2_d0.001_s43
![收敛](figures/N256_B2_d0.001_s43_convergence.png)
![诊断](figures/N256_B2_d0.001_s43_diagnostics.png)

### N256_B2_d0.001_s44
![收敛](figures/N256_B2_d0.001_s44_convergence.png)
![诊断](figures/N256_B2_d0.001_s44_diagnostics.png)

### N256_B4_d0.1_s42
![收敛](figures/N256_B4_d0.1_s42_convergence.png)
![诊断](figures/N256_B4_d0.1_s42_diagnostics.png)

### N256_B4_d0.1_s43
![收敛](figures/N256_B4_d0.1_s43_convergence.png)
![诊断](figures/N256_B4_d0.1_s43_diagnostics.png)

### N256_B4_d0.1_s44
![收敛](figures/N256_B4_d0.1_s44_convergence.png)
![诊断](figures/N256_B4_d0.1_s44_diagnostics.png)

### N256_B4_d0.01_s42
![收敛](figures/N256_B4_d0.01_s42_convergence.png)
![诊断](figures/N256_B4_d0.01_s42_diagnostics.png)

### N256_B4_d0.01_s43
![收敛](figures/N256_B4_d0.01_s43_convergence.png)
![诊断](figures/N256_B4_d0.01_s43_diagnostics.png)

### N256_B4_d0.01_s44
![收敛](figures/N256_B4_d0.01_s44_convergence.png)
![诊断](figures/N256_B4_d0.01_s44_diagnostics.png)

### N256_B4_d0.001_s42
![收敛](figures/N256_B4_d0.001_s42_convergence.png)
![诊断](figures/N256_B4_d0.001_s42_diagnostics.png)

### N256_B4_d0.001_s43
![收敛](figures/N256_B4_d0.001_s43_convergence.png)
![诊断](figures/N256_B4_d0.001_s43_diagnostics.png)

### N256_B4_d0.001_s44
![收敛](figures/N256_B4_d0.001_s44_convergence.png)
![诊断](figures/N256_B4_d0.001_s44_diagnostics.png)

## 复现与恢复

` .\scripts\run_ot.cmd scripts/run_sr1_study.py --config "results\20261110_114638_378013_sr1_controlled_795e0d/study_config.yaml" `

`.\scripts\run_ot.cmd scripts/run_sr1_study.py --config "results\20261110_114638_378013_sr1_controlled_795e0d/study_config.yaml" --resume "results\20261110_114638_378013_sr1_controlled_795e0d"`

`.\scripts\run_ot.cmd scripts/report_sr1.py --run-dir "results\20261110_114638_378013_sr1_controlled_795e0d"`

恢复检查配置、数学输入、初值指纹，跳过有完整产物的成功配对；未完成或失败配对新增 attempt，保留旧结果。
初值目录使用 SHA256 前 16 位防止 Windows 长路径，metadata 保存并核对完整 SHA256；不同问题的完整初始势分开保存。
