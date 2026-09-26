# θ 系统实验与固定 θ 消融

本报告只使用已保存的本轮实测数据；三次计时先在实例内取中位数，再跨实例汇总。

结果目录：[20260925_073951_550196_theta_balance_study_b8ea20](../results/20260925_073951_550196_theta_balance_study_b8ea20/)；[总体 CSV](../results/20260925_073951_550196_theta_balance_study_b8ea20/summary.csv)、[阈值 CSV](../results/20260925_073951_550196_theta_balance_study_b8ea20/thresholds.csv)、[共享索引](../results/20260925_073951_550196_theta_balance_study_b8ea20/manifest.json)。

## 实际覆盖与结论

| status | 正式重复次数 |
| --- | --- |
| converged | 627 |
| failed | 2 |
| max_iter | 8 |
| pending | 391 |
| skipped_repeat | 201 |
| timeout | 91 |

| block | 计划组合 | 已启动 | 三次收敛 | 未收敛 | 计时未齐 | 尚未启动 |
| --- | --- | --- | --- | --- | --- | --- |
| smoke | 8 | 8 | 8 | 0 | 0 | 0 |
| A_small | 96 | 96 | 88 | 8 | 0 | 0 |
| B | 72 | 66 | 58 | 6 | 2 | 6 |
| C | 180 | 165 | 76 | 87 | 2 | 15 |
| A_large | 96 | 0 | 0 | 0 | 0 | 96 |
| D | 12 | 0 | 0 | 0 | 0 | 12 |

块覆盖按问题/方法组合计数。A/B 的重叠行引用同一批实测结果，不重复执行或重复计入总体样本数。

计划 440 个问题/方法组合（含原 pilot 回归、高精度）；已执行 311 个，三次均收敛 206 个。未执行不计为失败；确定性失败、max_iter 或 timeout 后不机械补齐其余重复。

主实验 Sinkhorn/自适应计划 120 个独立问题；主对照已执行 69 个，其中双方三次均收敛的 61 个问题中，自适应实际总时间获胜 61 个。这些配对时间比的中位数为 1.816×，范围 [1.041, 4.300]×。

以下精确时间比仅包含双方完整收敛的配对，成功数另列，不能据此掩盖失败或未执行。

三次完整收敛的组合中，有 32/206 个的最大/最小耗时比超过1.2，跨度最大为 6.9。原始三次时间保留在 CSV；接近1的速度比需结合这一波动解读。

已记录的正式求解累计 17415.2s，其中固定 θ 占 14917.1s （85.7%）；这些时间不含预热、公共初始化和文件读写。

## 本轮判断与未完成范围

**最新两小时续跑：**本轮实验耗时7115.9秒（1小时58分36秒），低于7200秒预算；新增158次正式运行，其中114次收敛、42次超时、2次线搜索失败。原有570次结果全部保留，累计728次正式运行：627收敛、91超时、8达到迭代上限、2次线搜索失败。每个失败或超时后的剩余重复按协议跳过。

**已完成覆盖：**A的两种成本、n=100/500、四个ε、均匀边际均已执行。B/C的随机成本全部18个问题已按协议执行完；一维成本的均匀边际、ε=.01的random_positive/spiky及ε=.001的random_positive前两个种子均已执行完。B/C共36个问题，32个已执行完，1个只缺成功方法的第三次计时，3个尚未开始。完成执行不等于收敛：超时和失败同样保留。

**收益范围：**本轮新增8个双方三次均收敛的完整主对照，均为一维成本非均匀边际，自适应全部更快。ε=.01时，random_positive和spiky各3个种子的配对加速比中位数分别为3.901和4.244；ε=.001的random_positive前两个种子为3.936，第三个种子仅完成两次计时，不并入完整配对统计。累计61个完整配对均由自适应获胜，中位数1.816、范围1.041～4.300。原均匀边际44个成功配对未重跑，中位数仍为1.660；random_dense、n=100、ε=.05约1.047，收益较弱且需结合计时波动。

**明确失败边界：**random_dense、n=500、ε=.001、spiky的3个种子，Sinkhorn均超时；自适应在seed=42/2026上分别接受151/1258步后发生Armijo线搜索失败，最终L1残差约.015627/.001409，seed=123则超时。所有固定θ在这3个问题上也超时。不能为这些样本报告达标加速比，也不能把“61个成功配对全部更快”外推为全部问题都能收敛。原random_dense、n=100中ε=.003/seed=42和ε=.001/三个种子的双方20000步上限问题仍保留。

**固定θ不能统一替代自适应：**固定θ=1/1.5/2/4/8各已在33个B/C问题上执行，三次完整收敛数分别为28/29/14/3/2；各自完整配对子集内均慢于自适应。θ=1.5的29个完整配对中，自适应/固定耗时比中位数为0.5097，约快1.96倍。一维random_positive、ε=.01、seed=42的θ=2需9017步、计时中位数约91秒，而自适应65步、约.68秒。θ=1仍作为GS整步及实现开销核对；没有逐题选择最优固定θ作为默认算法。

**计时波动与混合状态：**一维random_positive、ε=.001、seed=42的固定θ=1第一次约29.65秒收敛，第二次在120秒超时，因此只保留两次已执行结果，第三次按协议跳过。相邻的固定θ=1.5计时为约19.81/86.77/21.14秒，自适应约7.36/27.28/7.35秒；说明此时存在明显计时波动。其原因未单独测量，不能直接归因于算法。三次中位数与原始时间均保留，只有双方三次完整收敛才给精确时间比。

**保护机制：**累计69个主实验的首条自适应轨迹，共111485个接受步；上界截断1297步、下界截断64步、h<1有813步、Sinkhorn回退64步。自由θ范围[-121.475, 788.371]，负值原样保留。新增非均匀边际在中后期也出现保护动作，不能沿用原均匀样本“保护仅在早期出现”的观察；存在保护仍可能线搜索失败或超时。这里统计接受步，终止时失败试步的原因另见事件记录。分段表为实例内比例再取跨实例中位数，表中0不代表所有实例都未触发保护。

**诊断图的范围：**规模、边际、固定θ和保护机制汇总包含本轮新增数据。代表性残差、θ轨迹和预测/真实L2散点仍沿用原4个均匀样本，不能据此解释所有新增失败；非均匀样本的负θ、范围和事件见本段、分段表及原始历史。预测与真实缩减均使用行L2，不能替代L1停机条件；高低交替等现象不是收敛定理。

**预算：**本次正式求解7050.6秒，其中固定θ耗时6187.6秒，占87.8%。执行顺序仍为smoke→A小规模→B/C→A大规模→D，没有为绕过慢分支而换序；本轮未发生检查点异常。只将总预算由3600改为7200秒，数值算法、单次120秒上限、三次计时协议与源码指纹保持一致。正常续跑默认预算为7200秒。

**还缺391次正式重复：**B/C剩余67次，包括一维成本n=500、ε=.001、spiky的3个种子×7种方法×3次计时（63次），以及一维random_positive、ε=.001、seed=2026的Sinkhorn、自适应、固定θ=1、1.5各第3次计时（4次；前两次均收敛）。A的n=1000/2000全部288次和D的1e-10子集全部36次尚未执行。另201次按失败/超时协议跳过，与391次待执行分开。原pilot的Dirichlet spiky只作回归，不与本轮spiky定义混用。


## 规模与 ε（A，均匀边际）

| cost | n | epsilon | problems | 状态 | Sinkhorn成功 | 自适应成功 | 配对数 | 时间比中位数 | S秒 | A秒 | S迭代 | A迭代 | Sproxy | Aproxy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1d_squared | 100 | 0.001 | 3 | 已执行 | 3 | 3 | 3 | 1.594 | 0.9044 | 0.5493 | 3291 | 1623 | 2.633e+04 | 2.112e+04 |
| 1d_squared | 100 | 0.003 | 3 | 已执行 | 3 | 3 | 3 | 1.651 | 0.2683 | 0.1731 | 1098 | 538 | 8790 | 7005 |
| 1d_squared | 100 | 0.01 | 3 | 已执行 | 3 | 3 | 3 | 1.528 | 0.08581 | 0.05618 | 323 | 163 | 2590 | 2132 |
| 1d_squared | 100 | 0.05 | 3 | 已执行 | 3 | 3 | 3 | 1.457 | 0.01477 | 0.00966 | 61 | 32 | 494 | 423 |
| 1d_squared | 500 | 0.001 | 3 | 已执行 | 3 | 3 | 3 | 1.906 | 35.06 | 18.15 | 3040 | 1523 | 2.433e+04 | 1.981e+04 |
| 1d_squared | 500 | 0.003 | 3 | 已执行 | 3 | 3 | 3 | 2.012 | 10.91 | 5.512 | 1020 | 514 | 8166 | 6689 |
| 1d_squared | 500 | 0.01 | 3 | 已执行 | 3 | 3 | 3 | 1.981 | 3.216 | 1.629 | 308 | 156 | 2470 | 2037 |
| 1d_squared | 500 | 0.05 | 3 | 已执行 | 3 | 3 | 3 | 1.956 | 0.6202 | 0.3486 | 60 | 33 | 486 | 440 |
| 1d_squared | 1000 | 0.001 | 3 | 未执行 | 0 | 0 | 0 | — | — | — | — | — | — | — |
| 1d_squared | 1000 | 0.003 | 3 | 未执行 | 0 | 0 | 0 | — | — | — | — | — | — | — |
| 1d_squared | 1000 | 0.01 | 3 | 未执行 | 0 | 0 | 0 | — | — | — | — | — | — | — |
| 1d_squared | 1000 | 0.05 | 3 | 未执行 | 0 | 0 | 0 | — | — | — | — | — | — | — |
| 1d_squared | 2000 | 0.001 | 3 | 未执行 | 0 | 0 | 0 | — | — | — | — | — | — | — |
| 1d_squared | 2000 | 0.003 | 3 | 未执行 | 0 | 0 | 0 | — | — | — | — | — | — | — |
| 1d_squared | 2000 | 0.01 | 3 | 未执行 | 0 | 0 | 0 | — | — | — | — | — | — | — |
| 1d_squared | 2000 | 0.05 | 3 | 未执行 | 0 | 0 | 0 | — | — | — | — | — | — | — |
| random_dense | 100 | 0.001 | 3 | 已执行 | 0 | 0 | 0 | — | — | — | — | — | — | — |
| random_dense | 100 | 0.003 | 3 | 已执行 | 2 | 2 | 2 | 2.8 | 3.575 | 1.295 | 1.382e+04 | 3864 | 1.106e+05 | 5.024e+04 |
| random_dense | 100 | 0.01 | 3 | 已执行 | 3 | 3 | 3 | 1.52 | 0.1453 | 0.09558 | 90 | 45 | 726 | 592 |
| random_dense | 100 | 0.05 | 3 | 已执行 | 3 | 3 | 3 | 1.047 | 0.02198 | 0.02047 | 13 | 9 | 110 | 124 |
| random_dense | 500 | 0.001 | 3 | 已执行 | 3 | 3 | 3 | 2.116 | 37.21 | 13.54 | 2587 | 863 | 2.07e+04 | 1.123e+04 |
| random_dense | 500 | 0.003 | 3 | 已执行 | 3 | 3 | 3 | 1.665 | 0.5916 | 0.3575 | 54 | 32 | 438 | 423 |
| random_dense | 500 | 0.01 | 3 | 已执行 | 3 | 3 | 3 | 1.396 | 0.1443 | 0.1034 | 13 | 9 | 110 | 124 |
| random_dense | 500 | 0.05 | 3 | 已执行 | 3 | 3 | 3 | 1.258 | 0.05936 | 0.04719 | 5 | 4 | 46 | 59 |
| random_dense | 1000 | 0.001 | 3 | 未执行 | 0 | 0 | 0 | — | — | — | — | — | — | — |
| random_dense | 1000 | 0.003 | 3 | 未执行 | 0 | 0 | 0 | — | — | — | — | — | — | — |
| random_dense | 1000 | 0.01 | 3 | 未执行 | 0 | 0 | 0 | — | — | — | — | — | — | — |
| random_dense | 1000 | 0.05 | 3 | 未执行 | 0 | 0 | 0 | — | — | — | — | — | — | — |
| random_dense | 2000 | 0.001 | 3 | 未执行 | 0 | 0 | 0 | — | — | — | — | — | — | — |
| random_dense | 2000 | 0.003 | 3 | 未执行 | 0 | 0 | 0 | — | — | — | — | — | — | — |
| random_dense | 2000 | 0.01 | 3 | 未执行 | 0 | 0 | 0 | — | — | — | — | — | — | — |
| random_dense | 2000 | 0.05 | 3 | 未执行 | 0 | 0 | 0 | — | — | — | — | — | — | — |

## 边际与固定 θ（B/C）

| method | 计划问题 | 已执行 | 三次收敛 | 有效配对 | 时间比中位数 | 获胜数 | 成功耗时秒 | 成功迭代中位数 | 成功proxy中位数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sinkhorn | 36 | 33 | 29 | 29 | 1 | 0 | 4.093 | 396 | 3174 |
| adaptive | 36 | 33 | 29 | 29 | 2.471 | 29 | 1.69 | 153 | 1996 |
| fixed_1 | 36 | 33 | 28 | 28 | 0.9882 | 11 | 4.038 | 364 | 4025 |
| fixed_1.5 | 36 | 33 | 29 | 29 | 1.423 | 23 | 2.846 | 261 | 2884 |
| fixed_2 | 36 | 33 | 14 | 14 | 0.4783 | 4 | 50.04 | 4482 | 4.983e+04 |
| fixed_4 | 36 | 33 | 3 | 3 | 0.3391 | 1 | 27.68 | 1543 | 2.199e+04 |
| fixed_8 | 36 | 33 | 2 | 2 | 0.1601 | 0 | 44.71 | 1783 | 3.128e+04 |

固定 θ=1 是 GS 整步核对及实现开销参照，不作为独立算法优势。每个固定值都保留成绩；固定的是提议 θ，Armijo 仍可改变 h，因此 θ=2/4/8 的 hθ 可能相同。没有逐题挑最优后包装成新方法。各方法的有效配对子集可能不同，须结合已执行数、完整收敛数判断稳定性。

额外选 θ 的代价是否值得，直接看自适应与各个全局固定值的配对：大于1表示固定值更快。

| method | 完整配对数 | 固定值更快的题数 | 自适应除以固定的时间比 | 最小比 | 最大比 |
| --- | --- | --- | --- | --- | --- |
| fixed_1 | 28 | 0 | 0.4189 | 0.2267 | 0.7687 |
| fixed_1.5 | 29 | 0 | 0.5097 | 0.3264 | 0.7904 |
| fixed_2 | 14 | 0 | 0.13 | 0.007466 | 0.6793 |
| fixed_4 | 3 | 0 | 0.1194 | 0.05895 | 0.6555 |
| fixed_8 | 2 | 0 | 0.05334 | 0.01633 | 0.09036 |

| cost | epsilon | marginal | 状态 | 计划 | S成功 | A成功 | 配对 | 时间比 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1d_squared | 0.001 | random_positive | 已执行 | 3 | 2 | 2 | 2 | 3.936 |
| 1d_squared | 0.001 | spiky | 未执行 | 3 | 0 | 0 | 0 | — |
| 1d_squared | 0.001 | uniform | 已执行 | 3 | 3 | 3 | 3 | 1.906 |
| 1d_squared | 0.003 | uniform | 已执行 | 3 | 3 | 3 | 3 | 2.012 |
| 1d_squared | 0.01 | random_positive | 已执行 | 3 | 3 | 3 | 3 | 3.901 |
| 1d_squared | 0.01 | spiky | 已执行 | 3 | 3 | 3 | 3 | 4.244 |
| 1d_squared | 0.01 | uniform | 已执行 | 3 | 3 | 3 | 3 | 1.981 |
| 1d_squared | 0.05 | uniform | 已执行 | 3 | 3 | 3 | 3 | 1.956 |
| random_dense | 0.001 | random_positive | 已执行 | 3 | 3 | 3 | 3 | 3.234 |
| random_dense | 0.001 | spiky | 已执行 | 3 | 0 | 0 | 0 | — |
| random_dense | 0.001 | uniform | 已执行 | 3 | 3 | 3 | 3 | 2.116 |
| random_dense | 0.003 | uniform | 已执行 | 3 | 3 | 3 | 3 | 1.665 |
| random_dense | 0.01 | random_positive | 已执行 | 3 | 3 | 3 | 3 | 1.364 |
| random_dense | 0.01 | spiky | 已执行 | 3 | 3 | 3 | 3 | 2.839 |
| random_dense | 0.01 | uniform | 已执行 | 3 | 3 | 3 | 3 | 1.396 |
| random_dense | 0.05 | uniform | 已执行 | 3 | 3 | 3 | 3 | 1.258 |

## 精度与时间口径

1e-4/1e-6/1e-8 为完整轨迹首次命中时间，不是独立求解总时间。D 的协议设定为从同一公共初值重新完整运行至 1e-10（不是把后半段耗时当总时间）；共享输入/初值，单独标记 highprecision。实际执行状态如下。

| target | S至少一次命中 | A至少一次命中 | S三次命中 | A三次命中 | 配对数 | 时间比中位数 |
| --- | --- | --- | --- | --- | --- | --- |
| 1e-08 | 62 | 62 | 61 | 61 | 61 | 1.817 |
| 1e-06 | 64 | 63 | 61 | 61 | 61 | 1.888 |
| 0.0001 | 68 | 67 | 61 | 61 | 61 | 1.811 |

| cost | marginal | method | status | elapsed_sec | n_iter | residual_l1 |
| --- | --- | --- | --- | --- | --- | --- |
| random_dense | uniform | sinkhorn | pending | — | — | — |
| random_dense | uniform | adaptive | pending | — | — | — |
| random_dense | random_positive | sinkhorn | pending | — | — | — |
| random_dense | random_positive | adaptive | pending | — | — | — |
| random_dense | spiky | sinkhorn | pending | — | — | — |
| random_dense | spiky | adaptive | pending | — | — | — |
| 1d_squared | uniform | sinkhorn | pending | — | — | — |
| 1d_squared | uniform | adaptive | pending | — | — | — |
| 1d_squared | random_positive | sinkhorn | pending | — | — | — |
| 1d_squared | random_positive | adaptive | pending | — | — | — |
| 1d_squared | spiky | sinkhorn | pending | — | — | — |
| 1d_squared | spiky | adaptive | pending | — | — | — |

## θ 与保护机制

每步 θ/h 与更新前 R 对齐；残差分段为 R>1e-2、1e-6<R≤1e-2、R≤1e-6。summary.csv 保存每条轨迹各段的 min/q10/median/q90/max 和截断/回溯/回退比例。下表先逐实例计算再取中位数，每个实例只用首条正式轨迹，避免把计时重复当独立样本。

| band | 实例数 | free q10 | free q50 | free q90 | θ中位数 | hθ中位数 | 上截断率 | 下截断率 | h<1比例 | 回退率 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| early_R>1e-2 | 69 | 1.465 | 1.897 | 2.257 | 1.897 | 1.629 | 0 | 0 | 0 | 0 |
| middle_1e-6<R<=1e-2 | 68 | 1.266 | 1.76 | 2.279 | 1.76 | 1.76 | 0 | 0 | 0 | 0 |
| late_R<=1e-6 | 63 | 1.255 | 1.915 | 2.169 | 1.915 | 1.915 | 0 | 0 | 0 | 0 |

比例以0～1表示。各残差段参与实例数不同，尚未到达后期的实例不进入后期统计。

所有已记录主实验方向的 θ_free 范围为 [-121.475, 788.371]，截断 θ 范围 [0, 10]，非回退 hθ 范围 [2.32269e-13, 10]。

预测值是 ||r−hθs||₂/||r||₂，真实值是 ||r_next||₂/||r||₂；η 对应 h=1 的线性预测。这些量不能直接解释 L1 停止指标。固定分支不计算 qS、s、θ_free、η 或预测比，空值表示不适用。回退的 hθ 为空；负 θ_free 原样保留；失败试步原因保存在 summary 的事件记录。

## 失败、超时、最慢与未执行

| label | method | repeat | status | stop_reason | elapsed_sec | residual_l1 |
| --- | --- | --- | --- | --- | --- | --- |
| random_dense_n100_e0.003_uniform_s42 | sinkhorn | 1 | max_iter | max_iter | 33.63 | 8.954e-07 |
| random_dense_n100_e0.003_uniform_s42 | adaptive | 1 | max_iter | max_iter | 37.24 | 1.358e-07 |
| random_dense_n100_e0.001_uniform_s42 | sinkhorn | 1 | max_iter | max_iter | 8.682 | 1.086e-05 |
| random_dense_n100_e0.001_uniform_s42 | adaptive | 1 | max_iter | max_iter | 10.55 | 2.304e-06 |
| random_dense_n100_e0.001_uniform_s123 | sinkhorn | 1 | max_iter | max_iter | 8.14 | 4.173e-06 |
| random_dense_n100_e0.001_uniform_s123 | adaptive | 1 | max_iter | max_iter | 9.611 | 1.27e-06 |
| random_dense_n100_e0.001_uniform_s2026 | sinkhorn | 1 | max_iter | max_iter | 8.127 | 6.344e-06 |
| random_dense_n100_e0.001_uniform_s2026 | adaptive | 1 | max_iter | max_iter | 9.716 | 1.808e-06 |
| random_dense_n500_e0.01_uniform_s42 | fixed_2 | 1 | timeout | timeout | 120 | 2.021e-08 |
| random_dense_n500_e0.01_uniform_s42 | fixed_4 | 1 | timeout | timeout | 120 | 3.538e-08 |
| random_dense_n500_e0.01_uniform_s42 | fixed_8 | 1 | timeout | timeout | 120 | 5.668e-08 |
| random_dense_n500_e0.01_uniform_s123 | fixed_2 | 1 | timeout | timeout | 120 | 1.201e-08 |
| random_dense_n500_e0.01_uniform_s123 | fixed_4 | 1 | timeout | timeout | 120 | 6.768e-08 |
| random_dense_n500_e0.01_uniform_s123 | fixed_8 | 1 | timeout | timeout | 120 | 3.323e-08 |
| random_dense_n500_e0.01_uniform_s2026 | fixed_2 | 1 | timeout | timeout | 120 | 2.194e-08 |
| random_dense_n500_e0.01_uniform_s2026 | fixed_4 | 1 | timeout | timeout | 120 | 4.748e-08 |
| random_dense_n500_e0.01_uniform_s2026 | fixed_8 | 1 | timeout | timeout | 120 | 7.559e-08 |
| random_dense_n500_e0.001_uniform_s42 | fixed_4 | 1 | timeout | timeout | 120 | 2.115e-06 |
| random_dense_n500_e0.001_uniform_s42 | fixed_8 | 1 | timeout | timeout | 120 | 6.515e-06 |
| random_dense_n500_e0.001_uniform_s123 | fixed_2 | 1 | timeout | timeout | 120 | 1.075e-07 |
| random_dense_n500_e0.001_uniform_s123 | fixed_4 | 1 | timeout | timeout | 120 | 2.045e-05 |
| random_dense_n500_e0.001_uniform_s123 | fixed_8 | 1 | timeout | timeout | 120 | 2.041e-05 |
| random_dense_n500_e0.001_uniform_s2026 | fixed_2 | 1 | timeout | timeout | 120 | 1.681e-08 |
| random_dense_n500_e0.001_uniform_s2026 | fixed_4 | 1 | timeout | timeout | 120 | 2.61e-06 |
| random_dense_n500_e0.001_uniform_s2026 | fixed_8 | 1 | timeout | timeout | 120 | 3.419e-06 |
| 1d_squared_n500_e0.01_uniform_s42 | fixed_4 | 1 | timeout | timeout | 120 | 6.896e-08 |
| 1d_squared_n500_e0.01_uniform_s42 | fixed_8 | 1 | timeout | timeout | 120 | 3.805e-08 |
| 1d_squared_n500_e0.01_uniform_s123 | fixed_4 | 1 | timeout | timeout | 120 | 5.684e-08 |
| 1d_squared_n500_e0.01_uniform_s123 | fixed_8 | 1 | timeout | timeout | 120 | 9.838e-08 |
| 1d_squared_n500_e0.01_uniform_s2026 | fixed_4 | 1 | timeout | timeout | 120 | 2.299e-08 |
| 1d_squared_n500_e0.01_uniform_s2026 | fixed_8 | 1 | timeout | timeout | 120 | 2.482e-05 |
| 1d_squared_n500_e0.001_uniform_s42 | fixed_8 | 1 | timeout | timeout | 120 | 7.731e-07 |
| 1d_squared_n500_e0.001_uniform_s123 | fixed_2 | 1 | timeout | timeout | 120 | 1.048e-08 |
| 1d_squared_n500_e0.001_uniform_s123 | fixed_4 | 1 | timeout | timeout | 120 | 1.381e-06 |
| 1d_squared_n500_e0.001_uniform_s123 | fixed_8 | 1 | timeout | timeout | 120 | 4.945e-06 |
| 1d_squared_n500_e0.001_uniform_s2026 | fixed_4 | 1 | timeout | timeout | 120 | 3.345e-05 |
| 1d_squared_n500_e0.001_uniform_s2026 | fixed_8 | 1 | timeout | timeout | 120 | 4.06e-08 |
| random_dense_n500_e0.01_random_positive_s42 | fixed_2 | 1 | timeout | timeout | 120 | 6.761e-08 |
| random_dense_n500_e0.01_random_positive_s42 | fixed_4 | 1 | timeout | timeout | 120 | 1.256e-07 |
| random_dense_n500_e0.01_random_positive_s42 | fixed_8 | 1 | timeout | timeout | 120 | 1.353e-06 |
| random_dense_n500_e0.01_random_positive_s123 | fixed_2 | 1 | timeout | timeout | 120 | 6.65e-07 |
| random_dense_n500_e0.01_random_positive_s123 | fixed_4 | 1 | timeout | timeout | 120 | 8.138e-07 |
| random_dense_n500_e0.01_random_positive_s123 | fixed_8 | 1 | timeout | timeout | 120 | 9.368e-07 |
| random_dense_n500_e0.01_random_positive_s2026 | fixed_2 | 1 | timeout | timeout | 120 | 1.649e-07 |
| random_dense_n500_e0.01_random_positive_s2026 | fixed_4 | 1 | timeout | timeout | 120 | 2.369e-07 |
| random_dense_n500_e0.01_random_positive_s2026 | fixed_8 | 1 | timeout | timeout | 120 | 3.055e-07 |
| random_dense_n500_e0.01_spiky_s42 | fixed_4 | 1 | timeout | timeout | 120 | 3.817e-08 |
| random_dense_n500_e0.01_spiky_s42 | fixed_8 | 1 | timeout | timeout | 120 | 3.41e-07 |
| random_dense_n500_e0.01_spiky_s123 | fixed_4 | 1 | timeout | timeout | 120 | 1.316e-07 |
| random_dense_n500_e0.01_spiky_s123 | fixed_8 | 1 | timeout | timeout | 120 | 1.627e-07 |
| random_dense_n500_e0.001_random_positive_s42 | fixed_2 | 1 | timeout | timeout | 120 | 1.571e-06 |
| random_dense_n500_e0.001_random_positive_s42 | fixed_4 | 1 | timeout | timeout | 120 | 2.17e-08 |
| random_dense_n500_e0.001_random_positive_s42 | fixed_8 | 1 | timeout | timeout | 120 | 6.041e-08 |
| random_dense_n500_e0.001_random_positive_s123 | fixed_8 | 1 | timeout | timeout | 120 | 1.523e-08 |
| random_dense_n500_e0.001_random_positive_s2026 | fixed_4 | 1 | timeout | timeout | 120 | 5.509e-07 |
| random_dense_n500_e0.001_random_positive_s2026 | fixed_8 | 1 | timeout | timeout | 120 | 5.528e-07 |
| random_dense_n500_e0.001_spiky_s42 | sinkhorn | 1 | timeout | timeout | 120 | 2.823e-05 |
| random_dense_n500_e0.001_spiky_s42 | adaptive | 1 | failed | armijo_line_search_failed | 3.058 | 0.01563 |
| random_dense_n500_e0.001_spiky_s42 | fixed_1 | 1 | timeout | timeout | 120 | 2.527e-05 |
| random_dense_n500_e0.001_spiky_s42 | fixed_1.5 | 1 | timeout | timeout | 120.1 | 2.022e-05 |
| random_dense_n500_e0.001_spiky_s42 | fixed_2 | 1 | timeout | timeout | 120.1 | 0.0002604 |
| random_dense_n500_e0.001_spiky_s42 | fixed_4 | 1 | timeout | timeout | 120.1 | 0.0009045 |
| random_dense_n500_e0.001_spiky_s42 | fixed_8 | 1 | timeout | timeout | 120.1 | 0.001081 |
| random_dense_n500_e0.001_spiky_s123 | sinkhorn | 1 | timeout | timeout | 120.1 | 0.0001851 |
| random_dense_n500_e0.001_spiky_s123 | adaptive | 1 | timeout | timeout | 120.1 | 3.993e-05 |
| random_dense_n500_e0.001_spiky_s123 | fixed_1 | 1 | timeout | timeout | 120 | 0.0005353 |
| random_dense_n500_e0.001_spiky_s123 | fixed_1.5 | 1 | timeout | timeout | 120 | 0.0001723 |
| random_dense_n500_e0.001_spiky_s123 | fixed_2 | 1 | timeout | timeout | 120 | 1.765e-08 |
| random_dense_n500_e0.001_spiky_s123 | fixed_4 | 1 | timeout | timeout | 120 | 1.866e-07 |
| random_dense_n500_e0.001_spiky_s123 | fixed_8 | 1 | timeout | timeout | 120 | 5.117e-06 |
| random_dense_n500_e0.001_spiky_s2026 | sinkhorn | 1 | timeout | timeout | 120 | 4.296e-07 |
| random_dense_n500_e0.001_spiky_s2026 | adaptive | 1 | failed | armijo_line_search_failed | 24.88 | 0.001409 |
| random_dense_n500_e0.001_spiky_s2026 | fixed_1 | 1 | timeout | timeout | 120 | 4.114e-07 |
| random_dense_n500_e0.001_spiky_s2026 | fixed_1.5 | 1 | timeout | timeout | 120 | 1.377e-07 |
| random_dense_n500_e0.001_spiky_s2026 | fixed_2 | 1 | timeout | timeout | 120 | 5.605e-08 |
| random_dense_n500_e0.001_spiky_s2026 | fixed_4 | 1 | timeout | timeout | 120 | 5.257e-07 |
| random_dense_n500_e0.001_spiky_s2026 | fixed_8 | 1 | timeout | timeout | 120 | 1.884e-06 |
| 1d_squared_n500_e0.01_random_positive_s42 | fixed_4 | 1 | timeout | timeout | 120 | 4.538e-08 |
| 1d_squared_n500_e0.01_random_positive_s42 | fixed_8 | 1 | timeout | timeout | 120 | 6.058e-08 |
| 1d_squared_n500_e0.01_random_positive_s123 | fixed_2 | 1 | timeout | timeout | 120 | 1.205e-08 |
| 1d_squared_n500_e0.01_random_positive_s123 | fixed_4 | 1 | timeout | timeout | 120 | 7.094e-08 |
| 1d_squared_n500_e0.01_random_positive_s123 | fixed_8 | 1 | timeout | timeout | 120 | 8.549e-08 |
| 1d_squared_n500_e0.01_random_positive_s2026 | fixed_2 | 1 | timeout | timeout | 120 | 1.417e-08 |
| 1d_squared_n500_e0.01_random_positive_s2026 | fixed_4 | 1 | timeout | timeout | 120 | 9.685e-07 |
| 1d_squared_n500_e0.01_spiky_s42 | fixed_2 | 1 | timeout | timeout | 120 | 2.179e-07 |
| 1d_squared_n500_e0.01_spiky_s42 | fixed_4 | 1 | timeout | timeout | 120 | 1.892e-05 |
| 1d_squared_n500_e0.01_spiky_s42 | fixed_8 | 1 | timeout | timeout | 120 | 4.231e-05 |
| 1d_squared_n500_e0.01_spiky_s123 | fixed_4 | 1 | timeout | timeout | 120 | 3.803e-08 |
| 1d_squared_n500_e0.01_spiky_s123 | fixed_8 | 1 | timeout | timeout | 120 | 2.092e-08 |
| 1d_squared_n500_e0.01_spiky_s2026 | fixed_2 | 1 | timeout | timeout | 120 | 1.255e-06 |
| 1d_squared_n500_e0.01_spiky_s2026 | fixed_4 | 1 | timeout | timeout | 120 | 9.387e-08 |
| 1d_squared_n500_e0.01_spiky_s2026 | fixed_8 | 1 | timeout | timeout | 120 | 4.509e-08 |
| 1d_squared_n500_e0.001_random_positive_s42 | fixed_1 | 2 | timeout | timeout | 120.1 | 4.389e-07 |
| 1d_squared_n500_e0.001_random_positive_s42 | fixed_2 | 1 | timeout | timeout | 120 | 1.526e-08 |
| 1d_squared_n500_e0.001_random_positive_s42 | fixed_4 | 1 | timeout | timeout | 120 | 2.776e-05 |
| 1d_squared_n500_e0.001_random_positive_s42 | fixed_8 | 1 | timeout | timeout | 120.1 | 3.139e-05 |
| 1d_squared_n500_e0.001_random_positive_s123 | fixed_4 | 1 | timeout | timeout | 120 | 3.971e-07 |
| 1d_squared_n500_e0.001_random_positive_s123 | fixed_8 | 1 | timeout | timeout | 120 | 1.055e-07 |
| 1d_squared_n500_e0.001_random_positive_s2026 | fixed_2 | 1 | timeout | timeout | 120 | 3.832e-05 |
| 1d_squared_n500_e0.001_random_positive_s2026 | fixed_4 | 1 | timeout | timeout | 120 | 3.493e-05 |
| 1d_squared_n500_e0.001_random_positive_s2026 | fixed_8 | 1 | timeout | timeout | 120 | 5.147e-08 |

耗时最长的十个已执行问题/方法组合（实例内中位数）：

| label | method | status | elapsed_sec | n_iter | residual_l1 |
| --- | --- | --- | --- | --- | --- |
| random_dense_n500_e0.001_spiky_s42 | fixed_8 | skipped_repeat,timeout | 120.1 | 732 | 0.001081 |
| 1d_squared_n500_e0.001_random_positive_s42 | fixed_8 | skipped_repeat,timeout | 120.1 | 4771 | 3.139e-05 |
| random_dense_n500_e0.001_spiky_s42 | fixed_4 | skipped_repeat,timeout | 120.1 | 841 | 0.0009045 |
| random_dense_n500_e0.001_spiky_s123 | adaptive | skipped_repeat,timeout | 120.1 | 1249 | 3.993e-05 |
| random_dense_n500_e0.001_spiky_s42 | fixed_2 | skipped_repeat,timeout | 120.1 | 1294 | 0.0002604 |
| random_dense_n500_e0.001_spiky_s42 | fixed_1.5 | skipped_repeat,timeout | 120.1 | 6627 | 2.022e-05 |
| random_dense_n500_e0.001_spiky_s123 | sinkhorn | skipped_repeat,timeout | 120.1 | 2130 | 0.0001851 |
| random_dense_n500_e0.001_uniform_s42 | fixed_8 | skipped_repeat,timeout | 120 | 3577 | 6.515e-06 |
| random_dense_n500_e0.001_spiky_s123 | fixed_1.5 | skipped_repeat,timeout | 120 | 1362 | 0.0001723 |
| random_dense_n500_e0.001_spiky_s123 | fixed_1 | skipped_repeat,timeout | 120 | 1392 | 0.0005353 |

尚未执行正式重复数：391；具体标签、方法和重复号见 summary.csv 的 pending 行。

## 配置、环境与复现

环境 `D:\miniconda_envs\OT\python.exe`；Python 3.9.24，NumPy 2.0.2，SciPy 1.13.1。CPU float64，BLAS 单线程；基础 commit `94db3aa0cb15550454b5d1ae82540638664f6fb2`，源码指纹 `033f5de7695153f52b641f6c81e4c35225fabd622cc46a824ef6311d320fbcf1`。

最大尺寸 2000，按 12 个稠密 float64 数组保守估计 366.2 MiB（非峰值实测，不含解释器/绘图）。串行运行，不保存每步 P。

成本复用原生成器并除以原最大值，原尺度、成本/边际/问题/初值指纹保存于 CSV/manifest。uniform、独立 Uniform(.1,1) 归一化、2% 位置分配90%质量的 spiky 为本轮边际；与旧 pilot 的 Dirichlet 边际不同，回归块单列。seeds=42/123/2026。

主容差 1e-08、max_iter=20000、每次 120 秒迭代边界上限，本会话预算 7200 秒；预算尾部不足一次完整上限时不启动新运行。θ_max=10，h初值1，Armijo=1e-4，收缩.5，最多60次回溯。每个方法20步/2秒有界预热，正式三次，方法顺序交替；重启进程重新预热。双方同一公共 GS 初值，初始化单列。

总时间含内部初始化、方向、全部试步、正性/Armijo、列更新、回退、共同监控、最终诊断；不含公共 GS、保存和绘图。两边逐步记录且关闭谱/SVD。成绩是当前统一监控框架内的性能。diagnostic_sec 继承旧格式，只累计谱诊断，本轮为0不代表共同监控没有成本。dense-pass proxy 为 P构造+全矩阵归约+LSE+P向量乘的未加权计数，不是 FLOPs 或等价 matvec；最终 transport objective、向量算术和分配不计入 proxy，但计入总时间。

累计输入处理及逐次结果写入时间约 91.085s（不含索引检查点写入，另列）；本次分析/绘图耗时见 manifest.analysis_sec。全体运行保留配置和运行环境；同问题只存一份输入。断点续跑按问题、初值、solver配置和源码指纹校验。

主运行命令：

```powershell
.\scripts\run_ot.cmd scripts/run_theta_balance_pilot.py --config configs/theta_balance_study.yaml
```

同一目录续跑 / 仅分析：

```powershell
.\scripts\run_ot.cmd scripts/run_theta_balance_pilot.py --resume D:/sinkhorn-lmflow/results/20260925_073951_550196_theta_balance_study_b8ea20
.\scripts\run_ot.cmd scripts/run_theta_balance_pilot.py --analyze D:/sinkhorn-lmflow/results/20260925_073951_550196_theta_balance_study_b8ea20
```

续跑默认使用索引中的原配置。需要更改本次预算时，只修改同一 YAML 的 budget_sec，并在 --resume 命令后添加 --config configs/theta_balance_study.yaml；数值协议改动会被拒绝。

没有矩形、真实数据、GPU 或额外文献基线。本轮不改变算法及保护规则，不把相关性写成普遍定理。

## 保存结果复核

已复核 728 份正式结果、73 份共享输入；320 份固定 θ 运行的 p_matvec_count 全为0，选参诊断全为空。首轮正式结果最终列 L1 残差最大 1.03e-14。前后残差对齐、成本/边际切片不变性和原 pilot 迭代数均通过检查。

元数据修正：259 份 runtime_metadata 最初继承了总控 seed=42，现已按共享问题的实际 seed 更正；问题数据、轨迹、状态和计时均未改动。

## 验证与本次文件改动

本次对全部728份正式结果、73份共享输入完成审计。检查输入/初值指纹、ε与边际切片不变性、前后残差对齐、历史结果读取及原pilot迭代数复现。320份固定θ结果的选参P向量乘计数全部为0，不适用诊断为空。首轮结果最终列L1残差最大约1.03e-14。此前71项单元测试通过，本次未改数值实现，未重复运行测试。

| 文件 | 本次改动 |
| --- | --- |
| `configs/theta_balance_study.yaml` | 总预算3600→7200秒；1行增加、1行删除。 |
| `docs/theta_balance_study.md` | 更新唯一主报告的累计数据、失败边界、覆盖、计时波动和续跑说明。 |
| 原结果目录 | 追加逐次结果和共享输入，更新CSV、manifest、审计及六张汇总图；未改旧运行的数值结果。 |
| 原结果目录的`scratch` | 保存570次运行的续跑前基线及本次报告核查脚本。 |

求解器、生成器、实验入口和测试源码本次均未修改；未自动提交代码。运行、同目录续跑及仅分析命令见上文。


## 总体图

代表曲线为 seed=42 的首条正式轨迹，展示随机成本 n=100/ε=.001 与 n=500/ε=.01，一维成本 n=500/ε=.01 与 .001，均为 uniform。此处按已观测的快速、缓慢及未收敛情形选图，总体统计仍保留全部实例。未执行的方法不补曲线。时间统计先取实例内三次中位数。保护率柱图先按实例计算比例再求均值，正文分段表取实例中位数。

![01_scale_epsilon](../results/20260925_073951_550196_theta_balance_study_b8ea20/figures/01_scale_epsilon.png)

![02_marginal_precision](../results/20260925_073951_550196_theta_balance_study_b8ea20/figures/02_marginal_precision.png)

![03_fixed_theta](../results/20260925_073951_550196_theta_balance_study_b8ea20/figures/03_fixed_theta.png)

![04_residuals](../results/20260925_073951_550196_theta_balance_study_b8ea20/figures/04_residuals.png)

![05_theta](../results/20260925_073951_550196_theta_balance_study_b8ea20/figures/05_theta.png)

![06_safeguards_prediction](../results/20260925_073951_550196_theta_balance_study_b8ea20/figures/06_safeguards_prediction.png)
