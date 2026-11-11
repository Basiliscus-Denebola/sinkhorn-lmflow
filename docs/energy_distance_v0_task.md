# 完整成本距离驱动 Sinkhorn：v0 实现与一小时实验计划

日期：2026-11-11  
用途：将本文件交给桌面 Codex，在现有 sinkhorn-lmflow 仓库中完成实现、验证、受预算约束的实验和中文报告。  
已核查的仓库：https://github.com/Basiliscus-Denebola/sinkhorn-lmflow  
阅读基准提交：61af801ed0b89a0c2b6c23ac1a29382f02a3e886  
建议新分支：codex/energy-distance-v0

## 0. 执行目标与冻结范围

请直接完成本文件规定的工作，不止输出设计或代码框架。先读取工作区的 AGENTS.md 和 git status；在保留用户改动的前提下创建独立分支，必要时使用 worktree。当前工作区若比上述提交更新，以实际源码为准，记录差异，不能为了匹配本计划而回退代码。

本轮检验一个明确候选：保持 EOT 诱导的镜像几何不变，用完整成本列距离构造 energy distance 失衡能量，将其梯度加入 Sinkhorn 对数残差驱动。它属于由成本几何构造的预条件梯度方法。不要把它宣传为“超出预条件梯度范畴”，也不要把已有 energy distance 理论写成新发现。

已经确定：

- 主比较只有三个方法：现有 sk_log_fast、现有 sk_scaling_fast、新增 energy_distance。
- 39 个正式输入，每个方法 2 次计时，共 234 个主实验计划条目。
- 校准 16 条、有限消融 12 条，总计 262 个计划条目；小型单元测试不计入轨道数量。
- 最大维数 2048；不运行 5000 或 10000 的求解实验，不恢复旧 SR1 的 5000 任务。
- 不把 SR1、L-BFGS、Newton、低秩、稀疏化、特征值搜索、多重网格、动量或投影近似加入本轮主实验。
- 不排序列，不建立一维链，不用平方距离代替距离。
- 用户希望约一小时跑出首轮报告。开发和排错时间不属于这一个小时的机器实验预算；正式实验必须有累计预算和部分完成时的报告。
- 使用独立结果目录、配置、实现指纹和预算账本；不得重置、覆盖或挪用旧 SR1 账本。
- 默认完成本地分支上的实现、结果与报告，不自动推送或合并远端。

研究成功与否以结果决定。即使没有加速，也必须交付完整失败分析与可复现产物。

## 1. 现有仓库中优先复用的部分

遵守当前 AGENTS.md。已读版本要求：

- 使用 OT Conda 环境，路径 D:\miniconda_envs\OT。
- 保持 Python 3.9 兼容。
- 使用现有 scripts/run_ot.cmd 启动；正式计时串行执行，BLAS 线程固定为 1。
- 环境不可用时报告实际阻碍，不擅自改用其他环境来产生可比较的正式结果。

优先阅读并复用：

| 文件 | 可复用内容 |
|---|---|
| src/sinkhorn_lmflow/core/semi_dual.py | z=-nu/epsilon 的符号、center、稳定目标差、计数器 |
| src/sinkhorn_lmflow/solvers/sinkhorn_fast.py | LogEvaluator、LightState、FastSnapshot、共享预热、两种 SK |
| src/sinkhorn_lmflow/problems/mainstream.py | M1—M4、M6、M7 的确定性输入、独立随机流、成本归一化 |
| src/sinkhorn_lmflow/problems/mnist.py | 固定图像配对、零支持删除、真实像素质量 |
| src/sinkhorn_lmflow/experiments/sr1_mainstream.py | 预算、恢复、资源检查、终点审计、结果持久化 |
| src/sinkhorn_lmflow/plotting/sr1_mainstream.py | 仅从落盘产物生成表格和图片的框架 |
| docs/sr1_v1_mainstream_report.md | 已有计时口径、基线表现、失败边界 |

不要直接调用旧的完整 main_cases() 后启动其全部计划：它包含本轮排除的 5000，以及本轮不需要的其他方法和消融。新建明确的 39 例清单。

建议新增文件，具体模块拆分可按仓库风格调整：

- core/cost_profile_distance.py
- solvers/energy_distance.py
- experiments/energy_distance_study.py
- plotting/energy_distance.py
- scripts/run_energy_distance_study.py
- scripts/report_energy_distance.py
- configs/energy_distance_v0.yaml
- tests/test_cost_profile_distance.py
- tests/test_energy_distance.py
- tests/test_energy_distance_study.py
- docs/energy_distance_v0_report.md
- docs/energy_distance_v0_resume.md

本文件可保存为 docs/energy_distance_v0_task.md，作为执行规范。

## 2. 数学定义：必须与实现一致

以下公式统一编号为 (51,*)，以便后续讨论定位。先实现一般正边际且总质量相等的表达；现有正式输入仍使用总质量 1。

### 2.1 行约束精确满足的列参数化

令 s=sum(a)=sum(b)>0，K=exp(-M/epsilon)，z=-nu/epsilon。始终用行归一化状态：

$$
P_{ij}(z)=a_i\frac{K_{ij}e^{z_j}}{\sum_\ell K_{i\ell}e^{z_\ell}},
\qquad c=P^\top\mathbf1,\qquad e=c-b.
\tag{51,1}
$$

定义约化对偶最小化目标：

$$
f(z)=\sum_i a_i\log\sum_j e^{-M_{ij}/\varepsilon+z_j}-b^\top z,
\qquad \nabla_z f=e.
\tag{51,2}
$$

常数规范用 Pi=I-11^T/m，代码中只减向量均值，不显式构造 Pi。

$$
\psi=\log(c/b),\qquad z_{\rm SK}^{+}=\operatorname{center}(z-\psi).
\tag{51,3}
$$

计算 psi 时使用已有 log_c-log(b)，不要先把极小 c 指数化后再取 log。

### 2.2 完整成本列距离

用 w=a/s 定义：

$$
\mu_j=\sum_iw_iM_{ij},\qquad
X_{ij}=\frac{\sqrt{w_i}}{\varepsilon}(M_{ij}-\mu_j),
\qquad
D_{j\ell}=\|X_{:j}-X_{:\ell}\|_2.
\tag{51,4}
$$

等价地：

$$
D_{j\ell}^2=
\varepsilon^{-2}\sum_iw_i
\left[M_{ij}-M_{i\ell}-\sum_kw_k(M_{kj}-M_{k\ell})\right]^2.
\tag{51,5}
$$

D 是固定的完整距离矩阵，由 M、a、epsilon 决定，与当前 c 无关。不要随迭代重建。

它对 M'=M+alpha*1^T+1*beta^T 不变。这类行列加常数只改变平衡 OT 的目标常数，因此该不变性是必要的验证项。

通用构造使用 Gram：

$$
C=X^\top X,\qquad
D_{j\ell}=\sqrt{\max(C_{jj}+C_{\ell\ell}-2C_{j\ell},0)}.
\tag{51,6}
$$

要求：

- 不生成 n×m×m 三维差分数组。
- 只允许对浮点舍入量级的负平方距离归零；明显负值必须报告错误。
- 对称化并将对角线明确设为 0。
- 可先对所有列特征减去共同的特征向量以改善 Gram 相消；这是精确平移，不改变距离。
- 允许复用工作区；不要为 X、C、D 和 D² 同时保留不必要的副本。
- 不对 D 做 kNN、阈值稀疏化、低秩近似或按列重排来改变方法。

### 2.3 二维平方欧氏成本的精确快速路径

当前 M2—M7 都具有 M_ij=||x_i-y_j||²/h 的结构，h 是输入生成器实际使用的成本归一化因子。

$$
\bar x=\sum_iw_ix_i,\qquad
\Sigma_a=\sum_iw_i(x_i-\bar x)(x_i-\bar x)^\top,
$$
$$
D_{j\ell}^2=
\frac{4}{\varepsilon^2h^2}
(y_j-y_\ell)^\top\Sigma_a(y_j-y_\ell).
\tag{51,7}
$$

- M2/M3/M4/M6/M7：h 使用 metadata.raw_cost_max，坐标取 reconstruction 中的 x、y。
- M5：坐标已经除以 27；h=2。a 必须用实际灰度质量。
- M4 的 Sigma_a 必须使用实际非均匀 a，不能换成均匀平均。
- 小型样例验证该路径与通用公式一致后，正式几何输入使用快速路径；这只是同一个 D 的精确实现，不是额外算法变体。
- M1 随机成本使用通用 Gram 路径。
- 不从 M 中昂贵地重建坐标；只使用已有并经过验证的几何元数据。元数据不可用时回退通用路径并记录。

### 2.4 能量、梯度与离散方向

$$
\mathcal G_D(c)=-\frac14 e^\top D e,\qquad
R_A(c)=\sum_j[c_j\log(c_j/b_j)-c_j+b_j]+\gamma\mathcal G_D(c),
$$
$$
u=-\frac12De,\qquad
V=\psi+\gamma u,\qquad
z^+=\operatorname{center}(z-\eta V).
\tag{51,8}
$$

欧氏距离矩阵在零和子空间上条件负半定，所以该能量非负。这里用距离 D，不是距离平方。误用平方距离会退化为特征一阶矩匹配：

$$
-\frac14\sum_{j,\ell}e_je_\ell\|X_{:j}-X_{:\ell}\|^2
=\frac12\left\|\sum_je_jX_{:j}\right\|^2.
\tag{51,9}
$$

浮点实现细节：

- 原始 e=c-b 必须保留用于真实残差验收。
- 辅助能量与方向可使用 e0=center(e)，u0=center(-0.5*(D@e0))，v=center(psi)+gamma*u0。它消除舍入造成的极小总质量偏差，与精确公式在可行质量超平面上等价。
- 不通过修改 c、b 或边际地板来制造可行性。
- v 是实际规范化方向；方向导数计算 e@v。
- 一次 D@e 同时服务于 u、能量和必要标量，不重复乘法。
- 连续理论保证 dot(f)=-e^T V<0、dot(R_A)=-V^T L_P V<=0；离散实现只通过线搜索验收 f，不要求每个离散步都让 R_A 单调。
- 不显式构造 L_P、Hessian、逆矩阵或谱分解。

### 2.5 gamma 的一次性校准与冻结

每个输入先完成共享的最多 20 步 SK。若已经达到 tol，直接返回，不构造 D、不计算 gamma，并标记 warmup_converged。

否则在共同预热终点计算 psi0=center(psi)、u0=center(-0.5*D@center(e))：

$$
\gamma=\kappa\frac{\|\psi_0\|_2}{\|u_0\|_2},
\qquad \kappa\in\{0.3,1,3\}.
\tag{51,10}
$$

每条轨道的 gamma 此后固定，即使迭代中两种驱动的范数比变化，也不重新定标。正式实验使用一个统一的 kappa，不逐输入挑选最优值。

退化处理明确记录：

- D 全零或 u0 为零：gamma=0，标记 geometry_degenerate。
- 可用 100*machine_epsilon*max(1,||D||_infinity)*||e0||_2 作为 u0 的数值不可分辨尺度；触发时禁用附加项并记录阈值、范数与原因，不以任意分母加小常数强行制造巨大 gamma。
- gamma 非有限或方向出现非有限值：记录数值异常并使用预先规定的 SK 回退；不得悄悄截断参数。

### 2.6 线搜索、回退与高精度目标差

固定 Armijo sigma=1e-4、beta=1/2，从 eta=1 开始，最多尝试 12 个步长。

$$
f(\operatorname{center}(z-\eta V))-f(z)
\le -\sigma\eta\,e^\top V.
\tag{51,11}
$$

不能在高精度阶段仅用两个完整 f 值直接相减判断接受；优先复用已有 stable_objective_difference。可以将同一稳定公式按行分块实现降低内存，但必须先做等价验证。复用已有函数若需要构造 P，该成本必须计入。

- 正常步：v 有限、e@v>0，按上述规则试探。
- 非下降、非有限、所有 12 次试探失败：执行一个真实单位 SK 更新 z_new=center(z-psi)，记录 fallback_reason、拒绝次数和费用。
- 若单位 SK 也产生非有限状态，则该轨道记 numerical_failure，不继续输出伪成功。
- 预算或单轨道超时时，不为完成回退而超额继续求解，保存最后有效状态。
- gamma=0 的线搜索对照采用同样的试探、稳定目标差和回退规则。
- 不使用旧 SR1 的曲率历史、secant 判断或额外 Bregman 诊断来生成方向。

新算法所需状态只有 z、c、log_c、D 和稳定目标差所需数据。优先复用轻量状态，不无意携带整个 SR1 工作区。每一种实际计算都应计费。

## 3. 正式输入：固定 39 例

使用现有生成器、成本归一化规则、独立 SeedSequence 流和 MNIST 配对。不得为了让方法好看而换数据、重采样失败边际或增加质量地板。

| 类别 | n×m | epsilon | 其他参数 | 数量 |
|---|---|---|---|---:|
| M1 随机成本 | 1024×1024 | 0.003, 0.001 | seeds=42,43 | 4 |
| M2 均匀二维点云 | 1024×1024 | 0.003, 0.001 | seeds=42,43 | 4 |
| M3 混合高斯 | 1024×1024 | 0.003, 0.001 | seeds=42,43 | 4 |
| M4 非均匀边际 | 1024×1024 | 0.003 | seeds=42,43; alpha=1,0.1 | 4 |
| M5 MNIST | 删除零质量后的真实支持 | 0.01, 0.001 | 现有固定十对图像 | 20 |
| M6 规模补充 | 512×512, 2048×2048 | 0.003 | seed=42，混合高斯 | 2 |
| M7 非方阵 | 1024×2048 | 0.003 | seed=42 | 1 |
| 总计 | | | | 39 |

M5 配对名固定为：

same_0, same_2, same_4, same_6, same_8,
cross_0_1, cross_2_3, cross_4_5, cross_6_7, cross_8_9。

继续使用 choose_pairs 的 seed=20261110 和同一原始测试图像。保存原始索引、掩码、支持数和文件哈希；MNIST 实际支持数不能写成统一 784。

1024 的规模点复用 M3_N1024_s42_e0.003，不重复生成额外任务。点池内部保留 5000 个二维坐标是现有生成器的廉价实现，不等于运行 5000 规模；禁止的是构造并求解 5000×5000 正式问题。

## 4. 三个阶段与精确条目数

### 4.1 校准：16 条

四个独立校准输入：

| 校准名 | 家族 | 尺寸 | seed | epsilon | alpha |
|---|---|---|---:|---:|---|
| cal_M1 | M1 | 512×512 | 41 | 0.003 | 均匀 |
| cal_M2 | M2 | 512×512 | 41 | 0.003 | 均匀 |
| cal_M3 | M3 | 512×512 | 41 | 0.003 | 均匀 |
| cal_M4 | M4 | 512×512 | 41 | 0.003 | 0.1 |

每例：sk_log_fast 一次，energy_distance 的三个 kappa 各一次，共 4×4=16 条。校准只重复一次，不作为正式泛化效果证据。

选择规则事先冻结：

1. 优先选成功校准例数最多的 kappa。
2. 若成功数相同，在这些候选共同成功、且预热未直接收敛的校准例上，选择相对 log SK 总时间比的几何平均最小者；所有输入等权。
3. 若没有共同成功例或数值上并列，优先较小 kappa。
4. 若校准预算耗尽导致三档未完整比较，正式实验使用预先指定的 kappa=1，并标记 calibration_incomplete；不凭已完成的有利片段挑参数。
5. 若所有校准例均在预热内收敛，也使用 kappa=1，并标记校准未提供区分信息。

保存全部参数结果、选择依据和 selected_kappa.json。正式实验开始后禁止更改 kappa。

### 4.2 主比较：234 个计划条目

39 输入 × 3 方法 × 2 次计时 = 234。

- sk_log_fast：保持现有轻量实现。
- sk_scaling_fast：保持现有原始缩放实现，冷启动，核构造计费。
- energy_distance：本文件冻结的新算法，使用统一选定的 kappa。

新方法和 log SK 使用相同新鲜共享预热状态。原始缩放 SK 沿用当前冷启动口径；比较总时间，不把它的冷启动时间与其他方法的 post 时间直接相比。

K=exp(-M/epsilon) 含 0 或非有限元素时，原始缩放 SK 记 not_applicable_kernel_underflow 或已有相应状态，不进行截断，不算该算法收敛失败。此类条目仍属于 234 个计划条目。

先执行所有输入的 repeat=0，再执行 repeat=1。首轮中按家族轮转，尽早覆盖每个家族；不要把所有难例或整个 MNIST 家族都排到最后。方法顺序按输入与重复确定性轮换，避免所有新方法都固定排在热缓存位置。

### 4.3 有限消融：12 条

固定以下六例，均来自正式输入：

1. M1_N1024_s42_e0.001
2. M2_N1024_s42_e0.003
3. M3_N1024_s42_e0.003
4. M4_N1024_s42_a0.1_e0.003
5. M5_same_0_e0.001
6. M5_cross_0_1_e0.001

每例增加以下两种方法，各一次，使用该例 repeat=0 的相同预热状态。完整方法的结果复用正式主比较，不再重跑。

A. energy_zero：gamma=0，保持同样线搜索与回退。无需构造 D，不人为给它加上 D 的费用。

B. energy_amp：使用完整方法在共同预热点确定的同一 gamma，但在自己的当前状态上计算：

$$
V_{\rm full}(c)=\psi(c)+\gamma u(c),\qquad
\alpha(c)=\frac{e^\top V_{\rm full}(c)}{e^\top\psi(c)},
\qquad V_{\rm amp}(c)=\alpha(c)\psi(c).
\tag{51,12}
$$

此消融匹配当前点原目标的一阶下降幅度，保留原方向。必须走自己的真实轨道，不复用完整方法的接受步长或未来状态。需要计算 D 和 u，费用照实计入。分母不可分辨时按记录明确的单位 SK 回退处理。

消融只用于有限机制判断，不能由一次计时声称统计显著性。

总数断言：16+234+12=262。新 solver 内部回退、线搜索试探和单元测试都不另算新轨道。

## 5. 一小时运行预算与恢复

正式运行阶段的目标预算为 3600 秒；开发、安装和修复代码不计入该实验账本。开始计时前应完成核心正确性测试，避免在带缺陷的实现上烧完预算。

| 阶段 | 计划上限 |
|---|---:|
| 实验启动小检查及校准 | 360 s |
| 主比较 | 2640 s |
| 有限消融 | 360 s |
| 审计汇总、出图、报告 | 240 s |
| 总计 | 3600 s |

执行规则：

- 同时维护累计总墙钟和各阶段预算；数据构造、算法 setup、求解、实验内审计和 I/O 都消耗机器运行预算，不能用“未计入算法时间”来延长整轮运行。
- 未用完的前一阶段时间可以结转后续阶段，但必须预留汇总报告时间。
- 主求解最迟在累计运行约 3360 秒前停止启动新任务；保证最终能保存部分报告。
- 单轨道：最大维数<=1024 时上限 180 秒；超过 1024 时上限 300 秒。实际 deadline 取单轨道、阶段、总预算的最小值。
- max_iter=50000，tol=1e-8。到时保存最后有效状态，不增加上限来追求成功。
- 数值核之间检查 deadline，允许已进入的单个核有限越界，记录超额；不要承诺绝对精确的 3600 秒完成。
- 完成第一轮覆盖优先于全部第二次计时。未完成项记 pending_budget，不得删除以冒充完整计划完成。
- 未启动的完整方法对应消融可以记 dependency_missing，不能自动换成“表现更好”的代表例。
- 实现/配置/输入指纹、选择的 kappa、预算和完成任务落盘。恢复不得重置总预算；修改实现必须新 revision，并保留同一任务累计账本。
- 旧实验报告中的 5000 用户排除仍保留；本轮无此任务，不需要触碰旧 policy。
- 若报告生成实际超过 240 秒，应先写出最小 CSV、coverage、中文结论并记录超额，不继续跑求解任务；最终报告不得因为部分实验未完成而缺失。

当前仓库旧结果中的工作量参照：上述 39 例前两次重复，两种 SK 总计约 905.36+13.54 秒；同子集旧 SR1 约 1318.13 秒。这些只是预算依据，不能当成本轮新方法的耗时或复用旧墙钟作为正式对照。

## 6. 计时与公平性

### 6.1 两种计时必须同时保留

- machine_elapsed：真实机器占用，用于一小时累计预算。
- algorithm_total：一个方法独立求解该输入的计费时间，用于比较。

log SK 和新方法共享预热在机器账本中只实际运行、计费一次；各自 algorithm_total 都包含这段预热成本。

新方法分解：

$$
T_{\rm total}
=T_{\rm warmup}+T_{\rm distance}
+T_{\rm gamma/setup}+T_{\rm iterate}.
\tag{51,13}
$$

若现有 summary 用 post 包含 setup，则同时保留互斥的分项和兼容字段，不能把已包含在 post 的距离/setup 再加一次。

- 每条正式 energy_distance 和 energy_amp 轨道在算法计时内构造自己的 D，不能跨方法、重复、epsilon 或 kappa 免费复用距离缓存。
- 共享预热内已收敛时，不构造 D；此类收益不能归因于新方法。
- M、原始数据和固定输入哈希属于共用输入准备，排除在算法计时之外，但消耗机器预算。
- 接受状态导出纳入算法计时的规则保持与现有仓库一致；审计、写盘、作图单独记录。
- 正式两种 SK 在本轮同机重新计时，不能直接拿旧结果相除。
- 核构造、稳定目标差、桥接状态、所有拒绝试探、回退都计入对应方法。
- 正式轨道不要做 Hessian、特征分解、谱诊断或完整逐步图像绘制。
- 对不足毫秒级的结果不要给过多小数或统计显著性表述。

### 6.2 统一成功判据与首次命中

沿用行归一化观测端点，记录相对边际 L1 残差：

$$
\rho=\frac{\max\{\|P\mathbf1-a\|_1,\|P^\top\mathbf1-b\|_1\}}{s}.
\tag{51,14}
$$

现有正式输入 s=1，必须与旧残差实现保持一致。每条轨道一直求到 1e-8 或真实停止，从同一接受轨道读取 1e-4、1e-6、1e-8 的首次命中时间，不为三个阈值重跑或调参。

终点由保存的 mu、nu 和原始问题独立分块重算，验证行列误差。不得只信 solver 自报 converged。预热内命中、后期命中必须区分。

## 7. 数值检查与必要测试

小型测试应在几十维以内完成，除了已有回归外，至少覆盖：

1. 通用 Gram 距离与直接按列差、加权方差计算一致。
2. 平方欧氏快速路径与通用路径一致，包含非均匀 a、非方阵、h 非 1 的情形。
3. 距离的行列成本平移不变性；测试时保持 epsilon 和成本比例固定，不在平移后重新归一化成本。
4. 对所有相关量一致重排列时，D、方向、实际运输更新等变。不能只重排 M 而不重排 a、b、z。
5. 小型 D 在零和子空间满足 e^T D e<=数值容差；G 的受限正半定性只在测试中检查。
6. 一维距离下 -e^T D e/4 与累计平方能量一致；梯度用零和扰动做有限差分验证。
7. gamma=0、eta=1 时，新更新与现有 sk_log_fast 的一次列更新一致，验证 z=-nu/epsilon 的符号。
8. 同一点的新方向满足 e^T V=e^T psi+2*gamma*G_D；对一正一负残差验证势差符号。
9. 稳定目标差与高精度/小问题参考一致；高精度阶段不把浮点 f 相消误判为连续接受。
10. 校准、39 例清单和 262 条计划数量正确；校准种子与正式种子分离；n、m 都不超过 2048。
11. 预算恢复不重置、校准不完整时默认档位冻结、数值失败/资源跳过不被报告为成功。
12. 原始缩放 SK 的核下溢标记保留；新算法不因复用其表示而改变 M、epsilon 或裁剪 K。

不要把矩阵平方根、伪逆、Hessian 求解塞进正式代码以通过测试。容差按数据尺度设置，明显错误不能靠放宽容差掩盖。

## 8. 记录哪些数据

每个接受状态至少保留：iteration、elapsed_total、post_elapsed、residual、eta、accept_reason、累计试探/回退次数、实际工作计数。保留现有可复现状态指纹策略，避免不必要的庞大数组逐步写盘。

每条轨道 summary 至少增加：

- family、case_id、n、m、epsilon、alpha、data_seed、repeat；
- kappa、gamma、gamma 初始化的两个范数及退化标志；
- distance_backend（generic_gram / exact_geometric）、distance_build_sec、distance_bytes；
- energy_matvec_count、energy_matvec_sec；
- warmup_sec、setup 各互斥分项、iteration_sec、total_sec；
- accepted_updates、line_search_trials、rejected_trials、fallback_count、fallback_reasons；
- logsumexp_count、kernel_matrix_vector_count、plan_build_count、objective_diff_count；
- endpoint_audit、status、stop_reason、last_residual；
- process_peak_rss 和实际工作区估计；
- source/config/input/warmup 指纹及计时环境。

不同大小、不同操作类型的 matvec 不能混成一个“等价计算次数”。D@e 与 K@v/K.T@u 分开计数；已有 Bregman 函数中的 P 向量乘法也不得遗漏或误称为零。

只对六个固定代表例记录必要的廉价方向诊断；可以在既有状态上抽样：

$$
\cos\theta=
\frac{\langle\Pi\psi,\Pi V\rangle}
{\|\Pi\psi\|_2\|\Pi V\|_2},
\qquad
q=\frac{\gamma\|\Pi u\|_2}{\|\Pi\psi\|_2}.
\tag{51,15}
$$

分母不可分辨时标记 undefined。不要为了诊断额外构造大矩阵或重复求解。若进行独立重放，必须另计机器预算并验证轨道前缀，不能扩张本轮正式轨道数量。

## 9. 报告、图表与结论边界

至少输出：

1. calibration.csv 与冻结的 selected_kappa.json。
2. runs.csv、threshold_costs.csv、coverage.csv、failures.csv、pending.csv。
3. 独立 endpoint_audit.json。
4. 输入描述、固定图像配对、环境、实现/配置指纹。
5. 中文 docs/energy_distance_v0_report.md 和恢复说明。
6. 以下四组图，PNG 为必需，预算允许时同时输出 PDF。

| 图组 | 要回答的问题 |
|---|---|
| 主性能 | 分别相对 log SK、原始缩放 SK 的总时间比；按合成/MNIST 分开；显示成功/计划覆盖 |
| 收敛与成本 | 代表例的残差—时间与残差—接受步曲线；距离 setup、迭代、试探的成本分解 |
| 机制与消融 | 完整方法、gamma=0、幅度匹配的比较；方向夹角、步长和回退情况 |
| 规模与内存 | 512/1024/2048 的时间/RSS；非方阵单列 |

时间比统一定义为 T_baseline/T_new，大于 1 表示新方法更快。只在双方同输入、验收成功、预算条件清楚时计算完成时间比；失败、超时和不适用必须留在覆盖分母，不把超时限额伪装成成功完成时间。

报告开头直接回答：

- 完整距离新驱动有没有总墙钟收益，在哪些输入/epsilon 上有或没有？
- 相对原始缩放 SK 的结果如何，而不仅是相对 log SK？
- 接受步减少是否被 D 构造、D@e、稳定目标差和拒绝试探抵消？
- 附加驱动是否明显改变方向；有限消融是否支持这种方向改变有用？

单独列出表现最差的例子，不能只展示赢家。两次计时重复不是独立数据样本，不报告由此虚构的置信区间。

明确理论边界：

- energy distance 是已有构造。
- 本方法保持原 EOT 平衡点并提供下降方向；离散更快没有先验证明。
- “可以写成预条件梯度”是方法的真实数学身份，不作为回避分析的理由。
- 成本距离固定，但实际质量响应仍通过 L_P；不声称绕过所有弱连接。
- 本轮只覆盖至 2048，不外推 5000 或更大规模。
- 部分预算完成时，先报告实际完成范围，不把 262 个计划条目说成 262 次成功。

## 10. 一条命令执行与最终交付

请实现一个主入口：核心验证通过后，自动依次校准、冻结参数、主比较、有限消融、审计、作图、写报告。默认串行、可恢复、预算累计。

建议命令：

~~~powershell
.\scripts\run_ot.cmd scripts/run_energy_distance_study.py --config configs/energy_distance_v0.yaml
~~~

同一命令再次执行应恢复有效未完成任务，不重置预算、不重跑已验证完成的匹配轨道。另提供只从现有结果重新生成报告的命令：

~~~powershell
.\scripts\run_ot.cmd scripts/report_energy_distance.py --run-dir results/<actual-directory> --audit --report docs/energy_distance_v0_report.md
~~~

建议配置必须明确包括：

~~~yaml
study: energy_distance_v0
repeats: 2
warmup_iters: 20
tol: 1.0e-8
max_iter: 50000
kappa_grid: [0.3, 1.0, 3.0]
kappa_if_calibration_incomplete: 1.0
armijo_sigma: 1.0e-4
backtrack_factor: 0.5
max_trial_steps: 12
total_budget_sec: 3600
calibration_budget_sec: 360
main_budget_sec: 2640
ablation_budget_sec: 360
report_reserve_sec: 240
track_limit_sec: 180
track_limit_large_sec: 300
max_dimension: 2048
memory_cap_gib: 8
memory_available_fraction: 0.6
budget_ledger: results/energy_distance_v0_budget.json
mnist_cache: data/mnist
download_mnist: true
~~~

内存估计按新实现真实持有的数组计算，包含 M、log 核、距离、临时状态与线搜索工作区；不要直接沿用 SR1 固定“14 张矩阵”的估计。资源不足时明确跳过，不偷偷缩小问题尺寸。

完成后向用户交付：

- 实现及测试结果，涉及的新增/修改文件；
- 实际使用环境和总实验运行时间；
- 262 条计划的状态统计，以及 39 个正式输入的完成覆盖；
- 固定 kappa、gamma 的跨输入分布；
- 各类数据相对两种 SK 的效果，最差反例与开销解释；
- 中文报告、图表、CSV、准确的结果目录与恢复命令；
- 尚未完成的内容和真实原因。

如果性能不理想，不自动切换成另一条算法路线或扩大实验。先完成本轮记录和诊断，让用户能够判断下一步。

## 11. 来源与已有证据

已核查的仓库版本：

- https://github.com/Basiliscus-Denebola/sinkhorn-lmflow/tree/61af801ed0b89a0c2b6c23ac1a29382f02a3e886
- https://github.com/Basiliscus-Denebola/sinkhorn-lmflow/blob/61af801ed0b89a0c2b6c23ac1a29382f02a3e886/docs/sr1_v1_mainstream_report.md
- https://github.com/Basiliscus-Denebola/sinkhorn-lmflow/blob/61af801ed0b89a0c2b6c23ac1a29382f02a3e886/src/sinkhorn_lmflow/problems/mainstream.py
- https://github.com/Basiliscus-Denebola/sinkhorn-lmflow/blob/61af801ed0b89a0c2b6c23ac1a29382f02a3e886/src/sinkhorn_lmflow/core/semi_dual.py

energy distance 与条件负定性的参考：

- Székely, G. J. & Rizzo, M. L. (2013), Energy statistics: A class of statistics based on distances. Journal of Statistical Planning and Inference 143, 1249–1272. https://doi.org/10.1016/j.jspi.2013.03.018
- Bellemare et al. (2017), The Cramer Distance as a Solution to Biased Wasserstein Gradients. https://arxiv.org/abs/1705.10743

这些文献支持能量构造及一维关系，不直接证明本计划对 EOT 的加速效果。
