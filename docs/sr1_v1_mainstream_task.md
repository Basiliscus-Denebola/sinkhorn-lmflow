# SR1 v1：面向硕士中期答辩的标准 EOT 实验任务书

日期：2026-11-10（北京时间）  
目标仓库：`Basiliscus-Denebola/sinkhorn-lmflow`  
编写时核对的最新提交：`0481d2f478c7302860cc6b42041ba244909d8152`  
执行环境：本地 Windows，现有 Conda `OT` 环境，Python 3.9。  
执行目标：在当前单对正 SR1 上补齐标准问题、实际速度、规模与设计消融证据；完成实现、测试、正式实验及中文报告。

> 给本地 Codex：请先阅读本任务书及仓库 `AGENTS.md`，检查本地现状，再自主完成以下工作。不要只交计划或冒烟结果。优先复用当前组件；若发现数值、计数或恢复问题，先修复并验证，再继续正式实验。到达明确预算时保存全部结果及 pending 清单，形成可继续恢复的阶段交付，不通过降低精度或删除失败例子制造完成。

## 1. 本轮要回答什么

当前 v0 的两块、四块实验已经证明正 SR1 可以形成有效校正，但核矩阵块内完全重复，整体秩至多为块数，不能作为一般 EOT 性能的主要证据。现有点云也仅覆盖 N=128、一个数据 seed、两个 epsilon。

本轮主结论必须建立在以下证据上：

1. 一般随机成本、普通点云及聚类点云上，SR1 是否更早达到同一边际精度？
2. 与轻量 Sinkhorn 比较，考虑预热、方向构造、回溯和数值保护后，实际时间是否仍有收益？
3. 在非均匀边际、图像质量分布、非方阵和数千点规模下，方法是否仍可用？
4. 收益来自方向改变还是主要来自幅度调整？是否强依赖预热 20 步？
5. 相对同一半对偶目标上的经典 L-BFGS，方法表现与代价如何？

这些是待检验问题，不预设 SR1 胜出。两块、四块结果保留为机制实验，不与新数据混算一个总加速倍数。

## 2. 工作范围与环境

### 2.1 冻结算法，扩充证据

- 保持现有单对正 SR1：每步重新构造当前 B0，只使用最近一对实际接受的历史，不累积多对记忆。
- 主实验保持 v0 的正分母门槛、六次步长试探、严格 Armijo、单位 Sinkhorn 回退及稳定目标差实现。
- 仅在明确命名的消融中改变预热长度或驱动形状。
- 允许缓存、延迟诊断和内存复用；应先验证它们不改变数学更新及接受判据。若改变了实际接受轨道，必须查清并单列说明。
- 不将实验扩展成新求解器研究；本轮不引入多记忆 SR1、Hessian 分块、特征向量检测或 GPU 重写。

### 2.2 尊重本地工作区

- 先执行 `git status`、查看分支和最近提交，阅读当地生效的 `AGENTS.md`。
- 不回退或覆盖用户的笔记、未提交修改和已有实验结果。不以本文件中的 SHA 强制重置仓库。
- 若本地已有本轮相关改动，检查后接续，避免重复建一套实现。
- 使用 `conda activate OT`、`conda run -n OT ...` 或 `D:\miniconda_envs\OT\python.exe`；优先沿用 `scripts/run_ot.cmd`。
- 不升级 Python、NumPy 或 SciPy。数据读取优先使用标准库和现有依赖；不为 MNIST 安装 PyTorch/TensorFlow。
- 交付本地代码、配置、结果与报告；Git 提交和推送按用户对发布的明确要求执行。

### 2.3 先读这些现有文件

- `docs/sr1_v0_task.md`、`docs/sr1_v0_report.md`
- `src/sinkhorn_lmflow/core/semi_dual.py`、`core/residuals.py`
- `src/sinkhorn_lmflow/solvers/mirror_sr1.py`、`sinkhorn.py`、`lbfgs.py`、`base.py`
- `src/sinkhorn_lmflow/experiments/sr1_study.py`
- `src/sinkhorn_lmflow/problems/generators.py`、`weak_coupling.py`、`instance.py`
- `src/sinkhorn_lmflow/plotting/sr1.py`、现有结果 I/O 和 tests

已核对的工程事实：`sr1_study.py` 的 transfer 阶段把 N=128、seed=42 写死；旧 `lbfgs.py` 优化完整双势；旧生成器默认对 Dirichlet 边际施加 floor。新实验需要显式参数化或新增薄适配，不能把这些默认行为当成已经满足本任务。

## 3. 符号、方法和统一精度

讨论中的列势 mu 对应仓库的 `nu`，仓库 `mu` 是行势。沿用当前内部 `z_col=-nu/epsilon`，每个接受态去均值。以下用 z 表示列坐标。

设 a、b 为总质量均为 1 的正边际，M 为成本，K=exp(-M/epsilon)。每个观察态先精确消去行势：

\[
P_{ij}(z)=a_i\frac{\exp(-M_{ij}/\varepsilon+z_j)}{\sum_\ell\exp(-M_{i\ell}/\varepsilon+z_\ell)},\quad
r=P\mathbf1=a,\quad c=P^\top\mathbf1,\quad e=c-b.
\tag{35,1}
\]

\[
f(z)=\sum_i a_i\log\sum_j\exp(-M_{ij}/\varepsilon+z_j)-b^\top z,
\qquad \nabla f(z)=c-b,\qquad \psi=\log(c/b).
\tag{35,2}
\]

正 SR1 的概念式如下；实际计算使用已验证的 log-domain 和连续延拓代码，不直接照抄不稳定商式：

\[
B_0=\operatorname{diag}(\beta_j),\quad
\beta_j=\frac{\log(c_j/b_j)}{c_j-b_j},\quad \beta_j\big|_{c_j=b_j}=\frac1{b_j},
\quad s=z-z_{\mathrm{prev}},\quad y=c-c_{\mathrm{prev}},\quad q=s-B_0y,
\quad V=\psi+q\frac{q^\top e}{q^\top y}.
\tag{35,3}
\]

保持 v0 的规范处理、合法性检查和稳定公式。仅在有限量、正分母且满足相对阈值时用 SR1；否则回退。更新 z+=center(z-hV)。默认参数：

| 参数 | 固定值 |
|---|---|
| `sr1_warmup_iters` | 20 |
| `sr1_secant_tol` | 1e-8 |
| 首次试探步长 | 1 |
| 回溯系数 | 0.5 |
| Armijo 常数 | 1e-4 |
| 试探集合 | 1、1/2、1/4、1/8、1/16、1/32 |
| 最大迭代数 | 50000，达到时间预算可提前记 timeout |
| 类型与硬件 | float64、CPU、单线程 BLAS，逐方法串行计时 |
| 主停止阈值 | 1e-8 |
| 额外阈值 | 同一轨道首次到达 1e-4、1e-6 的时刻 |

严格沿用仓库的残差定义，不能改成相对误差或另一种聚合：

\[
\mathrm{res}(P)=\max\{\|P\mathbf1-a\|_1,\|P^\top\mathbf1-b\|_1\}.
\tag{35,4}
\]

所有方法在一致的行归一化端点观察残差，不将旧列归一化半步直接与新轨道逐轮比较。库函数报告 success 不能替代此判据。最终对全部有效终点独立重建边际验收；大问题允许分块重建，不必再额外常驻一个 P。

普通数据没有已知 F*。不把最小观测目标当真值，不伪造 energy_gap。报告残差、目标变化与时间即可。

## 4. 方法对照：主表与补充表分开

### 4.1 全部主实例：`sk_log_fast` 对 `mirror_sr1`

`sk_log_fast` 是相同数学 Sinkhorn 迭代的轻量稳定实现：

- 复用 logsumexp、规范及残差组件，缓存行势或中间量；不为画图逐步计算完整 P、稳定目标差、Bregman 能量或完整双势导出。
- 在行归一化表示中，可以通过行 logsumexp 和列 logsumexp 得到 log_c，无须构造 P；实施这一优化时必须与 v0 端点逐步对齐。
- 每个接受态仍检查统一残差，记录必要的计时及计数。减少非算法诊断，不减少公平的停止检查。
- SR1 的线搜索和稳定目标差属于算法必要成本，必须计入；不能为了“相同诊断”强迫 SK 也做这些计算。

当前 `sinkhorn_reduced` 保留为诊断和等价性对照，不再独自承担“实际速度基线”。如新轻量 SK 更快，主表必须使用它。

### 4.2 核可稳定表示的主实例：额外报告 `sk_scaling_fast`

- 复用常规 K@v、K.T@u 的缩放更新和缓存，按相同行归一化端点测残差。
- K 构造成本计入本方法总时间；若缓存，明确区分 cold setup 与重复求解，主表采用包括所需 setup 的口径。
- 仅在全部 K 元素严格正且有限时进入原始缩放测试。若发生 exp 下溢成零，不以该 K 冒充原始问题；标 `not_applicable_kernel_underflow`，继续稳定 SK/SR1。
- 若初始 K 可表示但迭代中溢出，记录 numerical_failure；不能换成本或裁剪边际。
- 原始缩放只允许不改变 P 的规范重标定，不额外引入新的加速方法。
- 和稳定基线分别列出时间收益，不在各重复间事后挑最快结果拼成虚构基线。

### 4.3 十个预先指定的实例：`lbfgs_reduced`

复用 SciPy L-BFGS-B 和现有半对偶评估器，优化同一个 f(z)，使用最后一维固定为 0 的 m-1 维规范坐标，返回相应约化梯度。不要直接把旧完整双势 L-BFGS 当成同一参数化实验。

- `maxcor=20`、`maxls=50`；其他参数和实际版本写入配置。
- 在允许的库选项内避免 ftol/gtol 比主残差更早宣称收敛。若库因线搜索或舍入终止但残差未达标，保留其真实停止原因。
- 不向 L-BFGS 悄悄追加 SK 精修；若今后另做这种组合，应当是另一个方法名。
- 从相同的 20 步预热端点出发，各方法总成本均包含这 20 步；L-BFGS 初始无历史。
- 固定测试集：M1、M2、M3 的 seed=42、epsilon=0.01/0.001，共 6 例；MNIST `same_0`、`cross_0_1` 的两个 epsilon，共 4 例。
- 每例三次计时。这里只讨论这个预先选定子集，不称为全部问题上的 L-BFGS 比较。

## 5. 正式实验矩阵：约 57 个不同主实例

下面 M1–M7 是主实验，不再扩展人为分块数量。每个主实例默认三次性能计时，数据和初值保持不变。

| ID | 问题 | 规模 | 边际 | epsilon | 数据种子/样本 | 实例数 |
|---|---|---|---|---|---|---:|
| M1 | 独立随机成本 | 1024×1024 | 均匀 | 0.01、0.003、0.001 | 42、43、44 | 9 |
| M2 | 二维均匀点云 | 1024×1024 | 均匀 | 同上 | 42、43、44 | 9 |
| M3 | 二维 Gaussian mixture 点云 | 1024×1024 | 均匀 | 同上 | 42、43、44 | 9 |
| M4 | 与 M3 相同的成本 | 1024×1024 | 独立 Dirichlet，alpha=1、0.1 | 0.003 | 42、43、44 | 6 |
| M5 | MNIST 像素质量运输 | 原始 784×784；删除零质量支持后报告实际 n×m | 图像灰度归一化 | 0.01、0.001 | 预先固定 10 对图像 | 20 |
| M6 | M3 同类型规模扫描 | N=512、1024、2048、5000 | 均匀 | 0.003 | 42 | 4，1024 例与 M3 按指纹复用 |
| M7 | 非方阵聚类点云 | 1024×2048 | 均匀 | 0.003 | 42 | 1 |

M1–M5 共 53 个实例。M6 增加 3 个，M7 增加 1 个，合计 57 个。若不能实现 M6 的 1024 例与 M3 完全相同，保留单独实例并解释差异，不强行合并；计划数量应据此更新。不同方法、重复次数和 epsilon 不得混称为独立数据集数量。

### 5.1 M1：随机成本

先生成 M_raw 的独立 U(0,1) 元素，再由成本得到核。成本按该实例最大元素归一化。不要直接独立采样 K 元素，也不要预先把 K 平衡成双随机矩阵。

### 5.2 M2、M3：普通几何成本

M2：源点和目标点独立服从二维 [0,1]^2 均匀分布。

M3：采用以下固定分布，避免组内完全重复以及两端簇质量刻意匹配：

- 源簇中心：(0.2,0.2)、(0.8,0.2)、(0.5,0.8)，混合概率 (0.5,0.3,0.2)，各簇协方差 0.06² I。
- 目标簇中心：(0.25,0.25)、(0.75,0.25)、(0.5,0.75)，混合概率 (0.2,0.5,0.3)，各簇协方差 0.07² I。
- 每个点独立采样，不截断坐标、不把同簇成本设为常数。
- 统一采用平方欧氏成本。记录原始坐标、成本归一化因子及数据指纹。

\[
M^{\mathrm{raw}}_{ij}=\|x_i-y_j\|_2^2,\qquad
s_M=\max_{ij}M^{\mathrm{raw}}_{ij},\qquad M=M^{\mathrm{raw}}/s_M,\qquad
K=\exp(-M/\varepsilon).
\tag{35,5}
\]

这里 max 明确指最大元素，不是诱导矩阵无穷范数。所有算法读取同一份归一化后的 M。epsilon 是本任务中的熵系数，不直接照搬文献中可能定义为其倒数的 eta。

将几何、源权重、目标权重的 RNG 分开，固定 SeedSequence 派生规则。改变边际 alpha 或 epsilon 不得改变点位置。对每种几何类型、每个 seed 先生成最大长度为 5000 的两侧点池，再取前缀，以保证 M3/M6/M7 可复用相同点数据；不同规模允许有自己的最大成本归一化因子，并如实记录。

规模扫描是实测可扩展性，不声称“固定 epsilon 后所有规模条件数完全相同”，也不通过四个点拟合一个复杂度定理。

### 5.3 M4：只改变边际

使用 M3、epsilon=0.003 对应的同一份成本，分别生成独立 Dirichlet(alpha·1) 的 a、b，alpha=1 和 0.1。

- 数值上严格为正，总和归一到 1。保存 min/max、若干分位数及生成种子。
- 本轮不默认使用旧生成器的 1e-12 floor；增加显式无 floor 的生成入口或单独权重工具，保持旧入口兼容。
- 若极罕见地生成了数值零，按事先固定规则记录并处理为数据生成失败；不要静默裁剪或根据运行效果换 seed。

### 5.4 M5：MNIST 标准图像实验

采用 MNIST test split。先复用本地缓存；缺少时只下载 test images 和 test labels，使用 CVDF 维护的镜像入口：<https://github.com/cvdfoundation/mnist>。解析 gzip/IDX 可用标准库和 NumPy。记录下载来源、SHA256 和 IDX 形状。

在运行任何求解器之前，用 RNG seed=20261110 固定以下 10 对图像并保存 `mnist_pairs.json`：

- 同类 5 对：数字 0、2、4、6、8，各随机选两个不同测试图像，命名 `same_0`、`same_2` 等。
- 异类 5 对：(0,1)、(2,3)、(4,5)、(6,7)、(8,9)，各选一张，命名 `cross_0_1` 等。
- 按上述顺序抽样，整个列表不复用同一图像，记录 split、索引、标签、选择规则。不能先运行再筛选“加速明显”的图像对。

每幅图像的非负灰度归一化为质量；空间坐标为 (row/27, col/27)，成本为像素空间平方距离，不是灰度差。

- 精确移除质量为 0 的源/目标支持点，这不改变原问题在正质量支持上的解；两个方法使用相同掩码。
- 不加灰度背景，不阈值化微小正灰度，不对灰度插值，不将图像转成均匀边际。
- 使用完整 [0,1]^2 像素网格的最大平方距离 2 作固定成本归一化因子，再选取有效行列。不能因图像的零像素分布改变成本尺度。
- 同时记录原始网格尺寸 784、实际有效 n/m 和保留质量。论文表里不能把裁减后的实际计时称作完整 784×784 稠密计算。
- 数据暂时不可获得时标记 `blocked_data`，继续其他阶段，并给出恢复入口；不能用合成图像冒充 MNIST。

### 5.5 规模扩展边界

默认执行到 N=5000。N=10000 作为配置中明确关闭的扩展项，只有本地资源与用户追加预算支持时才启用，不作为本轮完成硬门槛。

单个 N=5000 float64 矩阵约 200 MB，真实求解含多个临时数组。检查实际峰值内存；不要将单个 P 的大小写成算法总内存。大规模仍使用完整问题，不为了跑通而偷偷做低秩近似、下采样或改变 epsilon。

## 6. 共享预热、初始化和性能口径

### 6.1 同一起点

- M1–M7 均从中心化 z=0 开始。当前讨论针对两阶段方法，不添加额外为 SR1 特制的初值。
- 同一实例同一重复，主方法共享一份真实执行的轻量稳定 SK 预热轨道，最多 20 步，提前达到 tol 则直接结束，不能强行再走满 20 步。
- 将同一预热成本完整记到每个两阶段方法上；后期单独计时并报告。快照、接口转换及必要的 SR1 初始 P 构造按各自实际成本计入。
- SR1 需要最近两个真实接受端点；快照不能只存最后一个态，也不能用另一问题的割线。
- 缓存和可变数组必须隔离，使用指纹验证。若采用每方法独立预热，需证明端点一致并分别实测，不再额外重复加共享预热费用。
- `sk_scaling_fast` 从相同数学初值独立运行，自己的 K/setup 和完整预热自然计入；不人为给它添加未执行的 log-domain 预热成本。

### 6.2 性能模式

主表计时从方法必需的初始化/setup 到首次达到对应残差阈值。包含：

- kernel/log-kernel 的方法特有准备、必要状态构造；
- 实际预热、方向计算、接受及拒绝试探、线搜索、保护和回退；
- 每步必要残差检查、计数与最小历史记录、快照读取和复制。

不包含：公共数据生成/下载/成本生成、最终审计、可选诊断重放、文件落盘、绘图。分别记录这些排除项的耗时，确保不是把算法必要工作误移出去。

固定单线程；计时前执行一次小型非正式预运行。三次重复只改变运行顺序和计时，不改变输入。方法顺序使用固定规则轮换，不并发跑性能试验。

以同一实例同一重复的 SK 时间除以 SR1 时间得到配对 speedup，报告三次比值的中位数及范围，不能用两个时间中位数相除替代配对统计。若方法过快、系统负载明显波动，保留原数并报告范围，不据一次不足 1% 的差别宣称退化或优势。

初始状态已经达标的 case 单独记录，避免零时间比。不要丢弃易例；整体与困难区间分别汇总。

### 6.3 诊断重放

只对预先指定的代表例做完整诊断，不让所有主实例承担重型观测成本：M1/M2/M3 的 seed=42、epsilon=0.003；M4 alpha=0.1、seed=42；MNIST `same_0` 与 `cross_0_1`、epsilon=0.001；原四块 delta=0.001、seed=42。

记录 SR1 合法性、分母、同点耗散增益、方向夹角、实际目标下降及接受步长。比较诊断/性能模式的接受轨道和终点，保证只是记录量不同。每个问题重放一次足够。

若 epsilon 或大动态范围导致浮点接受分支不一致，先核查执行表达式和误差，不默认为“等价”。数学路线改变的结果需单列，不能套用主性能表。

## 7. 两个有限消融

### 7.1 消融 A：方向改变与幅度放大

对当前态先按原 SR1 规则得到合法 V_SR1，再构造保持 SK 方向的对照：

\[
\alpha(c)=\frac{(c-b)^\top V_{\mathrm{SR1}}}{(c-b)^\top\psi},
\qquad V_{\mathrm{amp}}=\alpha(c)\psi.
\tag{35,6}
\]

它与 SR1 在同一点具有相同的一阶耗散，但方向仍为 SK。用同样的六次试探与 Armijo。每条完整轨道使用自己的当前态和历史；不能把另一条轨道算出的 alpha 直接借来。

- 若 SR1 对不合法，或分母不能可靠计算，按明确保护规则回到单位 SK，记录原因。
- alpha 构造所需的 SR1 计算必须计费。这个消融用于解释机制，不当作廉价调步长的最优实现。
- 固定三个例子：原两块 N=256/delta=0.001/initial_seed=42；原四块同参数；M3 seed=42/epsilon=0.003。
- 用本轮的轻量 SK、标准 SR1、`mirror_amp_ablation` 完整轨道对照，三次计时；旧 v0 计时不可直接混入新模式。
- 消融若显示主要收益来自幅度，也如实报告，不把“改变轨道方向”写成已经证实的贡献。

### 7.2 消融 B：预热长度

只选 M3 seed=42/epsilon=0.003 和 MNIST `cross_0_1`/epsilon=0.001，比较 warmup=0、5、20、50。所有总时间包括实际预热；0 步时没有历史就先走 SK，不能伪造割线。

每个设置三次计时，20 步结果按配置和指纹复用。保留提前收敛、跳过和回退。其余参数不变，不在看到结果后为主表挑选最佳 warmup。

## 8. 时间、内存预算与执行顺序

以下为本任务选择的执行预算，不是文献设定；全部做成显式配置，记录实际值：

| 项目 | 默认预算 |
|---|---|
| 一个正式方法轨道，max(n,m)≤2048 | 总求解时间 300 秒，含应计的预热/setup |
| 一个正式方法轨道，max(n,m)>2048 | 1200 秒 |
| 正式性能实验累计实际机器时间 | 6 小时 |
| 额外诊断重放累计时间 | 1 小时 |
| 单进程内存预警上限 | 8 GiB 与启动时可用物理内存 60% 的较小值 |
| N=10000 | 默认关闭 |

共享预热对各方法统计上计费，但总调度预算按机器真实运行时间统计，不能把一次共享预热重复扣除多次。时间检查覆盖每次试探；记录可能超过预算的单个不可中断数值核。失败不得阻止其他方法继续。

按以下顺序完成覆盖，随后再补重复，避免把预算都耗在前几个例子的三次运行上：

1. 回归检查与小规模 smoke：随机、点云、非均匀、非方阵、MNIST 各一个小例子。
2. M1–M4 的全部实例，先跑 repeat=0 的稳定 SK 和 SR1。
3. M5 全部 10 对图像及两个 epsilon，先跑 repeat=0。
4. M6/M7 首次运行，覆盖 N=5000。
5. 补主比较 repeat=1、2；每一轮覆盖整个主矩阵。
6. 对符合条件的实例执行原始缩放 SK 三次比较，优先覆盖较大规模和代表例；明确记录预算未覆盖项。
7. 十例 L-BFGS 对照、两个消融及诊断重放。

步骤 6 的原始缩放基线和步骤 7 的强基线是报告的重要内容；预算不足时标记未完成，不把仅完成步骤 2–5 称作“全部论文实验完成”。可以用同一 manifest 恢复剩余任务。

预算耗尽时保存 accepted state、已完成历史、终点残差、原因和 pending 项，生成阶段报告。不能用更大 tol、较大 epsilon 或更小 N 自动替换计划条目。内存预检查不通过记 `skipped_resource`，不能误计成数值不收敛。

## 9. 实现目录与组件复用

按仓库实际情况调整命名，但保持以下职责分离：

```text
docs/
  sr1_v1_mainstream_task.md
  sr1_v1_mainstream_report.md
configs/
  sr1_v1_smoke.yaml
  sr1_v1_main.yaml
  sr1_v1_mnist.yaml
  sr1_v1_scale.yaml
  sr1_v1_baselines.yaml
  sr1_v1_ablation.yaml
src/sinkhorn_lmflow/
  problems/mainstream.py          # 新场景、边际及固定点池；复用 OTProblem/成本工具
  problems/mnist.py               # 缓存、IDX、固定配对、正质量支持；不含求解逻辑
  core/semi_dual.py               # 原稳定核；按需增轻量评估或抽出专用模块
  solvers/sinkhorn_fast.py        # 稳定及可表示时的原始缩放基线
  solvers/lbfgs_reduced.py         # 同目标的 SciPy 薄适配
  solvers/mirror_sr1.py           # 原算法、可选记录等级；保护 v0 行为
  experiments/sr1_mainstream.py   # 计划、预算、重复、恢复、去重
  plotting/sr1_mainstream.py      # 仅读持久化结果，汇总图表
scripts/
  run_sr1_mainstream.py
  report_sr1_mainstream.py
tests/
  test_mainstream_problems.py
  test_mnist_problem.py
  test_sinkhorn_fast.py
  test_lbfgs_reduced.py
  test_sr1_mainstream.py
data/mnist/                      # 本地缓存，不提交整套数据
results/<timestamp>_sr1_v1_*/
  manifest.json
  study_config.yaml
  environment.json
  problems/                     # 去重实例/可重建描述和指纹
  runs/                         # 方法终点、历史、summary、配置
  tables/
  figures/
  report.md
```

薄 CLI、可复用模块优先。复用 v0 的指纹验证、attempt、失败保存、结果格式及独立审计，不复制数份求解代码。实验报告从持久化结果生成，不在 Markdown 手填与 CSV 脱离的数字。

大规模数据优先存坐标/质量/参数/指纹，成本按确定规则重建；若必须保存 M，用可 mmap 的格式或去重存储，不给每个方法复制一份成本。运行历史只存向量必要量和标量，不保存逐步 P。

## 10. 数据字段与统计规则

### 10.1 每个问题必须有

`case_id, family, n, m, epsilon, data_seed, weight_seed_a, weight_seed_b, marginal_type, alpha, raw_cost_max, normalization_policy, problem_sha256, initial_sha256`。

图像增加 split、原始索引、数字标签、pair_type、原始像素数、有效支持数、零质量掩码及数据文件哈希。规模实验增加点池 ID/前缀长度。可复现描述与实际数组哈希都要保留。

### 10.2 每次轨道必须有

- 方法名、repeat、attempt、配置指纹、源码提交和 dirty 状态。
- success/timeout/max_iter/numerical_failure/library_stop/blocked_data/skipped_resource/pending 等分开的状态。
- 最终残差、初始残差、迭代数，以及各阈值的首次命中步/时间。
- setup、warmup、post、total 时间；实际状态评估、目标/梯度、目标差、Bregman、logsumexp、matvec、P 构造次数。
- SR1 尝试、合法、接受、跳过、拒绝、回退和步长分布。
- 峰值 RSS 或可解释的可用内存指标，说明是测量值还是估计值。
- 独立终点审计是否通过及失败原因。

不同内核的 state_eval 不是等价 FLOPs；不能仅凭这个计数断言工作量相同。不要将继承的 `dense_pass_proxy` 当作新 fast 内核的精确成本，新增内核应有自己的实际计数或明确标为不可比较。

### 10.3 汇总不能隐藏失败

- 主表按 family、epsilon、边际类型分组，先列 planned/attempted/succeeded，再列成功配对上的时间比。
- 对失败/超时记录预算时的残差。一个方法成功、另一个超时时，报告成功率差异或时间比下界，不把 timeout 当准确完成时间。
- `not_applicable` 原始核下溢不是该稳定算法求解失败，单列覆盖率。
- 配对中位数和范围是描述统计，不称为统计显著性或置信区间。
- 保留 speedup<1 的例子，并解释它们的每步开销与迭代节省。
- 受控 toy、普通合成、图像数据和消融分别统计，不用两块超大倍数抬高总体指标。

## 11. 必要验证

先跑现有测试，记录基线；完成后跑相关新测试和全套回归。新增测试聚焦实际风险：

1. 轻量稳定 SK 与原 reduced SK 在同一初始状态上的多步 z/c 端点一致（规范对齐后比较），包括非方阵与非均匀边际。
2. 可表示例子上原始缩放 SK 与 log SK 端点和残差一致；下溢检测不静默改变问题。
3. 半对偶 L-BFGS 约化坐标梯度的有限差分，以及最终统一残差验收，不把 SciPy success 当充分条件。
4. SR1 性能/诊断模式的接受步一致，稳定 Armijo 和已有极小目标差测试保持通过。
5. 改 epsilon/边际不改点池；M3/M6 重叠实例指纹一致；重复计时不重采样数据。
6. MNIST IDX 小型夹具解析、归一化、删除零支持、完整网格固定成本尺度，以及图像配对清单确定性。
7. 共享预热不串改数组、0 步和提前收敛有正确语义；计时与预算不重复计算共享机器时间。
8. 超时、失败、缺数据、资源跳过和中断恢复都保留记录；无效产物不会被当作完成。

正式结果逐轨道回读：验证输入指纹、终点残差、成功判据、计时非负且口径一致、拒绝试探计费、预热一致。抽查一部分大实例的端点；若成本允许，对全部终点使用分块独立审计。

测试失败先定位。不得通过放宽残差、修改参考答案或关闭保护来换取绿灯。

## 12. 交付内容和答辩呈现

### 12.1 必须生成的表

1. `coverage.csv`：计划、实际运行、成功、失败、超时、资源跳过和 pending。
2. `threshold_costs.csv`：每方法每重复在三个精度下的真实成本。
3. `paired_speedups.csv`：逐配对相对稳定 SK 和可用原始缩放 SK 的比值/下界/不可比原因。
4. `main_summary.csv`：按问题族、epsilon、边际分组的成功率和速度。
5. `scale_summary.csv`：规模、时间、迭代、内存、实际 n/m。
6. `baseline_subset.csv`、`ablation_summary.csv`：L-BFGS 与两个消融。
7. `independent_audit.json`、`pending.csv`、`failures.csv`：可审查的完成状态。

### 12.2 必须生成的图

- 普通问题和图像问题的配对 speedup 分布，画出 speedup=1 参考线，显示失败覆盖而不只画成功点。
- 代表例的边际残差—迭代数、边际残差—总时间曲线；图例明确 warmup 是否包含。
- epsilon—迭代数/时间趋势，在一致输入上比较。
- 规模—时间/内存趋势，N=5000 若未完成要显示状态，不插值伪造点。
- 驱动幅度消融和预热消融图。
- MNIST 固定 10 对输入缩略图及有效支持数，便于答辩解释“问题是什么”。

PNG 和 PDF 均输出；字体、字号和图例适合直接放中期答辩。数据图从保存的轨道生成，不依赖求解时内存。

### 12.3 中文报告结构

`docs/sr1_v1_mainstream_report.md` 至少包括：

1. 一段直接结论：在哪些问题/精度有收益，哪些无收益或失败，本轮是否完成。
2. 方法和基线定义：单对 SR1、预热、轻量稳定 SK、原始缩放 SK、同半对偶 L-BFGS。
3. 实验设置：数据来源、矩阵规模、实际支持、边际、成本尺度、epsilon、种子、硬件、软件、计时。
4. 普通问题与 MNIST 主结果，按族分开。
5. 规模、非均匀边际、非方阵结果。
6. 两个消融：哪些结论由数据支持，哪些机制解释尚是假设。
7. 失败与适用范围：正分母失效、回溯、预算、数值问题分别说明。
8. 与 v0 的关系：toy 结果继续用于机制解释，新旧计时模式不可直接混比。
9. 可复制命令、结果目录、测试/审计结果与 pending 的恢复方式。

最后再给出“中期答辩可直接使用的四页实验提纲”：标准合成、图像数据、规模与边际、消融与边界。结论以实际数据为准，不预填“普遍加速 2–3 倍”。

## 13. 复现入口要求

实现下面这类入口（参数名可按现有框架统一，但报告必须给出真实可执行命令）：

```powershell
.\scripts\run_ot.cmd -m pytest -q -p no:cacheprovider
.\scripts\run_ot.cmd scripts/run_sr1_mainstream.py --config configs/sr1_v1_smoke.yaml
.\scripts\run_ot.cmd scripts/run_sr1_mainstream.py --config configs/sr1_v1_main.yaml
.\scripts\run_ot.cmd scripts/run_sr1_mainstream.py --config configs/sr1_v1_mnist.yaml
.\scripts\run_ot.cmd scripts/run_sr1_mainstream.py --config configs/sr1_v1_scale.yaml
.\scripts\run_ot.cmd scripts/run_sr1_mainstream.py --config configs/sr1_v1_baselines.yaml
.\scripts\run_ot.cmd scripts/run_sr1_mainstream.py --config configs/sr1_v1_ablation.yaml
.\scripts\run_ot.cmd scripts/report_sr1_mainstream.py --run-dir <实际总实验目录>
```

最好支持一个总计划入口统一管理多个配置的去重、累计 6 小时预算和执行优先级；分别运行子配置也必须共用同一个预算账本，不能每个子配置都重新获得 6 小时。

支持 `--resume <实际总实验目录>`，校验配置、问题、初值和实现指纹，跳过完整有效结果，未完成条目新增 attempt，保留旧尝试。若求解实现改变，不把旧新计时混成一次可比较重复；建立新的实验 revision。

## 14. 文献与数据出处

下列文献用于选择常见问题类型，并非要求逐项复制其全部代码、超参数或性能数字。本任务中的样本数、种子、预算和比较集合是本轮具体设计。

1. Lehmann et al., *A note on overrelaxation in the Sinkhorn algorithm*, Optimization Letters, 2022。包含 1000 点 RGB 经验分布和一维随机边际实验。  
   <https://link.springer.com/article/10.1007/s11590-021-01830-0>
2. Tang et al., *Accelerating Sinkhorn algorithm with sparse Newton iterations*, ICLR 2024。包含随机成本、均匀边际和 MNIST 像素质量运输。  
   <https://arxiv.org/abs/2401.12253>
3. Tang and Qiu, *Safe and Sparse Newton Method for Entropic-Regularized Optimal Transport*, NeurIPS 2024。包含图像质量、图像特征及至 10000 点的规模实验。  
   <https://proceedings.neurips.cc/paper_files/paper/2024/hash/ea8620683340facbd5f754dd169e0980-Abstract.html>
4. MNIST 的 CVDF 镜像及 IDX 格式说明：  
   <https://github.com/cvdfoundation/mnist>
5. 本轮仓库起点及 v0 报告：  
   <https://github.com/Basiliscus-Denebola/sinkhorn-lmflow/tree/0481d2f478c7302860cc6b42041ba244909d8152>  
   <https://github.com/Basiliscus-Denebola/sinkhorn-lmflow/blob/0481d2f478c7302860cc6b42041ba244909d8152/docs/sr1_v0_report.md>

## 15. 完成标准

- 当前 SR1 数学方法保持冻结，必要适配与性能优化均有验证。
- M1–M7 的计划清单和输入在运行前固定；成功、失败、资源/数据阻塞与 pending 都能追溯。
- 完成轻量稳定 SK 主比较、可用原始缩放 SK 比较和预定十例 L-BFGS；有限消融已运行或明确列入未完成项。
- 同一精度、包含预热和必要算法工作的配对成本完整；不以理想化分块核的巨大比值替代普通问题结论。
- 有真实 CSV、可读图、中文报告和复现/恢复入口。
- 若预算或资源不足，交付名称和结论明确写“阶段结果”，列出剩余工作，不能声称完整完成。

**本轮的终点是形成一组能回答答辩质询的实验事实：方法在哪里有效、付出多少成本、对哪些条件敏感。获得这些事实后收尾，不自动扩展新的驱动框架。**
