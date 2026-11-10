# sinkhorn-lmflow：Mirror 驱动正 SR1 第一轮实现与实验任务书

> 给本地 Codex：请直接在已有 `sinkhorn-lmflow` 工作区完成实现、检查、实验与报告。尽量复用已有组件，遵守现有目录分层。本文是可独立执行的任务说明，不需要再读取聊天历史或下载其他笔记。
>
> 本轮只研究 **Sinkhorn 基础驱动 + 单对历史正 SR1 校正**。主要比较对象是同一行归一化观测位置的标准 Sinkhorn。输出真实结果，包括无收益、失败与回退，不预设加速结论。

## 0. 仓库依据与开始方式

- 仓库：https://github.com/Basiliscus-Denebola/sinkhorn-lmflow
- 本任务书依据：2026-11-10 读取的 `main`，提交 `886445543600bef327b6a54ea1ed30ccef57d403`。
- 本地工作区可能更新或存在未提交修改，以本地实际代码为准；先记录差异，不强制回退到上述提交。
- 先阅读本地 `AGENTS.md`、`README.md`、`docs/math_spec.md`、`docs/experiment_protocol.md`，以及本任务涉及目录中的其他适用指令。
- 按已读取的 `AGENTS.md`：使用 Conda **OT** 环境，路径 `D:\miniconda_envs\OT`，兼容 Python 3.9；优先使用 `scripts/run_ot.cmd`。不另建环境，不升级 Python，不安装 GPU 框架。
- 先运行现有测试，记录原有失败。保留用户未提交修改；必要时创建本地工作分支。完成本地代码和实验即可，本任务不要求 push、PR 或改动远端。
- 将本说明放入 `docs/sr1_v0_task.md`。先检查本地是否已有同类实现，避免重复创建模块。

本轮要回答三件事：

1. SR1 历史校正是否经常可用，并作用到当前边际残差？
2. 同一点增加的瞬时耗散，是否转化为接受步的实际目标下降？
3. 将所有回溯与回退计入成本后，是否比 Sinkhorn 更早达到同一精度？

## 1. 已确认可复用的组件及适配边界

下列路径均相对于仓库根目录。不要复制整套旧求解器另起项目。

| 现有组件 | 本轮用途 | 需要留意的边界 |
|---|---|---|
| `problems/instance.py::OTProblem` | 沿用输入、float64、不可变数组和合法性检查 | 成本字段名为 `M` |
| `problems/generators.py::generate_problem` | 普通点云迁移检查 | 当前生成器会按成本最大值归一化；受控核不能走这条归一化路径 |
| `core/transport.py` | 根据完整势重建运输矩阵，核对新半对偶状态 | 复用底层计算；不要为同一接受状态重复构造矩阵 |
| `core/dual.py::dual_objective` | 保持原有完整对偶目标记录 | 它不是本文的无量纲镜像势函数，二者有常数和尺度差 |
| `core/residuals.py` | 保持统一边际残差 | 沿用 `max(||r-a||_1, ||c-b||_1)` |
| `core/gauge.py` | 导出标准完整势、验证规范不变性 | 现有函数固定行势均值；SR1 内部必须固定列势坐标的规范 |
| `solvers/sinkhorn.py` | 标准 Sinkhorn、稳定行列归一化与回归对照 | 当前 sweep 在列归一化后返回；本轮主轨道在行归一化后记录 |
| `solvers/base.py::Monitor` | 计时、停止条件、公共 history、构造 `SolverResult` | 当前 `observe` 每次重新构造 `P`，需增加向后兼容的缓存状态入口 |
| `policies/stepsize.py` | 沿用 Armijo 思路、回溯与明确失败方式 | 当前 `choose_step` 丢弃试探 `P` 且返回完整势；不要直接调用后再重复评估 |
| `types.py` | 沿用 `SolverConfig/IterationRecord/SolverResult` | 新字段有默认值，旧配置、旧结果继续可读；不要借用旧 `theta` 或 `balance_*` 字段 |
| `experiments/registry.py` | 注册新求解入口 | 保留原 `sinkhorn` 的语义和行为 |
| `io/problem_io.py` | NPZ 保存、问题指纹、回读 | 初值不在问题指纹中，需单独保存初值指纹 |
| `io/result_io.py`、`io/config.py` | 原有 CSV/JSON/YAML/NPZ 格式 | `load_result` 有硬编码整数及字符串字段集合，新字段需同步处理 |
| `experiments/reproducibility.py` | 记录运行环境和 git 状态 | 不伪造提交或计时信息 |
| `io/paths.py::create_run_dir` | 创建独立运行目录 | 每次运行新建目录，不覆盖旧结果 |
| `plotting/convergence.py`、`plotting/comparisons.py` | 保存 PNG/PDF、失败标记、汇总 | 原分组只含问题指纹和容差，本轮还必须区分初值及共享预热 |
| `scripts/run_ot.cmd`、`scripts/_bootstrap.py` | 环境、单线程 BLAS、薄 CLI 入口 | 从仓库根目录运行 |

仓库还有 `theta_balance` 及其长实验脚本，可参考计数、超时、报告和保存方式，但本轮不将 SR1 接入它的方向公式，不向那个脚本继续堆入实验逻辑。

## 2. 符号适配：必须先对齐

### 2.1 对话与仓库记号映射

| 本文／对话 | 仓库 | 含义 |
|---|---|---|
| 行势（可记为 \(\lambda\)） | `mu` | 长度等于行数 |
| 列势 \(\mu\) | `nu` | 长度等于列数；本轮实际优化的势 |
| \(z=-\mu/\varepsilon\) | 新局部变量 `z_col = -nu / epsilon` | 无量纲列势 |
| 成本 \(C\) | `problem.M` | 运输成本 |
| \(\Phi(z)\) | 新半对偶核中的 log-partition | 不等于旧文档称作 `Phi` 的完整对偶目标 |
| \(r=P\mathbf1, c=P^\top\mathbf1\) | 对应行和、列和 | 保持既有约定 |

本轮文档中的 \(B_0\) 是对角逆响应基础算子，**不是**旧算法的 Hessian 耦合参数。代码命名使用 `sr1_*`，不复用原 `theta`。

### 2.2 行归一化半对偶状态

记 \(\ell_i(z)=\operatorname{logsumexp}_j(-M_{ij}/\varepsilon+z_j)\)。采用

\[
\begin{aligned}
\nu_j&=-\varepsilon z_j,\\
\mu_i^{\rm repo}&=\varepsilon[\ell_i(z)-\log a_i],\\
P_{ij}(z)&=a_i\exp[-M_{ij}/\varepsilon+z_j-\ell_i(z)],\\
r&=a,\qquad c=P^\top\mathbf1.
\end{aligned}
\tag{28,1}
\]

新建共享的半对偶评估函数，例如 `evaluate_reduced(problem, z_col)`。返回一个轻量状态对象，包含规范化后的 `z_col`、完整势、`P`、`c`、`log_c`、`f`、公共残差以及需要复用的行归一化中间量。以本轮中小规模稠密实验为目标，允许当前状态和当前试探状态各缓存一个 `P`，不在历史中存储每步矩阵。

\[
\begin{aligned}
\Phi(z)&=\sum_i a_i\ell_i(z),\qquad f(z)=\Phi(z)-b^\top z,\\
e&=c-b=\nabla f(z),\qquad \psi=\log(c/b),\\
L_P&=D_c-P^\top D_a^{-1}P=\nabla^2f(z).
\end{aligned}
\tag{28,2}
\]

完整对偶目标在行归一化状态满足

\[
\frac{\Phi_{\rm full}(\mu^{\rm repo},\nu)}{\varepsilon}
=f(z)+1-\sum_i a_i\log a_i.
\tag{28,3}
\]

公共 `dual_objective` 字段仍记录旧完整目标，新增 `reduced_f` 记录 \(f\)。不能把这两个数混用。

### 2.3 规范与 Sinkhorn 的观测位置

- 内部每次存储势之前，统一 `z_col -= z_col.mean()`；两次历史差分必须处于同一个规范。
- 不使用旧 `normalize_gauge` 的输出直接构造未再中心化的 `s`。导出完整势时可以调用原规范函数，物理状态不变。
- 本轮基础更新为

\[
z^+=\Pi(z-\psi),\qquad c^+=\nabla\Phi(z^+),
\qquad \Pi=I-\frac1{n_c}\mathbf1\mathbf1^\top.
\tag{28,4}
\]

其中 \(n_c\) 是列数。它是原 Sinkhorn 的列更新，再完成下一次行归一化后的状态。

新增入口建议称为 `sinkhorn_reduced`，标注为“标准 Sinkhorn，行归一化后记录”。主实验让它和 SR1 共用评估核。保留原 `sinkhorn` 求解器，不悄悄改变它的观测位置和历史语义。

回归时应比较：从同一行归一化状态调用旧 `sinkhorn_log_sweep`，再补行归一化，所得 `P,c` 与式（28,4）一致；列势在中心化后也一致。不要把两个不同半步位置的残差强行逐轮比较。

## 3. SR1 v0：精确规定本轮模型

### 3.1 Sinkhorn 基础算子

\[
B_0(c)=\operatorname{diag}(\beta_j),\qquad
\beta_j=\begin{cases}
\log(c_j/b_j)/(c_j-b_j),&c_j\ne b_j,\\
1/b_j,&c_j=b_j,
\end{cases}
\quad B_0e=\psi.
\tag{28,5}
\]

只存对角向量。接近 \(c=b\) 时采用 `log1p(t)/t` 的稳定连续延拓，\(t=(c-b)/b\)；例如小 \(|t|\) 用相应泰勒展开。远离 1 的比值可使用 `log_c - log_b`，避免因 `t` 舍入为 -1 产生假的无穷值。不得偷偷加边际下限或裁剪残差。

### 3.2 最近一对历史、正分母、完整秩一校正

\[
\begin{aligned}
s_k&=z_k-z_{k-1},\qquad y_k=c_k-c_{k-1},\\
q_k&=s_k-B_0(c_k)y_k,\qquad d_k=q_k^\top y_k,\\
V_k&=\psi_k+q_k\frac{q_k^\top e_k}{d_k}.
\end{aligned}
\tag{28,6}
\]

v0 固定完整校正，即此前的 \(\theta_k=1\)。**每步重建当前 \(B_0(c_k)\)，只使用最近一对历史；不递推积累旧的 SR1 矩阵，不实现 L-SR1/BFGS/多对记忆。**

只有所有量有限且满足以下条件才使用校正：

\[
d_k>0,\qquad d_k\ge10^{-8}\|q_k\|_2\|y_k\|_2.
\tag{28,7}
\]

无历史、零历史、\(q=0\)、非正分母、过小分母或非有限校正均明确记录原因，当前步使用 Sinkhorn。零残差优先触发收敛，不再构造方向或计算增益比值。

原（23,6）应作为性质检查，不能当作已证明实际加速：

\[
\begin{aligned}
e_k^\top V_k&=J_k+\frac{(q_k^\top e_k)^2}{d_k}\ge J_k,
\qquad J_k=e_k^\top\psi_k,\\
\left(B_0+\frac{q_kq_k^\top}{d_k}\right)y_k&=s_k.
\end{aligned}
\tag{23,6}
\]

代码不用显式形成秩一矩阵，更不形成完整 Hessian、求逆、求特征值。分组信息和最优解也不进入方向计算。

### 3.3 离散、回溯和回退

先共享 20 次 Sinkhorn 更新，之后才尝试 SR1。合法 SR1 方向试探：

\[
z^{\rm trial}=\Pi(z_k-hV_k),\qquad
f(z^{\rm trial})\le f(z_k)-10^{-4}h\,e_k^\top V_k.
\tag{28,8}
\]

- 步长依次为 `1, 1/2, 1/4, 1/8, 1/16, 1/32`，总共至多 6 次试探。
- 现有 `choose_step` 的 `max_backtracks + 1` 语义需要注意；若沿用它的字段，这里应设置 `max_backtracks=5`，不是 6。
- 回溯用同一个方向，只改变 `h`；不要在被拒绝的试探点重新生成历史。
- 6 次都未通过，就从最后接受的当前状态执行式（28,4）的单位 Sinkhorn 步。所有已失败试探照常计入成本。
- 回退步不再套 SR1 的 Armijo 条件。回退后保留实际接受的最近两个状态作为下一轮历史，不每次清空历史。
- 任何被拒绝的状态不得覆盖当前状态；非有限试探计为拒绝。回退自身失败则以明确状态退出，保留最后有效解。
- `max_iter` 计算接受的总迭代数，包含预热；试探不算接受迭代，但独立计数。

### 3.4 高精度阶段的目标差

不能仅以两个较大的 `f` 直接相减判断极小下降，也不能给 Armijo 加一个固定宽容常数，让非下降步默默通过。

优先实现稳定的目标差接口，利用精确关系

\[
\Delta f=f(z-hV)-f(z)=-h\,e^\top V+D_\Phi(z-hV,z).
\tag{28,9}
\]

稳定计算的可用构造：令 \(p_{ij}=P_{ij}/a_i\)、\(u_j=-hV_j\)、\(m_i=\sum_jp_{ij}u_j\)、\(v_{ij}=u_j-m_i\)，则

\[
D_\Phi(z-hV,z)
=\sum_i a_i\log\!\left(\sum_jp_{ij}e^{v_{ij}}\right)
=\sum_i a_i\log\!\left[1+\sum_jp_{ij}(e^{v_{ij}}-1-v_{ij})\right].
\tag{28,10}
\]

最后的等式使用行条件分布的归一化及零均值性质。实现要检查这两项的浮点误差；若对行概率再归一化，要把它作为数值评估措施明确记录，不能改动求解目标或真实边际。

小步长区域用 `log1p` 与稳定的 `expm1(v)-v`（极小时用从二次项开始的级数）；较大步长用 logsumexp 表达，避免指数溢出。在小算例用高精度或独立有限差分检查这一辅助函数。额外的稠密遍历必须计入实际工作。

若保留仓库的“舍入误差内目标变化 + 残差严格减小”备用接受规则，必须单独记录 `accept_reason=roundoff_residual`，与真正 Armijo 接受分开统计；不能把这种接受写成通过严格 Armijo。严禁为达成收敛结果而放宽停止阈值。

## 4. 文件分工与最小接口

优先采用下表的落点；如果本地已有等价模块，复用并在报告中说明实际路径。无需为了逐字匹配文件名做大范围重构。

| 路径 | 操作与职责 |
|---|---|
| `src/sinkhorn_lmflow/core/semi_dual.py` | 新增行归一化评估、`z` 与完整势转换、稳定目标差；不含 SR1 控制逻辑 |
| `src/sinkhorn_lmflow/solvers/mirror_sr1.py` | 新增 SR1 方向纯函数和共用迭代过程；导出 `solve_mirror_sr1`、`solve_sinkhorn_reduced` |
| `src/sinkhorn_lmflow/problems/weak_coupling.py` | 新增受控核生成器和独立初值生成器 |
| `src/sinkhorn_lmflow/experiments/sr1_study.py` | 配对预热、实验调度、预算、检查点、汇总；复用原有 I/O 和运行信息 |
| `src/sinkhorn_lmflow/plotting/sr1.py` | 读取已存日志生成本轮图表，不调用求解器 |
| `configs/sr1_smoke.yaml` | 小规模实现核对 |
| `configs/sr1_controlled.yaml` | 18 个受控算例、重复计时与固定参数 |
| `configs/sr1_transfer.yaml` | 两个既有点云算例的迁移检查 |
| `scripts/run_sr1_study.py` | 薄 CLI：解析参数后调用实验模块 |
| `scripts/report_sr1.py` | 薄 CLI：从已有结果生成图表与报告 |
| `tests/test_semi_dual.py` | 符号、规范、半步对应和稳定目标差检查 |
| `tests/test_mirror_sr1.py` | 割线、耗散、保护、回溯与历史状态检查 |
| `tests/test_sr1_experiment.py` | 数据指纹、共享预热、计数、保存回读与失败保留 |
| `docs/sr1_v0_report.md` | 最终中文报告，附复现命令和真实结果路径 |

现有模块允许的必要扩展：

- `SolverConfig` 增加 `sr1_warmup_iters=20`、`sr1_secant_tol=1e-8` 等少量明确字段；共享已有 `tol/max_iter/step_size/backtrack_factor/armijo/wall_time_limit_sec`。在新 solver 中验证其实际支持的配置。
- `IterationRecord` 新增必要的可选 `sr1_*`、`reduced_f`、`energy_gap`、`state_eval_count` 字段；默认值保证旧结果可读。浮点缺失用 `None`，计数用整数，原因用字符串；同步更新 `load_result`。
- `Monitor` 增加接收已评估状态的入口，或向后兼容的可选缓存参数。普通旧调用行为不变。方向、回溯、日志只使用明确的当前／试探状态对象，不维护容易过期的全局缓存。
- `registry.py` 注册上述两个名称。若支持原 `run_experiment.py --solver`，同步更新其 argparse choices；不能只注册而遗漏 CLI 限制。
- 本轮配对实验使用 `sr1_study`，因为通用 runner 目前没有共享预热历史接口。保存和格式仍沿用现有组件。
- README 增加简短入口与报告链接，不改写旧实验说明。

建议求解器保留原来的 `(problem, config)` 调用，额外支持可选的、已验证的预热快照。快照包括最后两个规范一致的 `(z,c)`、当前评估状态、预热历史、计数、时间和指纹；两个求解器读取独立副本，不能互相修改。

### 缓存与计数契约

1. `evaluate_reduced` 每次真实执行，增加一次 `state_eval_count`；包含初始状态、预热、所有拒绝试探和回退评估。
2. 接受状态直接传入 Monitor 和下一轮。不得为日志、停止检查或下一轮初始化再次计算相同的 `P,c,f`。
3. 缓存结果传入 Monitor 不增加新的矩阵构造计数；普通旧 `observe` 仍按实际构造计数。
4. 目标差稳定计算、logsumexp、构造 P 等实际工作分别记录。`state_eval_count` 不是全部 FLOPs，额外目标差核的工作不能藏在这个计数之外。
5. 所有方法使用相同评估器、相同诊断频率。`compute_spectrum_every=0`，主计时中不调用谱诊断或完整 Hessian。
6. 不把 SR1 向量运算伪记成一次线性系统求解；不使用旧 `matrix_vector_count` 独自衡量本轮成本。

## 5. 实验数据：已知解、可调弱耦合

### 5.1 受控生成器

设总规模为 \(N\)，组数为 \(B\)，每组 \(m=N/B\)，要求能整除。为了不和成本 `M` 混淆，组间矩阵在代码中命名 `block_transition`，数学记为 \(T_\delta\)：

\[
\Gamma_{gh}=\begin{cases}(1+|g-h|)^{-1},&g\ne h,\\0,&g=h,\end{cases}
\qquad
T_\delta=I-\delta[\operatorname{diag}(\Gamma\mathbf1)-\Gamma].
\tag{28,11}
\]

\[
K_{ij}=\frac{(T_\delta)_{gh}}m\quad(i\in I_g,j\in I_h),\qquad
a=b=\frac1N\mathbf1,\qquad\varepsilon=1,\qquad M=-\log K.
\tag{28,12}
\]

验证严格正性及双随机性。当前参数下对角元为正；非法参数直接拒绝，不裁剪。其已知解为

\[
z_*=0,\qquad P_*=K/N,\qquad f_*=0,\qquad E=f-f_*.
\tag{28,13}
\]

实现直接构造 `OTProblem`。**不调用 `normalize_cost`**，否则式（28,13）和 \(\delta\) 的含义会被改变。

组标签、已知解只存在于生成／检查／报告层。求解器输入不包含用来构造方向的分组或精确解信息。求解后可用这些已知信息诊断。

### 5.2 初值与配对

以独立初值 seed 生成 \(\xi_g\sim N(0,1)\)、\(\eta_j\sim N(0,1)\)，取

\[
z_j^0=\xi_{g(j)}+0.1\eta_j,
\qquad z^0\leftarrow\Pi z^0.
\tag{28,14}
\]

不得使用零初值，这里零初值已经是最优解。将初始 `z` 以及对应完整势保存到独立 `initial_state.npz`，计算初值 SHA256；对三种 \(\delta\)，同组数、同 seed 使用同一初值。

每个 case、每个计时重复：

1. 从指定初值用 `sinkhorn_reduced` 预热 20 步，保留第 19、20 步及完整预热记录。
2. 复制同一预热快照，分别继续 Sinkhorn 和 SR1。
3. 若预热已达到容差，记为 `warmup_converged`，两方法都无需继续；该 case 不解释为 SR1 加速。
4. 每种方法的总成本为共享预热成本加各自后续成本；同时单独报告后续成本。共享预热只实际执行一次，但比较表中双方都承担这笔成本。
5. 校验问题指纹、初值指纹、预热快照指纹完全相同。只比较问题指纹不足以保证本轮配对成立。

受控问题的三个 seed 只改变初值，不改变同 \(N,B,\delta\) 的成本矩阵，因此报告中明确称为“初值种子”，不是三个独立数据集。

### 5.3 固定实验矩阵

| 阶段 | 数据与规模 | 重复 | 要求 |
|---|---|---|---|
| Smoke | \(N=32,B=2,\delta=10^{-2}\)，初值 seed 42 | 1 | 先验证端到端实现和保存绘图 |
| Controlled | \(N=256\)，\(B\in\{2,4\}\)，\(\delta\in\{10^{-1},10^{-2},10^{-3}\}\)，初值 seed \(\{42,43,44\}\) | 每个 case 3 次计时 | 共 18 个 case、108 条分叉后方法轨道 |
| Transfer | 原 `generate_problem(generator="2d_pointcloud", n=128, m=128, seed=42, clustered=True)`，\(\varepsilon\in\{0.05,0.01\}\) | 1 | 原有成本归一化和边际生成保持不变；初始列势为零，再行归一化；不需要精确最优值 |

全部统一：float64、CPU、单线程 BLAS、`record_every=1`、`compute_spectrum_every=0`、最终容差 `1e-8`、`max_iter=20000`（含预热）。本轮不做参数网格搜索，不引入其他算法候选。

保存达到 `1e-4, 1e-6, 1e-8` 的首次接受状态及其成本，不为三个精度重新独立运行；若起始状态已满足某阈值，记录时间／迭代零，不能据此计算有意义的加速倍数。

\[
\rho_k=\max(\|r_k-a\|_1,\|c_k-b\|_1)\le10^{-8}.
\tag{28,15}
\]

行归一化正确时 \(\rho_k\) 基本等于列残差，但仍记录两者以发现行规范错误。Transfer 的 `energy_gap` 为 `None`，不可把“该批最小观测目标”冒充 \(f_*\)。

受控问题在极小误差阶段也要防止能量差消减：可用已知参考态计算 `E = D_Phi(z, 0)`，复用稳定 Bregman 评估思路。若为每步日志额外计算，它属于诊断成本，双方采用同一频率并如实计入；也可只在约定检查点计算。不要把负的舍入误差直接裁剪成一个漂亮的正数并当成有效目标精度。

## 6. 记录什么，怎样解释

### 6.1 轨道记录

保留原 history 的字段，增加或明确以下数据：

| 数据 | 内容 |
|---|---|
| 状态 | 接受迭代数、预热／SR1／SK 回退 phase、行列残差、`reduced_f`、受控 case 的 `energy_gap` |
| 成本 | 包含预热的时间、后续时间、`state_eval_count`、目标差核调用及实际稠密工作计数、试探和拒绝数 |
| 割线 | `d`、相对分母 `d/(||q|| ||y||)`；零分母场景记缺失及原因 |
| 驱动 | `J`、`e_dot_V`、预测增益 `G`、规范投影后的驱动夹角 |
| 离散 | 接受步长、实际下降 `-delta_f`、兑现率 `Q`、接受原因 |
| 保护 | SR1 尝试数、有效方向数、接受数、各类跳过数、回溯耗尽数、SK 回退数 |

\[
G_k=\frac{e_k^\top V_k}{J_k}
=1+\frac{(q_k^\top e_k)^2}{d_kJ_k},
\qquad
Q_k=\frac{f(z_k)-f(z_k-h_kV_k)}{h_k e_k^\top V_k}.
\tag{28,16}
\]

只在分母可靠、SR1 方向有效时计算相应比值。接近零的分母给 `None` 并记录原因；不人为加一个大 epsilon 改变诊断。回退步不得沿用失败的 SR1 预测值，预测与实际必须对应同一次尝试／接受事件。

夹角可记录 `cos(Pi V, Pi psi)`，零向量时缺失。它用于判断修正主要是放大原方向还是改变方向；本轮不据此再加入一个新基线。

如果将进入当前状态的步骤诊断放在该状态所在行，应明确 `from_iteration`/`to_iteration` 或在字段文档说明。避免把 \(G_k\) 与下一步的残差或步长错配。

### 6.2 三类主图与表格

每个匹配的 case 生成：

1. 边际残差—累计评估次数；
2. 边际残差—时间，分别能看总成本和分叉后成本；
3. 受控 case 的能量差—累计评估次数；
4. 简洁诊断面板：`G`、`Q`、步长、回退／跳过事件（辅助图）。

主表列出每个 case／阈值的成功率、达到阈值的迭代数、完整评估次数、额外目标差核次数、总时间／后期时间，以及 SR1 接受和回退比例。三次计时报告中位数及范围，数据／初值种子与计时重复分别标识。

加速比定义为相同 case、相同阈值下 `T_SK / T_SR1`。任一方法未达阈值时记录缺失或截尾事实，不把失败时间伪装成完成时间。保留所有失败、超时及 pending 项，图上标记非收敛终点。

当前公共绘图分组没有初值维度，本轮应按 `(problem_sha256, initial_sha256, warmup协议, tol)` 分组，再区分 repeat 和 solver。不要把不同初值曲线当作同一条配对轨道。

### 6.3 报告需要作出的判断

- 跳过多：主要是哪类分母／数值问题？
- `G≈1`：当前残差与历史修正作用是否很弱？
- `G` 很大但 `Q` 很低、步长很小：是否被有限步长曲率代价抵消？
- 迭代减少但评估／时间增加：额外工作是否抵消收益？
- 两组与四组、三档耦合之间有什么可重复的差别？
- 点云迁移与受控问题的观察是否一致？

只作数据支持的解释。不能把（23,6）写成整条轨道优于 Sinkhorn 的定理，也不能以受控核表现推断一般 EOT 都加速。若收益不明显，给出证据和最直接的下一步假设，保留本轮 v0 结果；不要自动改成多记忆／另一种拟牛顿。

## 7. 必要验证与执行顺序

### 7.1 有意义的数学及集成检查

先补下列检查，再执行全部现有测试；这些检查验证独立数学关系和关键故障路径，不仅复述实现。

1. **半对偶状态**：使用非方阵、非均匀边际、小 \(\varepsilon\) 的小问题，验证 `r=a`、完整势重建一致、式（28,3）成立。
2. **符号与响应**：有限差分验证 \(\nabla_z f=c-b\)，小矩阵下验证 Jacobian 为 \(L_P\)；仅测试中显式形成 Hessian。
3. **Sinkhorn 端点**：式（28,4）与旧 sweep 加行归一化一致；多个连续步骤也一致。关闭校正的新参考流完全退化为该 Sinkhorn 流。
4. **规范不变性**：给输入列势加常数后，规范化轨道、`P,c`、校正结果一致。
5. **基础算子及 SR1**：稳定地验证 `B0*e=psi`，含 `c=b` 极限；合法正分母下验证割线与（23,6）。零、负、近零和非有限分母都走规定分支。
6. **稳定目标差**：温和步长时与直接目标差一致；微小步长时用高精度／独立参考验证符号和量级。检查稠密遍历被计数。
7. **回溯和状态**：强制拒绝一次／全部拒绝，验证有限试探预算、回退、旧状态不被污染、历史只来自接受状态、所有成本计入。
8. **缓存和保存**：被接受的试探不被 Monitor 再评估；新增整数／字符串／None 正确回读，旧 history 仍可读。
9. **已知解与配对**：受控核严格正且双随机，`z=0` 符合已知解；问题和初值指纹区分正确，共享预热的两份副本不互相污染。

### 7.2 本地执行

在既有环境中完成，建议最终提供这些真实可运行入口（如参数名不同，在报告中给出实际命令）：

```powershell
.\scripts\run_ot.cmd -m pytest -q
.\scripts\run_ot.cmd scripts/run_sr1_study.py --config configs/sr1_smoke.yaml
.\scripts\run_ot.cmd scripts/run_sr1_study.py --config configs/sr1_controlled.yaml
.\scripts\run_ot.cmd scripts/run_sr1_study.py --config configs/sr1_transfer.yaml
.\scripts\run_ot.cmd scripts/report_sr1.py --run-dir results/<实际运行目录>
```

首次性能测试前做一次不计入汇总的小规模预运行，避免导入和首次调用开销。重复计时交替两种方法的执行顺序，保证系统负载设置一致。日志在内存中收集，落盘、绘图、报告生成放在 solver 计时之外；算法必需的诊断和方向计算仍包含在 solver 时间中。

默认每条正式方法轨道后续求解时间上限 120 秒，每次实验命令总预算 3600 秒，均配置化。总预算包含预热和调度中的求解时间；逐试探检查预算，不能只有接受步结束才检查。预算耗尽保存已完成和部分结果，其余标记 pending，输出恢复命令，不静默缩小实验矩阵或放宽容差。

支持按配置与输入指纹跳过完整已完成 case 的恢复方式；不强求从某次 SR1 中间状态恢复。半途失败的 case 可作为新 attempt 重跑并保留旧记录。不要把部分日志当作完整成功结果。

## 8. 输出目录和交付要求

继续使用 `results/<时间戳>_sr1_*` 独立目录及原有运行文件格式。每个 study 至少包含：

| 输出 | 内容 |
|---|---|
| `study_config.yaml` / `manifest.json` | 完整参数、计划 case、repeat、状态、配对关系与恢复信息 |
| `problems/<fingerprint>/` | 原有 NPZ/JSON 数学输入，每个不同问题只保存一次 |
| `initial_states/<initial_hash>/` | 初值、初值参数、指纹 |
| `warmups/<pair_id>/` | 公共预热末两状态、记录、成本与指纹 |
| 每个方法的标准 run 目录 | `history.csv`、`summary.json`、`config_resolved.yaml`、`problem_ref.json`、`runtime_metadata.json`、`final_state.npz` |
| `tables/` | 所有运行汇总、各阈值成本表、配对加速比、跳过／回退统计 |
| `figures/` | 按 case 分组的 PNG/PDF，失败可见 |
| `report.md` | 该次 study 的结果解释、限制、命令和路径 |

`docs/sr1_v0_report.md` 汇总三阶段结果，至少写清：

1. 实际复用了什么，新加／修改了什么，和本说明的差异；
2. 采用的符号适配、基线半步位置、完整目标与半对偶目标的关系；
3. 测试结果、实验覆盖、成功率和失败原因；
4. 带真实数字的成本表与图的相对路径；
5. 理论增益是否兑现，哪里有效、哪里无效；
6. 实际复现／恢复命令、环境和 git 状态。

保持根目录整洁，不新增 `sr1_final_v2_tmp.py` 一类脚本。不复制已有 results 或笔记到新项目，不将 `P` 的逐步快照、临时图和运行产物堆入源码目录。遵守已有 `.gitignore` 的结果保存约定；本轮不要求提交大量实验文件。

**完成标准：代码能在既有 OT 环境运行，原测试与新增必要检查通过；三阶段按预算执行并保存可审查结果；报告明确回答成本收益与失败机制。只写代码或只给一个运行命令不算完成；遇到真实环境阻塞则准确记录已完成工作、阻塞点及剩余命令。**

## 附：本任务书核对过的源码入口

- [AGENTS.md](https://github.com/Basiliscus-Denebola/sinkhorn-lmflow/blob/886445543600bef327b6a54ea1ed30ccef57d403/AGENTS.md)
- [完整对偶与符号说明](https://github.com/Basiliscus-Denebola/sinkhorn-lmflow/blob/886445543600bef327b6a54ea1ed30ccef57d403/docs/math_spec.md)
- [Sinkhorn 实现](https://github.com/Basiliscus-Denebola/sinkhorn-lmflow/blob/886445543600bef327b6a54ea1ed30ccef57d403/src/sinkhorn_lmflow/solvers/sinkhorn.py)
- [Monitor 与统一结果构造](https://github.com/Basiliscus-Denebola/sinkhorn-lmflow/blob/886445543600bef327b6a54ea1ed30ccef57d403/src/sinkhorn_lmflow/solvers/base.py)
- [现有步长策略](https://github.com/Basiliscus-Denebola/sinkhorn-lmflow/blob/886445543600bef327b6a54ea1ed30ccef57d403/src/sinkhorn_lmflow/policies/stepsize.py)
- [现有类型定义](https://github.com/Basiliscus-Denebola/sinkhorn-lmflow/blob/886445543600bef327b6a54ea1ed30ccef57d403/src/sinkhorn_lmflow/types.py)
- [现有结果 I/O](https://github.com/Basiliscus-Denebola/sinkhorn-lmflow/blob/886445543600bef327b6a54ea1ed30ccef57d403/src/sinkhorn_lmflow/io/result_io.py)
- [现有问题生成器](https://github.com/Basiliscus-Denebola/sinkhorn-lmflow/blob/886445543600bef327b6a54ea1ed30ccef57d403/src/sinkhorn_lmflow/problems/generators.py)
