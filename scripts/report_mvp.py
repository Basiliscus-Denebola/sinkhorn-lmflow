"""Build the delivery report from saved artifacts, without rerunning solvers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import _bootstrap  # noqa: F401
import pandas as pd
from sinkhorn_lmflow.io.config import load_config
from sinkhorn_lmflow.plotting.comparisons import find_runs, summarize_runs


def table(frame: pd.DataFrame, columns: list[str]) -> str:
    def cell(value) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "—"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(cell(row[c]) for c in columns) + " |" for _, row in frame.iterrows())
    return "\n".join(lines)


def source_tree(root: Path) -> str:
    lines = ["sinkhorn-lmflow/"]
    excluded = {".git", "__pycache__", ".pytest_cache", ".codex", ".agents"}

    def walk(path: Path, prefix: str) -> None:
        entries = sorted((p for p in path.iterdir() if p.name not in excluded), key=lambda p: (p.is_file(), p.name))
        for index, entry in enumerate(entries):
            last = index == len(entries)-1
            lines.append(prefix + ("└── " if last else "├── ") + entry.name + ("/" if entry.is_dir() else ""))
            if entry.is_dir():
                child_prefix = prefix + ("    " if last else "│   ")
                if entry.name == "results":
                    lines.append(child_prefix + "[实验文件见 G；包含 .gitkeep、各 run 和 figures]")
                else:
                    walk(entry, child_prefix)
    walk(root, "")
    return "\n".join(lines)


def module_table(root: Path) -> str:
    descriptions = {
        "types": "统一问题导出、配置、迭代记录、结果和 policy state dataclass",
        "instance": "不可变 float64 OTProblem 与初始化校验",
        "validation": "维度、有限性、严格正边际、单位质量、epsilon 校验",
        "generators": "四类确定性问题与 Dirichlet 边际生成",
        "costs": "几何 cost 与生成器专用归一化",
        "transport": "K/P 重构及行列和",
        "dual": "共享 convex dual、梯度、完整 Hessian、transport objective",
        "residuals": "共享 marginal residual 与所有 solver 停止指标",
        "gauge": "mean(mu)=0 归一化和 gauge null vector",
        "spectral": "标准化 B、非平凡奇异值与谱图",
        "stability": "scaling positivity 检查及 scaling 导出",
        "linalg": "G_theta、对角/SPD/augmented gauge 直接求解",
        "base": "Solver Protocol、初始化、统一监测与结果构造",
        "sinkhorn": "primal/log 两种数值实现与单轮 GS 更新",
        "lbfgs": "SciPy reduced-dual L-BFGS-B wrapper",
        "newton": "完整 Hessian additive Newton + Armijo",
        "hard_switch": "残差阈值控制 Sinkhorn→Newton 永久切换",
        "lmflow": "fixed theta、两种离散格式与 GS endpoint",
        "theta": "ThetaPolicy 接口与 ConstantTheta",
        "stepsize": "固定请求步长/Armijo、positivity backtracking 与失败原因",
        "registry": "显式 solver 名称映射",
        "runner": "单实验编排、解析配置、保存问题和结果",
        "sweep": "有大小上限的参数扫描与共享 problem artifact",
        "reproducibility": "Python/包/BLAS/CPU/OS/git/seed/UTC 元数据",
        "config": "YAML 与严格 JSON 序列化",
        "problem_io": "problem NPZ/JSON 保存、加载与 SHA256",
        "result_io": "result CSV/JSON/NPZ roundtrip",
        "paths": "唯一 UTC run 目录命名",
        "convergence": "读取历史绘制残差-迭代、残差-耗时、theta 比较",
        "comparisons": "已保存结果发现、分组、失败标记与 CSV 汇总",
        "acc_sinkhorn": "有论文元数据的 NotImplementedError 占位",
        "regot_adapter": "第二阶段 RegOT/SSNS 占位，不安装外部实现",
    }
    lines = ["| 文件 | 职责 |", "| --- | --- |"]
    for path in sorted((root/"src").rglob("*.py")):
        desc = "包入口，无迭代或实验副作用" if path.stem == "__init__" else descriptions[path.stem]
        if path.parent.name == "core" and path.stem == "spectral":
            desc = "标准化 B、sigma1、最大非平凡奇异值与 gap 数学诊断"
        elif path.parent.name == "plotting" and path.stem == "spectral":
            desc = "从历史 CSV 绘制 sigma 与 gap，不运行 solver"
        lines.append(f"| `{path.relative_to(root).as_posix()}` | {desc} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--output", default="docs/mvp_report.md")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.touch(exist_ok=True)
    groups = {}
    for path in sorted(Path(args.results).iterdir()):
        if not path.is_dir():
            continue
        if (path/"sweep_config.yaml").exists():
            name = load_config(path/"sweep_config.yaml")["experiment"]["name"]
        elif (path/"summary.json").exists():
            name = load_config(path/"config_resolved.yaml")["experiment"]["name"]
        else:
            continue
        groups[name] = path
    all_frame = summarize_runs(args.results)
    all_frame.to_csv(Path(args.results)/"all_runs.csv", index=False)
    frames = {name: summarize_runs(path) for name, path in groups.items()}
    for name, frame in frames.items():
        frame.to_csv(groups[name]/"comparison.csv", index=False)
    lines = ["# 第一阶段 MVP 交付报告", "",
             "已使用 OT / Python 3.9.24 完成 Phase 0–6。Python 版本兼容例外由用户明确授权。",
             f"共保存 {len(all_frame)} 次实验：{int(all_frame.converged.sum())} 次达到各自 residual tolerance，"
             f"{int((~all_frame.converged).sum())} 次未收敛。所有失败记录保留。", "",
             "## A. 实际目录树", "", "缓存目录省略；实验目录在 G 中展开说明。", "", "```text",
             source_tree(root), "```", "", "## B. 已实现模块及职责", "", module_table(root), "",
             "`scripts/` 分别提供问题生成、单次运行、扫描、绘图、汇总和本报告生成；",
             "`run_ot.cmd`/`run_ot.ps1` 指定 OT 环境和单线程 BLAS；`_bootstrap.py` 支持源码直接运行。",
             "`configs/` 提供 smoke、smoke baselines、theta、epsilon、baseline 和 ablation 六组配置。",
             "`tests/` 覆盖验证、数学导数、gauge、两端点、谱、solver、坐标一致性及 I/O。", "",
             "## C. 测试报告", "", "```text", (root/"docs/test_report.txt").read_text(encoding="utf-8").strip(),
             "```", "", "## D. 数学核验", "",
             "| 项目 | 验证方法与结果 |", "| --- | --- |",
             "| Gradient | 中心差分，相对误差 < 1e-6，通过 |",
             "| Hessian-vector | 梯度方向差分，相对误差 < 1e-6，通过 |",
             "| Gauge | P、Phi、残差不变，mean(mu)=0，通过 |",
             "| theta=0 GS endpoint | 与显式 Sinkhorn 比值更新比较，rtol=1e-12/atol=1e-14，通过 |",
             "| theta=1 Newton endpoint | G1=H，Hd=rhs，q.T d=0；additive 轨迹一致，通过 |",
             "| Spectrum | 矩形标准化矩阵与 1±theta*sigma 特征值比较，通过 |",
             "| Scaling ODE | 坐标变换后的块方程与 forward Euler 一致，通过 |",
             "| Same optimum | Sinkhorn/Newton/LM 共享解，J+Phi≈0，通过 |", "",
             "## E. Smoke 实验", "", "n=m=20，epsilon=.05，seed=42，tol=1e-8。时间包含诊断。", ""]
    smoke = frames["smoke_baselines"].copy()
    labels = []
    for _, row in smoke.iterrows():
        cfg = load_config(Path(row.run_path)/"config_resolved.yaml")["solver"]
        name = row.solver_name
        if name == "sinkhorn":
            name += " " + cfg["numerical_mode"]
        elif name == "hard_switch":
            name += f" ({cfg['switch_tol']:g})"
        elif name == "lmflow":
            name += f" theta={cfg['theta']}"
        labels.append(name)
    smoke["solver"] = labels
    lines += [table(smoke, ["solver", "converged", "n_iter", "elapsed_sec", "residual_l1", "grad_norm_l2"]), "",
              "## F. Fixed-theta 实验", "",
              "n=m=100，epsilon=.01，seed=42，tol=1e-9；scaling-coordinate、Jacobi、Armijo。",
              "theta=1 此表是 scaling 离散格式，其方向等于 Newton，但不是 additive Newton 轨迹。", "",
              table(frames["theta_sweep"], ["problem", "theta", "n_iter", "elapsed_sec", "residual_l1", "safeguard_count"]), "",
              "这 28 个样本均收敛。外层迭代随 theta 增大减少，但 wall time 不单调；",
              "时间为单次运行且包含诊断与首次调用开销，不能据此宣称普遍加速。", "",
              "### 多 tolerance 基线", ""]
    base = frames["baseline_compare"].copy()
    base["method"] = [r.solver_name + (f" ({r.switch_tol:g})" if r.solver_name == "hard_switch" else "")
                      for _, r in base.iterrows()]
    lines += [table(base, ["method", "tol", "converged", "n_iter", "elapsed_sec", "residual_l1"]), "",
              "### Epsilon 扫描", "",
              table(frames["epsilon_sweep"], ["epsilon", "theta", "converged", "n_iter", "elapsed_sec", "residual_l1", "safeguard_count"]), "",
              "### Ablation", ""]
    ablation = frames["ablation"].copy()
    ablation["GS"] = [load_config(Path(p)/"config_resolved.yaml")["solver"]["gauss_seidel"] for p in ablation.run_path]
    ablation["Armijo"] = [load_config(Path(p)/"config_resolved.yaml")["solver"]["line_search"] for p in ablation.run_path]
    lines += [table(ablation, ["theta", "discretization", "GS", "Armijo", "converged", "n_iter", "residual_l1"]), "",
              "### 全部未收敛记录", "",
              table(all_frame[~all_frame.converged], ["solver_name", "epsilon", "theta", "tol", "n_iter", "residual_l1", "stop_reason"]), "",
              "三个 epsilon=.001 初始求解失败来自未缩放的 dense G 的条件数警告。此时标准化谱",
              "proxy 不能代表未缩放系统的条件数。没有加正则、隐藏 warm start 或改变问题。",
              "L-BFGS 自身终止并不代表通过共享残差阈值。固定 h=1 的 Jacobi 端点不等于 GS Sinkhorn。", "",
              "## G. 已生成文件", ""]
    for name, path in groups.items():
        link = "../" + path.as_posix()
        lines += [f"- **{name}**：[{path.as_posix()}]({link})；"
                  f"[comparison.csv]({link}/comparison.csv)。"]
        figures = list((path/"figures").rglob("*.png")) if (path/"figures").exists() else []
        if figures:
            lines.append(f"  PNG/PDF 各 {len(figures)} 个；[figures]({link}/figures)。")
    lines += ["", "每个 run 保存：`config_resolved.yaml`、`problem_ref.json`、`summary.json`、",
              "`history.csv`、`final_state.npz`、`runtime_metadata.json`。扫描共享问题保存在 `problems/`。",
              "[总表](../results/all_runs.csv) 包含全部实验（不排除失败）；",
              "[配置](../configs/) 和 [可复用 smoke problem](../data/smoke_problem.npz) 已生成。", "",
              "已目视检查 theta 收敛图和含失败标记的 baseline 耗时图；其余图由同一保存结果流程输出。", "",
              "## 当前边界", "",
              "Acc-Sinkhorn/RegOT/SSNS 仍为明确占位；closed-loop theta、Schur-PCG、稀疏 Hessian、GPU 未实现。",
              "未运行 n=50/200 的完整矩阵或多 seed 重复；本轮仅 n=20 smoke 和 n=100 MVP。",
              "矩阵向量计数目前只覆盖显式 primal K 乘法，不能独立用作跨方法复杂度结论。",
              "完整计数和 timing 限制见 [实验协议](experiment_protocol.md)。", ""]
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
