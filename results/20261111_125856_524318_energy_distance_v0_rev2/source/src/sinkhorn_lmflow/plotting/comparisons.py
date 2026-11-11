from __future__ import annotations

import json
import textwrap
from pathlib import Path
import pandas as pd
from ..io.config import load_config


def find_runs(path: str | Path) -> list[Path]:
    path = Path(path)
    if (path/"summary.json").exists():
        return [path]
    runs = sorted(p.parent for p in path.rglob("summary.json"))
    if not runs:
        raise ValueError(f"No saved runs in {path}")
    return runs


def run_label(path: str | Path) -> str:
    path = Path(path)
    cfg = load_config(path/"config_resolved.yaml")
    summary = json.loads((path/"summary.json").read_text(encoding="utf-8"))
    s = cfg.get("solver", {})
    label = summary["solver_name"]
    if label == "lmflow":
        label += f" theta={s.get('theta')} {s.get('discretization')}"
        if s.get("gauss_seidel"):
            label += " GS"
    elif label == "hard_switch":
        label += f" switch={s.get('switch_tol'):g}"
    elif label == "sinkhorn":
        label += f" {s.get('numerical_mode')}"
    if not summary["converged"]:
        label += "\n" + textwrap.fill(f"[NOT CONVERGED: {summary['stop_reason']}]", width=65)
    return label


def summarize_runs(path: str | Path) -> pd.DataFrame:
    rows = []
    for run in find_runs(path):
        row = json.loads((run/"summary.json").read_text(encoding="utf-8"))
        cfg = load_config(run/"config_resolved.yaml")
        row.update(run_path=str(run), problem=cfg.get("problem_resolved", {}).get("name"),
                   epsilon=cfg.get("problem_resolved", {}).get("epsilon"),
                   theta=cfg.get("solver", {}).get("theta") if row["solver_name"] == "lmflow" else None,
                   discretization=cfg.get("solver", {}).get("discretization"),
                   tol=cfg.get("solver", {}).get("tol"),
                   switch_tol=cfg.get("solver", {}).get("switch_tol"))
        rows.append(row)
    return pd.DataFrame(rows)


def group_runs(paths: list[Path]) -> dict[str, list[Path]]:
    """Separate unlike problems/tolerances so plots do not imply fair comparisons."""
    groups: dict[str, list[Path]] = {}
    for path in paths:
        cfg = load_config(path/"config_resolved.yaml")
        ref = json.loads((path/"problem_ref.json").read_text(encoding="utf-8"))
        key = f"{cfg['problem_resolved']['name']}_{ref['sha256'][:8]}_tol{cfg['solver']['tol']:g}"
        groups.setdefault(key, []).append(path)
    return groups
