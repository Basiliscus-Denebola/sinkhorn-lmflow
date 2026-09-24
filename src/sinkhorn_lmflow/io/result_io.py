from __future__ import annotations

from dataclasses import asdict, fields
import csv
import json
from pathlib import Path
import numpy as np
from ..types import IterationRecord, SolverResult
from .config import save_json, save_config


def save_result(result: SolverResult, path: str | Path, config: dict | None = None,
                problem_ref: dict | None = None, runtime_metadata: dict | None = None) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    summary = dict(result.summary)
    summary.update(solver_name=result.solver_name, converged=result.converged,
                   stop_reason=result.stop_reason, n_iter=result.n_iter, elapsed_sec=result.elapsed_sec)
    save_json(summary, path/"summary.json")
    save_config(config or {}, path/"config_resolved.yaml")
    save_json(problem_ref or {}, path/"problem_ref.json")
    save_json(runtime_metadata or {}, path/"runtime_metadata.json")
    np.savez_compressed(path/"final_state.npz", mu=result.mu, nu=result.nu, u=result.u, v=result.v)
    with (path/"history.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[x.name for x in fields(IterationRecord)])
        writer.writeheader()
        writer.writerows(asdict(row) for row in result.history)
    return path


def load_result(path: str | Path) -> SolverResult:
    path = Path(path)
    summary = json.loads((path/"summary.json").read_text(encoding="utf-8"))
    integer_fields = {"iteration", "linear_solve_iters", "safeguard_count", "linear_solve_count",
                      "matrix_vector_count", "backtrack_count", "balance_clip_count",
                      "balance_degenerate_count", "balance_fallback_count", "plan_build_count",
                      "reduction_count", "logsumexp_count", "p_matvec_count",
                      "column_update_count", "trial_count", "dense_pass_proxy"}
    history = []
    with (path/"history.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            values = {k: (v if k in ("phase", "balance_reason") else None if v == "" else int(v) if k in integer_fields else float(v))
                      for k, v in row.items()}
            history.append(IterationRecord(**values))
    keys = ("solver_name", "converged", "stop_reason", "n_iter", "elapsed_sec")
    meta = {k: summary.pop(k) for k in keys}
    with np.load(path/"final_state.npz", allow_pickle=False) as state:
        return SolverResult(**meta, mu=state["mu"], nu=state["nu"], u=state["u"], v=state["v"],
                            history=history, summary=summary)
