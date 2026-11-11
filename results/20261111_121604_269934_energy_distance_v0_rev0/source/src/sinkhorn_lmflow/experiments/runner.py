from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from ..types import OTProblem, SolverConfig
from ..problems.generators import generate_problem
from ..io.problem_io import save_problem, load_problem, problem_fingerprint
from ..io.result_io import save_result
from ..io.paths import create_run_dir
from .registry import get_solver
from .reproducibility import runtime_metadata


def make_problem(config: dict) -> OTProblem:
    if "path" in config:
        if len(config) != 1:
            raise ValueError("problem.path cannot be combined with generator overrides")
        return load_problem(config["path"])
    return generate_problem(**config)


def run_experiment(config: dict, output_root: str | Path = "results", *,
                   problem: OTProblem | None = None, problem_path: Path | None = None) -> Path:
    resolved = deepcopy(config)
    solver_spec = dict(resolved.get("solver", {}))
    name = solver_spec.pop("name", "lmflow")
    diagnostics = resolved.get("diagnostics", {})
    if set(diagnostics) - {"record_every", "compute_spectrum_every"}:
        raise ValueError("Unknown diagnostic setting")
    solver_spec.update(diagnostics)
    cfg = SolverConfig(**solver_spec)
    solver = get_solver(name)
    problem = problem if problem is not None else make_problem(resolved.get("problem", {}))
    resolved["solver"] = {"name": name, **asdict(cfg)}
    resolved["problem_resolved"] = {"name": problem.name, "shape": list(problem.M.shape),
                                    "epsilon": problem.epsilon, "metadata": problem.metadata}
    run_dir = create_run_dir(output_root, resolved.get("experiment", {}).get("name", name))
    if problem_path is None:
        problem_path = save_problem(problem, run_dir / "input")
    import os
    reference = {"path": os.path.relpath(problem_path, run_dir),
                 "sha256": problem_fingerprint(problem)}
    runtime = runtime_metadata(problem.metadata.get("seed"))
    result = solver(problem, cfg)
    save_result(result, run_dir, resolved, reference, runtime)
    return run_dir
