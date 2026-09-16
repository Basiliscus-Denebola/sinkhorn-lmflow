from __future__ import annotations

from copy import deepcopy
from itertools import product
from pathlib import Path
from ..io.config import save_config, save_json
from ..io.paths import create_run_dir
from ..io.problem_io import save_problem, problem_fingerprint
from .runner import run_experiment, make_problem


def expanded_configs(config: dict) -> list[dict]:
    """Explicit cases plus small named-axis products; refuse accidental explosions."""
    base = deepcopy(config)
    sweep = base.pop("sweep", {})
    cases = sweep.get("cases", [{}])
    axes = sweep.get("axes", {})
    keys = list(axes)
    values = list(product(*(axes[k] for k in keys)))
    if len(cases)*len(values) > sweep.get("max_runs", 200):
        raise ValueError("Sweep exceeds max_runs; split experiments explicitly")
    output = []
    for case in cases:
        for choice in values:
            current = deepcopy(base)
            for key, value in {**case, **dict(zip(keys, choice))}.items():
                parts = key.split(".")
                target = current
                for part in parts[:-1]:
                    target = target.setdefault(part, {})
                target[parts[-1]] = value
            output.append(current)
    return output


def run_sweep(config: dict, output_root: str | Path = "results") -> Path:
    cases = expanded_configs(config)
    parent = create_run_dir(output_root, config.get("experiment", {}).get("name", "sweep"))
    save_config(config, parent/"sweep_config.yaml")
    saved: dict[str, Path] = {}
    runs = []
    for index, current in enumerate(cases):
        problem = make_problem(current.get("problem", {}))
        key = problem_fingerprint(problem)
        if key not in saved:
            saved[key] = save_problem(problem, parent/"problems"/key[:16])
        run = run_experiment(current, parent, problem=problem, problem_path=saved[key])
        runs.append(str(run.relative_to(parent)))
        save_json({"runs": runs, "planned_runs": len(cases)}, parent/"manifest.json")
        print(f"[{index+1}/{len(cases)}] {run.name}", flush=True)
    return parent
