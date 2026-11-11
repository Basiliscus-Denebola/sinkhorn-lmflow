from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
from ..types import OTProblem
from .config import save_json


def problem_fingerprint(problem: OTProblem) -> str:
    """Hash mathematical inputs (including shapes), independent of metadata."""
    digest = hashlib.sha256()
    for x in (problem.a, problem.b, problem.M, np.asarray(problem.epsilon)):
        a = np.asarray(x, dtype="<f8")
        digest.update(str(a.shape).encode("ascii"))
        digest.update(a.tobytes())
    return digest.hexdigest()


def _npz_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.suffix == ".npz" else path / "problem.npz"


def save_problem(problem: OTProblem, path: str | Path) -> Path:
    path = _npz_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, a=problem.a, b=problem.b, M=problem.M, epsilon=problem.epsilon)
    save_json({"name": problem.name, "metadata": problem.metadata,
               "sha256": problem_fingerprint(problem)}, path.with_suffix(".json"))
    return path


def load_problem(path: str | Path) -> OTProblem:
    path = _npz_path(path)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(path, allow_pickle=False) as d:
        problem = OTProblem(d["a"], d["b"], d["M"], float(d["epsilon"]), meta["name"], meta["metadata"])
    if meta.get("sha256") != problem_fingerprint(problem):
        raise ValueError("Problem fingerprint mismatch")
    return problem
