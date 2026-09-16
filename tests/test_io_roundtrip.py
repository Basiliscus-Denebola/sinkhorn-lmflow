from dataclasses import asdict
import json
import numpy as np
from sinkhorn_lmflow.io.problem_io import save_problem, load_problem, problem_fingerprint
from sinkhorn_lmflow.io.result_io import save_result, load_result
from sinkhorn_lmflow.solvers.lmflow import solve_lmflow
from sinkhorn_lmflow.types import SolverConfig


def test_problem_roundtrip(problem, tmp_path):
    path = save_problem(problem, tmp_path/"problem")
    loaded = load_problem(path)
    for name in ("a", "b", "M"):
        np.testing.assert_array_equal(getattr(problem, name), getattr(loaded, name))
    assert loaded.epsilon == problem.epsilon
    assert loaded.metadata == problem.metadata
    assert problem_fingerprint(loaded) == problem_fingerprint(problem)


def test_result_roundtrip(problem, tmp_path):
    result = solve_lmflow(problem, SolverConfig(max_iter=3))
    save_result(result, tmp_path)
    loaded = load_result(tmp_path)
    for name in ("mu", "nu", "u", "v"):
        np.testing.assert_array_equal(getattr(result, name), getattr(loaded, name))
    assert result.summary == loaded.summary
    assert [asdict(r) for r in result.history] == [asdict(r) for r in loaded.history]
    assert result.stop_reason == loaded.stop_reason
    assert json.loads((tmp_path/"summary.json").read_text())["converged"] is False
