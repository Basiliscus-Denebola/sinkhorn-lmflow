import json
import numpy as np
import pytest
from sinkhorn_lmflow.problems.generators import generate_problem
from sinkhorn_lmflow.experiments.sweep import run_sweep, expanded_configs
from sinkhorn_lmflow.io.result_io import load_result


@pytest.mark.parametrize("generator", ["random_dense", "1d_squared", "2d_pointcloud", "spiky_marginal"])
def test_deterministic_generators(generator):
    x = generate_problem(generator, n=10, m=7, seed=10)
    y = generate_problem(generator, n=10, m=7, seed=10)
    np.testing.assert_array_equal(x.M, y.M)
    np.testing.assert_array_equal(x.a, y.a)
    assert x.M.shape == (10, 7)
    assert x.M.max() == 1
    assert np.all(x.a > 0)


def test_sweep_common_problem_and_failure_saved(tmp_path):
    config = {"problem": {"n": 4, "m": 3, "seed": 7},
              "solver": {"name": "lmflow", "max_iter": 1},
              "sweep": {"axes": {"solver.theta": [0, 1]}}}
    path = run_sweep(config, tmp_path)
    manifest = json.loads((path/"manifest.json").read_text())
    refs = []
    for run in manifest["runs"]:
        refs.append(json.loads((path/run/"problem_ref.json").read_text())["sha256"])
        result = load_result(path/run)
        assert not result.converged
        assert result.stop_reason == "max_iter"
        assert (path/run/"runtime_metadata.json").exists()
    assert len(set(refs)) == 1


def test_sweep_size_guard():
    with pytest.raises(ValueError, match="max_runs"):
        expanded_configs({"sweep": {"axes": {"solver.theta": list(range(201))}}})
