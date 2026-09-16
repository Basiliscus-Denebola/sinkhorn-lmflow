import numpy as np
import pytest
from sinkhorn_lmflow.types import SolverConfig
from sinkhorn_lmflow.core.stability import scaling_coordinate_update
from sinkhorn_lmflow.solvers.lmflow import solve_lmflow
from sinkhorn_lmflow.solvers.lbfgs import solve_lbfgs
from sinkhorn_lmflow.solvers.hard_switch import solve_hard_switch
from sinkhorn_lmflow.solvers.sinkhorn import solve_sinkhorn
from sinkhorn_lmflow.solvers.newton import solve_newton


@pytest.mark.parametrize("theta", [0, .25, .5, .75, .9, .99, 1])
@pytest.mark.parametrize("discretization", ["additive", "scaling_coordinate"])
def test_flow_convergence(problem, theta, discretization):
    result = solve_lmflow(problem, SolverConfig(theta=theta, discretization=discretization, tol=1e-10))
    assert result.converged, result.stop_reason
    assert abs(np.mean(result.mu)) < 1e-14
    objectives = np.array([r.dual_objective for r in result.history])
    assert np.max(np.diff(objectives)) < 1e-13


def test_scaling_rejects_and_backtracks(problem):
    with pytest.raises(FloatingPointError, match="positivity"):
        scaling_coordinate_update(np.zeros(2), np.ones(2), .1, 1)
    result = solve_lmflow(problem, SolverConfig(theta=.5, step_size=10, max_iter=1000))
    assert result.summary["safeguard_count"] > 0
    assert result.converged


@pytest.mark.parametrize("switch_tol", [1e-2, 1e-4, 1e-6])
def test_hard_switch(problem, switch_tol):
    result = solve_hard_switch(problem, SolverConfig(switch_tol=switch_tol, tol=1e-11))
    assert result.converged
    assert result.summary["switch_iteration"] is not None
    assert any(row.phase == "newton" for row in result.history)
    assert any(row.phase == "sinkhorn" for row in result.history)


def test_lbfgs(problem):
    result = solve_lbfgs(problem, SolverConfig(tol=1e-8))
    assert result.converged, result.stop_reason


@pytest.mark.parametrize("solver", [solve_sinkhorn, solve_newton, solve_lbfgs, solve_hard_switch, solve_lmflow])
def test_shared_stopping_and_final_record(problem, solver):
    result = solver(problem, SolverConfig(max_iter=2, record_every=10, compute_spectrum_every=0))
    assert not result.converged
    assert result.n_iter == 2
    assert result.history[-1].iteration == 2
    assert result.history[-1].residual_l1 == result.summary["residual_l1"]
