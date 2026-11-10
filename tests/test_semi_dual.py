from decimal import Decimal, localcontext
import numpy as np
import pytest
from sinkhorn_lmflow.problems.instance import OTProblem
from sinkhorn_lmflow.core.semi_dual import (evaluate_reduced, stable_objective_difference,
                                          ReducedWork, center, bregman_phi)
from sinkhorn_lmflow.core.transport import transport_plan_from_potentials
from sinkhorn_lmflow.solvers.sinkhorn import sinkhorn_log_sweep


def small_problem():
    return OTProblem(np.array([.3, .7]), np.array([.2, .3, .5]),
                     np.array([[.1, .4, .2], [.5, .2, .7]]), .03)


def test_state_sign_gauge_and_derivatives():
    p = small_problem()
    z = np.array([.3, -.1, -.2])
    s = evaluate_reduced(p, z)
    np.testing.assert_allclose(s.r, p.a, atol=1e-15)
    np.testing.assert_allclose(transport_plan_from_potentials(p.M, p.epsilon, s.mu, s.nu), s.P, atol=1e-15)
    assert s.full_objective/p.epsilon == pytest.approx(s.f+1-p.a@np.log(p.a))
    t = evaluate_reduced(p, z+14)
    np.testing.assert_allclose(t.P, s.P, atol=2e-15)
    h = 1e-5
    grad = np.array([(evaluate_reduced(p, z+h*v).f-evaluate_reduced(p, z-h*v).f)/(2*h) for v in np.eye(3)])
    jac = np.column_stack([(evaluate_reduced(p, z+h*v).c-evaluate_reduced(p, z-h*v).c)/(2*h) for v in np.eye(3)])
    np.testing.assert_allclose(grad, s.c-p.b, atol=1e-10)
    np.testing.assert_allclose(jac, np.diag(s.c)-s.P.T@(s.P/p.a[:, None]), atol=1e-10)


def test_sinkhorn_endpoint_multiple_steps():
    p = small_problem()
    s = evaluate_reduced(p, [.2, -.3, .1])
    for _ in range(8):
        _, nu = sinkhorn_log_sweep(p, s.mu, s.nu)
        old = evaluate_reduced(p, -nu/p.epsilon)
        new = evaluate_reduced(p, s.z_col-(s.log_c-np.log(p.b)))
        np.testing.assert_allclose(old.z_col, new.z_col, atol=1e-13)
        np.testing.assert_allclose(old.P, new.P, atol=1e-14)
        s = new


def decimal_f(p, z):
    D = lambda x: Decimal(str(float(x)))
    with localcontext() as ctx:
        ctx.prec = 65
        total = Decimal(0)
        for i in range(len(p.a)):
            total += D(p.a[i])*sum((-D(p.M[i,j])/D(p.epsilon)+z[j]).exp() for j in range(len(z))).ln()
        return total-sum(D(p.b[j])*z[j] for j in range(len(z)))


def test_stable_difference_against_high_precision():
    p = small_problem()
    z = np.array([.25, -.5, .25])
    s = evaluate_reduced(p, z)
    work = ReducedWork()
    for scale in (1., .01, 1e-10, 1e-15):
        u = np.array([.5, -.25, -.25])*scale
        value = stable_objective_difference(p, s, u, work)
        with localcontext() as ctx:
            ctx.prec = 65
            zd = [Decimal(str(x)) for x in z]
            ud = [Decimal(str(x)) for x in u]
            exact = float(decimal_f(p, [x+y for x,y in zip(zd,ud)])-decimal_f(p, zd))
        assert value == pytest.approx(exact, rel=2e-12, abs=1e-30)
    assert work.objective_diff_count == work.bregman_count == 4
    assert work.dense_pass_proxy >= 36
    assert work.probability_mass_error_max < 1e-13


def test_small_positive_bregman_not_lost():
    p = OTProblem(np.array([1.]), np.array([.5, .5]), np.zeros((1,2)), 1.)
    s = evaluate_reduced(p, [0., 0.])
    value = bregman_phi(p, s, [1e-12, -1e-12])
    assert value == pytest.approx(.5e-24, rel=1e-14, abs=0)
