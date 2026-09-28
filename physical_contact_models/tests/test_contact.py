"""Tests of the own numerical methods (on the test function p = 40 - 2(x-1)^2 - 3(y-2)^2) and of the
contact models (known answers)."""
import numpy as np

import data
from models import Hertz, Ring, distinct_exact, fit, ring_hull_max, select_largest_sigma
from numerics.nlls import levenberg_marquardt, numerical_jacobian
from numerics.optimize import newton_nd, parabolic_max, project_to_polygon, projected_steepest_ascent
from numerics.rootfind import bisection, newton_raphson, scan_brackets


def pt(x, y):
    return 40 - 2 * (x - 1) ** 2 - 3 * (y - 2) ** 2


ROOTS = (2 - np.sqrt(40 / 3), 2 + np.sqrt(40 / 3))   # p_t(1, y) = 0


def test_bisection_and_newton_on_the_test_function():
    f = lambda y: pt(1.0, y)  # noqa: E731
    df = lambda y: -6 * (y - 2)  # noqa: E731
    brackets = scan_brackets(f, -3, 7, 100)
    assert len(brackets) == 2
    for (a, b), exact in zip(brackets, ROOTS):
        r, k = bisection(f, a, b)
        assert abs(r - exact) < 1e-11 and k < 60
        rn, kn, ok = newton_raphson(f, df, b)
        assert ok and abs(rn - exact) < 1e-13 and kn < 10


def test_parabolic_interpolation_is_exact_for_a_quadratic():
    x, fx, it = parabolic_max(lambda x: pt(x, 2.0), -1.0, 4.0)
    assert abs(x - 1) < 1e-12 and abs(fx - 40) < 1e-12 and it <= 2
    x, fx, _ = parabolic_max(lambda x: pt(x, 2.0), 2.0, 4.0)     # maximum at the end point
    assert x == 2.0 and fx == 38.0


def test_newton_nd_finds_and_classifies_the_maximum():
    grad = lambda v: np.array([-4 * (v[0] - 1), -6 * (v[1] - 2)])  # noqa: E731
    hess = lambda v: np.diag([-4.0, -6.0])  # noqa: E731
    for st in [(0, 0), (2, 3), (-2, 4), (4, -1)]:
        r = newton_nd(grad, hess, st)
        np.testing.assert_allclose(r["x"], [1, 2], atol=1e-12)
        assert r["kind"] == "maximum" and r["iterations"] <= 2


def test_projected_steepest_ascent_unconstrained_and_on_the_hull():
    grad = lambda v: np.array([-4 * (v[0] - 1), -6 * (v[1] - 2)])  # noqa: E731
    r = projected_steepest_ascent(lambda v: pt(*v), grad, lambda v: v, np.array([0.0, 0.0]), alpha0=0.1)
    np.testing.assert_allclose(r["x"], [1, 2], atol=1e-8)
    # constrained: the unconstrained maximizer (1, 2) lies outside the hull, so the result must be the
    # hull point where p_t is largest, which is on the boundary (checked against a dense boundary scan)
    hull = data.hull_polygon()
    assert not data.in_hull(1.0, 2.0)
    r = projected_steepest_ascent(lambda v: pt(*v), grad, lambda v: project_to_polygon(v, hull), np.array([1.3, 1.1]))
    assert data.in_hull(*r["x"])
    t = np.linspace(0, 1, 2001)[:, None]
    border = np.vstack([a + t * (b - a) for a, b in zip(hull[:-1], hull[1:])])
    assert abs(r["f"] - pt(border[:, 0], border[:, 1]).max()) < 1e-6


def test_projection_onto_the_hull():
    hull = data.hull_polygon()
    np.testing.assert_allclose(project_to_polygon(np.array([1.3, 1.1]), hull), [1.3, 1.1])   # inside: unchanged
    q = project_to_polygon(np.array([0.0, 2.5]), hull)
    assert data.in_hull(*q) and data.dist_to_hull(*q) < 1e-9
    assert np.isclose(np.hypot(q[0] - 0.0, q[1] - 2.5), data.dist_to_hull(0.0, 2.5))


def test_levenberg_marquardt_on_a_known_nonlinear_problem():
    x = np.linspace(0, 2, 12)
    y = 3.0 * np.exp(-0.7 * x) + 0.5
    res = lambda t: t[0] * np.exp(-t[1] * x) + t[2] - y  # noqa: E731
    r = levenberg_marquardt(res, [1.0, 0.1, 0.0], [-10, -10, -10], [10, 10, 10])
    np.testing.assert_allclose(r["theta"], [3.0, 0.7, 0.5], atol=1e-8)


def test_levenberg_marquardt_respects_bounds_with_active_set():
    # unconstrained optimum a = 5 lies outside [0, 2]: LM must stop on the bound
    r = levenberg_marquardt(lambda t: np.array([t[0] - 5.0, t[1] - 1.0]), [1.0, 0.0], [0, -5], [2, 5])
    np.testing.assert_allclose(r["theta"], [2.0, 1.0], atol=1e-10)


def test_analytic_jacobians_match_finite_differences():
    th = np.array([40.0, 1.2, 1.1, 2.0, 1.6])
    Jh = Hertz.jac(th, data.X, data.Y)[:5]
    Jn = numerical_jacobian(lambda t: Hertz.residual(t, data.X, data.Y, data.P), th)[:5]
    np.testing.assert_allclose(Jh, Jn, atol=1e-5)
    tr = np.array([50.0, 1.2, 1.1, 1.0, 0.5])
    np.testing.assert_allclose(Ring.jac(tr, data.X, data.Y),
                               numerical_jacobian(lambda t: Ring.residual(t, data.X, data.Y, data.P), tr), atol=1e-5)


def test_hertz_recovers_known_parameters():
    true = np.array([40.0, 1.2, 1.1, 2.0, 1.6])
    runs, _ = fit(Hertz, data.X, data.Y, Hertz.f(true, data.X, data.Y), n_starts=60)
    assert min(np.abs(r["theta"] - true).max() for r in runs if r["cost"] < 1e-20) < 1e-6


def test_ring_truth_is_one_of_the_exact_fits_and_rule_picks_largest_sigma():
    true = np.array([50.0, 1.2, 1.1, 1.0, 0.5])
    p = Ring.f(true, data.X, data.Y)
    runs, _ = fit(Ring, data.X, data.Y, p, n_starts=400)
    sols, _ = distinct_exact(Ring, runs, data.X, data.Y, p)
    assert len(sols) >= 2 and min(np.abs(t - true).max() for t in sols) < 1e-6
    assert select_largest_sigma(sols)[4] == max(t[4] for t in sols)


def test_ring_hull_max_on_crest_circle():
    t = np.array([50.0, 1.2, 1.1, 0.5, 0.3])   # crest circle of radius 0.5 around S2 lies inside the hull
    mx, x, y, where = ring_hull_max(t)
    assert np.isclose(mx, 50.0) and "crest" in where and np.isclose(np.hypot(x - 1.2, y - 1.1), 0.5)


def test_s2_is_inside_the_hull_of_the_other_four():
    # premise of the S2 hold-out test: predicting S2 from the other sensors is interpolation
    assert data.s2_inside_hull_of_others()


def test_own_lm_and_scipy_reach_the_same_hertz_optimum():
    from run_contact import scipy_hertz_multistart
    from models import sse
    _, s = fit(Hertz, data.X, data.Y, data.P, n_starts=100, max_iter=3000)
    ref, _, on_bound = scipy_hertz_multistart()
    assert abs(sse(Hertz, s["theta"]) - sse(Hertz, ref)) < 1e-6
    assert "y0" in on_bound   # the optimum sits on the y0 bound: the ellipse wants to grow without limit
