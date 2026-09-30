"""Analytical references, not duplicated numerical solver implementations."""
import numpy as np
import pytest
from c.noise_robustness import fit_minimum_norm, predict, exact_hull_extrema, hull_vertices
from validation.tps import tps_fit
from numerics.rootfind import scan_brackets, bisection, newton_raphson
from validation.benchmarks import run_benchmarks


def test_all_known_answer_benchmarks():
    rows = run_benchmarks()
    assert len(rows) >= 15
    assert all(r['status'] == 'PASS' for r in rows), rows


def test_quadratic_training_is_underdetermined():
    from c.noise_robustness import X, quadratic_matrix
    z = quadratic_matrix((X - X.mean(0)) / X.std(0))
    assert z.shape == (5, 6) and np.linalg.matrix_rank(z) == 5


def test_quadratic_singular_hessian_boundary_extrema():
    # p=x^2: singular Hessian; extrema must still be covered by the boundary.
    lo, hi = exact_hull_extrema(np.array([0., 0, 0, 1, 0, 0]), np.zeros(2), np.ones(2))
    np.testing.assert_allclose([lo, hi], [0, 6.25], atol=1e-12)


def test_quadratic_constant_surface():
    lo, hi = exact_hull_extrema(np.array([7., 0, 0, 0, 0, 0]), np.zeros(2), np.ones(2))
    np.testing.assert_allclose([lo, hi], [7, 7], atol=1e-12)


def test_bisection_rejects_invalid_bracket():
    with pytest.raises(ValueError):
        bisection(lambda x: x*x + 1, -1, 1)


def test_scan_includes_both_endpoints_once():
    assert scan_brackets(lambda x: x*(x-1), 0, 1, 10) == [(0., 0.), (1., 1.)]


def test_newton_reports_zero_derivative_failure():
    _, _, ok = newton_raphson(lambda x: x*x-1, lambda x: 2*x, 0)
    assert not ok


def test_sign_scan_known_limitation_tangent_and_narrow_pair():
    # Passing this test DOCUMENTS incompleteness; it does not certify all roots.
    assert scan_brackets(lambda x: (x-.123)**2, 0, 1, 10) == []
    assert scan_brackets(lambda x: (x-.123)*(x-.127), 0, 1, 10) == []


def test_tps_rotation_and_uniform_scale_invariance():
    from c.noise_robustness import X, p_original
    angle = .37
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    q = np.array([[.4, .7], [1.4, 1.4]])
    transform = lambda x: 3.7*x@rot.T + [5, -2]
    a = tps_fit(X, p_original)[0](q)
    b = tps_fit(transform(X), p_original)[0](transform(q))
    np.testing.assert_allclose(a, b, atol=1e-10)
