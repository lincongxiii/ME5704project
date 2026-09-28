"""Optimization methods used in the chapters (course methods, own implementations).

parabolic_max(f, a, b)
    Maximum of a 1-D function on [a, b] by successive parabolic interpolation: fit a parabola
    through three points, jump to its vertex
        x* = x2 - 0.5 * [(x2-x1)^2 (f2-f3) - (x2-x3)^2 (f2-f1)] / [(x2-x1)(f2-f3) - (x2-x3)(f2-f1)],
    replace the worst point, repeat.  For a quadratic f the first vertex is exact.  The end points
    are always compared too, because a constrained maximum can sit on the boundary.

newton_nd(grad, hess, x0)
    x_{k+1} = x_k - H^{-1} grad.  Finds a STATIONARY point (maximum, minimum or saddle); the
    eigenvalues of the Hessian at the result classify it.  It ignores constraints.

projected_steepest_ascent(f, grad, project, x0)
    x_trial = P(x_k + alpha grad f(x_k)), P = projection onto the feasible set.  Backtracking:
    alpha is halved until f increases; the step size of the next iteration restarts at alpha0.
    A local method: the result depends on the starting point.

project_to_polygon(q, poly)
    Nearest point of a convex polygon (closed vertex array) to q; q itself if it is inside.
"""
import numpy as np


def parabolic_max(f, a, b, tol=1e-12, max_iter=100):
    """Returns (x, f(x), iterations) of the maximum of f on [a, b]."""
    x1, x2, x3 = a, 0.5 * (a + b), b
    f1, f2, f3 = f(x1), f(x2), f(x3)
    best = max(((x1, f1), (x2, f2), (x3, f3)), key=lambda t: t[1])
    it, stall = 0, 0
    for it in range(1, max_iter + 1):
        num = (x2 - x1) ** 2 * (f2 - f3) - (x2 - x3) ** 2 * (f2 - f1)
        den = (x2 - x1) * (f2 - f3) - (x2 - x3) * (f2 - f1)
        if den == 0:
            break
        xv = x2 - 0.5 * num / den
        if not (a <= xv <= b) or not np.isfinite(xv):
            break
        fv = f(xv)
        gain = fv - best[1]
        if fv > best[1]:
            best = (xv, fv)
        # a single non-improving parabola is normal; stop after three in a row, or when the gain is
        # negligible (converged vertex, or a monotone edge creeping towards its end point)
        stall = stall + 1 if gain <= 0 else 0
        if abs(xv - x2) < tol or stall >= 3 or 0 < gain < 1e-13 * max(1.0, abs(best[1])):
            break
        # keep the three best points (bracketing the vertex) for the next parabola
        pts = sorted([(x1, f1), (x2, f2), (x3, f3), (xv, fv)], key=lambda t: -t[1])[:3]
        pts.sort()
        (x1, f1), (x2, f2), (x3, f3) = pts
    for x in (a, b):  # constrained maximum may be an end point
        fx = f(x)
        if fx > best[1]:
            best = (x, fx)
    return best[0], best[1], it


def newton_nd(grad, hess, x0, tol=1e-12, max_iter=50):
    """Returns dict(x, iterations, converged, eig, kind)."""
    x = np.asarray(x0, dtype=float)
    it, conv = 0, False
    for it in range(1, max_iter + 1):
        step = np.linalg.solve(hess(x), grad(x))
        x = x - step
        if np.linalg.norm(step) < tol:
            conv = True
            break
    ev = np.linalg.eigvalsh(hess(x))
    zero = np.abs(ev) <= 1e-6 * max(1.0, np.abs(ev).max())
    if np.all(ev < 0) and not zero.any():
        kind = "maximum"
    elif np.all(ev > 0) and not zero.any():
        kind = "minimum"
    elif np.all((ev < 0) | zero) and not zero.all():
        kind = "degenerate maximum (semi-definite Hessian, non-isolated)"
    elif np.all((ev > 0) | zero) and not zero.all():
        kind = "degenerate minimum (semi-definite Hessian, non-isolated)"
    else:
        kind = "saddle" if (ev.min() < 0 < ev.max()) else "flat (all eigenvalues ~ 0)"
    return {"x": x, "iterations": it, "converged": conv, "eig": ev, "kind": kind}


def project_to_polygon(q, poly):
    """Nearest point of the convex polygon ``poly`` ((N+1, 2), closed, counter-clockwise) to q."""
    q = np.asarray(q, dtype=float)
    a, b = poly[:-1], poly[1:]
    ab = b - a
    cross = ab[:, 0] * (q[1] - a[:, 1]) - ab[:, 1] * (q[0] - a[:, 0])
    if np.all(cross >= -1e-12):  # left of every edge of a counter-clockwise polygon -> inside
        return q
    t = np.clip(((q - a) * ab).sum(1) / (ab ** 2).sum(1), 0.0, 1.0)
    cand = a + t[:, None] * ab
    return cand[np.argmin(((cand - q) ** 2).sum(1))]


def projected_steepest_ascent(f, grad, project, x0, alpha0=1.0, tol=1e-10, max_iter=1000, min_alpha=1e-14):
    """Returns dict(x, f, iterations, converged)."""
    x = project(np.asarray(x0, dtype=float))
    fx = f(x)
    it = 0
    for it in range(1, max_iter + 1):
        g = grad(x)
        alpha, moved = alpha0, False
        while alpha > min_alpha:
            xt = project(x + alpha * g)
            ft = f(xt)
            if ft > fx:
                moved = True
                break
            alpha *= 0.5
        if not moved or np.linalg.norm(xt - x) < tol:
            return {"x": x, "f": fx, "iterations": it, "converged": True}
        x, fx = xt, ft
    return {"x": x, "f": fx, "iterations": max_iter, "converged": False}
