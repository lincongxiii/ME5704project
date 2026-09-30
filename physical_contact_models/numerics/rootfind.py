"""One-dimensional root finding (course methods): bracket scan, bisection, Newton-Raphson.

bisection(f, a, b)
    Requires f(a) f(b) < 0.  Repeats  m = (a + b) / 2  and keeps the half-interval whose end
    values still differ in sign.  The bracket halves every step, so after k steps the error is at
    most (b - a) / 2^k: slow (linear) but guaranteed for a continuous f.

newton_raphson(f, df, x0)
    x_{k+1} = x_k - f(x_k) / f'(x_k).  Quadratic convergence near a simple root when the start is
    close and f' is not small; no guarantee otherwise (it may diverge or jump to another root).

scan_brackets(f, a, b, n)
    Evaluates f on n + 1 equally spaced points and returns the sub-intervals with a sign change;
    each one contains at least one root (intermediate value theorem).  A root pair closer together
    than the scan step can be missed, which is why a zero-free conclusion is always backed up by
    a minimum search (see the chapters).
"""
import numpy as np


def scan_brackets(f, a, b, n=400):
    xs = np.linspace(a, b, n + 1)
    fs = np.array([f(x) for x in xs])
    out = []
    for i in range(n):
        if fs[i] == 0.0:
            out.append((xs[i], xs[i]))
        elif fs[i] * fs[i + 1] < 0:
            out.append((xs[i], xs[i + 1]))
    if fs[-1] == 0.0:
        out.append((xs[-1], xs[-1]))
    return out


def bisection(f, a, b, tol=1e-12, max_iter=200):
    """Root of f in [a, b]; returns (root, iterations)."""
    fa, fb = f(a), f(b)
    if fa == 0:
        return a, 0
    if fb == 0:
        return b, 0
    if fa * fb > 0:
        raise ValueError("bisection needs a sign change on [a, b]")
    for k in range(1, max_iter + 1):
        m = 0.5 * (a + b)
        fm = f(m)
        if fm == 0 or 0.5 * (b - a) < tol:
            return m, k
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b), max_iter


def newton_raphson(f, df, x0, tol=1e-12, max_iter=50):
    """Returns (root, iterations, converged).  Stops when |step| < tol or f' vanishes."""
    x = float(x0)
    for k in range(1, max_iter + 1):
        d = df(x)
        if not np.isfinite(d) or d == 0:
            return x, k, False
        step = f(x) / d
        x -= step
        if not np.isfinite(x):
            return x, k, False
        if abs(step) < tol:
            return x, k, True
    return x, max_iter, False
