"""Contact models and the tools to fit them.

    Hertz  p = p0 sqrt(max(0, 1 - ((x-x0)/a)^2 - ((y-y0)/b)^2))          (single elliptical contact)
    Ring   p = A exp(-(r - R)^2 / (2 sigma^2)),  r = |(x, y) - (x0, y0)|   (ring-shaped contact)

Fitting: multi-start Levenberg-Marquardt (numerics/nlls.py) with analytic Jacobians.
Selection rule for the ring (fixed before looking at (b)/(c)): among the exact fits take the one
with the largest sigma.  Hull geometry helpers find the maximum / minimum over the sensor hull.
"""
import warnings

import numpy as np

import data
from numerics.nlls import multistart_lm
from numerics.optimize import parabolic_max, project_to_polygon, projected_steepest_ascent

SEED = 2024
X, Y, PR = data.X, data.Y, data.P
S2 = data.S2_INDEX
HULL = data.hull_polygon()                          # closed, counter-clockwise
EXACT_SSE = 1e-10                                   # "passes through the data"


def vertex_name(v):
    return data.NAMES[int(np.argmin(np.hypot(X - v[0], Y - v[1])))]


EDGES = [(vertex_name(a), vertex_name(b), a, b) for a, b in zip(HULL[:-1], HULL[1:])]


# =============================================================================== models
class Hertz:
    """p = p0 sqrt(max(0, 1 - q)),  q = ((x-x0)/a)^2 + ((y-y0)/b)^2;  theta = (p0, x0, y0, a, b)."""
    names = ("p0", "x0", "y0", "a", "b")
    lower = np.array([1e-6, -5.0, -5.0, 1e-3, 1e-3])
    upper = np.array([20 * 43.0, 7.5, 7.5, 50.0, 50.0])
    W, DELTA = 1e3, 1e-6   # penalty weight / margin for "every sensor inside the ellipse"

    @staticmethod
    def q(t, x, y):
        return ((x - t[1]) / t[3]) ** 2 + ((y - t[2]) / t[4]) ** 2

    @classmethod
    def f(cls, t, x, y):
        return t[0] * np.sqrt(np.maximum(0.0, 1.0 - cls.q(t, x, y)))

    @classmethod
    def residual(cls, t, x, y, p):
        qq = cls.q(t, x, y)
        return np.concatenate([t[0] * np.sqrt(np.maximum(0, 1 - qq)) - p,
                               cls.W * np.maximum(0, qq - (1 - cls.DELTA))])

    @classmethod
    def jac(cls, t, x, y):
        """Analytic Jacobian of ``residual`` (data rows, then penalty rows).

        A sensor outside the ellipse has p = 0 whatever theta is, so its data row is zero there."""
        p0, x0, y0, a, b = t
        qq = cls.q(t, x, y)
        s = np.sqrt(np.maximum(1e-12, 1 - qq))
        Jd = np.column_stack([s, p0 * (x - x0) / (a * a * s), p0 * (y - y0) / (b * b * s),
                              p0 * (x - x0) ** 2 / (a ** 3 * s), p0 * (y - y0) ** 2 / (b ** 3 * s)])
        Jd = Jd * (qq < 1)[:, None]
        dq = np.column_stack([np.zeros_like(x), -2 * (x - x0) / a ** 2, -2 * (y - y0) / b ** 2,
                              -2 * (x - x0) ** 2 / a ** 3, -2 * (y - y0) ** 2 / b ** 3])
        return np.vstack([Jd, cls.W * dq * ((qq - (1 - cls.DELTA)) > 0)[:, None]])

    @staticmethod
    def starts(rng, n, x, y, p):
        pm = float(np.max(p))
        return np.column_stack([rng.uniform(pm, 3 * pm, n), rng.uniform(x.min() - 1, x.max() + 1, n),
                                rng.uniform(y.min() - 1, y.max() + 1, n), rng.uniform(1, 6, n), rng.uniform(1, 6, n)])


class Ring:
    """p = A exp(-(r - R)^2 / (2 s^2)),  r = sqrt((x-x0)^2 + (y-y0)^2);  theta = (A, x0, y0, R, s)."""
    names = ("A", "x0", "y0", "R", "sigma")
    lower = np.array([0.0, -5.0, -5.0, 0.0, 0.05])
    upper = np.array([2000.0, 7.5, 7.5, 20.0, 20.0])

    @staticmethod
    def f(t, x, y):
        A, x0, y0, R, s = t
        return A * np.exp(-(np.hypot(x - x0, y - y0) - R) ** 2 / (2 * s * s))

    @classmethod
    def residual(cls, t, x, y, p):
        return cls.f(t, x, y) - p

    @staticmethod
    def jac(t, x, y):
        A, x0, y0, R, s = t
        r = np.maximum(np.hypot(x - x0, y - y0), 1e-12)
        d = r - R
        e = np.exp(-d * d / (2 * s * s))
        return np.column_stack([e, A * e * d * (x - x0) / (s * s * r), A * e * d * (y - y0) / (s * s * r),
                                A * e * d / (s * s), A * e * d * d / s ** 3])

    @classmethod
    def grad_xy(cls, t, v):
        A, x0, y0, R, s = t
        r = max(np.hypot(v[0] - x0, v[1] - y0), 1e-12)
        d = r - R
        dp_dr = -A * np.exp(-d * d / (2 * s * s)) * d / (s * s)
        return dp_dr * np.array([(v[0] - x0) / r, (v[1] - y0) / r])

    @staticmethod
    def starts(rng, n, x, y, p):
        return np.column_stack([rng.uniform(20, 150, n), rng.uniform(x.min() - 1, x.max() + 1, n),
                                rng.uniform(y.min() - 1, y.max() + 1, n), rng.uniform(0.1, 3, n),
                                rng.uniform(0.1, 2, n)])


def fit(model, x, y, p, n_starts, seed=SEED, warm=None, lower=None, upper=None, max_iter=1500):
    lo = model.lower if lower is None else lower
    hi = model.upper if upper is None else upper
    starts = model.starts(np.random.default_rng(seed), n_starts, x, y, p) if n_starts else np.zeros((0, 5))
    if warm is not None:
        starts = np.vstack([np.atleast_2d(warm), starts])
    return multistart_lm(lambda t: model.residual(t, x, y, p), starts, lo, hi,
                         jac=lambda t: model.jac(t, x, y), max_iter=max_iter)


def sse(model, t, x=X, y=Y, p=PR):
    return float(np.sum((model.f(t, x, y) - p) ** 2))


def distinct_exact(model, runs, x, y, p):
    """Distinct parameter vectors among the runs that pass through the data; with counts."""
    sols, counts = [], []
    for r in runs:
        if sse(model, r["theta"], x, y, p) > EXACT_SSE:
            continue
        for k, u in enumerate(sols):
            if np.allclose(r["theta"], u, rtol=1e-4, atol=1e-4):
                counts[k] += 1
                break
        else:
            sols.append(r["theta"])
            counts.append(1)
    return sols, counts


def select_largest_sigma(sols, x=None, y=None, p=None, n_refine=5, upper=None):
    """Selection rule fixed before looking at (b)/(c): among the exact fits take the broadest ring.

    With 5 points the exact fits are isolated, so the largest sampled sigma is the answer.  With 4
    points they form a continuous family and sampling only approximates the largest sigma; then
    (x, y, p given) the best candidates are refined by solving
        maximize sigma  subject to  f(theta; x_i, y_i) = p_i  for every training point
    with SLSQP (equality constraints), within the bounds Ring.lower .. ``upper`` (default Ring.upper).
    If the result sits on an artificial cap (see ``finite_rule_answer``), sigma would keep growing
    with a larger cap: the rule then has no finite answer for that data set.
    """
    if not sols:
        return None
    if x is None:
        return max(sols, key=lambda t: t[4])
    from scipy.optimize import minimize
    hi = Ring.upper if upper is None else np.asarray(upper, float)
    cons = {"type": "eq", "fun": lambda t: Ring.f(t, x, y) - p, "jac": lambda t: Ring.jac(t, x, y)}
    best = max(sols, key=lambda t: t[4])
    for t0 in sorted(sols, key=lambda t: -t[4])[:n_refine]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = minimize(lambda t: -t[4], t0, jac=lambda t: np.array([0, 0, 0, 0, -1.0]), method="SLSQP",
                         constraints=[cons], bounds=list(zip(Ring.lower, hi)),
                         options={"maxiter": 1000, "ftol": 1e-14})
        if sse(Ring, r.x, x, y, p) < 1e-8 and r.x[4] > best[4]:
            best = r.x
    return best


# R = 0 turns the ring into a single Gaussian bump and A = 0 is zero pressure: edges of the model
# family.  Every other bound is an arbitrary cap of the search box.
NATURAL_BOUNDS = {("A", "lower"), ("R", "lower")}


def bounds_hit(t, upper=None, rel=1e-6):
    """Ring parameters of t that sit on a bound, e.g. ["A upper", "R lower"]."""
    hi = Ring.upper if upper is None else np.asarray(upper, float)
    out = []
    for n, v, lo, up in zip(Ring.names, t, Ring.lower, hi):
        w = rel * (up - lo)
        if v - lo <= w:
            out.append(f"{n} lower")
        elif up - v <= w:
            out.append(f"{n} upper")
    return out


def finite_rule_answer(t, upper=None):
    """False if the largest-sigma fit sits on an artificial cap, i.e. the rule has no finite answer."""
    return all(tuple(b.split()) in NATURAL_BOUNDS for b in bounds_hit(t, upper))


# =============================================================================== geometry / extrema
def hull_y_interval(x):
    ys = []
    for _, _, a, b in EDGES:
        if min(a[0], b[0]) - 1e-12 <= x <= max(a[0], b[0]) + 1e-12 and a[0] != b[0]:
            ys.append(a[1] + (x - a[0]) / (b[0] - a[0]) * (b[1] - a[1]))
    return (min(ys), max(ys)) if len(ys) >= 2 else None


EDGE_SCAN = 40  # coarse samples per edge before the parabolic refinement


def edge_max(fun, parabola_on=None, scan=EDGE_SCAN):
    """Maximum along every hull edge: coarse scan, then successive parabolic interpolation on the
    bracket [t_(k-1), t_(k+1)] around the best sample (end points included).

    A plain three-point parabola over the whole edge is only reliable for a unimodal edge function
    (always true for a quadratic surface).  Along an edge that the ring crest crosses twice the
    pressure has two peaks, so the scan is needed to pick the right bracket first.
    """
    rows = []
    for na, nb, a, b in EDGES:
        pt = lambda t: a + t * (b - a)  # noqa: E731
        target = (lambda t: parabola_on(*pt(t))) if parabola_on else (lambda t: fun(*pt(t)))
        ts = np.linspace(0.0, 1.0, scan + 1)
        k = int(np.argmax([target(t) for t in ts]))
        lo, hi = ts[max(k - 1, 0)], ts[min(k + 1, scan)]
        tb, _, it = parabolic_max(target, lo, hi)
        xb = pt(tb)
        rows.append({"edge": f"{na}-{nb}", "t": tb, "x": xb[0], "y": xb[1], "p_max": float(fun(*xb)),
                     "scan_points": scan + 1, "parabolic_iterations": it})
    return rows


def ring_hull_max(t):
    """Max of a ring surface over the hull: the crest value A if the crest circle r = R crosses the
    hull interior, otherwise the best edge maximum (parabolic interpolation along the edges)."""
    A, x0, y0, R, s = t
    ang = np.linspace(0, 2 * np.pi, 3600, endpoint=False)
    cx, cy = x0 + R * np.cos(ang), y0 + R * np.sin(ang)
    inside = data.in_hull(cx, cy)
    if inside.any():
        k = np.flatnonzero(inside)[0]
        return A, cx[k], cy[k], "crest circle r = R crosses the hull"
    fun = lambda x, y: Ring.f(t, np.atleast_1d(x), np.atleast_1d(y))[0]  # noqa: E731
    e = max(edge_max(fun), key=lambda r: r["p_max"])
    return e["p_max"], e["x"], e["y"], f"edge {e['edge']}"


def ring_hull_min(t):
    """Min over the hull: grid search refined by projected steepest DESCENT (ascent on -p)."""
    gx, gy = data.hull_grid_points(121)
    z = Ring.f(t, gx, gy)
    k = int(np.argmin(z))
    r = projected_steepest_ascent(lambda v: -Ring.f(t, v[0], v[1]), lambda v: -Ring.grad_xy(t, v),
                                  lambda v: project_to_polygon(v, HULL), np.array([gx[k], gy[k]]), alpha0=0.1)
    return -r["f"], r["x"][0], r["x"][1]


