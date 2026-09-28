"""Sensor data and convex-hull geometry.

Coordinates are (x, y) positions of the five pressure sensors on the robot hand and p is the
measured pressure.  Everything that needs to know "is this point supported by data?" goes through
``in_hull``, so the definition lives in exactly one place.
"""
import numpy as np
from scipy.spatial import ConvexHull

NAMES = ["S1", "S2", "S3", "S4", "S5"]
X = np.array([0.0, 1.3, 1.9, 2.5, 0.7])
Y = np.array([0.5, 1.1, 0.1, 2.3, 1.8])
P = np.array([2.0, 17.0, 43.0, 28.0, 36.0])
POINTS = np.column_stack([X, Y])

REGION = (0.0, 2.5, 0.0, 2.5)  # region of interest [0, 2.5]^2 (plotting grids)
P_ZERO = 0.0                   # p <= 0 means damage
P_LIMIT = 50.0                 # pain threshold
S2_INDEX = 1                   # the only sensor inside the convex hull of the other four


def _hull_equations(points):
    """Facet equations n.x + c <= 0 describing the convex hull of ``points``."""
    return ConvexHull(np.asarray(points, dtype=float)).equations


def in_hull(x, y, points=POINTS, tol=1e-9):
    """Boolean mask: is (x, y) inside (or on the boundary of) the convex hull of ``points``?

    q is inside iff n_k . q + c_k <= 0 for every facet k (scipy returns outward normals n_k);
    ``tol`` keeps the hull vertices themselves "inside" despite round-off.
    """
    x = np.asarray(x, dtype=float)
    q = np.column_stack([x.ravel(), np.asarray(y, dtype=float).ravel()])
    eq = _hull_equations(points)
    return np.all(q @ eq[:, :2].T + eq[:, 2] <= tol, axis=1).reshape(x.shape)


def hull_polygon(points=POINTS):
    """Closed polygon (N+1, 2) of the hull vertices in counter-clockwise order."""
    pts = np.asarray(points, dtype=float)
    poly = pts[ConvexHull(pts).vertices]  # counter-clockwise in 2-D
    return np.vstack([poly, poly[:1]])


def hull_bbox(points=POINTS):
    pts = np.asarray(points, dtype=float)
    return pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()


def region_grid(n, region=REGION):
    """n x n meshgrid over the region of interest."""
    x0, x1, y0, y1 = region
    return np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))


def hull_grid_points(n, points=POINTS):
    """Flat arrays (xs, ys) of an n x n bounding-box grid restricted to the hull."""
    x0, x1, y0, y1 = hull_bbox(points)
    gx, gy = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
    mask = in_hull(gx, gy, points)
    return gx[mask], gy[mask]


def dist_to_hull(x, y):
    """Euclidean distance of (x, y) to the convex hull of the sensors (0 inside)."""
    x = np.asarray(x, dtype=float)
    q = np.column_stack([x.ravel(), np.asarray(y, dtype=float).ravel()])
    poly = hull_polygon()
    a, b = poly[:-1], poly[1:]
    ab = b - a
    t = np.clip(((q[:, None, :] - a[None]) * ab[None]).sum(-1) / (ab ** 2).sum(-1)[None], 0.0, 1.0)
    nearest = a[None] + t[..., None] * ab[None]
    d = np.sqrt(((q[:, None, :] - nearest) ** 2).sum(-1)).min(axis=1)
    d[in_hull(q[:, 0], q[:, 1])] = 0.0
    return d.reshape(x.shape)


def s2_inside_hull_of_others():
    """True if S2 lies inside the convex hull of S1, S3, S4, S5 (premise of the S2 hold-out test)."""
    others = np.delete(POINTS, S2_INDEX, axis=0)
    return bool(in_hull(X[S2_INDEX], Y[S2_INDEX], points=others))
