import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull, Delaunay


# ============================================================
# Original sensor data
# ============================================================

X = np.array([
    [0.0, 0.5],
    [1.3, 1.1],
    [1.9, 0.1],
    [2.5, 2.3],
    [0.7, 1.8]
], dtype=float)

p = np.array([
    2.0,
    17.0,
    43.0,
    28.0,
    36.0
])


# ============================================================
# 1. Convex hull of all five sensors
# ============================================================

hull = ConvexHull(X)

print("Convex-hull sensor indices:")
print(hull.vertices + 1)  # +1 so sensor numbering starts at S1


# ============================================================
# 2. LOOCV geometry check
#
# For each sensor:
# remove it, construct the convex hull of the remaining four,
# then test whether the omitted sensor lies inside that hull.
# ============================================================

print("\nLOOCV Geometry Check")
print("------------------------------------------------")
print("Sensor    Inside remaining convex hull?")
print("------------------------------------------------")


for i in range(len(X)):

    mask = np.ones(len(X), dtype=bool)
    mask[i] = False

    X_train = X[mask]
    X_test = X[i]

    # Delaunay triangulation can test whether a point lies
    # inside the convex hull of the training points.
    triangulation = Delaunay(X_train)

    inside = triangulation.find_simplex(X_test) >= 0

    if inside:
        interpretation = "YES  -> interpolation"
    else:
        interpretation = "NO   -> extrapolation"

    print(
        f"S{i+1:<8}{interpretation}"
    )


# ============================================================
# 3. Plot all sensors and their convex hull
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    X[:, 0],
    X[:, 1],
    s=80
)

# Sensor labels and pressure values
for i in range(len(X)):
    plt.text(
        X[i, 0] + 0.03,
        X[i, 1] + 0.03,
        f"S{i+1} (p={p[i]:.0f})"
    )

# Draw convex hull
for simplex in hull.simplices:

    plt.plot(
        X[simplex, 0],
        X[simplex, 1],
        "k-"
    )

plt.xlabel("x")
plt.ylabel("y")
plt.title("Pressure Sensor Locations and Convex Hull")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()