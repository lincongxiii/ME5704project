import numpy as np
from scipy.spatial import ConvexHull
from matplotlib.path import Path


# ============================================================
# Sensor data
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
# Full quadratic design matrix
# ============================================================

def quadratic_matrix(X):

    x = X[:, 0]
    y = X[:, 1]

    return np.column_stack([
        np.ones(len(X)),
        x,
        y,
        x**2,
        x*y,
        y**2
    ])


# ============================================================
# Minimum-norm quadratic fit
# ============================================================

def minimum_norm_fit(X, p):

    Z = quadratic_matrix(X)

    coefficients, _, _, _ = np.linalg.lstsq(
        Z,
        p,
        rcond=None
    )

    return coefficients


def predict(X, coefficients):

    return (
        quadratic_matrix(X)
        @ coefficients
    )


# ============================================================
# PART A
# S2 withheld validation
# ============================================================

test_index = 1

mask = np.ones(len(p), dtype=bool)
mask[test_index] = False

X_train = X[mask]
p_train = p[mask]

X_test = X[test_index].reshape(1, -1)

actual_S2 = p[test_index]


# ------------------------------------------------------------
# A1. Original-coordinate model
# ------------------------------------------------------------

coef_original = minimum_norm_fit(
    X_train,
    p_train
)

pred_original = predict(
    X_test,
    coef_original
)[0]


# ------------------------------------------------------------
# A2. Standardised-coordinate model
#
# IMPORTANT:
# Scaling parameters are calculated ONLY from the
# four training sensors to avoid data leakage.
# ------------------------------------------------------------

mean_train = np.mean(
    X_train,
    axis=0
)

std_train = np.std(
    X_train,
    axis=0
)

X_train_scaled = (
    (X_train - mean_train)
    / std_train
)

X_test_scaled = (
    (X_test - mean_train)
    / std_train
)


coef_scaled_validation = minimum_norm_fit(
    X_train_scaled,
    p_train
)

pred_scaled = predict(
    X_test_scaled,
    coef_scaled_validation
)[0]


print("S2 WITHHELD VALIDATION")
print("=" * 65)

print(
    f"Actual S2 pressure               = "
    f"{actual_S2:.6f}"
)

print(
    f"Original-coordinate prediction  = "
    f"{pred_original:.6f}"
)

print(
    f"Original absolute error          = "
    f"{abs(pred_original-actual_S2):.6f}"
)

print()

print(
    f"Standardised-coordinate prediction = "
    f"{pred_scaled:.6f}"
)

print(
    f"Standardised absolute error         = "
    f"{abs(pred_scaled-actual_S2):.6f}"
)


# ============================================================
# PART B
# Fit all five measurements
# ============================================================

# ------------------------------------------------------------
# B1. Original coordinates
# ------------------------------------------------------------

coef_full_original = minimum_norm_fit(
    X,
    p
)


# ------------------------------------------------------------
# B2. Standardised coordinates
# ------------------------------------------------------------

mean_full = np.mean(
    X,
    axis=0
)

std_full = np.std(
    X,
    axis=0
)

X_scaled = (
    (X - mean_full)
    / std_full
)

coef_full_scaled = minimum_norm_fit(
    X_scaled,
    p
)


# Verify sensor fitting

pred_full_original = predict(
    X,
    coef_full_original
)

pred_full_scaled = predict(
    X_scaled,
    coef_full_scaled
)


rmse_original = np.sqrt(
    np.mean(
        (p - pred_full_original)**2
    )
)

rmse_scaled = np.sqrt(
    np.mean(
        (p - pred_full_scaled)**2
    )
)


print("\nFULL-DATA FIT")
print("=" * 65)

print(
    f"Original-coordinate RMSE     = "
    f"{rmse_original:.10f}"
)

print(
    f"Standardised-coordinate RMSE = "
    f"{rmse_scaled:.10f}"
)


# ============================================================
# PART C
# Compare both surfaces inside the sensor convex hull
# ============================================================

x_grid = np.linspace(
    X[:, 0].min(),
    X[:, 0].max(),
    500
)

y_grid = np.linspace(
    X[:, 1].min(),
    X[:, 1].max(),
    500
)

XX, YY = np.meshgrid(
    x_grid,
    y_grid
)

grid_points = np.column_stack([
    XX.ravel(),
    YY.ravel()
])


# ------------------------------------------------------------
# Convex-hull mask
# ------------------------------------------------------------

hull = ConvexHull(X)

hull_points = X[
    hull.vertices
]

hull_path = Path(
    hull_points
)

inside = hull_path.contains_points(
    grid_points,
    radius=1e-10
)


# ------------------------------------------------------------
# Original-coordinate predictions
# ------------------------------------------------------------

P_original = predict(
    grid_points,
    coef_full_original
)


# ------------------------------------------------------------
# Standardised-coordinate predictions
# ------------------------------------------------------------

grid_scaled = (
    (grid_points - mean_full)
    / std_full
)

P_scaled = predict(
    grid_scaled,
    coef_full_scaled
)


# Only points inside convex hull

P_original_inside = P_original[
    inside
]

P_scaled_inside = P_scaled[
    inside
]


print("\nPRESSURE RANGE INSIDE SENSOR CONVEX HULL")
print("=" * 65)

print("Original-coordinate minimum:")
print(
    f"  {np.min(P_original_inside):.6f}"
)

print("Original-coordinate maximum:")
print(
    f"  {np.max(P_original_inside):.6f}"
)

print()

print("Standardised-coordinate minimum:")
print(
    f"  {np.min(P_scaled_inside):.6f}"
)

print("Standardised-coordinate maximum:")
print(
    f"  {np.max(P_scaled_inside):.6f}"
)


# ============================================================
# PART D
# Compare the two predicted surfaces directly
# ============================================================

surface_difference = (
    P_scaled_inside
    - P_original_inside
)

rmse_between_surfaces = np.sqrt(
    np.mean(
        surface_difference**2
    )
)

max_abs_difference = np.max(
    np.abs(
        surface_difference
    )
)


print("\nSURFACE SENSITIVITY TO COORDINATE SCALING")
print("=" * 65)

print(
    f"RMSE between surfaces = "
    f"{rmse_between_surfaces:.6f}"
)

print(
    f"Maximum absolute difference = "
    f"{max_abs_difference:.6f}"
)


# ============================================================
# PART E
# Engineering conclusions
# ============================================================

original_min = np.min(
    P_original_inside
)

original_max = np.max(
    P_original_inside
)

scaled_min = np.min(
    P_scaled_inside
)

scaled_max = np.max(
    P_scaled_inside
)


print("\nENGINEERING CONCLUSION SENSITIVITY")
print("=" * 65)

print(
    "Original model:"
)

print(
    f"  Damage predicted (p <= 0)? "
    f"{original_min <= 0}"
)

print(
    f"  Pain predicted (p > 50)? "
    f"{original_max > 50}"
)


print(
    "\nStandardised model:"
)

print(
    f"  Damage predicted (p <= 0)? "
    f"{scaled_min <= 0}"
)

print(
    f"  Pain predicted (p > 50)? "
    f"{scaled_max > 50}"
)