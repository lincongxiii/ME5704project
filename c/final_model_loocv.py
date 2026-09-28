import numpy as np
from scipy.spatial import ConvexHull
from matplotlib.path import Path


# ============================================================
# ME5704 Group Project 1
#
# LOOCV of Standardised Minimum-Norm Full Quadratic
#
# Purpose:
# Separate interpolation and extrapolation performance.
# ============================================================


# ------------------------------------------------------------
# 1. Sensor data
# ------------------------------------------------------------

X = np.array([
    [0.0, 0.5],   # S1
    [1.3, 1.1],   # S2
    [1.9, 0.1],   # S3
    [2.5, 2.3],   # S4
    [0.7, 1.8]    # S5
], dtype=float)

p = np.array([
    2.0,
    17.0,
    43.0,
    28.0,
    36.0
])


# ============================================================
# 2. Quadratic design matrix
#
# p = b0 + b1*xs + b2*ys
#     + b3*xs^2 + b4*xs*ys + b5*ys^2
# ============================================================

def quadratic_matrix(X_scaled):

    xs = X_scaled[:, 0]
    ys = X_scaled[:, 1]

    return np.column_stack([
        np.ones(len(X_scaled)),
        xs,
        ys,
        xs**2,
        xs*ys,
        ys**2
    ])


# ============================================================
# 3. Fit standardised minimum-norm quadratic
#
# IMPORTANT:
# mean/std are calculated using TRAINING DATA ONLY.
# ============================================================

def fit_model(X_train, p_train):

    mean_xy = np.mean(
        X_train,
        axis=0
    )

    std_xy = np.std(
        X_train,
        axis=0
    )

    X_scaled = (
        (X_train - mean_xy)
        / std_xy
    )

    Z = quadratic_matrix(
        X_scaled
    )

    coefficients, _, _, _ = np.linalg.lstsq(
        Z,
        p_train,
        rcond=None
    )

    return (
        coefficients,
        mean_xy,
        std_xy
    )


# ============================================================
# 4. Prediction
# ============================================================

def predict(
    X_test,
    coefficients,
    mean_xy,
    std_xy
):

    X_scaled = (
        (X_test - mean_xy)
        / std_xy
    )

    Z_test = quadratic_matrix(
        X_scaled
    )

    return (
        Z_test @ coefficients
    )


# ============================================================
# 5. Check whether test point is inside training convex hull
# ============================================================

def point_inside_training_hull(
    X_train,
    test_point
):

    hull = ConvexHull(
        X_train
    )

    hull_vertices = X_train[
        hull.vertices
    ]

    hull_path = Path(
        hull_vertices
    )

    return hull_path.contains_point(
        test_point,
        radius=1e-10
    )


# ============================================================
# 6. Leave-One-Out Cross Validation
# ============================================================

results = []


print(
    "STANDARDISED MINIMUM-NORM FULL QUADRATIC: LOOCV"
)

print(
    "=" * 95
)

print(
    f"{'Sensor':<10}"
    f"{'Actual':>12}"
    f"{'Predicted':>14}"
    f"{'Error':>14}"
    f"{'Abs Error':>14}"
    f"{'Type':>20}"
)

print(
    "-" * 95
)


for test_index in range(
    len(X)
):

    mask = np.ones(
        len(X),
        dtype=bool
    )

    mask[test_index] = False


    X_train = X[mask]
    p_train = p[mask]

    X_test = X[
        test_index
    ].reshape(
        1,
        -1
    )

    actual = p[
        test_index
    ]


    # Fit only on four training sensors
    (
        coefficients,
        mean_xy,
        std_xy
    ) = fit_model(
        X_train,
        p_train
    )


    # Predict omitted sensor
    predicted = predict(
        X_test,
        coefficients,
        mean_xy,
        std_xy
    )[0]


    error = (
        predicted
        - actual
    )

    abs_error = abs(
        error
    )


    # Interpolation or extrapolation?
    inside = point_inside_training_hull(
        X_train,
        X_test[0]
    )


    prediction_type = (
        "Interpolation"
        if inside
        else "Extrapolation"
    )


    results.append({
        "sensor": test_index + 1,
        "actual": actual,
        "predicted": predicted,
        "error": error,
        "abs_error": abs_error,
        "type": prediction_type
    })


    print(
        f"{'S'+str(test_index+1):<10}"
        f"{actual:>12.6f}"
        f"{predicted:>14.6f}"
        f"{error:>14.6f}"
        f"{abs_error:>14.6f}"
        f"{prediction_type:>20}"
    )


# ============================================================
# 7. Overall LOOCV
# ============================================================

all_errors = np.array([
    r["error"]
    for r in results
])


overall_rmse = np.sqrt(
    np.mean(
        all_errors**2
    )
)

overall_mae = np.mean(
    np.abs(
        all_errors
    )
)


# ============================================================
# 8. Extrapolation-only performance
#
# S1, S3, S4, S5 should fall here.
# ============================================================

extrapolation_results = [
    r
    for r in results
    if r["type"] == "Extrapolation"
]


extrapolation_errors = np.array([
    r["error"]
    for r in extrapolation_results
])


extrapolation_rmse = np.sqrt(
    np.mean(
        extrapolation_errors**2
    )
)

extrapolation_mae = np.mean(
    np.abs(
        extrapolation_errors
    )
)


# ============================================================
# 9. Interpolation-only performance
#
# With this geometry this should contain S2 only.
# ============================================================

interpolation_results = [
    r
    for r in results
    if r["type"] == "Interpolation"
]


interpolation_errors = np.array([
    r["error"]
    for r in interpolation_results
])


interpolation_rmse = np.sqrt(
    np.mean(
        interpolation_errors**2
    )
)

interpolation_mae = np.mean(
    np.abs(
        interpolation_errors
    )
)


# ============================================================
# 10. Print summary
# ============================================================

print(
    "\n"
    + "=" * 95
)

print(
    "SUMMARY"
)

print(
    "=" * 95
)


print(
    f"\nOverall LOOCV RMSE = "
    f"{overall_rmse:.6f}"
)

print(
    f"Overall LOOCV MAE  = "
    f"{overall_mae:.6f}"
)


print(
    "\nInterpolation-only validation:"
)

print(
    f"Number of folds = "
    f"{len(interpolation_results)}"
)

print(
    f"RMSE = "
    f"{interpolation_rmse:.6f}"
)

print(
    f"MAE  = "
    f"{interpolation_mae:.6f}"
)


print(
    "\nExtrapolation-only validation:"
)

print(
    f"Number of folds = "
    f"{len(extrapolation_results)}"
)

print(
    f"RMSE = "
    f"{extrapolation_rmse:.6f}"
)

print(
    f"MAE  = "
    f"{extrapolation_mae:.6f}"
)


# ============================================================
# 11. Relative errors for extrapolation folds
# ============================================================

print(
    "\nEXTRAPOLATION FOLDS"
)

print(
    "=" * 75
)

print(
    f"{'Sensor':<10}"
    f"{'Actual':>12}"
    f"{'Predicted':>14}"
    f"{'Abs Error':>14}"
    f"{'Rel Error %':>16}"
)

print(
    "-" * 75
)


for r in extrapolation_results:

    relative_error = (
        r["abs_error"]
        / abs(r["actual"])
        * 100.0
    )

    print(
        f"{'S'+str(r['sensor']):<10}"
        f"{r['actual']:>12.6f}"
        f"{r['predicted']:>14.6f}"
        f"{r['abs_error']:>14.6f}"
        f"{relative_error:>16.2f}"
    )