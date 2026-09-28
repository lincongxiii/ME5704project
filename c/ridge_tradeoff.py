import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.path import Path


# ============================================================
# ME5704 Group Project 1
#
# Ridge-Regularised Full Quadratic
# Interpolation / Extrapolation Trade-off
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
# 2. Full quadratic matrix
# ============================================================

def quadratic_matrix(Xs):

    x = Xs[:, 0]
    y = Xs[:, 1]

    return np.column_stack([
        np.ones(len(Xs)),
        x,
        y,
        x**2,
        x*y,
        y**2
    ])


# ============================================================
# 3. Ridge model
#
# Standardise coordinates using TRAINING DATA ONLY.
#
# Objective:
#
# ||Zb - p||^2 + lambda ||b_non-intercept||^2
#
# Intercept is not penalised.
# ============================================================

def fit_ridge(X_train, p_train, lam):

    mean_xy = np.mean(
        X_train,
        axis=0
    )

    std_xy = np.std(
        X_train,
        axis=0
    )

    Xs = (
        (X_train - mean_xy)
        / std_xy
    )

    Z = quadratic_matrix(Xs)

    penalty = np.eye(6)

    # Do not penalise intercept
    penalty[0, 0] = 0.0

    coefficients = np.linalg.solve(
        Z.T @ Z
        + lam * penalty,
        Z.T @ p_train
    )

    return (
        coefficients,
        mean_xy,
        std_xy
    )


def predict(
    X_test,
    coefficients,
    mean_xy,
    std_xy
):

    Xs = (
        (X_test - mean_xy)
        / std_xy
    )

    Z = quadratic_matrix(Xs)

    return Z @ coefficients


# ============================================================
# 4. Determine interpolation/extrapolation fold
# ============================================================

def inside_training_hull(
    X_train,
    test_point
):

    hull = ConvexHull(X_train)

    polygon = X_train[
        hull.vertices
    ]

    path = Path(polygon)

    return path.contains_point(
        test_point,
        radius=1e-10
    )


# ============================================================
# 5. Pre-identify fold geometry
# ============================================================

fold_types = []


for test_index in range(len(X)):

    mask = np.ones(
        len(X),
        dtype=bool
    )

    mask[test_index] = False

    inside = inside_training_hull(
        X[mask],
        X[test_index]
    )

    fold_types.append(
        "Interpolation"
        if inside
        else "Extrapolation"
    )


print("Fold geometry")
print("=" * 45)

for i, fold_type in enumerate(
    fold_types
):
    print(
        f"S{i+1}: {fold_type}"
    )


# ============================================================
# 6. Lambda values
#
# Include very weak through strong regularisation.
# ============================================================

lambda_values = np.logspace(
    -8,
    4,
    241
)


s2_errors = []

extrap_rmse_values = []

extrap_mae_values = []

overall_rmse_values = []

training_rmse_values = []


# ============================================================
# 7. Evaluate every lambda
# ============================================================

for lam in lambda_values:

    fold_errors = []

    interpolation_errors = []

    extrapolation_errors = []


    # --------------------------------------------------------
    # LOOCV
    # --------------------------------------------------------

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
        ].reshape(1, -1)

        actual = p[
            test_index
        ]


        (
            coefficients,
            mean_xy,
            std_xy
        ) = fit_ridge(
            X_train,
            p_train,
            lam
        )


        predicted = predict(
            X_test,
            coefficients,
            mean_xy,
            std_xy
        )[0]


        error = (
            predicted - actual
        )

        fold_errors.append(
            error
        )


        if (
            fold_types[test_index]
            == "Interpolation"
        ):

            interpolation_errors.append(
                error
            )

        else:

            extrapolation_errors.append(
                error
            )


    fold_errors = np.array(
        fold_errors
    )

    extrapolation_errors = np.array(
        extrapolation_errors
    )

    interpolation_errors = np.array(
        interpolation_errors
    )


    overall_rmse = np.sqrt(
        np.mean(
            fold_errors**2
        )
    )

    extrap_rmse = np.sqrt(
        np.mean(
            extrapolation_errors**2
        )
    )

    extrap_mae = np.mean(
        np.abs(
            extrapolation_errors
        )
    )

    s2_error = np.abs(
        interpolation_errors[0]
    )


    overall_rmse_values.append(
        overall_rmse
    )

    extrap_rmse_values.append(
        extrap_rmse
    )

    extrap_mae_values.append(
        extrap_mae
    )

    s2_errors.append(
        s2_error
    )


    # --------------------------------------------------------
    # Full-data training RMSE
    # --------------------------------------------------------

    (
        coefficients_full,
        mean_full,
        std_full
    ) = fit_ridge(
        X,
        p,
        lam
    )


    p_train_pred = predict(
        X,
        coefficients_full,
        mean_full,
        std_full
    )


    train_rmse = np.sqrt(
        np.mean(
            (p_train_pred-p)**2
        )
    )


    training_rmse_values.append(
        train_rmse
    )


# Convert to arrays

s2_errors = np.array(
    s2_errors
)

extrap_rmse_values = np.array(
    extrap_rmse_values
)

extrap_mae_values = np.array(
    extrap_mae_values
)

overall_rmse_values = np.array(
    overall_rmse_values
)

training_rmse_values = np.array(
    training_rmse_values
)


# ============================================================
# 8. Identify useful lambda values
# ============================================================

best_s2_index = np.argmin(
    s2_errors
)

best_extrap_index = np.argmin(
    extrap_rmse_values
)

best_overall_index = np.argmin(
    overall_rmse_values
)


# Normalised compromise score:
#
# Equal weight to interpolation and extrapolation.
#
# This is NOT declared the "correct" model.
# It is only a diagnostic compromise.
# ============================================================

normalised_s2 = (
    s2_errors
    / np.min(s2_errors)
)

normalised_extrap = (
    extrap_rmse_values
    / np.min(extrap_rmse_values)
)

compromise_score = (
    normalised_s2
    + normalised_extrap
)

best_compromise_index = np.argmin(
    compromise_score
)


# ============================================================
# 9. Print important results
# ============================================================

def print_result(
    title,
    index
):

    print(
        f"\n{title}"
    )

    print(
        "-" * 65
    )

    print(
        f"lambda = "
        f"{lambda_values[index]:.8g}"
    )

    print(
        f"S2 interpolation abs error = "
        f"{s2_errors[index]:.6f}"
    )

    print(
        f"Extrapolation RMSE = "
        f"{extrap_rmse_values[index]:.6f}"
    )

    print(
        f"Extrapolation MAE = "
        f"{extrap_mae_values[index]:.6f}"
    )

    print(
        f"Overall LOOCV RMSE = "
        f"{overall_rmse_values[index]:.6f}"
    )

    print(
        f"Full-data training RMSE = "
        f"{training_rmse_values[index]:.6f}"
    )


print(
    "\nRIDGE REGULARISATION TRADE-OFF"
)

print(
    "=" * 65
)


print_result(
    "Best S2 interpolation",
    best_s2_index
)

print_result(
    "Best extrapolation RMSE",
    best_extrap_index
)

print_result(
    "Best overall LOOCV RMSE",
    best_overall_index
)

print_result(
    "Equal-weight diagnostic compromise",
    best_compromise_index
)


# ============================================================
# 10. Print selected lambda table
# ============================================================

selected_lambdas = [
    1e-8,
    1e-6,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1.0,
    10.0,
    100.0,
    1000.0,
    10000.0
]


print(
    "\nSELECTED LAMBDA VALUES"
)

print(
    "=" * 100
)

print(
    f"{'lambda':>12}"
    f"{'S2 error':>16}"
    f"{'Extrap RMSE':>18}"
    f"{'Overall RMSE':>18}"
    f"{'Train RMSE':>16}"
)

print(
    "-" * 100
)


for target_lambda in selected_lambdas:

    index = np.argmin(
        np.abs(
            np.log10(lambda_values)
            - np.log10(target_lambda)
        )
    )


    print(
        f"{lambda_values[index]:>12.4g}"
        f"{s2_errors[index]:>16.6f}"
        f"{extrap_rmse_values[index]:>18.6f}"
        f"{overall_rmse_values[index]:>18.6f}"
        f"{training_rmse_values[index]:>16.6f}"
    )


# ============================================================
# 11. Plot interpolation/extrapolation trade-off
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.semilogx(
    lambda_values,
    s2_errors,
    label="S2 interpolation absolute error"
)

plt.semilogx(
    lambda_values,
    extrap_rmse_values,
    label="Extrapolation RMSE"
)

plt.xlabel(
    "Regularisation parameter, lambda"
)

plt.ylabel(
    "Prediction error"
)

plt.title(
    "Interpolation-Extrapolation Trade-off"
)

plt.grid(
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "ridge_interpolation_extrapolation_tradeoff.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 12. Plot training vs validation
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.semilogx(
    lambda_values,
    training_rmse_values,
    label="Training RMSE"
)

plt.semilogx(
    lambda_values,
    overall_rmse_values,
    label="Overall LOOCV RMSE"
)

plt.xlabel(
    "Regularisation parameter, lambda"
)

plt.ylabel(
    "RMSE"
)

plt.title(
    "Effect of Regularisation Strength"
)

plt.grid(
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "ridge_regularisation_sensitivity.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()