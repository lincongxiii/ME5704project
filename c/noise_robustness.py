import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull
from matplotlib.path import Path


# ============================================================
# ME5704 Group Project 1
#
# Monte Carlo Noise Robustness Analysis
#
# Model:
# Standardised Minimum-Norm Full Quadratic
# ============================================================


# ============================================================
# 1. Original sensor data
# ============================================================

X = np.array([
    [0.0, 0.5],   # S1
    [1.3, 1.1],   # S2
    [1.9, 0.1],   # S3
    [2.5, 2.3],   # S4
    [0.7, 1.8]    # S5
], dtype=float)

p_original = np.array([
    2.0,
    17.0,
    43.0,
    28.0,
    36.0
], dtype=float)


# ============================================================
# 2. Quadratic design matrix
# ============================================================

def quadratic_matrix(X_scaled):

    x = X_scaled[:, 0]
    y = X_scaled[:, 1]

    return np.column_stack([
        np.ones(len(X_scaled)),
        x,
        y,
        x**2,
        x*y,
        y**2
    ])


# ============================================================
# 3. Fit standardised minimum-norm full quadratic
# ============================================================

def fit_minimum_norm(
    X_train,
    p_train
):

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
# 4. Prediction function
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

    Z = quadratic_matrix(
        X_scaled
    )

    return Z @ coefficients


# ============================================================
# 5. Exact extrema inside the sensor convex hull
#
# We evaluate model stability only inside the original
# sensor convex hull.
#
# A quadratic attains its minimum and maximum over a convex
# polygon at one of these candidates:
#   (i)   a hull vertex,
#   (ii)  a stationary point along a hull edge,
#   (iii) the interior stationary point, if inside the hull.
#
# Evaluating every candidate gives the EXACT extrema.
# A rectangular grid misses the hull vertices S1 and S3,
# where the clean extrema p = 2 and p = 43 occur, so grid
# extrema are biased (minimum too high, maximum too low).
# ============================================================

hull = ConvexHull(X)

hull_vertices = X[
    hull.vertices
]

hull_path = Path(
    hull_vertices
)


def exact_hull_extrema(
    coefficients,
    mean_xy,
    std_xy
):
    """Exact (min, max) of the fitted quadratic over the
    sensor convex hull, worked in standardised coordinates."""

    b0, b1, b2, b3, b4, b5 = coefficients

    # p(z) = b0 + g.z + 0.5 z^T H z
    g = np.array([b1, b2])

    H = np.array([
        [2*b3, b4],
        [b4, 2*b5]
    ])

    corners = (
        (hull_vertices - mean_xy)
        / std_xy
    )

    candidates = list(corners)

    # (ii) edge z(t) = zA + t d, stationary where dp/dt = 0
    for i in range(len(corners)):

        zA = corners[i]
        zB = corners[(i + 1) % len(corners)]

        d = zB - zA

        curvature = d @ H @ d

        if abs(curvature) > 1e-12:

            t = -np.dot(g + H @ zA, d) / curvature

            if 0.0 < t < 1.0:
                candidates.append(zA + t*d)

    # (iii) interior stationary point H z = -g
    if abs(np.linalg.det(H)) > 1e-12:

        z_star = np.linalg.solve(H, -g)

        if hull_path.contains_point(
            z_star*std_xy + mean_xy,
            radius=1e-10
        ):
            candidates.append(z_star)

    values = (
        quadratic_matrix(np.array(candidates))
        @ coefficients
    )

    return np.min(values), np.max(values)


# ============================================================
# 6. Monte Carlo settings
# ============================================================

noise_levels = [
    0.00,
    0.02,
    0.05,
    0.10
]

n_simulations = 1000

rng = np.random.default_rng(
    5704
)


# ============================================================
# IMPORTANT:
#
# Noise model:
#
# p_noisy = p + epsilon
#
# epsilon_i ~ N(0, sigma^2)
#
# sigma = noise_level * pressure range
#
# Pressure range = 43 - 2 = 41
#
# This uses the SAME absolute noise scale for every sensor.
# ============================================================

pressure_range = (
    np.max(p_original)
    - np.min(p_original)
)


# ============================================================
# 7. Storage
# ============================================================

summary = {}

all_s2_predictions = {}

all_hull_minima = {}

all_hull_maxima = {}


# ============================================================
# 8. Monte Carlo simulation
# ============================================================

for noise_level in noise_levels:

    s2_predictions = []

    hull_minima = []

    hull_maxima = []


    sigma = (
        noise_level
        * pressure_range
    )


    for simulation in range(
        n_simulations
    ):

        # ----------------------------------------------------
        # Generate noisy measurements
        # ----------------------------------------------------

        if noise_level == 0.0:

            p_noisy = (
                p_original.copy()
            )

        else:

            noise = rng.normal(
                loc=0.0,
                scale=sigma,
                size=len(p_original)
            )

            p_noisy = (
                p_original + noise
            )


        # ====================================================
        # A. Full-data model:
        #    exact pressure extrema inside original convex hull
        # ====================================================

        (
            coefficients_full,
            mean_full,
            std_full
        ) = fit_minimum_norm(
            X,
            p_noisy
        )


        hull_min, hull_max = exact_hull_extrema(
            coefficients_full,
            mean_full,
            std_full
        )


        hull_minima.append(
            hull_min
        )

        hull_maxima.append(
            hull_max
        )


        # ====================================================
        # B. S2 withheld interpolation validation
        #
        # Train using S1, S3, S4, S5 only.
        # Predict ORIGINAL S2 pressure location.
        # ====================================================

        s2_index = 1

        mask = np.ones(
            len(X),
            dtype=bool
        )

        mask[s2_index] = False


        (
            coefficients_s2,
            mean_s2,
            std_s2
        ) = fit_minimum_norm(
            X[mask],
            p_noisy[mask]
        )


        s2_prediction = predict(
            X[s2_index].reshape(
                1,
                -1
            ),
            coefficients_s2,
            mean_s2,
            std_s2
        )[0]


        s2_predictions.append(
            s2_prediction
        )


    # Convert to arrays

    s2_predictions = np.array(
        s2_predictions
    )

    hull_minima = np.array(
        hull_minima
    )

    hull_maxima = np.array(
        hull_maxima
    )


    all_s2_predictions[
        noise_level
    ] = s2_predictions

    all_hull_minima[
        noise_level
    ] = hull_minima

    all_hull_maxima[
        noise_level
    ] = hull_maxima


    # --------------------------------------------------------
    # S2 error is evaluated against the original known
    # measurement p = 17.
    # --------------------------------------------------------

    s2_errors = (
        s2_predictions
        - p_original[1]
    )


    # --------------------------------------------------------
    # Engineering-conclusion frequencies
    # --------------------------------------------------------

    damage_frequency = np.mean(
        hull_minima <= 0
    )

    pain_frequency = np.mean(
        hull_maxima > 50
    )


    summary[
        noise_level
    ] = {

        "sigma": sigma,

        "s2_mean":
            np.mean(
                s2_predictions
            ),

        "s2_std":
            np.std(
                s2_predictions
            ),

        "s2_mae":
            np.mean(
                np.abs(
                    s2_errors
                )
            ),

        "hull_min_mean":
            np.mean(
                hull_minima
            ),

        "hull_min_std":
            np.std(
                hull_minima
            ),

        "hull_max_mean":
            np.mean(
                hull_maxima
            ),

        "hull_max_std":
            np.std(
                hull_maxima
            ),

        "damage_frequency":
            damage_frequency,

        "pain_frequency":
            pain_frequency
    }


# ============================================================
# 9. Print summary table
# ============================================================

print(
    "MONTE CARLO NOISE ROBUSTNESS"
)

print(
    "=" * 125
)

print(
    f"{'Noise':>8}"
    f"{'sigma':>10}"
    f"{'S2 mean':>12}"
    f"{'S2 std':>12}"
    f"{'S2 MAE':>12}"
    f"{'Hull min':>14}"
    f"{'Hull max':>14}"
    f"{'Damage %':>12}"
    f"{'Pain %':>10}"
)

print(
    "-" * 125
)


for noise_level in noise_levels:

    s = summary[
        noise_level
    ]


    print(
        f"{100*noise_level:>7.1f}%"
        f"{s['sigma']:>10.4f}"
        f"{s['s2_mean']:>12.4f}"
        f"{s['s2_std']:>12.4f}"
        f"{s['s2_mae']:>12.4f}"
        f"{s['hull_min_mean']:>14.4f}"
        f"{s['hull_max_mean']:>14.4f}"
        f"{100*s['damage_frequency']:>11.2f}%"
        f"{100*s['pain_frequency']:>9.2f}%"
    )


# ============================================================
# 10. More detailed quantiles
# ============================================================

print(
    "\n95% MONTE CARLO INTERVALS"
)

print(
    "=" * 100
)


for noise_level in noise_levels:

    s2_values = all_s2_predictions[
        noise_level
    ]

    min_values = all_hull_minima[
        noise_level
    ]

    max_values = all_hull_maxima[
        noise_level
    ]


    s2_interval = np.percentile(
        s2_values,
        [2.5, 97.5]
    )

    min_interval = np.percentile(
        min_values,
        [2.5, 97.5]
    )

    max_interval = np.percentile(
        max_values,
        [2.5, 97.5]
    )


    print(
        f"\nNoise level = "
        f"{100*noise_level:.1f}%"
    )

    print(
        f"S2 prediction 95% interval: "
        f"[{s2_interval[0]:.4f}, "
        f"{s2_interval[1]:.4f}]"
    )

    print(
        f"Hull minimum 95% interval: "
        f"[{min_interval[0]:.4f}, "
        f"{min_interval[1]:.4f}]"
    )

    print(
        f"Hull maximum 95% interval: "
        f"[{max_interval[0]:.4f}, "
        f"{max_interval[1]:.4f}]"
    )


# ============================================================
# 11. Plot S2 prediction distributions
# ============================================================

plt.figure(
    figsize=(9, 5)
)


plot_data = [
    all_s2_predictions[level]
    for level in noise_levels
]


plt.boxplot(
    plot_data,
    tick_labels=[
        f"{100*level:.0f}%"
        for level in noise_levels
    ],
    showfliers=False
)


plt.axhline(
    17.0,
    linestyle="--",
    label="Actual S2 pressure = 17"
)

plt.xlabel(
    "Measurement noise level"
)

plt.ylabel(
    "Predicted pressure at S2"
)

plt.title(
    "Noise Sensitivity of S2 Interpolation"
)

plt.grid(
    alpha=0.25
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "noise_sensitivity_S2.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 12. Plot convex-hull extrema distributions
# ============================================================

plt.figure(
    figsize=(9, 5)
)


max_plot_data = [
    all_hull_maxima[level]
    for level in noise_levels
]


plt.boxplot(
    max_plot_data,
    tick_labels=[
        f"{100*level:.0f}%"
        for level in noise_levels
    ],
    showfliers=False
)


plt.axhline(
    50.0,
    linestyle="--",
    label="Pain threshold = 50"
)

plt.xlabel(
    "Measurement noise level"
)

plt.ylabel(
    "Maximum pressure inside convex hull"
)

plt.title(
    "Noise Sensitivity of Convex-Hull Maximum Pressure"
)

plt.grid(
    alpha=0.25
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "noise_sensitivity_hull_maximum.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()