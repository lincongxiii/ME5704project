import numpy as np


# ============================================================
# ME5704 Group Project 1
#
# Projected Steepest Ascent
# constrained to the SENSOR CONVEX HULL
#
# Purpose:
# 1. Compare different initial points
# 2. Study initial step-size sensitivity
# 3. Compare with the reference global maximum obtained
#    using Boundary Parabolic Interpolation
# ============================================================


# ============================================================
# 1. Sensor data
# ============================================================

X = np.array([
    [0.0, 0.5],   # S1
    [1.3, 1.1],   # S2
    [1.9, 0.1],   # S3
    [2.5, 2.3],   # S4
    [0.7, 1.8]    # S5
], dtype=float)

p_sensor = np.array([
    2.0,
    17.0,
    43.0,
    28.0,
    36.0
])


# ============================================================
# 2. Final pressure model
# ============================================================

a0 = 3.3231532886
a1 = 23.4770637201
a2 = -12.0757539776
a3 = -0.1198214306
a4 = -18.3054344341
a5 = 18.8588948009


def pressure(x, y):

    return (
        a0
        + a1*x
        + a2*y
        + a3*x**2
        + a4*x*y
        + a5*y**2
    )


def gradient(x, y):

    return np.array([
        a1 + 2*a3*x + a4*y,
        a2 + a4*x + 2*a5*y
    ])


# ============================================================
# 3. Convex-hull vertices
#
# Counter-clockwise order:
#
# S1 -> S3 -> S4 -> S5
# ============================================================

hull_vertices = np.array([
    [0.0, 0.5],   # S1
    [1.9, 0.1],   # S3
    [2.5, 2.3],   # S4
    [0.7, 1.8]    # S5
], dtype=float)


# ============================================================
# 4. Check whether a point lies inside/on convex hull
#
# For a CCW convex polygon, all edge cross products
# should be non-negative.
# ============================================================

def inside_convex_hull(
    point,
    tolerance=1e-10
):

    for i in range(
        len(hull_vertices)
    ):

        A = hull_vertices[i]

        B = hull_vertices[
            (i + 1)
            % len(hull_vertices)
        ]

        edge = B - A

        relative = point - A

        cross = (
            edge[0]*relative[1]
            - edge[1]*relative[0]
        )

        if cross < -tolerance:
            return False

    return True


# ============================================================
# 5. Closest point on a line segment
# ============================================================

def closest_point_on_segment(
    point,
    A,
    B
):

    AB = B - A

    denominator = np.dot(
        AB,
        AB
    )

    if denominator == 0:

        return A.copy()

    t = np.dot(
        point - A,
        AB
    ) / denominator

    t = np.clip(
        t,
        0.0,
        1.0
    )

    return (
        A + t*AB
    )


# ============================================================
# 6. Projection onto convex hull
#
# If point is already feasible:
#     return point
#
# Otherwise:
#     find closest point among all polygon edges.
# ============================================================

def project_to_convex_hull(
    point
):

    point = np.asarray(
        point,
        dtype=float
    )


    if inside_convex_hull(
        point
    ):

        return point.copy()


    candidates = []


    for i in range(
        len(hull_vertices)
    ):

        A = hull_vertices[i]

        B = hull_vertices[
            (i + 1)
            % len(hull_vertices)
        ]


        candidate = (
            closest_point_on_segment(
                point,
                A,
                B
            )
        )


        distance = np.linalg.norm(
            point - candidate
        )


        candidates.append(
            (
                distance,
                candidate
            )
        )


    best = min(
        candidates,
        key=lambda item:
            item[0]
    )


    return best[1]


# ============================================================
# 7. Projected Steepest Ascent
#
# Search direction:
#
# d = grad(p) / ||grad(p)||
#
# Trial points are projected onto the sensor convex hull.
#
# Backtracking reduces alpha if pressure does not increase.
# ============================================================

def projected_steepest_ascent(
    initial_point,
    initial_alpha=1.0,
    tolerance=1e-10,
    max_iterations=1000
):

    point = project_to_convex_hull(
        np.asarray(
            initial_point,
            dtype=float
        )
    )


    history = [
        (
            0,
            point[0],
            point[1],
            pressure(
                point[0],
                point[1]
            )
        )
    ]


    for iteration in range(
        1,
        max_iterations + 1
    ):

        grad = gradient(
            point[0],
            point[1]
        )

        grad_norm = np.linalg.norm(
            grad
        )


        if grad_norm < tolerance:

            break


        direction = (
            grad / grad_norm
        )


        current_pressure = pressure(
            point[0],
            point[1]
        )


        alpha = initial_alpha

        improved = False


        # ----------------------------------------------------
        # Backtracking line search
        # ----------------------------------------------------

        while alpha > 1e-12:

            trial_unconstrained = (
                point
                + alpha*direction
            )


            trial = (
                project_to_convex_hull(
                    trial_unconstrained
                )
            )


            trial_pressure = pressure(
                trial[0],
                trial[1]
            )


            if (
                trial_pressure
                > current_pressure
                + 1e-12
            ):

                improved = True
                break


            alpha *= 0.5


        # ----------------------------------------------------
        # No feasible improving step:
        # constrained stationary solution reached
        # ----------------------------------------------------

        if not improved:
            break


        movement = np.linalg.norm(
            trial - point
        )


        point = trial


        history.append(
            (
                iteration,
                point[0],
                point[1],
                trial_pressure
            )
        )


        if movement < tolerance:
            break


    return (
        point,
        pressure(
            point[0],
            point[1]
        ),
        iteration,
        history
    )


# ============================================================
# 8. Reference result
#
# Already obtained from exact convex-hull
# boundary optimisation.
# ============================================================

reference_point = np.array([
    1.9,
    0.1
])

reference_pressure = 43.0


# ============================================================
# 9. Initial-point sensitivity
#
# All initial points chosen inside convex hull.
# ============================================================

initial_guesses = [
    np.array([0.5, 0.5]),
    np.array([1.0, 1.0]),
    np.array([1.5, 0.5]),
    np.array([1.5, 1.5]),
    np.array([2.0, 1.0])
]


print(
    "PROJECTED STEEPEST ASCENT"
)

print(
    "CONVEX-HULL INITIAL-POINT SENSITIVITY"
)

print(
    "=" * 105
)

print(
    f"Reference maximum: "
    f"p = {reference_pressure:.8f} "
    f"at "
    f"({reference_point[0]:.8f}, "
    f"{reference_point[1]:.8f})"
)

print()


print(
    f"{'Initial point':<22}"
    f"{'Final x':>12}"
    f"{'Final y':>12}"
    f"{'Pressure':>14}"
    f"{'p error':>14}"
    f"{'Iterations':>14}"
)

print(
    "-" * 105
)


for initial in initial_guesses:

    (
        point,
        p_value,
        iterations,
        history
    ) = projected_steepest_ascent(
        initial,
        initial_alpha=1.0
    )


    initial_text = (
        f"({initial[0]:.2f}, "
        f"{initial[1]:.2f})"
    )


    print(
        f"{initial_text:<22}"
        f"{point[0]:>12.6f}"
        f"{point[1]:>12.6f}"
        f"{p_value:>14.6f}"
        f"{abs(p_value-reference_pressure):>14.6f}"
        f"{iterations:>14}"
    )


# ============================================================
# 10. Step-size sensitivity
#
# Fixed initial point:
#
# (1.0, 1.0)
#
# This point lies inside sensor convex hull.
# ============================================================

alpha_values = [
    0.05,
    0.10,
    0.25,
    0.50,
    1.00,
    2.00,
    5.00
]


fixed_initial = np.array([
    1.0,
    1.0
])


print(
    "\n\nPROJECTED STEEPEST ASCENT"
)

print(
    "CONVEX-HULL STEP-SIZE SENSITIVITY"
)

print(
    "=" * 105
)

print(
    "Fixed initial point = "
    "(1.00, 1.00)"
)

print()


print(
    f"{'Initial alpha':<18}"
    f"{'Final x':>12}"
    f"{'Final y':>12}"
    f"{'Pressure':>14}"
    f"{'p error':>14}"
    f"{'Iterations':>14}"
)

print(
    "-" * 105
)


for alpha0 in alpha_values:

    (
        point,
        p_value,
        iterations,
        history
    ) = projected_steepest_ascent(
        fixed_initial,
        initial_alpha=alpha0
    )


    print(
        f"{alpha0:<18.4f}"
        f"{point[0]:>12.6f}"
        f"{point[1]:>12.6f}"
        f"{p_value:>14.6f}"
        f"{abs(p_value-reference_pressure):>14.6f}"
        f"{iterations:>14}"
    )


# ============================================================
# 11. Detailed iteration history
#
# Representative run:
#
# initial point = (1,1)
# alpha0 = 1
# ============================================================

(
    point_demo,
    p_demo,
    iterations_demo,
    history_demo
) = projected_steepest_ascent(
    np.array([
        1.0,
        1.0
    ]),
    initial_alpha=1.0
)


print(
    "\n\nREPRESENTATIVE ITERATION HISTORY"
)

print(
    "=" * 75
)

print(
    "Initial point = (1.00, 1.00)"
)

print(
    "Initial alpha = 1.00"
)

print()


print(
    f"{'Iter':<10}"
    f"{'x':>14}"
    f"{'y':>14}"
    f"{'p(x,y)':>18}"
)

print(
    "-" * 60
)


for row in history_demo:

    (
        iteration,
        x_value,
        y_value,
        p_value
    ) = row


    print(
        f"{iteration:<10}"
        f"{x_value:>14.8f}"
        f"{y_value:>14.8f}"
        f"{p_value:>18.8f}"
    )


# ============================================================
# 12. Final interpretation
# ============================================================

print(
    "\n"
    + "=" * 105
)

print(
    "REFERENCE"
)

print(
    "=" * 105
)

print(
    f"Boundary Parabolic maximum = "
    f"{reference_pressure:.8f}"
)

print(
    f"Reference location = "
    f"({reference_point[0]:.8f}, "
    f"{reference_point[1]:.8f})"
)

print(
    "\nAll results above are constrained "
    "to the sensor convex hull."
)