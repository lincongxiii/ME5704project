import numpy as np
from scipy.spatial import ConvexHull


# ============================================================
# ME5704 Group Project 1
# Part (c)
# Boundary optimisation using Parabolic Interpolation
# ============================================================


# ------------------------------------------------------------
# Sensor locations
# ------------------------------------------------------------

X = np.array([
    [0.0, 0.5],
    [1.3, 1.1],
    [1.9, 0.1],
    [2.5, 2.3],
    [0.7, 1.8]
], dtype=float)


# ------------------------------------------------------------
# Final pressure model
# ------------------------------------------------------------

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


# ============================================================
# Parabolic interpolation
# ============================================================

def parabolic_maximum(
    function,
    x1,
    x2,
    x3,
    tolerance=1e-10,
    max_iterations=100
):

    print(
        f"{'Iter':<7}"
        f"{'x1':>12}"
        f"{'x2':>12}"
        f"{'x3':>12}"
        f"{'x4':>12}"
        f"{'f(x4)':>14}"
    )

    print("-" * 69)


    for iteration in range(max_iterations):

        f1 = function(x1)
        f2 = function(x2)
        f3 = function(x3)


        numerator = (
            (x2-x1)**2 * (f2-f3)
            - (x2-x3)**2 * (f2-f1)
        )

        denominator = (
            (x2-x1) * (f2-f3)
            - (x2-x3) * (f2-f1)
        )


        # Degenerate parabola
        if abs(denominator) < 1e-14:
            break


        x4 = (
            x2
            - 0.5
            * numerator
            / denominator
        )


        # Do not allow extrapolation outside edge
        x4 = np.clip(
            x4,
            0.0,
            1.0
        )


        f4 = function(x4)


        print(
            f"{iteration:<7}"
            f"{x1:>12.6f}"
            f"{x2:>12.6f}"
            f"{x3:>12.6f}"
            f"{x4:>12.6f}"
            f"{f4:>14.6f}"
        )


        # Convergence check
        if abs(x4-x2) < tolerance:
            x2 = x4
            break


        # Keep three points surrounding the
        # highest function value

        points = [
            (x1, f1),
            (x2, f2),
            (x3, f3),
            (x4, f4)
        ]

        points.sort(
            key=lambda item: item[0]
        )


        # Find point with maximum pressure
        best_index = max(
            range(len(points)),
            key=lambda i: points[i][1]
        )


        # Select neighbouring points
        if best_index == 0:

            selected = points[:3]

        elif best_index == len(points)-1:

            selected = points[-3:]

        else:

            selected = points[
                best_index-1:
                best_index+2
            ]


        x1 = selected[0][0]
        x2 = selected[1][0]
        x3 = selected[2][0]


    return x2, function(x2)


# ============================================================
# Convex-hull boundary
# ============================================================

hull = ConvexHull(X)

vertices = X[hull.vertices]


global_candidates = []


# ============================================================
# Optimise every edge
# ============================================================

for edge_index in range(len(vertices)):

    A = vertices[edge_index]

    B = vertices[
        (edge_index + 1)
        % len(vertices)
    ]


    def edge_pressure(t):

        point = (
            A + t*(B-A)
        )

        return pressure(
            point[0],
            point[1]
        )


    print(
        "\n"
        + "="*69
    )

    print(
        f"EDGE {edge_index+1}: "
        f"({A[0]:.2f},{A[1]:.2f}) "
        f"-> "
        f"({B[0]:.2f},{B[1]:.2f})"
    )

    print(
        "="*69
    )


    # Three initial guesses on the edge
    t_opt, p_opt = parabolic_maximum(
        edge_pressure,
        0.0,
        0.5,
        1.0
    )


    # Explicitly check endpoints because the
    # maximum may lie exactly at a boundary vertex

    candidates = [
        (
            edge_pressure(0.0),
            0.0
        ),
        (
            edge_pressure(1.0),
            1.0
        ),
        (
            p_opt,
            t_opt
        )
    ]


    best = max(
        candidates,
        key=lambda item: item[0]
    )


    best_p = best[0]
    best_t = best[1]

    best_point = (
        A + best_t*(B-A)
    )


    print(
        f"\nBest pressure on edge = "
        f"{best_p:.8f}"
    )

    print(
        f"Location = "
        f"({best_point[0]:.8f}, "
        f"{best_point[1]:.8f})"
    )


    global_candidates.append(
        (
            best_p,
            best_point
        )
    )


# ============================================================
# Global maximum
# ============================================================

global_best = max(
    global_candidates,
    key=lambda item: item[0]
)


print(
    "\n"
    + "="*69
)

print(
    "GLOBAL MAXIMUM FROM "
    "BOUNDARY PARABOLIC SEARCH"
)

print(
    "="*69
)


print(
    f"Maximum pressure = "
    f"{global_best[0]:.8f}"
)

print(
    f"Location = "
    f"({global_best[1][0]:.8f}, "
    f"{global_best[1][1]:.8f})"
)


print(
    f"Pain threshold exceeded? "
    f"{global_best[0] > 50}"
)