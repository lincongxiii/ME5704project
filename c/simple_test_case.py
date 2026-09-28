import numpy as np


# ============================================================
# ME5704 Group Project 1
#
# SIMPLE TEST CASE FOR PROGRAMME ACCURACY
#
# Artificial quadratic pressure surface:
#
# p(x,y) = 40 - 2(x-1)^2 - 3(y-2)^2
#
# Analytical results:
#
# Global maximum:
#   (x,y) = (1,2)
#   p_max = 40
#
# Zero-pressure contour:
#   2(x-1)^2 + 3(y-2)^2 = 40
#
# At x = 1:
#   y = 2 +/- sqrt(40/3)
# ============================================================


# ============================================================
# 1. Test pressure surface
# ============================================================

def pressure(x, y):

    return (
        40.0
        - 2.0*(x - 1.0)**2
        - 3.0*(y - 2.0)**2
    )


# ============================================================
# 2. Gradient and Hessian
# ============================================================

def gradient(x, y):

    return np.array([
        -4.0*(x - 1.0),
        -6.0*(y - 2.0)
    ])


H = np.array([
    [-4.0, 0.0],
    [0.0, -6.0]
])


# ============================================================
# PART A
# ROOT-FINDING VALIDATION
#
# Fix x = 1.
#
# Then:
#
# f(y) = p(1,y)
#      = 40 - 3(y-2)^2
#
# Analytical roots:
#
# y = 2 +/- sqrt(40/3)
# ============================================================

x_fixed = 1.0


def f_root(y):

    return pressure(
        x_fixed,
        y
    )


def df_root(y):

    return (
        -6.0*(y - 2.0)
    )


# ============================================================
# 3. Bisection Method
# ============================================================

def bisection(
    lower,
    upper,
    tolerance=1e-10,
    max_iterations=200
):

    f_lower = f_root(lower)
    f_upper = f_root(upper)


    if f_lower * f_upper > 0:

        raise ValueError(
            "Initial interval does not bracket a root."
        )


    for iteration in range(
        1,
        max_iterations + 1
    ):

        midpoint = (
            lower + upper
        ) / 2.0

        f_mid = f_root(
            midpoint
        )


        if (
            abs(f_mid) < tolerance
            or
            abs(upper-lower) < tolerance
        ):

            return (
                midpoint,
                iteration
            )


        if f_lower * f_mid < 0:

            upper = midpoint

        else:

            lower = midpoint
            f_lower = f_mid


    return (
        midpoint,
        max_iterations
    )


# ============================================================
# 4. Newton-Raphson Method
# ============================================================

def newton_raphson(
    initial_guess,
    tolerance=1e-10,
    max_iterations=100
):

    y = float(
        initial_guess
    )


    for iteration in range(
        1,
        max_iterations + 1
    ):

        f_value = f_root(
            y
        )

        derivative = df_root(
            y
        )


        if abs(f_value) < tolerance:

            return (
                y,
                iteration
            )


        if abs(derivative) < 1e-14:

            raise ValueError(
                "Derivative is too close to zero."
            )


        y_new = (
            y
            - f_value/derivative
        )


        if abs(y_new-y) < tolerance:

            y = y_new

            return (
                y,
                iteration
            )


        y = y_new


    return (
        y,
        max_iterations
    )


# ============================================================
# 5. Analytical roots
# ============================================================

root_exact_lower = (
    2.0
    - np.sqrt(40.0/3.0)
)

root_exact_upper = (
    2.0
    + np.sqrt(40.0/3.0)
)


# ============================================================
# 6. Numerical roots
#
# lower root approx -1.65148
# upper root approx  5.65148
# ============================================================

root_b_lower, iter_b_lower = (
    bisection(
        -3.0,
        0.0
    )
)

root_n_lower, iter_n_lower = (
    newton_raphson(
        -2.0
    )
)


root_b_upper, iter_b_upper = (
    bisection(
        4.0,
        7.0
    )
)

root_n_upper, iter_n_upper = (
    newton_raphson(
        6.0
    )
)


# ============================================================
# 7. Root-finding results
# ============================================================

print(
    "SIMPLE QUADRATIC PRESSURE TEST CASE"
)

print(
    "=" * 100
)

print(
    "p_test(x,y) = "
    "40 - 2(x-1)^2 - 3(y-2)^2"
)

print(
    "\nROOT-FINDING VALIDATION"
)

print(
    "=" * 100
)

print(
    "Fixed x = 1"
)

print(
    f"Analytical lower root = "
    f"{root_exact_lower:.12f}"
)

print(
    f"Analytical upper root = "
    f"{root_exact_upper:.12f}"
)

print()


print(
    f"{'Method':<24}"
    f"{'Root':>18}"
    f"{'Abs Error':>18}"
    f"{'Residual':>18}"
    f"{'Iterations':>16}"
)

print(
    "-" * 100
)


root_results = [

    (
        "Bisection lower",
        root_b_lower,
        root_exact_lower,
        iter_b_lower
    ),

    (
        "Newton lower",
        root_n_lower,
        root_exact_lower,
        iter_n_lower
    ),

    (
        "Bisection upper",
        root_b_upper,
        root_exact_upper,
        iter_b_upper
    ),

    (
        "Newton upper",
        root_n_upper,
        root_exact_upper,
        iter_n_upper
    )
]


for (
    method,
    root_value,
    exact_value,
    iterations
) in root_results:

    abs_error = abs(
        root_value
        - exact_value
    )

    residual = abs(
        f_root(
            root_value
        )
    )


    print(
        f"{method:<24}"
        f"{root_value:>18.12f}"
        f"{abs_error:>18.3e}"
        f"{residual:>18.3e}"
        f"{iterations:>16}"
    )


# ============================================================
# PART B
# NEWTON OPTIMISATION VALIDATION
#
# Analytical maximum:
#
# (1,2)
# p = 40
# ============================================================

def newton_optimisation(
    initial_point,
    tolerance=1e-12,
    max_iterations=50
):

    point = np.array(
        initial_point,
        dtype=float
    )


    for iteration in range(
        1,
        max_iterations + 1
    ):

        grad = gradient(
            point[0],
            point[1]
        )


        if (
            np.linalg.norm(grad)
            < tolerance
        ):

            break


        step = np.linalg.solve(
            H,
            grad
        )


        point = (
            point - step
        )


    return (
        point,
        pressure(
            point[0],
            point[1]
        ),
        iteration
    )


# ============================================================
# PART C
# STEEPEST ASCENT VALIDATION
# ============================================================

def steepest_ascent(
    initial_point,
    alpha=0.1,
    tolerance=1e-10,
    max_iterations=2000
):

    point = np.array(
        initial_point,
        dtype=float
    )


    for iteration in range(
        1,
        max_iterations + 1
    ):

        grad = gradient(
            point[0],
            point[1]
        )


        if (
            np.linalg.norm(grad)
            < tolerance
        ):

            break


        new_point = (
            point
            + alpha*grad
        )


        if (
            np.linalg.norm(
                new_point-point
            )
            < tolerance
        ):

            point = new_point
            break


        point = new_point


    return (
        point,
        pressure(
            point[0],
            point[1]
        ),
        iteration
    )


# ============================================================
# 8. Optimisation test initial points
# ============================================================

initial_points = [
    [0.0, 0.0],
    [2.0, 3.0],
    [-2.0, 4.0],
    [4.0, -1.0]
]


exact_max_point = np.array([
    1.0,
    2.0
])

exact_max_pressure = 40.0


# ============================================================
# 9. Newton optimisation results
# ============================================================

print(
    "\n\nNEWTON OPTIMISATION VALIDATION"
)

print(
    "=" * 100
)

print(
    "Analytical maximum:"
)

print(
    "Location = (1.000000000000, "
    "2.000000000000)"
)

print(
    "Pressure = 40.000000000000"
)

print()


print(
    f"{'Initial':<18}"
    f"{'Final x':>16}"
    f"{'Final y':>16}"
    f"{'Pressure':>16}"
    f"{'Location error':>18}"
    f"{'Iterations':>14}"
)

print(
    "-" * 100
)


for initial in initial_points:

    (
        point,
        p_value,
        iterations
    ) = newton_optimisation(
        initial
    )


    location_error = np.linalg.norm(
        point
        - exact_max_point
    )


    initial_text = (
        f"({initial[0]:.1f},"
        f"{initial[1]:.1f})"
    )


    print(
        f"{initial_text:<18}"
        f"{point[0]:>16.12f}"
        f"{point[1]:>16.12f}"
        f"{p_value:>16.12f}"
        f"{location_error:>18.3e}"
        f"{iterations:>14}"
    )


# ============================================================
# 10. Steepest Ascent results
# ============================================================

print(
    "\n\nSTEEPEST ASCENT VALIDATION"
)

print(
    "=" * 100
)

print(
    "Analytical maximum:"
)

print(
    "Location = (1.000000000000, "
    "2.000000000000)"
)

print(
    "Pressure = 40.000000000000"
)

print()


print(
    f"{'Initial':<18}"
    f"{'Final x':>16}"
    f"{'Final y':>16}"
    f"{'Pressure':>16}"
    f"{'Location error':>18}"
    f"{'Iterations':>14}"
)

print(
    "-" * 100
)


for initial in initial_points:

    (
        point,
        p_value,
        iterations
    ) = steepest_ascent(
        initial,
        alpha=0.1
    )


    location_error = np.linalg.norm(
        point
        - exact_max_point
    )


    initial_text = (
        f"({initial[0]:.1f},"
        f"{initial[1]:.1f})"
    )


    print(
        f"{initial_text:<18}"
        f"{point[0]:>16.12f}"
        f"{point[1]:>16.12f}"
        f"{p_value:>16.12f}"
        f"{location_error:>18.3e}"
        f"{iterations:>14}"
    )


# ============================================================
# PART D
# PARABOLIC INTERPOLATION VALIDATION
#
# Test along y = 2.
#
# Then:
#
# p(x,2) = 40 - 2(x-1)^2
#
# Exact maximum:
#
# x = 1
# p = 40
# ============================================================

def edge_function(x):

    return pressure(
        x,
        2.0
    )


def parabolic_vertex(
    x1,
    x2,
    x3
):

    f1 = edge_function(
        x1
    )

    f2 = edge_function(
        x2
    )

    f3 = edge_function(
        x3
    )


    numerator = (
        (x2-x1)**2
        * (f2-f3)
        -
        (x2-x3)**2
        * (f2-f1)
    )


    denominator = (
        (x2-x1)
        * (f2-f3)
        -
        (x2-x3)
        * (f2-f1)
    )


    x4 = (
        x2
        - 0.5
        * numerator
        / denominator
    )


    return (
        x4,
        edge_function(x4)
    )


x_parabolic, p_parabolic = (
    parabolic_vertex(
        0.0,
        0.5,
        2.0
    )
)


print(
    "\n\nPARABOLIC INTERPOLATION VALIDATION"
)

print(
    "=" * 100
)

print(
    "Test line: y = 2"
)

print(
    "Analytical maximum:"
)

print(
    "x = 1.000000000000"
)

print(
    "p = 40.000000000000"
)

print()

print(
    f"Parabolic x = "
    f"{x_parabolic:.12f}"
)

print(
    f"Parabolic pressure = "
    f"{p_parabolic:.12f}"
)

print(
    f"x absolute error = "
    f"{abs(x_parabolic-1.0):.3e}"
)

print(
    f"pressure absolute error = "
    f"{abs(p_parabolic-40.0):.3e}"
)


# ============================================================
# 11. Overall statement
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "PROGRAMME VALIDATION SUMMARY"
)

print(
    "=" * 100
)

print(
    "The analytical roots and maximum of the "
    "simple quadratic pressure surface are known."
)

print(
    "The numerical results above can therefore "
    "be compared directly with exact solutions "
    "to demonstrate programme accuracy."
)