import numpy as np


# ============================================================
# ME5704 Group Project 1
# Part (c) - Multidimensional Newton Optimisation
# ============================================================


# ------------------------------------------------------------
# Final candidate pressure model coefficients
# ------------------------------------------------------------

a0 = 3.3231532886
a1 = 23.4770637201
a2 = -12.0757539776
a3 = -0.1198214306
a4 = -18.3054344341
a5 = 18.8588948009


# ------------------------------------------------------------
# Pressure function
# ------------------------------------------------------------

def pressure(x, y):

    return (
        a0
        + a1*x
        + a2*y
        + a3*x**2
        + a4*x*y
        + a5*y**2
    )


# ------------------------------------------------------------
# Gradient
#
# dp/dx = a1 + 2*a3*x + a4*y
# dp/dy = a2 + a4*x + 2*a5*y
# ------------------------------------------------------------

def gradient(x, y):

    dp_dx = (
        a1
        + 2*a3*x
        + a4*y
    )

    dp_dy = (
        a2
        + a4*x
        + 2*a5*y
    )

    return np.array([
        dp_dx,
        dp_dy
    ])


# ------------------------------------------------------------
# Hessian
# ------------------------------------------------------------

H = np.array([
    [2*a3, a4],
    [a4, 2*a5]
])


# ------------------------------------------------------------
# Newton optimisation
# ------------------------------------------------------------

def newton_optimisation(
    x0,
    y0,
    tolerance=1e-8,
    max_iterations=20
):

    point = np.array(
        [x0, y0],
        dtype=float
    )

    print("\nNewton Iterations")
    print("=" * 75)

    print(
        f"{'Iter':<8}"
        f"{'x':>14}"
        f"{'y':>14}"
        f"{'p(x,y)':>16}"
        f"{'||grad||':>16}"
    )

    print("-" * 75)


    for iteration in range(
        max_iterations + 1
    ):

        x, y = point

        grad = gradient(x, y)

        grad_norm = np.linalg.norm(
            grad
        )

        p_value = pressure(
            x,
            y
        )


        print(
            f"{iteration:<8}"
            f"{x:>14.6f}"
            f"{y:>14.6f}"
            f"{p_value:>16.6f}"
            f"{grad_norm:>16.8f}"
        )


        # Stopping criterion
        if grad_norm < tolerance:
            break


        # Newton step:
        #
        # x_new = x_old - H^(-1) grad
        #
        # Solve H * step = grad instead of
        # explicitly calculating H inverse.

        step = np.linalg.solve(
            H,
            grad
        )

        point = point - step


    return point


# ============================================================
# Run Newton from an initial point
# ============================================================

initial_x = 1.0
initial_y = 1.0

solution = newton_optimisation(
    initial_x,
    initial_y
)

x_solution = solution[0]
y_solution = solution[1]

p_solution = pressure(
    x_solution,
    y_solution
)


# ============================================================
# Hessian classification
# ============================================================

det_H = np.linalg.det(H)


if det_H > 0 and H[0, 0] < 0:

    point_type = "local maximum"

elif det_H > 0 and H[0, 0] > 0:

    point_type = "local minimum"

elif det_H < 0:

    point_type = "saddle point"

else:

    point_type = "inconclusive"


print("\nFinal Newton result")
print("=" * 75)

print(
    f"x = {x_solution:.8f}"
)

print(
    f"y = {y_solution:.8f}"
)

print(
    f"p = {p_solution:.8f}"
)

print(
    f"det(H) = {det_H:.8f}"
)

print(
    f"Stationary-point type = "
    f"{point_type}"
)