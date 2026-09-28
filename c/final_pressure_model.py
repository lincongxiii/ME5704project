import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize_scalar


# ============================================================
# ME5704 Group Project 1
# Final Candidate Pressure Model
#
# Standardised-coordinate minimum-norm
# full quadratic surface
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


# ------------------------------------------------------------
# 2. Standardise x and y
# ------------------------------------------------------------

mean_xy = np.mean(X, axis=0)
std_xy = np.std(X, axis=0)

mean_x, mean_y = mean_xy
std_x, std_y = std_xy

X_scaled = (X - mean_xy) / std_xy


print("Coordinate standardisation")
print("=" * 65)

print(f"mean_x = {mean_x:.10f}")
print(f"mean_y = {mean_y:.10f}")

print(f"std_x  = {std_x:.10f}")
print(f"std_y  = {std_y:.10f}")


# ------------------------------------------------------------
# 3. Full quadratic design matrix
#
# p = b0 + b1*xs + b2*ys
#     + b3*xs^2 + b4*xs*ys + b5*ys^2
# ------------------------------------------------------------

def quadratic_matrix_scaled(Xs):

    xs = Xs[:, 0]
    ys = Xs[:, 1]

    return np.column_stack([
        np.ones(len(Xs)),
        xs,
        ys,
        xs**2,
        xs*ys,
        ys**2
    ])


Z = quadratic_matrix_scaled(X_scaled)


# ------------------------------------------------------------
# 4. Minimum-norm solution
# ------------------------------------------------------------

b, _, rank, _ = np.linalg.lstsq(
    Z,
    p,
    rcond=None
)

b0, b1, b2, b3, b4, b5 = b


print("\nStandardised quadratic coefficients")
print("=" * 65)

for name, value in zip(
    ["b0", "b1", "b2", "b3", "b4", "b5"],
    b
):
    print(f"{name} = {value:.10f}")

print(f"\nMatrix rank = {rank}")


# ------------------------------------------------------------
# 5. Convert standardised quadratic back to original x,y
#
# xs = (x - mean_x) / std_x
# ys = (y - mean_y) / std_y
#
# Final form:
#
# p(x,y) =
# a0 + a1*x + a2*y
# + a3*x^2 + a4*x*y + a5*y^2
# ------------------------------------------------------------

a3 = b3 / (std_x**2)

a4 = b4 / (std_x * std_y)

a5 = b5 / (std_y**2)

a1 = (
    b1 / std_x
    - 2*b3*mean_x/(std_x**2)
    - b4*mean_y/(std_x*std_y)
)

a2 = (
    b2 / std_y
    - 2*b5*mean_y/(std_y**2)
    - b4*mean_x/(std_x*std_y)
)

a0 = (
    b0
    - b1*mean_x/std_x
    - b2*mean_y/std_y
    + b3*(mean_x**2)/(std_x**2)
    + b4*mean_x*mean_y/(std_x*std_y)
    + b5*(mean_y**2)/(std_y**2)
)

a = np.array([
    a0, a1, a2, a3, a4, a5
])


print("\nFINAL MODEL IN ORIGINAL COORDINATES")
print("=" * 65)

for name, value in zip(
    ["a0", "a1", "a2", "a3", "a4", "a5"],
    a
):
    print(f"{name} = {value:.10f}")


print("\nPressure distribution:")

print(
    f"p(x,y) = {a0:.8f}"
    f" + ({a1:.8f})x"
    f" + ({a2:.8f})y"
    f" + ({a3:.8f})x^2"
    f" + ({a4:.8f})xy"
    f" + ({a5:.8f})y^2"
)


# ------------------------------------------------------------
# 6. Final pressure function
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
# 7. Verify original measurements again
# ------------------------------------------------------------

p_check = np.array([
    pressure(x, y)
    for x, y in X
])

errors = p_check - p

rmse = np.sqrt(
    np.mean(errors**2)
)


print("\nSensor verification")
print("=" * 65)

print(
    f"{'Sensor':<10}"
    f"{'Actual':>12}"
    f"{'Predicted':>14}"
    f"{'Error':>14}"
)

print("-" * 65)

for i in range(len(p)):

    print(
        f"{'S' + str(i+1):<10}"
        f"{p[i]:>12.6f}"
        f"{p_check[i]:>14.6f}"
        f"{errors[i]:>14.8f}"
    )

print("-" * 65)

print(
    f"Training RMSE = {rmse:.10f}"
)


# ============================================================
# 8. Stationary point and Hessian
# ============================================================

H = np.array([
    [2*a3, a4],
    [a4, 2*a5]
])

gradient_constant = np.array([
    a1,
    a2
])

stationary = np.linalg.solve(
    H,
    -gradient_constant
)

x_star, y_star = stationary

p_star = pressure(
    x_star,
    y_star
)

det_H = np.linalg.det(H)


print("\nStationary-point analysis")
print("=" * 65)

print(f"x* = {x_star:.8f}")
print(f"y* = {y_star:.8f}")
print(f"p* = {p_star:.8f}")

print("\nHessian:")
print(H)

print(
    f"\ndet(H) = {det_H:.8f}"
)


if det_H > 0 and H[0, 0] < 0:

    point_type = "local maximum"

elif det_H > 0 and H[0, 0] > 0:

    point_type = "local minimum"

elif det_H < 0:

    point_type = "saddle point"

else:

    point_type = "inconclusive"

print(
    f"Stationary-point type = {point_type}"
)


# ============================================================
# 9. Exact boundary optimisation on sensor convex hull
# ============================================================

hull = ConvexHull(X)

vertices = X[hull.vertices]

boundary_results = []


print("\nBoundary optimisation")
print("=" * 65)


for i in range(len(vertices)):

    A = vertices[i]
    B = vertices[(i + 1) % len(vertices)]

    def boundary_pressure(t):

        point = A + t*(B-A)

        return pressure(
            point[0],
            point[1]
        )


    # Interior candidate for edge maximum
    max_result = minimize_scalar(
        lambda t: -boundary_pressure(t),
        bounds=(0.0, 1.0),
        method="bounded"
    )

    # Interior candidate for edge minimum
    min_result = minimize_scalar(
        boundary_pressure,
        bounds=(0.0, 1.0),
        method="bounded"
    )


    candidates = [
        (
            boundary_pressure(0.0),
            A
        ),
        (
            boundary_pressure(1.0),
            B
        ),
        (
            boundary_pressure(max_result.x),
            A + max_result.x*(B-A)
        ),
        (
            boundary_pressure(min_result.x),
            A + min_result.x*(B-A)
        )
    ]


    edge_max = max(
        candidates,
        key=lambda item: item[0]
    )

    edge_min = min(
        candidates,
        key=lambda item: item[0]
    )


    boundary_results.append({
        "max": edge_max,
        "min": edge_min
    })


    print(
        f"\nEdge {i+1}: "
        f"({A[0]:.2f},{A[1]:.2f}) "
        f"-> "
        f"({B[0]:.2f},{B[1]:.2f})"
    )

    print(
        f"  Maximum: p = {edge_max[0]:.8f} "
        f"at "
        f"({edge_max[1][0]:.8f}, "
        f"{edge_max[1][1]:.8f})"
    )

    print(
        f"  Minimum: p = {edge_min[0]:.8f} "
        f"at "
        f"({edge_min[1][0]:.8f}, "
        f"{edge_min[1][1]:.8f})"
    )


# ------------------------------------------------------------
# 10. Global boundary extrema
# ------------------------------------------------------------

global_max = max(
    [result["max"] for result in boundary_results],
    key=lambda item: item[0]
)

global_min = min(
    [result["min"] for result in boundary_results],
    key=lambda item: item[0]
)


print("\n" + "=" * 65)

print(
    "FINAL RESULTS WITHIN SENSOR CONVEX HULL"
)

print("=" * 65)


print(
    f"\nMaximum pressure = "
    f"{global_max[0]:.8f}"
)

print(
    f"Maximum location = "
    f"({global_max[1][0]:.8f}, "
    f"{global_max[1][1]:.8f})"
)


print(
    f"\nMinimum pressure = "
    f"{global_min[0]:.8f}"
)

print(
    f"Minimum location = "
    f"({global_min[1][0]:.8f}, "
    f"{global_min[1][1]:.8f})"
)


print(
    f"\nDamage predicted (p <= 0)? "
    f"{global_min[0] <= 0}"
)

print(
    f"Pain threshold exceeded (p > 50)? "
    f"{global_max[0] > 50}"
)