"""Thin-plate spline (TPS) reconstruction for robotic-hand pressure data."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path
from scipy.spatial import ConvexHull


def tps_kernel(radius):
    # TPS radial basis function: r^2 log(r)
    value = np.zeros_like(radius, dtype=float)
    mask = radius > 0
    value[mask] = radius[mask] ** 2 * np.log(radius[mask])
    return value


def pairwise_distance(a, b):
    # Pairwise distances between two sets of 2D points
    squared = np.maximum(0.0, np.sum(a * a, axis=1)[:, None]
                         + np.sum(b * b, axis=1)[None, :] - 2 * a @ b.T)
    return np.sqrt(squared)


def fit_tps(x, y, z, smoothing=0.0):
    # Solve the augmented system for TPS coefficients
    x, y, z = (np.asarray(value, dtype=float).ravel() for value in (x, y, z))
    if len(x) < 3 or len(x) != len(y) or len(x) != len(z):
        raise ValueError("x, y and z must contain at least three matching entries.")
    if smoothing < 0 or len(np.unique(np.c_[x, y], axis=0)) != len(x):
        raise ValueError("Locations must be distinct and smoothing non-negative.")

    # Normalize coordinates to improve conditioning
    centre = np.array([x.mean(), y.mean()])
    scale = max(np.ptp(x), np.ptp(y), np.finfo(float).eps)
    points = (np.c_[x, y] - centre) / scale
    n = len(z)
    kernel = tps_kernel(pairwise_distance(points, points))
    affine_matrix = np.c_[np.ones(n), points]
    if np.linalg.matrix_rank(affine_matrix) < 3:
        raise ValueError("TPS locations must not be collinear.")
    # [K + lambda*I, P; P^T, 0] [w; a] = [z; 0]
    system = np.block([[kernel + smoothing * np.eye(n), affine_matrix],
                       [affine_matrix.T, np.zeros((3, 3))]])
    rhs = np.r_[z, np.zeros(3)]
    rcond = 1.0 / np.linalg.cond(system)
    coefficient = np.linalg.pinv(system) @ rhs if rcond < 1e-12 else np.linalg.solve(system, rhs)
    return {
        "centre": centre, "scale": scale, "points": points,
        "weight": coefficient[:n], "affine": coefficient[n:],
        "diagnostics": {
            "system_size": n + 3, "reciprocal_condition": rcond,
            "solver": "Moore-Penrose pseudoinverse fallback" if rcond < 1e-12 else "NumPy dense linear solve",
            "linear_system_residual": np.linalg.norm(system @ coefficient - rhs),
            "side_condition_residual": np.linalg.norm(affine_matrix.T @ coefficient[:n]),
            "bending_energy": max(0.0, float(coefficient[:n] @ kernel @ coefficient[:n])),
        },
    }


def evaluate_tps(model, xq, yq):
    # Evaluate pressure at query points
    xq, yq = np.broadcast_arrays(xq, yq)
    shape = xq.shape
    query = (np.c_[xq.ravel(), yq.ravel()] - model["centre"]) / model["scale"]
    values = tps_kernel(pairwise_distance(query, model["points"])) @ model["weight"]
    values += np.c_[np.ones(len(query)), query] @ model["affine"]
    return values.reshape(shape)


def evaluate_tps_derivatives(model, xq, yq):
    # Evaluate the TPS value, gradient, and Hessian
    query = (np.array([xq, yq]) - model["centre"]) / model["scale"]
    difference = query - model["points"]
    radius = np.linalg.norm(difference, axis=1)
    differentiable = bool(np.all(radius > 1e-10))
    gradient = model["affine"][1:].copy()
    hessian = np.zeros((2, 2))
    for delta, r, weight in zip(difference, radius, model["weight"]):
        if r <= 1e-10:
            continue
        factor = 2 * np.log(r) + 1
        gradient += weight * factor * delta
        hessian += weight * (factor * np.eye(2) + 2 * np.outer(delta, delta) / r**2)
    value = evaluate_tps(model, xq, yq).item()
    return value, gradient / model["scale"], hessian / model["scale"]**2, differentiable


def hull_polygon(x, y):
    # Closed convex hull of the sensor locations
    points = np.c_[x, y]
    hull = ConvexHull(points)
    return np.vstack([points[hull.vertices], points[hull.vertices[0]]])


def inside_polygon(x, y, polygon):
    return Path(polygon).contains_points(np.c_[np.asarray(x).ravel(), np.asarray(y).ravel()], radius=1e-10).reshape(np.shape(x))


def percentile(values, percentage):
    return np.percentile(np.asarray(values)[np.isfinite(values)], percentage, method="linear")


def bisection_root(a, b, fa, fb, model, level, tolerance, iterations):
    # One-dimensional bisection root solve on a grid edge
    if abs(fa) <= tolerance:
        return a
    if abs(fb) <= tolerance:
        return b
    if fa * fb > 0:
        return None
    lo, hi = 0.0, 1.0
    direction = b - a
    for _ in range(iterations):
        mid = (lo + hi) / 2
        point = a + mid * direction
        fm = evaluate_tps(model, *point).item() - level
        if abs(fm) <= tolerance or hi - lo <= tolerance:
            return point
        if fa * fm < 0:
            hi, fb = mid, fm
        else:
            lo, fa = mid, fm
    return a + (lo + hi) / 2 * direction


def contains_point(points, point, tolerance):
    return any(np.linalg.norm(existing - point) <= tolerance for existing in points)


def connect_segments(segments, tolerance):
    # Join short segments from adjacent grid edges
    lines, used = [], np.zeros(len(segments), dtype=bool)
    for initial, segment in enumerate(segments):
        if used[initial]:
            continue
        line = list(segment)
        used[initial] = True
        joined = True
        while joined:
            joined = False
            for index, candidate in enumerate(segments):
                if used[index]:
                    continue
                a, b = candidate
                if np.linalg.norm(line[-1] - a) <= tolerance:
                    line.append(b)
                elif np.linalg.norm(line[-1] - b) <= tolerance:
                    line.append(a)
                elif np.linalg.norm(line[0] - b) <= tolerance:
                    line.insert(0, a)
                elif np.linalg.norm(line[0] - a) <= tolerance:
                    line.insert(0, b)
                else:
                    continue
                used[index] = True
                joined = True
        lines.append(np.asarray(line))
    return lines


def bisection_contours(x_grid, y_grid, z, model, level, tolerance, iterations):
    # Construct the p = level contour from bisection roots
    segments = []
    merge_tolerance = max(10 * tolerance, 1e-7 * max(np.ptp(x_grid), np.ptp(y_grid), 1))
    edges = ((0, 1), (1, 2), (2, 3), (3, 0))
    for row in range(len(y_grid) - 1):
        for column in range(len(x_grid) - 1):
            corners = np.array([[x_grid[column], y_grid[row]], [x_grid[column + 1], y_grid[row]],
                                [x_grid[column + 1], y_grid[row + 1]], [x_grid[column], y_grid[row + 1]]])
            values = np.array([z[row, column], z[row, column + 1],
                               z[row + 1, column + 1], z[row + 1, column]]) - level
            roots, edge_ids = [], []
            for edge_id, (first, second) in enumerate(edges):
                point = bisection_root(corners[first], corners[second], values[first], values[second],
                                       model, level, tolerance, iterations)
                if point is not None and not contains_point(roots, point, merge_tolerance):
                    roots.append(point)
                    edge_ids.append(edge_id)
            if len(roots) == 2:
                segments.append(roots)
            elif len(roots) == 4:
                order = np.argsort(edge_ids)
                roots = [roots[i] for i in order]
                centre_sign = (evaluate_tps(model, *corners.mean(axis=0)).item() - level) * values[0] >= 0
                pairs = ((0, 1), (2, 3)) if centre_sign else ((0, 3), (1, 2))
                segments.extend([[roots[a], roots[b]] for a, b in pairs])
    return connect_segments(segments, merge_tolerance)


def plot_segments(segments, *args, **kwargs):
    for points in segments:
        plt.plot(points[:, 0], points[:, 1], *args, **kwargs)


def grid_extremum_seeds(x_grid, y_grid, z, inside, kind):
    # Generate Newton initial guesses from local grid extrema
    seeds = []
    for row in range(1, z.shape[0] - 1):
        for column in range(1, z.shape[1] - 1):
            if not inside[row, column]:
                continue
            neighbourhood = z[row - 1:row + 2, column - 1:column + 2]
            if (kind == "min" and z[row, column] <= neighbourhood.min()) or (kind == "max" and z[row, column] >= neighbourhood.max()):
                seeds.append(np.array([x_grid[column], y_grid[row]]))
    return seeds


def newton_stationary(model, start, polygon, tolerance, iterations):
    # Solve grad(p) = 0 inside the convex hull
    point = start.copy()
    for _ in range(iterations):
        _, gradient, hessian, differentiable = evaluate_tps_derivatives(model, *point)
        if not differentiable or 1 / np.linalg.cond(hessian) < 1e-12:
            return None, None
        if np.linalg.norm(gradient) <= tolerance:
            break
        step = -np.linalg.solve(hessian, gradient)
        current_norm = np.linalg.norm(gradient)
        for _ in range(30):
            candidate = point + step
            _, candidate_gradient, _, valid = evaluate_tps_derivatives(model, *candidate)
            if valid and inside_polygon(candidate[0], candidate[1], polygon) and np.linalg.norm(candidate_gradient) < current_norm:
                point = candidate
                break
            step *= 0.5
        else:
            return None, None
    _, gradient, hessian, valid = evaluate_tps_derivatives(model, *point)
    if not valid or np.linalg.norm(gradient) > tolerance or not inside_polygon(point[0], point[1], polygon):
        return None, None
    determinant = np.linalg.det(hessian)
    classification = "min" if determinant > 0 and hessian[0, 0] > 0 else "max" if determinant > 0 else None
    return point, classification


def newton_edge_stationary(model, a, b, start, tolerance, iterations):
    # One-dimensional Newton search on a hull edge
    fraction, direction = start, b - a
    for _ in range(iterations):
        point = a + fraction * direction
        _, gradient, hessian, valid = evaluate_tps_derivatives(model, *point)
        first, second = gradient @ direction, direction @ hessian @ direction
        if not valid or abs(second) < 1e-12:
            return None, None
        if abs(first) <= tolerance:
            break
        step = -first / second
        for _ in range(30):
            candidate = fraction + step
            if 0 < candidate < 1:
                _, candidate_gradient, _, valid = evaluate_tps_derivatives(model, *(a + candidate * direction))
                if valid and abs(candidate_gradient @ direction) < abs(first):
                    fraction = candidate
                    break
            step *= 0.5
        else:
            return None, None
    point = a + fraction * direction
    _, gradient, hessian, valid = evaluate_tps_derivatives(model, *point)
    first, second = gradient @ direction, direction @ hessian @ direction
    if not valid or abs(first) > tolerance:
        return None, None
    return point, "min" if second > 0 else "max"


def optimise_inside_hull(model, x_grid, y_grid, z, inside, polygon, kind, tolerance, iterations):
    # Compare interior, boundary, and vertex candidates
    vertices = polygon[:-1]
    candidates, interior, edge = [point.copy() for point in vertices], [], []
    seeds = grid_extremum_seeds(x_grid, y_grid, z, inside, kind)
    if not seeds:
        valid_values = np.where(inside, z, np.inf if kind == "min" else -np.inf)
        row, column = np.unravel_index(np.argmin(valid_values) if kind == "min" else np.argmax(valid_values), z.shape)
        seeds = [np.array([x_grid[column], y_grid[row]])]
    for seed in seeds:
        point, classification = newton_stationary(model, seed, polygon, tolerance, iterations)
        if classification == kind and not contains_point(interior, point, 1e-7):
            interior.append(point)
    for a, b in zip(polygon[:-1], polygon[1:]):
        for start in np.linspace(0.1, 0.9, 9):
            point, classification = newton_edge_stationary(model, a, b, start, tolerance, iterations)
            if classification == kind and not contains_point(edge, point, 1e-7):
                edge.append(point)
    candidates = np.asarray(candidates + interior + edge + seeds)
    values = evaluate_tps(model, candidates[:, 0], candidates[:, 1])
    index = np.argmin(values) if kind == "min" else np.argmax(values)
    return values[index], candidates[index], {"interior": len(interior), "edge": len(edge)}


def loocv(x, y, pressure, smoothing):
    # Leave-one-out cross-validation and hull-membership check
    prediction = np.zeros(len(pressure))
    interpolation = np.zeros(len(pressure), dtype=bool)
    for index in range(len(pressure)):
        keep = np.arange(len(pressure)) != index
        prediction[index] = evaluate_tps(fit_tps(x[keep], y[keep], pressure[keep], smoothing), x[index], y[index])
        interpolation[index] = inside_polygon(x[index], y[index], hull_polygon(x[keep], y[keep]))
    return prediction, prediction - pressure, interpolation


def main():
    # 1. Sensor measurements and parameters
    x = np.array([0.0, 1.3, 1.9, 2.5, 0.7])
    y = np.array([0.5, 1.1, 0.1, 2.3, 1.8])
    pressure = np.array([2.0, 17.0, 43.0, 28.0, 36.0])
    smoothing, pain_threshold, padding, grid_size = 0.0, 50.0, 0.15, 241
    root_tolerance, root_iterations = 1e-10, 80
    stationary_tolerance, newton_iterations = 1e-8, 80
    rng = np.random.default_rng(20260927)
    noise_fractions, trials, validation_sensor = np.array([0.02, 0.05, 0.10]), 1000, 1

    # 2. Plotting grid and trusted convex-hull region
    x_limits = np.array([x.min() - padding * np.ptp(x), x.max() + padding * np.ptp(x)])
    y_limits = np.array([y.min() - padding * np.ptp(y), y.max() + padding * np.ptp(y)])
    x_grid, y_grid = np.linspace(*x_limits, grid_size), np.linspace(*y_limits, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    polygon = hull_polygon(x, y)
    inside = inside_polygon(X, Y, polygon)

    # 3. TPS fit and training residual
    model = fit_tps(x, y, pressure, smoothing)
    Z = evaluate_tps(model, X, Y)
    training_rmse = np.sqrt(np.mean((evaluate_tps(model, x, y) - pressure) ** 2))
    d = model["diagnostics"]
    print("\nTPS LINEAR-SYSTEM DIAGNOSTICS")
    print(f"System size                         : {d['system_size']} x {d['system_size']}")
    print(f"Solver used                         : {d['solver']}")
    print(f"Reciprocal condition estimate       : {d['reciprocal_condition']:.6e}")
    print(f"||A*c-rhs||_2                       : {d['linear_system_residual']:.6e}")
    print(f"Side-condition residual ||P^T*w||_2 : {d['side_condition_residual']:.6e}")
    print(f"Training RMSE                       : {training_rmse:.6e}")
    print(f"TPS bending energy w^T*K*w          : {d['bending_energy']:.6e}")

    # 4. S2 holdout validation and full LOOCV
    loo_prediction, loo_error, interpolation = loocv(x, y, pressure, smoothing)
    table = pd.DataFrame({"Sensor": np.arange(1, len(pressure) + 1), "x": x, "y": y, "Measured": pressure,
                          "Predicted": loo_prediction, "Error": loo_error, "Absolute_error": abs(loo_error),
                          "Inside_remaining_hull": interpolation})
    s2_error = loo_error[validation_sensor]
    print("\nTPS WITHHELD-POINT VALIDATION\n", table.to_string(index=False))
    print(f"Primary interpolation validation: S2 predicted {loo_prediction[1]:.6f} versus measured {pressure[1]:.6f}; absolute error {abs(s2_error):.6f} ({100 * abs(s2_error) / pressure[1]:.3f}%).")
    print(f"Overall five-fold LOOCV RMSE (mixed interpolation/extrapolation): {np.sqrt(np.mean(loo_error**2)):.6f}")

    # 5. p = 0 contour and continuous extrema inside the hull
    zero_segments = bisection_contours(x_grid, y_grid, Z, model, 0, root_tolerance, root_iterations)
    min_value, min_location, min_info = optimise_inside_hull(model, x_grid, y_grid, Z, inside, polygon, "min", stationary_tolerance, newton_iterations)
    max_value, max_location, max_info = optimise_inside_hull(model, x_grid, y_grid, Z, inside, polygon, "max", stationary_tolerance, newton_iterations)
    print("\nPART (b): ZERO-PRESSURE / MINIMUM-PRESSURE RESULT")
    print(f"{len(zero_segments)} p=0 contour segment(s) occur in the displayed rectangle.")
    print(f"Global TPS minimum over sensor hull: p_min = {min_value:.6f} at (x,y) = ({min_location[0]:.6f}, {min_location[1]:.6f}).")
    print(f"Zero/negative pressure predicted inside hull: {min_value <= 0}")
    print("\nPART (c): MAXIMUM-PRESSURE RESULT")
    print(f"p_max = {max_value:.6f} at (x,y) = ({max_location[0]:.6f}, {max_location[1]:.6f}) inside the sensor hull.")
    print(f"Threshold p>{pain_threshold:.1f} exceeded: {max_value > pain_threshold}")

    # 6. Main figures: surface, contours, and LOOCV
    pain_segments = bisection_contours(x_grid, y_grid, Z, model, pain_threshold, root_tolerance, root_iterations)
    figure = plt.figure("TPS pressure surface", figsize=(8, 6)); axis = figure.add_subplot(projection="3d")
    surface = axis.plot_surface(X, Y, Z, cmap="turbo", linewidth=0, alpha=0.94)
    axis.scatter(x, y, pressure, c="k", s=70, edgecolors="white")
    axis.plot(polygon[:, 0], polygon[:, 1], evaluate_tps(model, polygon[:, 0], polygon[:, 1]), "w-", lw=2)
    axis.scatter(*max_location, max_value, marker="p", s=180, c="yellow", edgecolors="red")
    axis.scatter(*min_location, min_value, marker="v", s=100, c="cyan", edgecolors="blue")
    axis.set(xlabel="x", ylabel="y", zlabel="pressure p", title=f"TPS surface, λ = {smoothing:g}")
    figure.colorbar(surface, ax=axis, shrink=0.65)

    plt.figure("TPS pressure contour", figsize=(7, 6)); plt.contourf(X, Y, Z, 22, cmap="turbo")
    plot_segments(zero_segments, "w--", lw=2.4); plot_segments(pain_segments, "m-", lw=2.4)
    plt.plot(polygon[:, 0], polygon[:, 1], "w-", lw=2, label="sensor hull")
    plt.scatter(x, y, c=pressure, s=65, edgecolors="k", label="measurements")
    plt.plot(*max_location, "rp", ms=14, mfc="yellow", label="maximum"); plt.plot(*min_location, "bv", ms=9, mfc="cyan", label="minimum")
    plt.axis("equal"); plt.xlim(x_limits); plt.ylim(y_limits); plt.xlabel("x"); plt.ylabel("y"); plt.title("TPS pressure contours"); plt.colorbar(); plt.legend()

    plt.figure("TPS LOOCV errors", figsize=(7, 4)); plt.bar(np.arange(1, 6), loo_error, color=(.30, .58, .82)); plt.axhline(0, color="k")
    plt.xlabel("left-out sensor"); plt.ylabel("prediction error"); plt.grid(); plt.title(f"TPS leave-one-out errors; S2 |e| = {abs(s2_error):.3f}")

    # 7. Monte Carlo stability under additive Gaussian noise
    noise_std_list = np.ptp(pressure) * noise_fractions
    clean_peak_index = np.argmax(np.where(inside, Z, -np.inf)); clean_peak_location = np.array([X.flat[clean_peak_index], Y.flat[clean_peak_index]])
    noise_rows, mean_field, m2_field = [], np.zeros_like(Z), np.zeros_like(Z)
    for level, noise_std in enumerate(noise_std_list):
        s2_predictions, surface_changes, amplification, peaks, shifts, damage, pain = [], [], [], [], [], [], []
        for trial in range(1, trials + 1):
            noise = noise_std * rng.standard_normal(len(pressure)); noisy = pressure + noise
            keep = np.arange(len(pressure)) != validation_sensor
            s2_model = fit_tps(x[keep], y[keep], noisy[keep], smoothing)
            s2_predictions.append(evaluate_tps(s2_model, x[1], y[1]).item())
            noisy_model = fit_tps(x, y, noisy, smoothing); noisy_z = evaluate_tps(noisy_model, X, Y); values = noisy_z[inside]
            change = np.sqrt(np.mean((values - Z[inside])**2)); surface_changes.append(change); amplification.append(change / max(np.sqrt(np.mean(noise**2)), np.finfo(float).eps))
            peak_index = np.argmax(np.where(inside, noisy_z, -np.inf)); peaks.append(noisy_z.flat[peak_index]); shifts.append(np.hypot(X.flat[peak_index] - clean_peak_location[0], Y.flat[peak_index] - clean_peak_location[1]))
            damage.append(values.min() <= 0); pain.append(values.max() > pain_threshold)
            if level == 1:  # Retain the uncertainty field only for 5% noise
                delta = noisy_z - mean_field; mean_field += delta / trial; m2_field += delta * (noisy_z - mean_field)
        prediction = np.asarray(s2_predictions)
        noise_rows.append([100 * noise_fractions[level], noise_std, trials, prediction.mean(), prediction.std(ddof=1), np.mean(abs(prediction - pressure[1])), percentile(prediction, 2.5), percentile(prediction, 97.5), 100*np.mean(damage), 100*np.mean(pain), np.mean(amplification), percentile(surface_changes, 95), np.std(peaks, ddof=1), np.sqrt(np.mean(np.asarray(shifts)**2))])
    columns = ["Noise_percent_of_range", "Noise_std", "Trials", "S2_mean_prediction", "S2_prediction_std", "S2_MAE", "S2_CI95_low", "S2_CI95_high", "Damage_rate_percent", "Pain_rate_percent", "Mean_noise_amplification", "P95_surface_RMS_change", "Peak_pressure_std", "Peak_location_RMS_shift"]
    stability = pd.DataFrame(noise_rows, columns=columns); print("\nTPS NOISE-STABILITY RESULTS\n", stability.to_string(index=False))

    fig, axes = plt.subplots(2, 2, num="TPS common-noise stability", figsize=(10, 7), constrained_layout=True); percent = 100 * noise_fractions
    axes[0, 0].errorbar(percent, stability.S2_mean_prediction, yerr=1.96*stability.S2_prediction_std, fmt="o-"); axes[0, 0].axhline(pressure[1], color="k", ls="--"); axes[0, 0].set(title="S2 withheld interpolation", xlabel="noise std (% of range)", ylabel="predicted S2 pressure")
    axes[0, 1].plot(percent, stability.Damage_rate_percent, "o-", label="p_min ≤ 0"); axes[0, 1].plot(percent, stability.Pain_rate_percent, "s-", label="p_max > 50"); axes[0, 1].legend(); axes[0, 1].set(title="Decision sensitivity", xlabel="noise std (% of range)", ylabel="decision rate (%)")
    axes[1, 0].plot(percent, stability.Mean_noise_amplification, "o-"); axes[1, 0].set(title="Whole-field amplification", xlabel="noise std (% of range)", ylabel="mean amplification")
    axes[1, 1].plot(percent, stability.Peak_location_RMS_shift, "o-"); axes[1, 1].set(title="Peak-location stability", xlabel="noise std (% of range)", ylabel="RMS shift")
    for axis in axes.flat: axis.grid()
    uncertainty = np.sqrt(m2_field / (trials - 1)); uncertainty[~inside] = np.nan
    plt.figure("TPS pointwise uncertainty at 5% noise", figsize=(7, 6)); plt.contourf(X, Y, uncertainty, 18, cmap="turbo"); plt.plot(polygon[:, 0], polygon[:, 1], "w-", lw=1.8); plt.scatter(x, y, c="k", s=45); plt.axis("equal"); plt.xlim(x_limits); plt.ylim(y_limits); plt.colorbar(); plt.title("Pointwise prediction standard deviation (5% noise)")

    # 8. Sensitivity to the smoothing parameter lambda
    lambdas = np.array([0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1])
    sensitivity_rows, base_rms = [], np.sqrt(np.mean(Z[inside]**2)); keep = np.arange(len(pressure)) != validation_sensor
    for lam in lambdas:
        candidate = fit_tps(x, y, pressure, lam); candidate_z = evaluate_tps(candidate, X, Y); pred, error, _ = loocv(x, y, pressure, lam)
        s2 = evaluate_tps(fit_tps(x[keep], y[keep], pressure[keep], lam), x[1], y[1]).item()
        peak_index = np.argmax(np.where(inside, candidate_z, -np.inf))
        sensitivity_rows.append([lam, np.sqrt(np.mean((evaluate_tps(candidate, x, y)-pressure)**2)), s2, abs(s2-pressure[1]), np.sqrt(np.mean(error**2)), 100*np.sqrt(np.mean((candidate_z[inside]-Z[inside])**2))/base_rms, candidate["diagnostics"]["bending_energy"], candidate_z[inside].min(), candidate_z[inside].max(), np.hypot(X.flat[peak_index]-clean_peak_location[0], Y.flat[peak_index]-clean_peak_location[1]), candidate["diagnostics"]["reciprocal_condition"]])
    sensitivity = pd.DataFrame(sensitivity_rows, columns=["Lambda", "Training_RMSE", "S2_prediction", "S2_absolute_error", "LOOCV_RMSE_mixed", "Relative_surface_change_percent", "Bending_energy", "Minimum_pressure", "Maximum_pressure", "Peak_location_shift", "System_rcond"]); print("\nTPS LAMBDA-SENSITIVITY RESULTS\n", sensitivity.to_string(index=False))
    fig, axes = plt.subplots(2, 2, num="TPS lambda sensitivity", figsize=(10, 7), constrained_layout=True); labels = ["0"] + [f"{value:.0e}" for value in lambdas[1:]]; index = np.arange(len(lambdas))
    axes[0,0].plot(index, sensitivity.Training_RMSE, "o-", label="training RMSE"); axes[0,0].plot(index, sensitivity.S2_absolute_error, "s-", label="S2 absolute error"); axes[0,0].legend(); axes[0,0].set(title="Data fit versus interior validation", ylabel="pressure error")
    axes[0,1].semilogy(index, np.maximum(sensitivity.Bending_energy, np.finfo(float).eps), "o-"); axes[0,1].set(title="Bending energy", ylabel="wᵀKw")
    axes[1,0].plot(index, sensitivity.Relative_surface_change_percent, "o-"); axes[1,0].set(title="Change from exact TPS", ylabel="relative field change (%)")
    axes[1,1].plot(index, sensitivity.Minimum_pressure, "v-", label="minimum"); axes[1,1].plot(index, sensitivity.Maximum_pressure, "^-", label="maximum"); axes[1,1].axhline(0, color="k", ls="--"); axes[1,1].axhline(pain_threshold, color="m", ls="--"); axes[1,1].legend(); axes[1,1].set(title="Part (b)/(c) sensitivity", ylabel="pressure")
    for axis in axes.flat: axis.set_xticks(index, labels, rotation=25); axis.set_xlabel("lambda"); axis.grid()
    plt.show()


if __name__ == "__main__":
    main()
