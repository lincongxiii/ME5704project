"""Physical contact models for the robot-hand pressure data: Hertz (lead-in) and ring Gaussian (main).

    python run_contact.py            # full run (1000 Monte-Carlo runs per noise level), ~11 minutes
    python run_contact.py --mc 50    # quick run

Part A  Hertz contact: fit (own Levenberg-Marquardt, cross-checked with scipy), proof that no concave
        surface fits the data, and the degenerate "infinite ellipse" limit of the least-squares problem.
Part B  Ring Gaussian contact: all exact fits, selection rule (largest sigma), S2 hold-out and
        leave-one-out, noise Monte Carlo (Gaussian noise, sigma = 2 / 5 / 10 % of the pressure range),
        (b) zero pressure, (c) maximum pressure with several optimization methods, and a test case
        with a known answer.

Outputs: results/*.csv and results/numbers.json;  figures/*.png  (--out DIR writes both under DIR)
"""
import argparse
import csv
import json
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares, linprog

import data
import plotting as P
from models import (HULL, PR, S2, SEED, X, Y, Hertz, Ring, distinct_exact, edge_max, fit,
                    hull_y_interval, ring_hull_max, ring_hull_min, select_largest_sigma, sse)
from numerics.nlls import levenberg_marquardt, multistart_lm
from numerics.optimize import newton_nd, project_to_polygon, projected_steepest_ascent
from numerics.rootfind import bisection, newton_raphson, scan_brackets

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
FIG = ROOT / "figures"
RANGE = float(data.P.max() - data.P.min())          # 41
NOISE_LEVELS = (0.02, 0.05, 0.10)                  # noise sigma as a fraction of the pressure range


# =============================================================================== part A: Hertz
def scipy_hertz_multistart(n_starts=100, seed=SEED, rel_tol=0.01):
    """Cross-check: the same Hertz problem with scipy's trust-region reflective solver.

    Same starting points (seed), bounds, penalty residuals as the own LM; scipy tolerances 1e-10,
    at most 500 function evaluations per run.  Returns (best theta, fraction of runs ending within
    rel_tol of the best cost, names of parameters on a bound).
    """
    lo, hi = Hertz.lower, Hertz.upper
    starts = Hertz.starts(np.random.default_rng(seed), n_starts, X, Y, PR)
    runs = []
    for x0 in starts:
        x0 = np.clip(x0, lo + 1e-12, hi - 1e-12)
        try:
            r = least_squares(lambda t: Hertz.residual(t, X, Y, PR), x0, bounds=(lo, hi), x_scale="jac",
                              xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=500)
        except (ValueError, FloatingPointError):
            continue
        runs.append((float(r.cost), r.x.copy()))
    runs.sort(key=lambda t: t[0])
    best_cost, best = runs[0]
    hit = np.mean([c <= best_cost * (1 + rel_tol) + 1e-10 for c, _ in runs])
    width = hi - lo
    on_bound = [n for n, v, a, b, w in zip(Hertz.names, best, lo, hi, width) if min(v - a, b - v) < 1e-6 * w]
    return best, float(hit), on_bound


def a1_hertz_fit(nums):
    t0 = time.time()
    _, s = fit(Hertz, X, Y, PR, n_starts=100, max_iter=3000)
    t_lm = time.time() - t0
    t0 = time.time()
    ref, ref_hit, ref_bound = scipy_hertz_multistart()
    t_sc = time.time() - t0
    th = s["theta"]
    rows = [{"solver": "Levenberg-Marquardt (own, analytic Jacobian)", **dict(zip(Hertz.names, th)),
             "SSE": sse(Hertz, th), "best_hit_rate": s["best_hit_rate"],
             "params_on_bound": ", ".join(Hertz.names[k] for k in s["at_bound"]), "time_s": t_lm},
            {"solver": "scipy least_squares (trust-region reflective)", **dict(zip(Hertz.names, ref)),
             "SSE": sse(Hertz, ref), "best_hit_rate": ref_hit,
             "params_on_bound": ", ".join(ref_bound), "time_s": t_sc}]
    others = [i for i in range(5) if i != S2]
    lp = linprog(PR[others], A_eq=np.vstack([data.POINTS[others].T, np.ones(4)]),
                 b_eq=[X[S2], Y[S2], 1.0], bounds=[(0, None)] * 4)
    v = HULL[:-1]
    nums["hertz"] = {**dict(zip(Hertz.names, th)), "SSE": sse(Hertz, th), "RMSE": np.sqrt(sse(Hertz, th) / 5),
                     "pred": Hertz.f(th, X, Y).tolist(), "SSE_scipy": sse(Hertz, ref),
                     "concave_bound_S2": lp.fun,
                     "concave_bound_weights": dict(zip([data.NAMES[i] for i in others], lp.x)),
                     "hull_min_vertices": float(Hertz.f(th, v[:, 0], v[:, 1]).min())}
    return th, rows


def a2_hertz_limit(nums):
    """No finite minimizer: profile with the centre y0 fixed, and the 4-parameter limit model."""
    def res_lim(t):
        return np.sqrt(np.maximum(0, t[0] - t[1] * (X - t[2]) ** 2 - t[3] * Y)) - PR
    rng = np.random.default_rng(SEED)
    st = np.column_stack([rng.uniform(500, 3000, 60), rng.uniform(10, 500, 60), rng.uniform(0, 3, 60),
                          rng.uniform(-200, 200, 60)])
    _, lim = multistart_lm(res_lim, st, [0, 0, -5, -1e4], [1e5, 1e5, 7.5, 1e4])
    A, B, xl, C = lim["theta"]
    rows = []
    for y0 in (-1.0, -2.0, -5.0, -10.0, -20.0, -50.0, -100.0, -1000.0):
        # exact Hertz <-> limit mapping as the starting point:  p0^2 = A - C y0/2,  a^2 = p0^2/B,  b^2 = -2 y0 p0^2/C
        p02 = A - C * y0 / 2
        start = np.array([np.sqrt(p02), xl, y0, np.sqrt(p02 / B), np.sqrt(-2 * y0 * p02 / C)])
        lo, hi = Hertz.lower.copy(), Hertz.upper.copy()
        lo[2], hi[2] = y0 - 1e-9, y0 + 1e-9
        hi[0], hi[3], hi[4] = 1e6, 1e6, 1e6
        r = levenberg_marquardt(lambda t: Hertz.residual(t, X, Y, PR), start, lo, hi,
                                jac=lambda t: Hertz.jac(t, X, Y), max_iter=5000)
        rows.append({"y0_fixed": y0, **dict(zip(Hertz.names, r["theta"])), "SSE": sse(Hertz, r["theta"]),
                     "iterations": r["iterations"]})
    nums["hertz_limit"] = {"A": A, "B": B, "x0": xl, "C": C, "SSE": float(np.sum(res_lim(lim["theta"]) ** 2)),
                           "pred": (res_lim(lim["theta"]) + PR).tolist()}
    return rows


# =============================================================================== part B: ring
def b1_solutions(nums, n_starts):
    t0 = time.time()
    runs, s = fit(Ring, X, Y, PR, n_starts=n_starts)
    sols, counts = distinct_exact(Ring, runs, X, Y, PR)
    chosen = select_largest_sigma(sols)
    rows = []
    for t, c in sorted(zip(sols, counts), key=lambda z: -z[0][4]):
        mx, mxx, mxy, where = ring_hull_max(t)
        mn, mnx, mny = ring_hull_min(t)
        rows.append({"selected": t is chosen, **dict(zip(Ring.names, t)), "times_found": c,
                     "hull_max": mx, "hull_max_x": mxx, "hull_max_y": mxy, "max_location": where,
                     "exceeds_50": mx > 50, "hull_min": mn, "hull_min_x": mnx, "hull_min_y": mny})
    nums["ring_solutions"] = {"starts": n_starts, "exact_runs": int(sum(counts)), "distinct": len(sols),
                              "time_s": time.time() - t0, "selected": dict(zip(Ring.names, chosen)),
                              "hull_max_range": [min(r["hull_max"] for r in rows), max(r["hull_max"] for r in rows)],
                              "n_exceeding_50": int(sum(r["exceeds_50"] for r in rows))}
    return chosen, rows


def b2_validation(nums, n_starts):
    """4 training points, 5 parameters: a family of exact fits.  Report its prediction range and the
    prediction of the fit selected by the same rule (largest sigma)."""
    rows = []
    for i in range(5):
        tr = np.array([j for j in range(5) if j != i])
        runs, _ = fit(Ring, X[tr], Y[tr], PR[tr], n_starts=n_starts)
        sols, _ = distinct_exact(Ring, runs, X[tr], Y[tr], PR[tr])
        preds = np.array([Ring.f(t, X[i], Y[i]) for t in sols]) if sols else np.array([np.nan])
        ch = select_largest_sigma(sols, X[tr], Y[tr], PR[tr])
        sel = float(Ring.f(ch, X[i], Y[i])) if ch is not None else np.nan
        rows.append({"left_out": data.NAMES[i], "type": "interpolation" if i == S2 else "extrapolation",
                     "true_p": PR[i], "pred_selected": sel, "abs_error_selected": abs(sel - PR[i]),
                     "sigma_selected": ch[4] if ch is not None else np.nan,
                     "selected_on_sigma_bound": bool(ch is not None and ch[4] > 0.999 * Ring.upper[4]),
                     "n_distinct_exact_fits": len(sols), "pred_min": float(np.nanmin(preds)),
                     "pred_median": float(np.nanmedian(preds)), "pred_max": float(np.nanmax(preds))})
    ext = [r for r in rows if r["type"] == "extrapolation"]
    e = np.array([r["pred_selected"] - r["true_p"] for r in ext])
    nums["ring_validation"] = {"S2": rows[S2], "RMSE_ext": float(np.sqrt(np.mean(e ** 2))),
                               "MAE_ext": float(np.mean(np.abs(e)))}
    return rows


def b3_noise(nums, chosen, n_mc, n_check=30, check_starts=300):
    """Gaussian noise; the selected solution is followed by continuation (warm start from the
    noise-free selected fit).  A subsample is re-solved from scratch to check whether the
    largest-sigma solution is still the continued one."""
    tr = np.array([j for j in range(5) if j != S2])
    runs4, _ = fit(Ring, X[tr], Y[tr], PR[tr], n_starts=1000)
    sols4, _ = distinct_exact(Ring, runs4, X[tr], Y[tr], PR[tr])
    warm_hold = select_largest_sigma(sols4, X[tr], Y[tr], PR[tr])
    rng = np.random.default_rng(SEED)
    rows, raw = [], {}
    for lvl in NOISE_LEVELS:
        sig = lvl * RANGE
        s2, hmax, hmin, exact_ok, same_branch = [], [], [], [], []
        for k in range(n_mc):
            pn = PR + rng.normal(0, sig, 5)
            _, sa = fit(Ring, X, Y, pn, n_starts=0, warm=chosen, max_iter=500)
            _, sh = fit(Ring, X[tr], Y[tr], pn[tr], n_starts=0, warm=warm_hold, max_iter=500)
            s2.append(float(Ring.f(sh["theta"], X[S2], Y[S2])))
            hmax.append(ring_hull_max(sa["theta"])[0])
            hmin.append(float(Ring.f(sa["theta"], *data.hull_grid_points(61)).min()))
            exact_ok.append(sse(Ring, sa["theta"], X, Y, pn) < 1e-6)
            if k < n_check:  # full re-solve: is the continued fit still the largest-sigma exact fit?
                runs, _ = fit(Ring, X, Y, pn, n_starts=check_starts, seed=SEED + k)
                sols, _ = distinct_exact(Ring, runs, X, Y, pn)
                best = select_largest_sigma(sols)
                same_branch.append(best is not None and np.allclose(best, sa["theta"], rtol=1e-3, atol=1e-3))
        s2, hmax, hmin = map(np.array, (s2, hmax, hmin))
        raw[lvl] = {"s2": s2, "hmax": hmax}
        rows.append({"noise_frac_of_range": lvl, "sigma": sig, "runs": n_mc,
                     "S2_mean": s2.mean(), "S2_std": s2.std(), "S2_MAE": np.mean(np.abs(s2 - 17)),
                     "S2_CI95_low": np.percentile(s2, 2.5), "S2_CI95_high": np.percentile(s2, 97.5),
                     "hull_max_mean": hmax.mean(), "hull_max_CI95_low": np.percentile(hmax, 2.5),
                     "hull_max_CI95_high": np.percentile(hmax, 97.5), "frac_hull_max_over_50": np.mean(hmax > 50),
                     "frac_hull_min_le_0": np.mean(hmin <= 0), "frac_exact_fit_kept": np.mean(exact_ok),
                     "branch_check_runs": len(same_branch),
                     "frac_selected_branch_unchanged": np.mean(same_branch) if same_branch else np.nan})
    nums["ring_noise"] = rows
    return rows, raw


def b4_zero(nums, t):
    """(b): with no offset the ring is positive everywhere; confirm numerically inside the hull."""
    n_brackets = 0
    for x in np.linspace(HULL[:, 0].min(), HULL[:, 0].max(), 401):
        iv = hull_y_interval(x)
        if iv and iv[1] > iv[0]:
            n_brackets += len(scan_brackets(lambda yy: Ring.f(t, x, yy), iv[0], iv[1], 200))
    mn, mnx, mny = ring_hull_min(t)
    gx, gy = data.region_grid(201)
    zr = Ring.f(t, gx, gy)
    k = np.unravel_index(np.argmin(zr), zr.shape)
    nums["ring_zero"] = {"hull_brackets_found": n_brackets, "hull_min": mn, "hull_min_at": [mnx, mny],
                         "region_min": float(zr[k]), "region_min_at": [float(gx[k]), float(gy[k])],
                         "frac_hull_below_lowest_datum_2": float(np.mean(Ring.f(t, *data.hull_grid_points(161)) < 2))}


def b5_max(nums, t):
    """(c): crest circle, edge parabolic interpolation, multidimensional Newton, projected steepest ascent."""
    fun = lambda x, y: Ring.f(t, np.atleast_1d(x), np.atleast_1d(y))[0]  # noqa: E731
    grad = lambda v: Ring.grad_xy(t, v)  # noqa: E731

    def hess(v, h=1e-6):
        return np.column_stack([(grad(v + h * e) - grad(v - h * e)) / (2 * h) for e in np.eye(2)])

    edge_rows = edge_max(fun)
    mx, mxx, mxy, where = ring_hull_max(t)
    A, x0, y0, R, s = t
    ang = np.linspace(0, 2 * np.pi, 3600, endpoint=False)
    inside = data.in_hull(x0 + R * np.cos(ang), y0 + R * np.sin(ang))
    newton_rows, psa_rows = [], []
    starts = [(0.5, 0.5), (1.0, 1.0), (1.5, 0.5), (1.5, 1.5), (2.0, 1.0)]
    for st in starts:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                r = newton_nd(grad, hess, st, max_iter=100)
                eig = r["eig"]
                row = {"start": st, "x": r["x"][0], "y": r["x"][1], "p": fun(*r["x"]), "iterations": r["iterations"],
                       "converged": r["converged"], "kind": r["kind"], "hessian_eigenvalues": np.round(eig, 4).tolist(),
                       "radius_from_centre": float(np.hypot(r["x"][0] - x0, r["x"][1] - y0)),
                       "inside_hull": bool(data.in_hull(*r["x"]))}
            except np.linalg.LinAlgError:
                row = {"start": st, "converged": False, "kind": "singular Hessian"}
        newton_rows.append(row)
        rp = projected_steepest_ascent(lambda v: fun(*v), grad, lambda v: project_to_polygon(v, HULL),
                                       np.array(st), alpha0=1.0)
        psa_rows.append({"start": st, "x": rp["x"][0], "y": rp["x"][1], "p": rp["f"], "gap_to_max": mx - rp["f"],
                         "radius_from_centre": float(np.hypot(rp["x"][0] - x0, rp["x"][1] - y0)),
                         "iterations": rp["iterations"]})
    step_rows = []
    for alpha in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0):
        rp = projected_steepest_ascent(lambda v: fun(*v), grad, lambda v: project_to_polygon(v, HULL),
                                       np.array([1.0, 1.0]), alpha0=alpha)
        step_rows.append({"alpha0": alpha, "x": rp["x"][0], "y": rp["x"][1], "p": rp["f"], "iterations": rp["iterations"]})
    nums["ring_max"] = {"hull_max": mx, "at": [mxx, mxy], "where": where, "exceeds_50": mx > 50,
                        "crest_arc_fraction_inside_hull": float(inside.mean()),
                        "edge_max": max(r["p_max"] for r in edge_rows)}
    return edge_rows, newton_rows, psa_rows, step_rows


def b6_test_case(nums):
    """Known answer: synthetic ring data (A, x0, y0, R, sigma) = (50, 1.2, 1.1, 1.0, 0.5).

    parameter recovery at the five sensor positions (the truth must be one of the exact fits) and
    with 12 extra points (unique fit); roots of p = A/2 on the line y = y0 (analytic: x0 +- (R +-
    sigma sqrt(2 ln 2))); maximum = A on the crest circle r = R.
    """
    true = np.array([50.0, 1.2, 1.1, 1.0, 0.5])
    runs, _ = fit(Ring, X, Y, Ring.f(true, X, Y), n_starts=1000)
    sols, _ = distinct_exact(Ring, runs, X, Y, Ring.f(true, X, Y))
    err5 = min(np.abs(t - true).max() for t in sols) if sols else np.nan
    gx, gy = np.meshgrid(np.linspace(0.2, 2.3, 4), np.linspace(0.2, 2.2, 3))
    xs, ys = np.concatenate([X, gx.ravel()]), np.concatenate([Y, gy.ravel()])
    runs17, s17 = fit(Ring, xs, ys, Ring.f(true, xs, ys), n_starts=200)
    sols17, _ = distinct_exact(Ring, runs17, xs, ys, Ring.f(true, xs, ys))
    level = 25.0
    g = lambda x: Ring.f(true, x, 1.1) - level  # noqa: E731
    dg = lambda x: (g(x + 1e-7) - g(x - 1e-7)) / 2e-7  # noqa: E731
    half = 0.5 * np.sqrt(2 * np.log(2))
    exact_roots = sorted([1.2 - 1.0 - half, 1.2 - 1.0 + half, 1.2 + 1.0 - half, 1.2 + 1.0 + half])
    root_rows = []
    for ex in exact_roots:
        br = next((b for b in scan_brackets(g, -1.0, 3.5, 900) if b[0] - 1e-9 <= ex <= b[1] + 1e-9), None)
        rb, kb = bisection(g, *br)
        rn, kn, ok = newton_raphson(g, dg, ex + 0.05)
        root_rows += [{"exact_root": ex, "method": "Bisection", "value": rb, "abs_error": abs(rb - ex), "iterations": kb},
                      {"exact_root": ex, "method": "Newton-Raphson", "value": rn, "abs_error": abs(rn - ex),
                       "iterations": kn, "converged": ok}]
    grad = lambda v: Ring.grad_xy(true, v)  # noqa: E731
    opt_rows = []
    for st in [(0.5, 0.5), (1.5, 1.5), (2.0, 1.0), (0.8, 2.0)]:
        # alpha0 = 0.001: near the crest |grad p| ~ A (r - R) / sigma^2, so a larger step overshoots the
        # crest and the backtracking oscillates around it
        r = projected_steepest_ascent(lambda v: Ring.f(true, v[0], v[1]), grad, lambda v: v, np.array(st),
                                      alpha0=0.001, tol=1e-13, max_iter=20000)
        opt_rows.append({"start": st, "x": r["x"][0], "y": r["x"][1], "p": r["f"], "abs_error_p": abs(r["f"] - 50.0),
                         "radius_error": abs(np.hypot(r["x"][0] - 1.2, r["x"][1] - 1.1) - 1.0), "iterations": r["iterations"]})
    nums["ring_test"] = {"theta_true": true.tolist(), "exact_fits_5_points": len(sols), "max_param_error_5": err5,
                         "exact_fits_17_points": len(sols17),
                         "max_param_error_17": float(np.abs(s17["theta"] - true).max())}
    return root_rows, opt_rows


# =============================================================================== figures
def fig_hertz_limit(rows, nums):
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.plot([-r["y0_fixed"] for r in rows], [r["SSE"] for r in rows], "o-", color=P.BLUE, lw=2,
            label="Hertz fit with the centre y0 fixed")
    ax.axhline(nums["hertz_limit"]["SSE"], color=P.RED, ls="--", lw=1.3,
               label=f"limit model sqrt(A - B(x-x0)^2 - Cy): {nums['hertz_limit']['SSE']:.2f}")
    ax.set_xscale("log")
    ax.set_xlabel("distance of the ellipse centre below the region, -y0")
    ax.set_ylabel("sum of squared residuals")
    ax.set_title("Hertz: no finite best fit - the error keeps falling as the ellipse moves away", fontsize=10)
    ax.grid(True)
    ax.legend(fontsize=8)
    P.save(fig, FIG / "hertz_profile_y0.png")


class RingSurface:
    """A fitted ring as a drawable surface (predict(x, y))."""

    def __init__(self, t):
        self.t = t

    def predict(self, x, y):
        return Ring.f(self.t, np.asarray(x, float), np.asarray(y, float))


class HertzSurface:
    """A fitted Hertz model as a drawable surface; ``ellipse`` gives the exact zero line."""

    def __init__(self, t):
        self.t = t

    def predict(self, x, y):
        return Hertz.f(self.t, np.asarray(x, float), np.asarray(y, float))

    def ellipse(self):
        return tuple(float(v) for v in self.t[1:])


def fig_ring_solutions(sol_rows):
    n = len(sol_rows)
    fig, axes = plt.subplots(1, n, figsize=(4.3 * n, 4.6), squeeze=False)
    for ax, r in zip(axes[0], sol_rows):
        t = np.array([r[k] for k in Ring.names])
        title = f"{'SELECTED: ' if r['selected'] else ''}sigma={t[4]:.3f}, max={r['hull_max']:.1f}"
        P.draw_contour(ax, RingSurface(t), title, colorbar=False, small=True,
                       hull_max=(r["hull_max"], r["hull_max_x"], r["hull_max_y"]))
        ang = np.linspace(0, 2 * np.pi, 400)
        ax.plot(t[1] + t[3] * np.cos(ang), t[2] + t[3] * np.sin(ang), color=P.ORANGE, lw=1, ls=":")
    fig.suptitle("Ring Gaussian: all exact fits through the five sensors (dotted: crest circle r = R)", fontsize=11)
    fig.tight_layout()
    P.save(fig, FIG / "ring_all_exact_fits.png")


def fig_ring_selected(t):
    fig = plt.figure(figsize=(12.5, 5.8))
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    P.draw_surface(ax3, RingSurface(t), "3-D surface: ring Gaussian (selected fit)")
    ax2 = fig.add_subplot(1, 2, 2)
    mx, mxx, mxy, _ = ring_hull_max(t)
    P.draw_contour(ax2, RingSurface(t), "Contour map: ring Gaussian (selected fit)", hull_max=(mx, mxx, mxy))
    ang = np.linspace(0, 2 * np.pi, 400)
    ax2.plot(t[1] + t[3] * np.cos(ang), t[2] + t[3] * np.sin(ang), color=P.ORANGE, lw=1.2, ls=":")
    P.save(fig, FIG / "ring_selected_fit.png")


def fig_noise(raw):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    lv = sorted(raw)
    for ax, key, ref, lab in ((axes[0], "s2", 17, "actual S2 = 17"), (axes[1], "hmax", 50, "pain threshold = 50")):
        ax.boxplot([raw[l][key] for l in lv], showfliers=False, medianprops=dict(color=P.ORANGE))
        ax.set_xticks(range(1, len(lv) + 1), [f"{l:.0%}" for l in lv])
        ax.axhline(ref, color=P.BLUE, ls="--", label=lab)
        ax.set_xlabel("noise level (sigma as % of the pressure range 41)")
        ax.grid(True)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("S2 predicted from S1, S3, S4, S5")
    axes[1].set_ylabel("maximum pressure inside the hull")
    fig.suptitle("Ring Gaussian under Gaussian measurement noise (selected fit followed by continuation)", fontsize=11)
    fig.tight_layout()
    P.save(fig, FIG / "ring_noise.png")


# =============================================================================== main
def write_csv(name, rows):
    """Write a list of dicts to results/<name> (numbers rounded to 6 significant digits)."""
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))

    def fmt(v):
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else "no"
        if isinstance(v, (float, np.floating)):
            return "" if not np.isfinite(v) else f"{v:.6g}"
        return v

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r.get(k, "")) for k in fields})


def main():
    global OUT, FIG
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", type=int, default=1000, help="Monte-Carlo runs per noise level")
    ap.add_argument("--starts", type=int, default=2000, help="multi-start runs for the ring fit")
    ap.add_argument("--out", default=None, help="write results/ and figures/ under this folder instead")
    args = ap.parse_args()
    if args.out:
        OUT, FIG = Path(args.out) / "results", Path(args.out) / "figures"
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    nums = {"noise_runs": args.mc}
    print("[A1] Hertz fit and concavity bound")
    th, rows = a1_hertz_fit(nums)
    write_csv("a1_hertz_fit.csv", rows)
    print("[A2] Hertz degenerate limit")
    prof = a2_hertz_limit(nums)
    write_csv("a2_hertz_profile_y0.csv", prof)
    print(f"[B1] ring: exact fits ({args.starts} starts)")
    chosen, sol_rows = b1_solutions(nums, args.starts)
    write_csv("b1_ring_exact_fits.csv", sol_rows)
    print("[B2] ring: hold-out / leave-one-out")
    write_csv("b2_ring_validation.csv", b2_validation(nums, 1000))
    print(f"[B3] ring: noise Monte Carlo ({args.mc} runs per level)")
    noise, raw = b3_noise(nums, chosen, args.mc)
    write_csv("b3_ring_noise.csv", noise)
    print("[B4] ring: zero pressure")
    b4_zero(nums, chosen)
    print("[B5] ring: maximum pressure")
    edges, newton, psa, steps = b5_max(nums, chosen)
    write_csv("b5_ring_max_edges.csv", edges)
    write_csv("b5_ring_max_newton.csv", newton)
    write_csv("b5_ring_max_projected_ascent.csv", psa)
    write_csv("b5_ring_max_step_size.csv", steps)
    print("[B6] test case")
    roots, opts = b6_test_case(nums)
    write_csv("b6_test_roots.csv", roots)
    write_csv("b6_test_optimizer.csv", opts)
    print("[fig] figures")
    info = [f"p0 = {th[0]:.3g}, centre = ({th[1]:.3g}, {th[2]:.3g}), a = {th[3]:.3g}, b = {th[4]:.3g}",
            f"RMS residual = {nums['hertz']['RMSE']:.3g};  best-hit {rows[0]['best_hit_rate']:.0%} of 100 starts"]
    if rows[0]["params_on_bound"]:
        info.append(f"best fit on the parameter bound ({rows[0]['params_on_bound']}): "
                    "the ellipse wants to grow without limit")
    P.plot_hertz(HertzSurface(th), info, FIG / "hertz_surface_and_ellipse.png")
    fig_hertz_limit(prof, nums)
    fig_ring_solutions(sol_rows)
    fig_ring_selected(chosen)
    fig_noise(raw)
    (OUT / "numbers.json").write_text(json.dumps(nums, indent=2, default=float), encoding="utf-8")
    print(f"[done] {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
