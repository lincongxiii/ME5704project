"""Nonlinear least squares by Levenberg-Marquardt (own implementation) with multi-start.

Minimize  S(theta) = sum_i r_i(theta)^2.  With the Jacobian J = dr/dtheta, one LM step solves

    (J^T J + mu * diag(J^T J)) delta = -J^T r,

i.e. Gauss-Newton (mu -> 0) blended with scaled gradient descent (mu large).  A step that lowers
S is accepted and mu is divided by 3; otherwise mu is multiplied by 4 and the step retried.
Box bounds (active set): a parameter sitting on a bound whose gradient pushes it outward is frozen
for that step (the system is solved for the free parameters only), and the trial point is clipped
to [lower, upper].  Without freezing, a clipped step keeps pointing out of the box and the method
creeps along the bound.
Stops when the relative decrease of S or the step length falls below ``tol``.
"""
import numpy as np


def numerical_jacobian(residual, theta, h=1e-7):
    r0 = residual(theta)
    J = np.empty((len(r0), len(theta)))
    for k in range(len(theta)):
        d = np.zeros_like(theta)
        d[k] = h * max(1.0, abs(theta[k]))
        J[:, k] = (residual(theta + d) - residual(theta - d)) / (2 * d[k])
    return J


def levenberg_marquardt(residual, theta0, lower, upper, jac=None, tol=1e-12, max_iter=500, mu0=1e-3):
    """Returns dict(theta, cost, iterations, converged); cost = sum r^2."""
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    th = np.clip(np.asarray(theta0, float), lower, upper)
    r = residual(th)
    S = float(r @ r)
    mu = mu0
    for it in range(1, max_iter + 1):
        J = jac(th) if jac is not None else numerical_jacobian(residual, th)
        A = J.T @ J
        g = J.T @ r
        width = upper - lower
        frozen = ((th - lower <= 1e-10 * width) & (g > 0)) | ((upper - th <= 1e-10 * width) & (g < 0))
        free = ~frozen
        if not free.any():
            return {"theta": th, "cost": S, "iterations": it, "converged": True}
        Af, gf = A[np.ix_(free, free)], g[free]
        accepted = False
        for _ in range(30):
            try:
                df = np.linalg.solve(Af + mu * np.diag(np.maximum(np.diag(Af), 1e-12)), -gf)
            except np.linalg.LinAlgError:
                mu *= 4
                continue
            delta = np.zeros_like(th)
            delta[free] = df
            tn = np.clip(th + delta, lower, upper)
            rn = residual(tn)
            Sn = float(rn @ rn)
            if np.isfinite(Sn) and Sn < S:
                accepted = True
                break
            mu *= 4
        if not accepted:
            return {"theta": th, "cost": S, "iterations": it, "converged": True}
        step = np.linalg.norm(tn - th) / (np.linalg.norm(th) + 1e-12)
        dec = (S - Sn) / max(S, 1e-300)
        th, r, S, mu = tn, rn, Sn, mu / 3
        if dec < tol or step < tol or S < 1e-28:
            return {"theta": th, "cost": S, "iterations": it, "converged": True}
    return {"theta": th, "cost": S, "iterations": max_iter, "converged": False}


def multistart_lm(residual, starts, lower, upper, jac=None, rel_tol=0.01, **kw):
    """Run LM from every start.  Returns (runs sorted by cost, summary dict).

    summary: best theta / cost, convergence rate, fraction of runs within rel_tol of the best cost,
    parameter spread among those near-best runs, and parameters of the best run on a bound.
    """
    runs = [levenberg_marquardt(residual, s, lower, upper, jac=jac, **kw) for s in starts]
    runs.sort(key=lambda r: r["cost"])
    best = runs[0]
    lim = best["cost"] * (1 + rel_tol) + 1e-12
    near = np.array([r["theta"] for r in runs if r["cost"] <= lim])
    width = np.asarray(upper, float) - np.asarray(lower, float)
    at_bound = [k for k, (v, lo, hi, w) in enumerate(zip(best["theta"], lower, upper, width))
                if min(v - lo, hi - v) < 1e-6 * w]
    summary = {"theta": best["theta"], "cost": best["cost"], "n_starts": len(runs),
               "convergence_rate": np.mean([r["converged"] for r in runs]),
               "best_hit_rate": len(near) / len(runs), "spread": near.max(0) - near.min(0),
               "at_bound": at_bound, "median_iterations": float(np.median([r["iterations"] for r in runs]))}
    return runs, summary
