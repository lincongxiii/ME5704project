"""Independent re-implementation of the TPS chapter (no repo code exists)."""
import numpy as np


def phi(r):
    return np.where(r > 0, r * r * np.log(np.where(r > 0, r, 1.0)), 0.0)


def tps_fit(Xt, pt, lam=0.0, s=1.0):
    """Centre on training mean, divide by common length s (as the doc)."""
    m = Xt.mean(0)
    U = (Xt - m) / s
    n = len(U)
    K = phi(np.linalg.norm(U[:, None] - U[None], axis=2))
    Pm = np.column_stack([np.ones(n), U])
    A = np.block([[K + lam * np.eye(n), Pm], [Pm.T, np.zeros((3, 3))]])
    sol = np.linalg.solve(A, np.r_[pt, 0, 0, 0])
    w, c = sol[:n], sol[n:]

    def f(q):
        Q = (np.atleast_2d(q) - m) / s
        return phi(np.linalg.norm(Q[:, None] - U[None], axis=2)) @ w + c[0] + Q @ c[1:]
    return f, w @ K @ w, np.linalg.cond(A)
