"""Monte Carlo of the continuous-state linear-Gaussian LQG loop."""

import numpy as np

from .plant import closed_loop


def simulate(plant, n, T, seed):
    rng = np.random.default_rng(seed)
    cw = np.linalg.cholesky(plant.SW + 1e-18 * np.eye(plant.n))
    cv = np.linalg.cholesky(plant.SV + 1e-18 * np.eye(plant.p))
    s = np.tile(plant.s0, (n, 1))
    z = np.tile(plant.z0, (n, 1))
    S = np.empty((n, T + 1, plant.n))
    Z = np.empty((n, T + 1, plant.n))
    U = np.empty((n, T, plant.m))
    C = np.zeros((n, T + 1))
    S[:, 0], Z[:, 0] = s, z
    for t in range(T):
        v = rng.standard_normal((n, plant.p)) @ cv.T
        w = rng.standard_normal((n, plant.n)) @ cw.T
        y = s @ plant.H.T + v
        s_hat = z + (y - z @ plant.H.T) @ plant.L.T
        u = -(s_hat @ plant.K.T)
        stage = np.einsum("...i,ij,...j->...", s, plant.Qc, s) + np.einsum("...i,ij,...j->...", u, plant.Rc, u)
        C[:, t + 1] = C[:, t] + stage
        z = s_hat @ plant.A.T + u @ plant.B.T
        s = s @ plant.A.T + u @ plant.B.T + w
        S[:, t + 1], Z[:, t + 1], U[:, t] = s, z, u
    return {"S": S, "Z": Z, "U": U, "C": C, "err": S - Z}


def sample_quadratic_form(J, b, d0, n, seed):
    """Independent samples of zeta' J zeta + 2 b' zeta + d0."""
    rng = np.random.default_rng(seed)
    if J.size == 0:
        return np.full(n, d0)
    k = J.shape[0]
    z = rng.standard_normal((n, k))
    # cost = (Jz)' z wait: z' J z + 2 b' z + d
    Jz = z @ J
    return np.einsum("ij,ij->i", z, Jz) + 2.0 * z @ b + d0


def wilson(k, n, z=1.96):
    phat = k / n
    den = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / den
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / den
    return float(center - half), float(center + half)
