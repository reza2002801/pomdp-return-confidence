"""
Exact MGF of the accumulated quadratic cost for a continuous-state linear-Gaussian LQG loop.

The stacked vector X_T = (S_0, U_0, ..., S_{T-1}, U_{T-1}) is Gaussian. With a block-diagonal
quadratic form D_T, the cost is a Gaussian quadratic form whose MGF has the closed form
used in cartpole_observer/gaussian_cost.py (verified there against moments and Monte Carlo).

This module generalizes that construction to multi-input plants and H not necessarily I.
"""

import numpy as np

from .plant import closed_loop


def _joint(plant, T):
    """Means of X_t=(S_t,Z_t) and Cov(X_t, X_s), plus V_t -> X_{t+1+k} impulse responses."""
    F, Gw, Gv, Hx, Hv = closed_loop(plant)
    n = plant.n
    m = np.zeros((T + 1, 2 * n))
    m[0, :n] = plant.s0
    m[0, n:] = plant.z0
    Sigma = np.zeros((T + 1, 2 * n, 2 * n))
    noise = Gw @ plant.SW @ Gw.T + Gv @ plant.SV @ Gv.T
    for t in range(T):
        m[t + 1] = F @ m[t]
        Sigma[t + 1] = F @ Sigma[t] @ F.T + noise
    cross = np.zeros((T + 1, T + 1, 2 * n, 2 * n))
    for s in range(T + 1):
        cross[s, s] = Sigma[s]
        acc = Sigma[s]
        for t in range(s + 1, T + 1):
            acc = F @ acc
            cross[t, s] = acc
            cross[s, t] = acc.T
    # Cov(V_t, X_{t+1+k}) = SV Gv^T (F^T)^k
    base = plant.SV @ Gv.T
    lag = np.zeros((T, plant.p, 2 * n))
    acc = base.copy()
    for k in range(T):
        lag[k] = acc
        acc = acc @ F.T
    return m, cross, Hx, Hv, lag


def stage_moments(plant, T):
    """Exact E[C_t] and per-step contributions for t = 0..T."""
    m, cross, Hx, Hv, _ = _joint(plant, T)
    n, nu = plant.n, plant.m
    Qc, Rc = plant.Qc, plant.Rc
    mean = np.zeros(T + 1)
    for t in range(T):
        ms = m[t, :n]
        Ss = cross[t, t, :n, :n]
        mu_u = Hx @ m[t]
        Su = Hx @ cross[t, t] @ Hx.T + Hv @ plant.SV @ Hv.T
        mean[t + 1] = mean[t] + float(
            ms @ Qc @ ms + np.trace(Qc @ Ss) + mu_u @ Rc @ mu_u + np.trace(Rc @ Su)
        )
    return mean


def quadratic_form(plant, T):
    """
    Build C_T = zeta' J zeta + 2 b' zeta + d with zeta ~ N(0,I).

    Returns J, b, d, and the factor L such that Cov(X_T) = L L^T (columns for positive
    eigenvalues only), together with the mean vector of X_T.
    """
    m, cross, Hx, Hv, lag = _joint(plant, T)
    n, nu = plant.n, plant.m
    block = n + nu
    d = block * T
    mu = np.zeros(d)
    Cov = np.zeros((d, d))
    for t in range(T):
        mu[block * t:block * t + n] = m[t, :n]
        mu[block * t + n:block * t + block] = Hx @ m[t]
    for t in range(T):
        for s in range(t, T):
            Bts = np.zeros((block, block))
            Bts[:n, :n] = cross[t, s, :n, :n]
            Bts[:n, n:] = cross[t, s, :n, :] @ Hx.T
            Bts[n:, :n] = Hx @ cross[t, s, :, :n]
            Bts[n:, n:] = Hx @ cross[t, s] @ Hx.T
            if t == s:
                Bts[n:, n:] = Bts[n:, n:] + Hv @ plant.SV @ Hv.T
            else:
                cov_vx = lag[s - t - 1]
                # Cov(U_t, S_s) gains Hv Cov(V_t, X_s)[:, :n]
                Bts[n:, :n] = Hx @ cross[t, s, :, :n] + (Hv @ cov_vx)[:, :n]
                Bts[n:, n:] = Hx @ cross[t, s] @ Hx.T + (Hv @ cov_vx) @ Hx.T
                # Cov(S_t, U_s) stays cross[t,s,:n,:] @ Hx.T (V_s independent of S_t)
            Cov[block * t:block * t + block, block * s:block * s + block] = Bts
            if s != t:
                Cov[block * s:block * s + block, block * t:block * t + block] = Bts.T
    Cov = 0.5 * (Cov + Cov.T)
    evals, evecs = np.linalg.eigh(Cov)
    thresh = 1e-10 * max(float(evals.max()), 1.0)
    keep = evals > thresh
    Lfac = evecs[:, keep] * np.sqrt(evals[keep])

    # D is block-diagonal (Qc, Rc, ...)
    # J = L' D L, b = L' D mu, d = mu' D mu
    Dmu = np.zeros(d)
    DL = np.zeros_like(Lfac)
    for t in range(T):
        sl = slice(block * t, block * t + n)
        ul = slice(block * t + n, block * t + block)
        Dmu[sl] = plant.Qc @ mu[sl]
        Dmu[ul] = plant.Rc @ mu[ul]
        DL[sl, :] = plant.Qc @ Lfac[sl, :]
        DL[ul, :] = plant.Rc @ Lfac[ul, :]
    J = Lfac.T @ DL
    b = Lfac.T @ Dmu
    d0 = float(mu @ Dmu)
    return J, b, d0, Lfac, mu, Cov


def moments_from_form(J, b, d0):
    mean = float(np.trace(J) + d0)
    var = float(2.0 * np.trace(J @ J) + 4.0 * b @ b)
    return mean, var


def log_mgf_from_form(J, b, d0, lams):
    """log M(lambda) with domain I - 2 lambda J > 0."""
    lams = np.asarray(lams, dtype=float)
    out = np.full(len(lams), np.nan)
    valid = np.zeros(len(lams), dtype=bool)
    if J.size == 0:
        # deterministic cost
        out[:] = lams * d0
        valid[:] = True
        return out, valid, np.inf
    evals = np.linalg.eigvalsh(J)
    lam_max = 0.5 / evals.max() if evals.max() > 0 else np.inf
    eye = np.eye(J.shape[0])
    # work in the eigenbasis of J for stability
    je, U = np.linalg.eigh(J)
    bt = U.T @ b
    for i, lam in enumerate(lams):
        if lam > 0 and lam >= lam_max - 1e-15:
            continue
        factors = 1.0 - 2.0 * lam * je
        if np.any(factors <= 0):
            continue
        logdet = float(np.sum(np.log(factors)))
        # b' (I - 2 lam J)^{-1} b
        quad = float(np.sum(bt * bt / factors))
        out[i] = lam * d0 - 0.5 * logdet + 2.0 * lam * lam * quad
        valid[i] = True
    return out, valid, float(lam_max)


def chernoff_bounds(J, b, d0, delta=0.025, n_grid=60):
    """
    Pointwise two-sided bounds with failure probability delta on each side.
    Returns L, U, and diagnostic dict.
    """
    evals = np.linalg.eigvalsh(J) if J.size else np.array([0.0])
    lam_max = 0.5 / evals.max() if evals.max() > 0 else 1e3
    # positive grid: (0, lam_max)
    pos = np.geomspace(1e-4, 0.99 * lam_max, n_grid) if np.isfinite(lam_max) else np.geomspace(1e-4, 10.0, n_grid)
    # negative grid: large |lambda| can make I - 2 lambda J fail when J has negative eigenvalues;
    # for PSD J every lambda < 0 is valid.
    neg = -np.geomspace(1e-4, 1e3, n_grid)
    lams = np.concatenate([neg, pos])
    logm, valid, _ = log_mgf_from_form(J, b, d0, lams)
    log_one_over = np.log(1.0 / delta)

    obj_u = np.where(valid & (lams > 0), (logm + log_one_over) / lams, np.inf)
    j_u = int(np.argmin(obj_u))
    U = float(obj_u[j_u])
    # refine around the best positive lambda
    if np.isfinite(U) and lams[j_u] > 0:
        lo = lams[j_u] / 1.5 if j_u > 0 else lams[j_u] / 2
        hi = min(lams[j_u] * 1.5, 0.999 * lam_max) if np.isfinite(lam_max) else lams[j_u] * 1.5
        refine = np.linspace(lo, hi, 40)
        lm, va, _ = log_mgf_from_form(J, b, d0, refine)
        ou = np.where(va, (lm + log_one_over) / refine, np.inf)
        jr = int(np.argmin(ou))
        if ou[jr] < U:
            U, j_u_lam = float(ou[jr]), float(refine[jr])
        else:
            j_u_lam = float(lams[j_u])
    else:
        j_u_lam = float("nan")

    obj_l = np.where(valid & (lams < 0), (logm + log_one_over) / lams, -np.inf)
    j_l = int(np.argmax(obj_l))
    L = float(obj_l[j_l])
    if np.isfinite(L) and lams[j_l] < 0:
        lo = lams[j_l] * 1.5
        hi = lams[j_l] / 1.5
        refine = np.linspace(lo, hi, 40)
        lm, va, _ = log_mgf_from_form(J, b, d0, refine)
        ol = np.where(va, (lm + log_one_over) / refine, -np.inf)
        jr = int(np.argmax(ol))
        if ol[jr] > L:
            L, j_l_lam = float(ol[jr]), float(refine[jr])
        else:
            j_l_lam = float(lams[j_l])
    else:
        j_l_lam = float("nan")

    return L, U, {
        "lambda_upper": j_u_lam,
        "lambda_lower": j_l_lam,
        "lambda_max_domain": float(lam_max),
        "delta": delta,
    }


def analyze_horizon(plant, T, delta=0.025):
    J, b, d0, *_ = quadratic_form(plant, T)
    mean, var = moments_from_form(J, b, d0)
    stage = stage_moments(plant, T)
    L, U, info = chernoff_bounds(J, b, d0, delta=delta)
    # derivative check at 0
    eps = 1e-4
    lm, va, _ = log_mgf_from_form(J, b, d0, np.array([-eps, 0.0, eps]))
    d1 = (lm[2] - lm[0]) / (2 * eps) if va[0] and va[2] else float("nan")
    d2 = (lm[2] - 2 * lm[1] + lm[0]) / eps ** 2 if va.all() else float("nan")
    return {
        "T": T,
        "mean": mean,
        "var": var,
        "stage_mean": float(stage[T]),
        "mean_gap_stage": abs(mean - float(stage[T])),
        "lower": L,
        "upper": U,
        "mgf_d1": d1,
        "mgf_d2": d2,
        "log_M0": float(lm[1]) if va[1] else float("nan"),
        **info,
        "J": J,
        "b": b,
        "d0": d0,
    }
