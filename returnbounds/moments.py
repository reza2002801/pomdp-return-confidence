"""
Paper moment recursion and exact cost support on a FiniteModel edge list.

Uses Proposition (moments): xi^{(k)} forward with P^{(j)} = edges weighted by p * c^j.
Support [L_t, U_t] = min/max of G_t over positive-probability paths.
"""

from math import comb

import numpy as np
import scipy.sparse as sp

from . import polybound as pb


def _edge_matrix(fm, weights):
    """Sparse matrix M with M[dst, src] += weight (for xi' = M @ xi column vectors)."""
    return sp.csr_matrix(
        (weights, (fm.dst, fm.src)), shape=(fm.n_states, fm.n_states)
    )


def support_interval(fm, T):
    """
    L[t], U[t] for t=1..T: min/max of G_t = sum of t stage costs on positive-prob paths.
    L[0]=U[0]=0.
    """
    n = fm.n_states
    lo = np.full(n, np.inf)
    hi = np.full(n, -np.inf)
    lo[fm.x0] = hi[fm.x0] = 0.0
    L = np.zeros(T + 1)
    U = np.zeros(T + 1)
    # Pre-group edges by destination for vectorized updates is harder; use maximum.at
    src, dst, cost, prob = fm.src, fm.dst, fm.cost, fm.prob
    live = prob > 0
    src, dst, cost = src[live], dst[live], cost[live]
    for t in range(T):
        reach = np.isfinite(lo)
        # stage cost from current state: G_{t+1} = G_t + c(edge)
        # First record range of G before moving? Paper G_t is return at time t.
        # Cost accrued when taking edge at time t contributes to G_{t+1}.
        # At horizon t (after t steps), G_t has t costs. Match mgf indexing:
        # M_0=1 (G_0=0), M_1 after one tilted step includes first cost.
        cand_lo = lo[src] + cost
        cand_hi = hi[src] + cost
        # only from reachable src
        ok = np.isfinite(lo[src])
        nxt_lo = np.full(n, np.inf)
        nxt_hi = np.full(n, -np.inf)
        np.minimum.at(nxt_lo, dst[ok], cand_lo[ok])
        np.maximum.at(nxt_hi, dst[ok], cand_hi[ok])
        lo, hi = nxt_lo, nxt_hi
        reach = np.isfinite(lo)
        L[t + 1] = lo[reach].min()
        U[t + 1] = hi[reach].max()
    return L, U


def raw_moments(fm, T, degree, stage_cost=None):
    """
    m[t, k] = E[G_t^k] for t=0..T, k=0..degree via the paper moment recursion.

    If stage_cost is given (length n_edges), it replaces fm.cost. Use this with
    shifted/scaled stage costs so Y=(G-c)/h has O(1) moments (avoids float blow-up
    when |G| is hundreds and degree is large).
    """
    degree = int(degree)
    n = fm.n_states
    cost = fm.cost if stage_cost is None else np.asarray(stage_cost, dtype=float)
    Pj = []
    for j in range(degree + 1):
        w = fm.prob * (cost ** j)
        Pj.append(_edge_matrix(fm, w))
    xi = [np.zeros(n) for _ in range(degree + 1)]
    xi[0][fm.x0] = 1.0
    m = np.zeros((T + 1, degree + 1))
    m[0, 0] = 1.0
    binom = [[comb(k, j) for j in range(k + 1)] for k in range(degree + 1)]
    for t in range(T):
        new = [np.zeros(n) for _ in range(degree + 1)]
        for k in range(degree + 1):
            acc = np.zeros(n)
            for j in range(k + 1):
                acc = acc + binom[k][j] * (Pj[j] @ xi[k - j])
            new[k] = acc
        xi = new
        for k in range(degree + 1):
            m[t + 1, k] = float(xi[k].sum())
    return m


def scaled_moments_at_t(fm, t, degree, center, scale):
    """
    Moments of Y=(G_t - center)/scale via the paper recursion with per-step
    costs (r - center/t)/scale, matching poly_bounds.scaled_moments.
    Numbers stay O(1) on the support map to [-1,1].
    """
    t = int(t)
    if t <= 0:
        raise ValueError("t must be positive")
    if scale <= 0:
        raise ValueError("scale must be positive")
    stage = (fm.cost - center / t) / scale
    return raw_moments(fm, t, degree, stage_cost=stage)[t]


def _poly_one_scaled(args):
    """Worker: needs edge arrays to rebuild a light FiniteModel-like namespace."""
    (t, lo_s, hi_s, degree, alpha,
     src, dst, prob, cost, x0, n_states) = args
    from types import SimpleNamespace
    fm = SimpleNamespace(
        src=src, dst=dst, prob=prob, cost=cost, x0=int(x0), n_states=int(n_states),
    )
    if not np.isfinite(lo_s) or not np.isfinite(hi_s) or hi_s <= lo_s:
        # fall back: first moment on original cost
        mu = float(raw_moments(fm, t, 1)[t, 1])
        return t, mu, mu
    c = 0.5 * (lo_s + hi_s)
    h = 0.5 * (hi_s - lo_s)
    my = scaled_moments_at_t(fm, t, degree, c, h)
    # my[0] should be ~1; my are already E[Y^k]
    tau = pb.cheb_moments(my)
    y_lo, y_hi = pb.poly_interval_t(tau, degree, alpha)
    return t, float(np.clip(c + h * y_lo, lo_s, hi_s)), float(np.clip(c + h * y_hi, lo_s, hi_s))


def poly_intervals(fm, T, degree, alpha=0.025, times=None, log=print, n_workers=None):
    """
    Two-sided polynomial intervals using exact support and scaled paper moments.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    times = list(range(1, T + 1) if times is None else times)
    log("  exact support DP")
    Lsup, Usup = support_interval(fm, T)
    # Low-order raw moments for reporting / mean cross-check (degree 2 is cheap & safe)
    log("  paper moment recursion (orders 0..2 for diagnostics)")
    moments_diag = raw_moments(fm, T, min(2, degree))
    moments = np.zeros((T + 1, degree + 1))
    moments[:, : moments_diag.shape[1]] = moments_diag

    lower = np.full(T + 1, np.nan)
    upper = np.full(T + 1, np.nan)
    edge = (fm.src, fm.dst, fm.prob, fm.cost, fm.x0, fm.n_states)
    payloads = [
        (int(t), float(Lsup[t]), float(Usup[t]), int(degree), float(alpha), *edge)
        for t in times
    ]
    if n_workers is None:
        n_workers = max(1, min(len(payloads), (os.cpu_count() or 2) - 1))
    log(f"  polynomial LPs for {len(payloads)} horizons (d={degree}, scaled moments, workers={n_workers})")
    if n_workers <= 1:
        for p in payloads:
            t, L, U = _poly_one_scaled(p)
            lower[t], upper[t] = L, U
            # store scaled first moment check into moments[t,1] already from diag
            if log and (t == times[0] or t == times[-1] or t % 25 == 0):
                log(f"  poly T={t}: L={L:.4f} U={U:.4f}  width={U-L:.4f}")
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futs = [pool.submit(_poly_one_scaled, p) for p in payloads]
            done = 0
            for fut in as_completed(futs):
                t, L, U = fut.result()
                lower[t], upper[t] = L, U
                done += 1
                if log and (done == 1 or done == len(payloads) or t % 25 == 0):
                    log(f"  poly T={t}: L={L:.4f} U={U:.4f}  width={U-L:.4f}  ({done}/{len(payloads)})")
    return lower, upper, Lsup, Usup, moments
