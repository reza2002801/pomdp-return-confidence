"""
Polynomial moment bound of the paper (eq. moment-tail-bound) for G_t, t = 1..T:

    P(G_t >= b) <= B_{t,d}(b) = min_a sum_k a_k E[G_t^k]
                  s.t. p_a >= 0 on [L_t, U_t],  p_a >= 1 on [b, U_t].

[L_t, U_t] is the exact range of G_t over positive-probability paths. Everything is
solved for Y_t = (G_t - c_t) / h_t in [-1, 1], whose moments come from the moment
recursion (Proposition moments) with rewards shifted and scaled per horizon, and the
polynomial is written in the Chebyshev basis.

The semi-infinite constraints are handled by cutting planes: start from a coarse grid,
add the most violated points of a fine check grid and re-solve; the remaining
violation on the check grid is added to a_0, so the returned value is an upper bound
up to the check-grid resolution. Without further restriction the optimal polynomial
grows to ~1e5 where Y has (almost) no mass and the LP loses precision, so the
polynomials are additionally restricted to p <= P_CAP on [-1, 1]. Restricting the
feasible set keeps every returned value a valid upper bound, possibly a looser one.
"""

from math import comb

import numpy as np
from numpy.polynomial import chebyshev as C
from scipy.optimize import linprog

P_CAP = 1e3
COARSE_GRID = np.unique(np.concatenate([np.cos(np.linspace(0.0, np.pi, 401)), np.linspace(-1, 1, 401)]))
CHECK_GRID = np.unique(np.concatenate([np.cos(np.linspace(0.0, np.pi, 100001)), np.linspace(-1, 1, 100001)]))


def support_range(P_t, r_t, s0):
    """L[t-1], U[t-1] = min / max of G_t over paths with positive probability."""
    T, n, _ = P_t.shape
    lo = np.full(n, np.inf)
    hi = np.full(n, -np.inf)
    lo[s0] = hi[s0] = 0.0
    L, U = np.zeros(T), np.zeros(T)
    for k in range(T):
        reach = np.isfinite(lo)
        L[k], U[k] = (lo + r_t[k])[reach].min(), (hi + r_t[k])[reach].max()
        edge = P_t[k] > 0
        lo = np.where(edge, (lo + r_t[k])[:, None], np.inf).min(axis=0)
        hi = np.where(edge, (hi + r_t[k])[:, None], -np.inf).max(axis=0)
    return L, U


def scaled_moments(P_t, r_t, s0, t, center, scale, d):
    """E[Y^k], k = 0..d, for Y = (G_t - center) / scale, via the moment recursion."""
    n = P_t.shape[1]
    xi = np.zeros((d + 1, n))
    xi[0, s0] = 1.0
    binom = [[comb(k, j) for j in range(k + 1)] for k in range(d + 1)]
    for k in range(t):
        r = (r_t[k] - center / t) / scale
        rp = r[None, :] ** np.arange(d + 1)[:, None]
        new = np.empty_like(xi)
        for m in range(d + 1):
            new[m] = sum(binom[m][j] * rp[j] * xi[m - j] for j in range(m + 1))
        xi = new @ P_t[k]
    return xi.sum(axis=1)


def cheb_moments(moments_y):
    """E[T_k(Y)] from E[Y^k]."""
    d = len(moments_y) - 1
    return np.array([C.cheb2poly(np.eye(d + 1)[k]) @ moments_y[: k + 1] for k in range(d + 1)])


class TailLP:
    """B_d(beta) >= P(Y >= beta) for Y in [-1, 1] with Chebyshev moments tau[0..d]."""

    def __init__(self, d, tol=1e-6, max_rounds=40, ridge=1e-7):
        self.d = d
        self.tol, self.max_rounds = tol, max_rounds
        self.Vc = C.chebvander(CHECK_GRID, d)
        self.coarse = np.searchsorted(CHECK_GRID, COARSE_GRID).clip(0, len(CHECK_GRID) - 1)
        self.warm = self.coarse
        # ridge * (integral of p over [-1, 1]): where Y has (almost) no mass the LP is
        # degenerate and returns wildly oscillating polynomials; this selects a tame one.
        self.integral = ridge * np.array([0.0 if k % 2 else 2.0 / (1.0 - k * k) for k in range(d + 1)])

    def bound(self, tau, beta):
        if beta <= -1.0:
            return 1.0
        if beta > 1.0:
            return 0.0
        ind = (CHECK_GRID >= beta).astype(float)
        j = np.searchsorted(CHECK_GRID, beta)
        active = np.union1d(self.warm, [max(j - 1, 0), min(j, len(CHECK_GRID) - 1)])
        V_beta = C.chebvander(np.array([beta]), self.d)
        free = [(None, None)] * (self.d + 1)
        a = None
        for _ in range(self.max_rounds):
            V = self.Vc[active]
            res = linprog(tau + self.integral, A_ub=np.vstack([-V, -V_beta, V]),
                          b_ub=np.concatenate([-ind[active], [-1.0], np.full(len(active), P_CAP)]),
                          bounds=free, method="highs")
            if not res.success:
                break
            a = res.x
            slack = self.Vc @ a - ind
            if slack.min() >= -self.tol:
                break
            s = np.concatenate([[np.inf], slack, [np.inf]])
            local_min = np.flatnonzero((s[1:-1] <= s[:-2]) & (s[1:-1] <= s[2:]) & (slack < -self.tol))
            violated = np.flatnonzero(slack < -self.tol)
            spread = violated[np.linspace(0, len(violated) - 1, min(200, len(violated))).astype(int)]
            new = np.concatenate([local_min - 1, local_min, local_min + 1, spread]).clip(0, len(slack) - 1)
            active = np.union1d(active, new)
        if a is None:
            return 1.0
        self.warm = np.union1d(self.coarse, active[np.abs(self.Vc[active] @ a - ind[active]) < 1e-3])
        eps = max(0.0, float(-np.min(self.Vc @ a - ind)), float(1.0 - V_beta @ a))
        return float(np.clip(tau @ a + eps, 0.0, 1.0))

    def endpoint(self, tau, alpha, start, iters=16):
        """Smallest beta in [start, 1] (up to bisection tolerance) with B_d(beta) <= alpha."""
        lo, hi = start, 1.0
        self.warm = self.coarse
        if self.bound(tau, hi) > alpha:
            return 1.0
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if self.bound(tau, mid) <= alpha:
                hi = mid
            else:
                lo = mid
        return hi


def poly_interval_t(tau, d, alpha, lp=None):
    """[lo, hi] in Y units with P(Y > hi) <= alpha and P(Y < lo) <= alpha."""
    lp = lp or TailLP(d)
    tau = tau[: d + 1]
    flip = tau * (-1.0) ** np.arange(d + 1)
    mean_y = tau[1]
    hi = lp.endpoint(tau, alpha, mean_y)
    lo = -lp.endpoint(flip, alpha, -mean_y)
    return lo, hi
