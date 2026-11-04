"""
Exact cost MGF of the finite model and the resulting Chernoff upper bounds.

K_lambda(x, x') = sum over merged edges e: x -> x' of p_e exp(lambda c_e), stored as a sparse
matrix whose sparsity pattern (unique (x, x') pairs) is computed once; for every lambda only the
data vector is recomputed. Row-vector recursion xi_{t+1} = xi_t K_lambda, M_t = xi_t 1.

Numerical scheme (same convention as ``bounds.log_mgf`` of the paper code): xi_t = exp(a_t) w_t
with max(w_t) = 1, and the tilt of each row is shifted by m_x = max_e lambda c_e over the edges
leaving x, so every exponential that is evaluated is ideally <= 1. Tiny shifted weights and tiny
edge factors are rounded UP to e^CLAMP. Every term is nonnegative, so rounding up can only
overestimate M_t; Chernoff levels computed from an overestimated M_t remain valid. The ``valid``
mask now only guards against non-finite values (sticky once False for a lambda).
"""

import time

import numpy as np
import scipy.sparse as sp

# Each clamped factor is >= e^-300, so a product of two factors times an edge probability
# stays well above double-precision underflow.
CLAMP = -300.0


class TiltedOperator:
    def __init__(self, fm):
        self.fm = fm
        n = fm.n_states
        pair = fm.src * n + fm.dst
        self.pairs, self.slot = np.unique(pair, return_inverse=True)
        self.rows, self.cols = self.pairs // n, self.pairs % n
        self.n = n
        self.cmax = np.full(n, -np.inf)
        self.cmin = np.full(n, np.inf)
        np.maximum.at(self.cmax, fm.src, fm.cost)
        np.minimum.at(self.cmin, fm.src, fm.cost)
        # transpose pattern so that xi_{t+1} = K^T xi_t is a CSR mat-vec
        self._KT0 = sp.csr_matrix((np.ones(len(self.pairs)), (self.cols, self.rows)), shape=(n, n))
        self._order = self._csr_order()

    def _csr_order(self):
        """Permutation mapping slot data to the CSR data array of the transposed pattern."""
        probe = sp.csr_matrix((np.arange(1, len(self.pairs) + 1, dtype=float), (self.cols, self.rows)),
                              shape=(self.n, self.n))
        return probe.data.astype(np.int64) - 1

    def matrix_T(self, lam, shifted=True):
        """K_lambda^T (rows shifted by exp(-m_x) if shifted)."""
        fm = self.fm
        shift = (lam * (self.cmax if lam >= 0 else self.cmin))[fm.src] if shifted else 0.0
        expo = lam * fm.cost - shift
        data = np.bincount(self.slot, weights=fm.prob * np.exp(np.maximum(expo, CLAMP)),
                           minlength=len(self.pairs))
        KT = self._KT0.copy()
        KT.data = data[self._order]
        return KT

    def row_shift(self, lam):
        m = lam * (self.cmax if lam >= 0 else self.cmin)
        return np.where(np.isfinite(m), m, 0.0)


def initial_vector(n, init):
    """init: an augmented-state index (deterministic start) or a probability vector."""
    if np.ndim(init) == 0:
        nu = np.zeros(n)
        nu[int(init)] = 1.0
        return nu
    return np.asarray(init, dtype=float)


def reachable_sets(op, x0, T):
    """reach[t] = states with positive probability at time t (identical for every lambda)."""
    KT = op._KT0
    r = initial_vector(op.n, x0) > 0
    out = [r]
    for _ in range(T):
        r = (KT @ r.astype(float)) > 0
        out.append(r)
    return out


def check_stochastic(op, reach, T, tol=1e-9):
    """Assert every state reachable at some t < T has outgoing probabilities summing to 1."""
    fm = op.fm
    row_sum = np.bincount(fm.src, weights=fm.prob, minlength=op.n)
    mask = np.zeros(op.n, dtype=bool)
    for t in range(T):
        mask |= reach[t]
    bad = mask & (np.abs(row_sum - 1.0) > tol)
    if not bad.any():
        return
    n_bad = int(bad.sum())
    diffs = np.abs(row_sum[bad] - 1.0)
    worst_sum = float(row_sum[bad][np.argmax(diffs)])
    raise AssertionError(
        f"non-stochastic rows: {n_bad} states; worst row sum = {worst_sum}"
    )


def log_mgf(op, lams, T, x0, reach):
    """log M_t(lambda) for t = 0..T and all lambdas; valid[t, j] sticky on non-finite values."""
    out = np.zeros((T + 1, len(lams)))
    valid = np.ones((T + 1, len(lams)), dtype=bool)
    nu0 = initial_vector(op.n, x0)
    for j, lam in enumerate(lams):
        KT, m = op.matrix_T(lam), op.row_shift(lam)
        w = nu0 / nu0.max()
        a, ok = float(np.log(nu0.max())), True
        for t in range(T):
            with np.errstate(divide="ignore"):
                ell = np.log(w) + m
            b = ell[reach[t]].max()
            q = np.where(reach[t], np.exp(np.maximum(ell - b, CLAMP)), 0.0)
            v = KT @ q
            d = v.max()
            w = v / d
            a += b + np.log(d)
            out[t + 1, j] = a + np.log(w.sum())
            ok = ok and bool(np.isfinite(out[t + 1, j]))
            valid[t + 1, j] = ok
    return out, valid


def exact_moments(op, x0, T):
    """E[C_t], Var[C_t] for t = 0..T and occupancies, by the forward moment recursion."""
    fm = op.fm
    w = lambda x: sp.csr_matrix((np.bincount(op.slot, weights=x, minlength=len(op.pairs)),
                                 (op.cols, op.rows)), shape=(op.n, op.n))
    P0, P1, P2 = w(fm.prob), w(fm.prob * fm.cost), w(fm.prob * fm.cost ** 2)
    nu = initial_vector(op.n, x0)
    m1, m2 = np.zeros(op.n), np.zeros(op.n)
    mean, var, occ = [0.0], [0.0], [nu]
    for _ in range(T):
        m2 = P0 @ m2 + 2 * (P1 @ m1) + P2 @ nu
        m1 = P0 @ m1 + P1 @ nu
        nu = P0 @ nu
        occ.append(nu)
        mean.append(m1.sum())
        var.append(m2.sum() - m1.sum() ** 2)
    return np.array(mean), np.array(var), np.array(occ)


def expected_cost_from_occupancies(op, occ):
    """E[C_t] = sum_{k<t} sum_x P(X_k = x) cbar(x), cbar(x) = sum over edges leaving x of p_e c_e."""
    fm = op.fm
    cbar = np.bincount(fm.src, weights=fm.prob * fm.cost, minlength=op.n)
    return np.concatenate([[0.0], np.cumsum(occ[:-1] @ cbar)])


def clipping_probabilities(op, occ, T):
    """
    Exact clipping diagnostics of the finite model: per-step probabilities that the transition
    clips s', clips z', or saturates the torque, and the probability that at least one of the
    first T transitions clips s' or z' (sub-stochastic recursion on the non-clipping edges).
    """
    fm = op.fm
    per = lambda w: np.bincount(fm.src, weights=w, minlength=op.n)
    ps, pz, pu = per(fm.p_state_clip), per(fm.p_memory_clip), per(fm.p_saturation)
    step = {"state_clip": occ[:T] @ ps, "memory_clip": occ[:T] @ pz, "saturation": occ[:T] @ pu}
    keep = fm.prob * ((fm.p_state_clip == 0) & (fm.p_memory_clip == 0))
    KT = sp.csr_matrix((np.bincount(op.slot, weights=keep, minlength=len(op.pairs)), (op.cols, op.rows)),
                       shape=(op.n, op.n))
    nu = occ[0].copy()
    no_clip = [1.0]
    for _ in range(T):
        nu = KT @ nu
        no_clip.append(nu.sum())
    step["p_any_clip"] = 1.0 - np.array(no_clip)
    return step


def chernoff(log_m, valid, lams, delta):
    """U = min over valid lambda > 0 of (log M + log(1/delta)) / lambda; returns (U, argmin index)."""
    obj = np.where(valid & (lams > 0), (log_m + np.log(1.0 / delta)) / lams, np.inf)
    if not np.isfinite(obj).any():
        return float("inf"), -1
    j = int(np.argmin(obj))
    return float(obj[j]), j


def chernoff_lower(log_m, valid, lams, delta):
    """L = max over valid lambda < 0 of (log M + log(1/delta)) / lambda; returns (L, argmax index)."""
    obj = np.where(valid & (lams < 0), (log_m + np.log(1.0 / delta)) / lams, -np.inf)
    if not np.isfinite(obj).any():
        return float("nan"), -1
    j = int(np.argmax(obj))
    return float(obj[j]), j


def lower_bounds(op, x0, T, reach, lams_pos, delta, horizons, per_decade=12, max_expansions=3):
    """
    One-sided Chernoff lower bounds at the same delta as ``chernoff``, on the mirrored grid
    -|lambda|. The grid grows by a decade when a requested horizon attains its maximum at an
    endpoint. Returns L[0..T] (nan at 0) and the negative lambdas used.
    """
    def decade(a, b):
        n = max(2, int(round(np.log10(b / a) * per_decade)) + 1)
        return np.geomspace(a, b, n)

    mags = np.unique(np.abs(lams_pos))
    mags = mags[mags > 0]
    for _ in range(max_expansions + 1):
        lams = -mags[::-1]
        log_m, valid = log_mgf(op, lams, T, x0, reach)
        idx = [chernoff_lower(log_m[t], valid[t], lams, delta)[1] for t in horizons]
        grow_large = 0 in idx
        grow_small = (len(lams) - 1) in idx
        if not (grow_large or grow_small):
            break
        extra = []
        if grow_large:
            extra.append(decade(mags[-1], mags[-1] * 10)[1:])
        if grow_small:
            extra.append(decade(mags[0] / 10, mags[0])[:-1])
        mags = np.unique(np.concatenate([mags, *extra]))
    L = np.full(T + 1, np.nan)
    for t in range(1, T + 1):
        L[t] = chernoff_lower(log_m[t], valid[t], lams, delta)[0]
    return L, lams


def bounds_with_expansion(op, x0, T_max, horizons, deltas, lam_cfg, reach, log=print):
    """
    Evaluate log M_t on a log-spaced lambda grid and extend the grid by a decade whenever some
    (horizon, delta) attains its minimum at an endpoint (at most lam_cfg['max_expansions'] times;
    an upper endpoint cannot be extended past the last valid lambda).
    """
    lo, hi, per_decade = lam_cfg["min"], lam_cfg["max"], lam_cfg["points_per_decade"]
    grid = lambda a, b: np.geomspace(a, b, max(2, int(round(np.log10(b / a) * per_decade)) + 1))
    lams = grid(lo, hi)
    t0 = time.perf_counter()
    log_m, valid = log_mgf(op, lams, T_max, x0, reach)
    expansions = []
    for _ in range(lam_cfg["max_expansions"]):
        idx = [chernoff(log_m[T], valid[T], lams, d)[1] for T in horizons for d in deltas]
        idx = [i for i in idx if i >= 0]
        if not idx:
            break
        last_valid = np.flatnonzero(valid[max(horizons)])
        at_lo = min(idx) == 0
        at_hi = bool(len(last_valid)) and max(idx) == last_valid[-1] and last_valid[-1] == len(lams) - 1
        if not (at_lo or at_hi):
            break
        new = []
        if at_lo:
            new.append(grid(lams[0] / 10, lams[0])[:-1])
        if at_hi:
            new.append(grid(lams[-1], lams[-1] * 10)[1:])
        new = np.concatenate(new)
        lm, va = log_mgf(op, new, T_max, x0, reach)
        lams = np.concatenate([lams, new])
        order = np.argsort(lams)
        lams, log_m, valid = lams[order], np.hstack([log_m, lm])[:, order], np.hstack([valid, va])[:, order]
        expansions.append({"low": bool(at_lo), "high": bool(at_hi), "range": [float(lams[0]), float(lams[-1])]})
        log(f"    lambda grid expanded: {expansions[-1]}")
    return lams, log_m, valid, expansions, time.perf_counter() - t0


def ess_sup(op, x0, T):
    """Largest accumulated cost over positive-probability paths (max-plus recursion)."""
    fm = op.fm
    best = np.where(initial_vector(op.n, x0) > 0, 0.0, -np.inf)
    out = [0.0]
    for _ in range(T):
        cand = best[fm.src] + fm.cost
        nxt = np.full(op.n, -np.inf)
        np.maximum.at(nxt, fm.dst, cand)
        best = nxt
        out.append(best.max())
    return np.array(out)
