"""Tests for clamp-based MGF numerics in returnbounds.mgf."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from returnbounds import mgf  # noqa: E402


LOG_FLOOR = -600.0


def _old_matrix_T(op, lam, shifted=True):
    fm = op.fm
    shift = (lam * (op.cmax if lam >= 0 else op.cmin))[fm.src] if shifted else 0.0
    data = np.bincount(op.slot, weights=fm.prob * np.exp(lam * fm.cost - shift),
                       minlength=len(op.pairs))
    KT = op._KT0.copy()
    KT.data = data[op._order]
    return KT


def old_log_mgf(op, lams, T, x0, reach):
    out = np.zeros((T + 1, len(lams)))
    valid = np.ones((T + 1, len(lams)), dtype=bool)
    nu0 = mgf.initial_vector(op.n, x0)
    for j, lam in enumerate(lams):
        KT, m = _old_matrix_T(op, lam), op.row_shift(lam)
        w = nu0 / nu0.max()
        a, ok = float(np.log(nu0.max())), True
        for t in range(T):
            with np.errstate(divide="ignore"):
                ell = np.log(w) + m
            b = ell[reach[t]].max()
            ok &= bool(ell[reach[t]].min() - b >= LOG_FLOOR)
            v = KT @ np.exp(ell - b)
            d = v.max()
            w = v / d
            a += b + np.log(d)
            with np.errstate(divide="ignore"):
                ok &= bool(np.log(w[reach[t + 1]]).min() >= LOG_FLOOR)
            out[t + 1, j] = a + np.log(w.sum())
            valid[t + 1, j] = ok
    return out, valid


@dataclass
class TinyFM:
    n_states: int
    src: np.ndarray
    dst: np.ndarray
    prob: np.ndarray
    cost: np.ndarray
    x0: int = 0


def toy_chain():
    src = np.array([0, 0, 1, 1, 2, 3], dtype=np.int64)
    dst = np.array([1, 1, 2, 2, 3, 3], dtype=np.int64)
    prob = np.array([0.6, 0.4, 0.5, 0.5, 1.0, 1.0], dtype=float)
    cost = np.array([1.0, 11.0, 0.0, 10.0, 2.0, 0.0], dtype=float)
    return TinyFM(n_states=4, src=src, dst=dst, prob=prob, cost=cost, x0=0)


def enumerate_log_mgf(fm, lam, T):
    adj = [[] for _ in range(fm.n_states)]
    for s, d, p, c in zip(fm.src, fm.dst, fm.prob, fm.cost):
        adj[int(s)].append((int(d), float(p), float(c)))
    cur = [(int(fm.x0), 0.0, 0.0)]
    logs = [0.0]
    for _ in range(T):
        nxt, terms = [], []
        for state, lp, cs in cur:
            for d, p, c in adj[state]:
                lp2, cs2 = lp + np.log(p), cs + c
                nxt.append((d, lp2, cs2))
                terms.append(lp2 + lam * cs2)
        cur = nxt
        logs.append(float(np.logaddexp.reduce(np.asarray(terms, dtype=float))))
    return np.array(logs)


def test_brute_force_toy_chain():
    fm = toy_chain()
    op = mgf.TiltedOperator(fm)
    T = 6
    reach = mgf.reachable_sets(op, fm.x0, T)
    mgf.check_stochastic(op, reach, T)
    lams = np.array([-5.0, -0.5, 0.01, 0.5, 5.0, 50.0])
    log_m, valid = mgf.log_mgf(op, lams, T, fm.x0, reach)
    assert valid[T].all()
    assert np.isfinite(log_m[T, -1])
    for j, lam in enumerate(lams):
        exact = enumerate_log_mgf(fm, lam, T)
        for t in range(1, T + 1):
            assert log_m[t, j] >= exact[t] - 1e-9
            if abs(lam) <= 5.0:
                assert abs(log_m[t, j] - exact[t]) <= 1e-6


def test_lambda_zero():
    fm = toy_chain()
    op = mgf.TiltedOperator(fm)
    T = 6
    reach = mgf.reachable_sets(op, fm.x0, T)
    log_m, valid = mgf.log_mgf(op, np.array([0.0]), T, fm.x0, reach)
    assert valid.all()
    assert np.allclose(log_m, 0.0, atol=1e-12)


def _msd_op(T=20):
    from returnbounds.io import load_finite_model
    path = ROOT / "lqg_benchmark" / "outputs_paper" / "mass_spring_damper" / "finite_model.npz"
    if not path.exists():
        path = ROOT / "experiments" / "lqg" / "outputs_paper" / "mass_spring_damper" / "finite_model.npz"
    if not path.exists():
        pytest.skip(f"missing finite_model.npz")
    fm = load_finite_model(path)
    op = mgf.TiltedOperator(fm)
    reach = mgf.reachable_sets(op, fm.x0, T)
    return op, reach, fm


def test_agree_with_old_where_valid():
    T = 30
    op, reach, fm = _msd_op(T)
    lams = np.geomspace(0.01, 10.0, 12)
    new_lm, new_v = mgf.log_mgf(op, lams, T, fm.x0, reach)
    old_lm, old_v = old_log_mgf(op, lams, T, fm.x0, reach)
    both = np.isfinite(new_lm) & np.isfinite(old_lm)
    assert np.all(new_lm[both] >= old_lm[both] - 1e-10)
    mask = old_v & np.isfinite(old_lm)
    assert np.allclose(new_lm[mask], old_lm[mask], atol=1e-8, rtol=0.0)


def test_mean_from_mgf_derivative():
    T = 20
    op, reach, fm = _msd_op(T)
    mean, _, _ = mgf.exact_moments(op, fm.x0, T)
    h = 1e-4
    log_m, valid = mgf.log_mgf(op, np.array([h, -h]), T, fm.x0, reach)
    assert valid[T].all()
    approx = (log_m[:, 0] - log_m[:, 1]) / (2 * h)
    for t in range(1, T + 1):
        rel = abs(approx[t] - mean[t]) / max(abs(mean[t]), 1e-12)
        assert rel < 1e-4
