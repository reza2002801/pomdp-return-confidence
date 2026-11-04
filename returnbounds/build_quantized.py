"""
Reachable finite model on X=(S,Z) for a QuantizedSetup.
Same construction as cartpole_observer/finite_model.py, for general n and m.
"""

from dataclasses import dataclass
import time

import numpy as np


@dataclass
class FiniteModel:
    n_states: int
    src: np.ndarray
    dst: np.ndarray
    prob: np.ndarray
    cost: np.ndarray
    p_state_clip: np.ndarray
    p_memory_clip: np.ndarray
    p_saturation: np.ndarray
    x0: int
    n_raw_edges: int
    build_seconds: float
    state_keys: np.ndarray
    dims: tuple
    pilot: dict
    n_s: int
    n_u_actions: int

    @property
    def n_edges(self):
        return len(self.src)


def _strides(dims):
    st = np.ones(len(dims), dtype=np.int64)
    for i in range(len(dims) - 2, -1, -1):
        st[i] = st[i + 1] * dims[i + 1]
    return st


def _pack(idx, strides):
    return (idx.astype(np.int64) * strides).sum(axis=-1)


def _unpack(keys, dims):
    keys = np.array(keys, dtype=np.int64, copy=True)
    out = np.empty(keys.shape + (len(dims),), dtype=np.int64)
    for i in range(len(dims) - 1, -1, -1):
        out[..., i] = keys % dims[i]
        keys //= dims[i]
    return out


def _action_code(iu, u_dims):
    """Pack multi-dim action indices into a single integer."""
    iu = np.asarray(iu, dtype=np.int64)
    if iu.ndim == 1:
        return iu
    code = np.zeros(iu.shape[:-1], dtype=np.int64)
    for j, d in enumerate(u_dims):
        code = code * int(d) + iu[..., j]
    return code


def build(setup, batch=64, max_states=25000, log=print):
    t0 = time.perf_counter()
    n = setup.n
    dims = tuple(gr.n for gr in setup.S_grids) + tuple(gr.n for gr in setup.Z_grids)
    u_dims = tuple(gr.n for gr in setup.U_grids)
    nA = int(np.prod(u_dims))
    full = int(np.prod(dims))
    branches = len(setup.pV) * len(setup.pW)
    strides = _strides(dims)
    log(f"  full Cartesian grid: {full:,} states, {branches:,} noise branches/state, "
        f"{nA} quantized actions")

    s0, z0, _, _ = setup.initial_indices()
    origin = int(_pack(np.concatenate([s0, z0]), strides))
    id_of = {origin: 0}
    keys = [origin]
    frontier = [0]
    edge_src, edge_dst, edge_prob, edge_cost = [], [], [], []
    edge_sc, edge_zc, edge_sat = [], [], []
    n_raw = 0
    p_vw = (setup.pV[:, None] * setup.pW[None, :]).ravel()

    while frontier:
        batch_ids = frontier
        frontier = []
        for lo in range(0, len(batch_ids), batch):
            ids = np.array(batch_ids[lo:lo + batch])
            idx = _unpack(np.array([keys[i] for i in ids]), dims)
            s = setup.values_from_idx(idx[:, :n], setup.S_grids)
            z = setup.values_from_idx(idx[:, n:], setup.Z_grids)
            # V broadcast: (B, nV, n)
            _, iu, u, sat, iz, zclip = setup.control(
                s[:, None, :], z[:, None, :], setup.V[None, :, :])
            # shapes: s (B,1,n), V (1,nV,n) -> u (B,nV,m), iz (B,nV,n), sat (B,nV)
            is_, sclip = setup.physical(
                s[:, None, None, :], u[:, :, None, :], setup.W[None, None, :, :])
            # is_ (B,nV,nW,n), sclip (B,nV,nW)
            B, nV, nW = is_.shape[:3]
            dst_idx = np.concatenate(
                [is_, np.broadcast_to(iz[:, :, None, :], is_.shape)], axis=-1)
            dst_pack = _pack(dst_idx.reshape(-1, 2 * n), strides)
            iu_b = np.broadcast_to(iu[:, :, None, :], (B, nV, nW, setup.m))
            iu_r = _action_code(iu_b, u_dims).ravel()
            sat_b = np.broadcast_to(sat[:, :, None], (B, nV, nW))
            zclip_b = np.broadcast_to(zclip[:, :, None], (B, nV, nW))
            flag = (sclip.astype(np.int64) * 4
                    + zclip_b.astype(np.int64) * 2
                    + sat_b.astype(np.int64)).ravel()
            cost = np.broadcast_to(
                setup.cost(s[:, None, :], u)[:, :, None], (B, nV, nW)).ravel()
            src = np.repeat(ids, nV * nW)
            n_raw += src.size
            key = ((src.astype(np.int64) * (full + 1) + dst_pack) * nA + iu_r) * 8 + flag
            uniq, inv = np.unique(key, return_inverse=True)
            prob = np.bincount(inv, weights=np.tile(p_vw, B))
            order = np.argsort(inv, kind="stable")
            first = np.concatenate([[True], inv[order][1:] != inv[order][:-1]])
            take = order[first]
            dst_u = dst_pack[take]
            new_ids = np.empty(len(uniq), dtype=np.int64)
            fresh = []
            for j, dp in enumerate(dst_u.tolist()):
                i = id_of.get(dp)
                if i is None:
                    i = len(keys)
                    id_of[dp] = i
                    keys.append(dp)
                    fresh.append(i)
                new_ids[j] = i
            frontier.extend(fresh)
            if len(keys) > max_states:
                raise RuntimeError(
                    f"reachable set exceeded {max_states:,} "
                    f"({n_raw:,} raw pairs; Cartesian {full:,})")
            edge_src.append(src[take])
            edge_dst.append(new_ids)
            edge_prob.append(prob)
            edge_cost.append(cost[take])
            pw = np.tile(p_vw, B)
            edge_sc.append(np.bincount(inv, weights=sclip.ravel().astype(float) * pw))
            edge_zc.append(np.bincount(inv, weights=zclip_b.ravel().astype(float) * pw))
            edge_sat.append(np.bincount(inv, weights=sat_b.ravel().astype(float) * pw))
        log(f"    reachable {len(keys):,}, edges {sum(len(e) for e in edge_src):,}, "
            f"raw {n_raw:,}")

    fm = FiniteModel(
        n_states=len(keys),
        src=np.concatenate(edge_src), dst=np.concatenate(edge_dst),
        prob=np.concatenate(edge_prob), cost=np.concatenate(edge_cost),
        p_state_clip=np.concatenate(edge_sc),
        p_memory_clip=np.concatenate(edge_zc),
        p_saturation=np.concatenate(edge_sat),
        x0=0, n_raw_edges=n_raw, build_seconds=time.perf_counter() - t0,
        state_keys=np.array(keys, dtype=np.int64), dims=dims,
        pilot={"full_cartesian": full, "noise_branches": branches,
               "reachable": len(keys),
               "merged_edges": int(sum(len(e) for e in edge_src)),
               "raw_pairs": n_raw},
        n_s=n, n_u_actions=nA,
    )
    fm.id_of = id_of
    order = np.argsort(fm.state_keys)
    fm.sorted_keys = fm.state_keys[order]
    fm.sorted_index = order.astype(np.int64)
    log(f"  built: {fm.n_states:,} states, {fm.n_edges:,} edges, "
        f"{n_raw:,} raw, {fm.build_seconds:.2f}s")
    return fm


def simulate(fm, n_traj, T, seed):
    """Monte Carlo on the merged edge list (the model the MGF is for)."""
    rng = np.random.default_rng(seed)
    order = np.argsort(fm.src, kind="stable")
    src, dst, prob = fm.src[order], fm.dst[order], fm.prob[order]
    cost = fm.cost[order]
    counts = np.bincount(src, minlength=fm.n_states)
    ind = np.zeros(fm.n_states + 1, dtype=np.int64)
    ind[1:] = np.cumsum(counts)
    cdf = np.cumsum(prob)
    cdf = cdf - np.repeat(np.concatenate([[0.0], cdf[ind[1:-1] - 1]]), counts)
    x = np.full(n_traj, fm.x0, dtype=np.int64)
    C = np.zeros((n_traj, T + 1))
    for t in range(T):
        u = rng.random(n_traj)
        nxt = np.empty(n_traj, dtype=np.int64)
        add = np.zeros(n_traj)
        for state in np.unique(x):
            mask = x == state
            a, b = ind[state], ind[state + 1]
            j = a + np.searchsorted(cdf[a:b], u[mask], side="right")
            j = np.minimum(j, b - 1)
            nxt[mask] = dst[j]
            add[mask] = cost[j]
        C[:, t + 1] = C[:, t] + add
        x = nxt
    return C
