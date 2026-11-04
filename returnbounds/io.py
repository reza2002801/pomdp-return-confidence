"""Save / load numerical artifacts so plots can be regenerated without recomputing MGF."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def save_finite_model(fm, path: Path):
    path = Path(path)
    np.savez_compressed(
        path,
        src=fm.src, dst=fm.dst, prob=fm.prob, cost=fm.cost,
        p_state_clip=fm.p_state_clip, p_memory_clip=fm.p_memory_clip,
        p_saturation=fm.p_saturation, state_keys=fm.state_keys,
        dims=np.array(fm.dims, dtype=np.int64),
        x0=np.array([fm.x0], dtype=np.int64),
        n_states=np.array([fm.n_states], dtype=np.int64),
        n_s=np.array([fm.n_s], dtype=np.int64),
        n_u_actions=np.array([fm.n_u_actions], dtype=np.int64),
        n_raw_edges=np.array([fm.n_raw_edges], dtype=np.int64),
        build_seconds=np.array([fm.build_seconds], dtype=np.float64),
    )


def load_finite_model(path: Path):
    from .build_quantized import FiniteModel
    z = np.load(path, allow_pickle=False)
    fm = FiniteModel(
        n_states=int(z["n_states"][0]),
        src=z["src"], dst=z["dst"], prob=z["prob"], cost=z["cost"],
        p_state_clip=z["p_state_clip"], p_memory_clip=z["p_memory_clip"],
        p_saturation=z["p_saturation"],
        x0=int(z["x0"][0]),
        n_raw_edges=int(z["n_raw_edges"][0]),
        build_seconds=float(z["build_seconds"][0]),
        state_keys=z["state_keys"],
        dims=tuple(int(x) for x in z["dims"]),
        pilot={},
        n_s=int(z["n_s"][0]),
        n_u_actions=int(z["n_u_actions"][0]),
    )
    return fm


def save_run_arrays(out_dir: Path, *, mean, var, lower, upper, poly_lower, poly_upper,
                    Lsup, Usup, moments, C, lams=None, log_m=None, valid=None):
    out_dir = Path(out_dir)
    np.savez_compressed(
        out_dir / "arrays.npz",
        mean=np.asarray(mean), var=np.asarray(var),
        mgf_lower=np.asarray(lower), mgf_upper=np.asarray(upper),
        poly_lower=np.asarray(poly_lower), poly_upper=np.asarray(poly_upper),
        support_L=np.asarray(Lsup), support_U=np.asarray(Usup),
        moments=np.asarray(moments),
        C=np.asarray(C),
    )
    if lams is not None:
        np.savez_compressed(
            out_dir / "mgf_grid.npz",
            lams=np.asarray(lams),
            log_m=np.asarray(log_m),
            valid=np.asarray(valid),
        )


def load_run_arrays(out_dir: Path):
    out_dir = Path(out_dir)
    z = np.load(out_dir / "arrays.npz")
    out = {k: z[k] for k in z.files}
    grid = out_dir / "mgf_grid.npz"
    if grid.exists():
        g = np.load(grid)
        out.update({k: g[k] for k in g.files})
    return out
