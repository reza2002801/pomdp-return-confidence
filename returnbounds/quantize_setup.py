"""
Quantized Setup on top of a continuous Plant: grids, finite-support noise, and
one-step quantized dynamics. The paper MGF recursion applies to the finite model
built from this Setup, not to the underlying continuous plant.
"""

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.hermite_e import hermegauss

from .plant import Plant, build_plant


def noise_nodes(name):
    n = {"gh3": 3, "gh5": 5}[name]
    nodes, weights = hermegauss(n)
    return nodes, weights / weights.sum()


def product_noise(std, rule):
    std = np.asarray(std, dtype=float)
    nodes, probs = noise_nodes(rule)
    grids = np.meshgrid(*([nodes * s for s in std]), indexing="ij")
    outcomes = np.column_stack([g.ravel() for g in grids])
    p = np.ones(1)
    for _ in std:
        p = np.multiply.outer(p, probs).ravel()
    return outcomes, p


@dataclass
class Grid1D:
    n: int
    r: float

    @property
    def h(self):
        return 0.0 if self.n == 1 else 2.0 * self.r / (self.n - 1)

    @property
    def points(self):
        return np.array([0.0]) if self.n == 1 else np.linspace(-self.r, self.r, self.n)

    def quantize(self, x):
        x = np.asarray(x, dtype=float)
        if self.n == 1:
            return np.zeros(x.shape, dtype=np.int64), np.abs(x) > 0
        k = np.floor((x + self.r) / self.h + 0.5).astype(np.int64)
        clipped = (k < 0) | (k > self.n - 1)
        return np.clip(k, 0, self.n - 1), clipped


class QuantizedSetup:
    """Finite-support observer LQG on X=(S,Z) with quantized U and Z'."""

    def __init__(self, plant: Plant, grid_cfg: dict, noise_rule: str = "gh3"):
        self.plant = plant
        self.cfg_grid = grid_cfg
        n, m = plant.n, plant.m
        g = grid_cfg
        if len(g["state_points"]) != n or len(g["state_range"]) != n:
            raise ValueError("state grid length must match plant.n")
        if len(g["memory_points"]) != n or len(g["memory_range"]) != n:
            raise ValueError("memory grid length must match plant.n")
        self.S_grids = [Grid1D(int(p), float(r)) for p, r in zip(g["state_points"], g["state_range"])]
        self.Z_grids = [Grid1D(int(p), float(r)) for p, r in zip(g["memory_points"], g["memory_range"])]
        # Action: one Grid1D per input channel (odd length so 0 is included).
        ap = g["action_points"]
        lim = g["action_limit"]
        if np.ndim(ap) == 0:
            ap = [int(ap)] * m
            lim = [float(lim)] * m
        if len(ap) != m:
            raise ValueError("action grid length must match plant.m")
        for a in ap:
            if int(a) % 2 == 0:
                raise ValueError("each action grid must contain zero (odd number of points)")
        self.U_grids = [Grid1D(int(a), float(r)) for a, r in zip(ap, lim)]
        self.V, self.pV = product_noise(np.sqrt(np.maximum(np.diag(plant.SV), 0.0)), noise_rule)
        # Process noise: W = L eps with eps product Gauss-Hermite on R^n (exact if SW=LL^T).
        Lfac = np.linalg.cholesky(plant.SW + 1e-14 * np.eye(n))
        eps, self.pW = product_noise(np.ones(n), noise_rule)
        self.W = eps @ Lfac.T
        self.n = n
        self.m = m

    def quantize_vec(self, x, grids):
        idx = []
        clipped = np.zeros(x.shape[:-1], dtype=bool)
        for i, gr in enumerate(grids):
            k, c = gr.quantize(x[..., i])
            idx.append(k)
            clipped |= c
        return np.stack(idx, axis=-1), clipped

    def values_from_idx(self, idx, grids):
        return np.stack([grids[i].points[idx[..., i]] for i in range(len(grids))], axis=-1)

    def quantize_action(self, u_raw):
        """u_raw (..., m) -> indices (..., m), values (..., m), saturation mask."""
        iu = []
        sat = np.zeros(u_raw.shape[:-1], dtype=bool)
        for j, gr in enumerate(self.U_grids):
            k, c = gr.quantize(u_raw[..., j])
            iu.append(k)
            sat |= c
        iu = np.stack(iu, axis=-1)
        u = self.values_from_idx(iu, self.U_grids)
        return iu, u, sat

    def control(self, s, z, v):
        """y=s+v, s_hat=z+L(y-z), u=Q(-K s_hat), z'=Q(A s_hat+B u)."""
        plant = self.plant
        s_hat = z + (s + v - z) @ plant.L.T
        u_raw = -(s_hat @ plant.K.T)
        iu, u, sat = self.quantize_action(u_raw)
        z_raw = s_hat @ plant.A.T + u @ plant.B.T
        iz, zclip = self.quantize_vec(z_raw, self.Z_grids)
        return s_hat, iu, u, sat, iz, zclip

    def physical(self, s, u, w):
        raw = s @ self.plant.A.T + u @ self.plant.B.T + w
        return self.quantize_vec(raw, self.S_grids)

    def cost(self, s, u):
        Qc, Rc = self.plant.Qc, self.plant.Rc
        return np.einsum("...i,ij,...j->...", s, Qc, s) + np.einsum("...i,ij,...j->...", u, Rc, u)

    def initial_indices(self):
        s = np.asarray(self.plant.s0, dtype=float)
        z = np.asarray(self.plant.z0, dtype=float)
        is_, cs = self.quantize_vec(s, self.S_grids)
        iz, cz = self.quantize_vec(z, self.Z_grids)
        if np.any(cs) or np.any(cz):
            raise ValueError("initial condition lies outside the quantization grids")
        return is_.astype(np.int64), iz.astype(np.int64), s, z


def setup_from_config(cfg):
    plant = build_plant(cfg)
    if "grid" not in cfg:
        raise KeyError("config needs a 'grid' block for the paper-recursion finite model")
    rule = cfg.get("noise", {}).get("rule", "gh3")
    return QuantizedSetup(plant, cfg["grid"], noise_rule=rule), plant
