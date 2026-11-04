"""
Fixed LQG plant: discrete linear dynamics, LQR feedback, steady-state Kalman update.

Timing:
  Y_t = H S_t + V_t
  S_hat = Z_t + L (Y_t - H Z_t)
  U_t = -K S_hat
  Z_{t+1} = A S_hat + B U_t
  S_{t+1} = A S_t + B U_t + W_{t+1}
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .linalg_utils import (
    as_matrix,
    discrete_process_cov,
    kalman_gain,
    lqr_gain,
    spectral_radius,
    zoh,
)
from .systems import continuous_model


@dataclass
class Plant:
    name: str
    description: str
    Ac: np.ndarray
    Bc: np.ndarray
    A: np.ndarray
    B: np.ndarray
    H: np.ndarray
    SW: np.ndarray
    SV: np.ndarray
    Qc: np.ndarray
    Rc: np.ndarray
    K: np.ndarray
    L: np.ndarray
    P_lqr: np.ndarray
    P_pred: np.ndarray
    P_post: np.ndarray
    dt: float
    s0: np.ndarray
    z0: np.ndarray
    state_labels: list
    state_units: list
    input_labels: list
    input_units: list
    cfg: dict
    notes: list
    process_channel: Optional[np.ndarray] = None

    @property
    def n(self):
        return self.A.shape[0]

    @property
    def m(self):
        return self.B.shape[1]

    @property
    def p(self):
        return self.H.shape[0]


def closed_loop(plant):
    """Augmented X=(S,Z): X'=F X + Gw W + Gv V, U = Hx X + Hv V."""
    A, B, K, L, H = plant.A, plant.B, plant.K, plant.L, plant.H
    n = plant.n
    I = np.eye(n)
    LH = L @ H
    I_LH = I - LH
    Hx = np.hstack([-K @ LH, -K @ I_LH])
    Hv = -(K @ L)
    F = np.zeros((2 * n, 2 * n))
    F[:n, :] = np.hstack([A, np.zeros((n, n))]) + B @ Hx
    AK = A - B @ K
    F[n:, :] = AK @ np.hstack([LH, I_LH])
    Gw = np.zeros((2 * n, n))
    Gw[:n, :] = np.eye(n)
    Gv = np.zeros((2 * n, plant.p))
    Gv[:n, :] = B @ Hv
    Gv[n:, :] = AK @ L
    return F, Gw, Gv, Hx, Hv


def design_report(plant):
    I = np.eye(plant.n)
    Fcl = plant.A - plant.B @ plant.K
    P = plant.P_lqr
    ric = plant.A.T @ P @ plant.A - plant.A.T @ P @ plant.B @ plant.K + plant.Qc - P
    gain_res = plant.L @ (plant.H @ plant.P_pred @ plant.H.T + plant.SV) - plant.P_pred @ plant.H.T
    pred_res = plant.A @ plant.P_post @ plant.A.T + plant.SW - plant.P_pred
    F, *_ = closed_loop(plant)
    return {
        "name": plant.name,
        "n": plant.n,
        "m": plant.m,
        "dt": plant.dt,
        "spectral_radius_A": spectral_radius(plant.A),
        "spectral_radius_A_minus_BK": spectral_radius(Fcl),
        "spectral_radius_A(I-LH)": spectral_radius(plant.A @ (I - plant.L @ plant.H)),
        "spectral_radius_joint": spectral_radius(F),
        "lqr_riccati_residual": float(np.max(np.abs(ric))),
        "kalman_gain_residual": float(np.max(np.abs(gain_res))),
        "kalman_pred_residual": float(np.max(np.abs(pred_res))),
        "K": plant.K.tolist(),
        "L": plant.L.tolist(),
        "notes": plant.notes,
        "disclaimer": (
            "Spectral radii and Riccati residuals are for the continuous-state linear "
            "Gaussian model with the fixed steady-state gains. They do not apply to "
            "nonlinear plants or quantized controllers."
        ),
    }


def _process_cov(Ac, noise, meta, dt, n):
    if "process_cov" in noise:
        return as_matrix(noise["process_cov"], (n, n)), "discrete process_cov from config"
    if "process_std" in noise:
        return np.diag(np.square(noise["process_std"])), "discrete process_std from config"
    if "process_intensity" not in noise:
        raise ValueError("noise needs process_cov, process_std, or process_intensity")

    # Continuous white intensity through a channel G.
    if "process_channel" in noise:
        G = np.asarray(noise["process_channel"], dtype=float)
    elif meta.get("process_channel") is not None:
        G = np.asarray(meta["process_channel"], dtype=float)
    else:
        G = np.eye(n)

    if G.ndim == 1:
        G = G.reshape(-1, 1)
    q = noise["process_intensity"]
    if np.ndim(q) == 0:
        Qw = float(q) * np.eye(G.shape[1])
    else:
        Qw = as_matrix(q, (G.shape[1], G.shape[1]))
    SW = discrete_process_cov(Ac, G, Qw, dt)
    return SW, "discrete SW from continuous intensity via Van Loan"


def build_plant(cfg):
    Ac, Bc, meta = continuous_model(cfg)
    dt = float(cfg["physics"]["dt"])
    A, B = zoh(Ac, Bc, dt)
    n, m = A.shape[0], B.shape[1]
    H_cfg = cfg.get("observation", {}).get("H", "I")
    if isinstance(H_cfg, str) or H_cfg is None:
        H = np.eye(n)
    else:
        H = np.asarray(H_cfg, dtype=float)
        if H.shape == (n,):
            H = np.diag(H)
        if H.ndim != 2 or H.shape[1] != n:
            H = np.eye(n)

    SW, cov_note = _process_cov(Ac, cfg["noise"], meta, dt, n)
    if "measurement_cov" in cfg["noise"]:
        SV = as_matrix(cfg["noise"]["measurement_cov"], (H.shape[0], H.shape[0]))
    else:
        SV = np.diag(np.square(cfg["noise"]["measurement_std"]))
    SV = SV * float(cfg.get("measurement_noise_multiplier", 1.0))

    Qc = as_matrix(cfg["cost"]["Q"], (n, n))
    Rc = as_matrix(cfg["cost"]["R"], (m, m))
    Q_lqr = as_matrix(cfg["lqr"]["Q"], (n, n))
    R_lqr = as_matrix(cfg["lqr"]["R"], (m, m))

    if cfg.get("gains") and "K" in cfg["gains"] and "L" in cfg["gains"]:
        K = np.asarray(cfg["gains"]["K"], dtype=float)
        L = np.asarray(cfg["gains"]["L"], dtype=float)
        P_lqr = np.zeros((n, n))
        P_pred = np.zeros((n, n))
        P_post = np.zeros((n, n))
        gain_note = "K and L held fixed from the baseline design"
    else:
        K, P_lqr = lqr_gain(A, B, Q_lqr, R_lqr)
        P_pred, L, P_post = kalman_gain(A, H, SW, SV)
        gain_note = "K and L redesigned for this configuration"

    s0 = np.asarray(cfg["init"]["s"], dtype=float)
    z0 = np.asarray(cfg["init"].get("z", np.zeros(n)), dtype=float)
    notes = list(meta.get("notes", [])) + [cov_note, gain_note,
        "Accumulated cost is sum_t (S_t' Qc S_t + U_t' Rc U_t) with no factor of dt."]

    return Plant(
        name=cfg["name"], description=cfg.get("description", ""),
        Ac=Ac, Bc=Bc, A=A, B=B, H=H, SW=SW, SV=SV,
        Qc=Qc, Rc=Rc, K=K, L=L, P_lqr=P_lqr, P_pred=P_pred, P_post=P_post,
        dt=dt, s0=s0, z0=z0,
        state_labels=meta["state_labels"], state_units=meta["state_units"],
        input_labels=meta["input_labels"], input_units=meta["input_units"],
        cfg=cfg, notes=notes,
        process_channel=meta.get("process_channel"),
    )
