"""Publication figures for the LQG finite-model paper suite."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BAND_STYLE = {
    "Exact MGF": {"color": "#2ca02c", "ls": (0, (4, 1.5)), "lw": 2.4},
    "Polynomial (d=16)": {"color": "#9467bd", "ls": "-", "lw": 2.2},
}

MGF = BAND_STYLE["Exact MGF"]
POLY = BAND_STYLE["Polynomial (d=16)"]
HIST = dict(color="#a3b1c6", edgecolor="#4c566a", linewidth=0.4)


def _style(ax):
    ax.grid(True, alpha=0.28, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def physical_trajectories(plant, sim, path, n_show=3):
    """True state and observer prediction memory for a few trajectories, plus applied inputs."""
    T = sim["S"].shape[1] - 1
    t = np.arange(T + 1) * plant.dt
    rows = plant.n + plant.m
    fig, axes = plt.subplots(rows, 1, figsize=(8.4, 1.35 * rows + 1.2), sharex=True)
    if rows == 1:
        axes = [axes]
    show = min(n_show, sim["S"].shape[0])
    for k, ax in enumerate(axes[:plant.n]):
        for i in range(show):
            ax.plot(t, sim["S"][i, :, k], color="#2e3440", lw=1.3 if i == 0 else 0.8,
                    label=("true $S$" if i == 0 else None))
            ax.plot(t, sim["Z"][i, :, k], color="#d62728", lw=1.1 if i == 0 else 0.7,
                    label=("observer memory $Z$" if i == 0 else None))
        ax.set_ylabel(f"{plant.state_labels[k]}\n[{plant.state_units[k]}]")
        _style(ax)
    for j, ax in enumerate(axes[plant.n:]):
        for i in range(show):
            ax.step(t[:-1], sim["U"][i, :, j], where="post", color="#1f77b4",
                    lw=1.3 if i == 0 else 0.8)
        ax.set_ylabel(f"{plant.input_labels[j]}\n[{plant.input_units[j]}]")
        _style(ax)
    axes[-1].set_xlabel(f"time [s]  (dt={plant.dt:g})")
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=9)
    fig.suptitle(f"{plant.name}: physical trajectories (linear-Gaussian model)", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def cost_trajectories(C, mean, lower, upper, path, title, n_draw=800,
                      poly_lower=None, poly_upper=None,
                      support_L=None, support_U=None):
    n = min(n_draw, C.shape[0])
    idx = np.linspace(0, C.shape[0] - 1, n).astype(int)
    t = np.arange(C.shape[1])
    centered = C[idx] - mean
    q_lo = np.quantile(C, 0.025, axis=0) - mean
    q_hi = np.quantile(C, 0.975, axis=0) - mean
    # Chernoff can numerically leave exact support; clip for display.
    lo = np.asarray(lower, dtype=float).copy()
    hi = np.asarray(upper, dtype=float).copy()
    if support_L is not None and support_U is not None:
        lo = np.maximum(lo, support_L)
        hi = np.minimum(hi, support_U)
    plo = None if poly_lower is None else np.asarray(poly_lower, dtype=float).copy()
    phi = None if poly_upper is None else np.asarray(poly_upper, dtype=float).copy()
    if plo is not None and support_L is not None and support_U is not None:
        plo = np.clip(plo, support_L, support_U)
        phi = np.clip(phi, support_L, support_U)
    fig, ax = plt.subplots(figsize=(7.8, 5.3))
    ax.plot(t, centered.T, color="#4c566a", alpha=1.0, lw=0.25, rasterized=True, zorder=1)
    ax.plot(t, np.zeros_like(t, dtype=float), color="#2e3440", lw=2.0, label="exact mean", zorder=3)
    ax.plot(t, q_lo, color="#2e3440", ls=":", lw=1.8, label="empirical 2.5%", zorder=4)
    ax.plot(t, q_hi, color="#2e3440", ls=(0, (1, 1.2)), lw=1.8, label="empirical 97.5%", zorder=4)
    ax.plot(t[1:], (lo - mean)[1:], color=MGF["color"], ls=MGF["ls"], lw=MGF["lw"],
            label="MGF bounds", zorder=5)
    ax.plot(t[1:], (hi - mean)[1:], color=MGF["color"], ls=MGF["ls"], lw=MGF["lw"], zorder=5)
    span = [
        float(np.nanmax(np.abs(lo[1:] - mean[1:]))),
        float(np.nanmax(np.abs(hi[1:] - mean[1:]))),
        float(np.abs(centered).max()),
    ]
    if plo is not None and phi is not None:
        ax.plot(t[1:], (plo - mean)[1:], color=POLY["color"], ls=POLY["ls"], lw=POLY["lw"],
                label="polynomial (d=16)", zorder=6)
        ax.plot(t[1:], (phi - mean)[1:], color=POLY["color"], ls=POLY["ls"], lw=POLY["lw"],
                zorder=6)
        span.extend([
            float(np.nanmax(np.abs(plo[1:] - mean[1:]))),
            float(np.nanmax(np.abs(phi[1:] - mean[1:]))),
        ])
    lim = 1.08 * max(span)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"time step $t$")
    ax.set_ylabel(r"$C_t-\mathbb{E}[C_t]$")
    ax.set_title("Pointwise 95% bounds and empirical percentiles", pad=28)
    ncol = 6 if plo is not None else 5
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=ncol, frameon=False, fontsize=8,
              handlelength=2.4, columnspacing=0.9, borderaxespad=0.15)
    _ = title  # kept for call-site compatibility; LaTeX captions name the plant
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def histogram(C_T, mean_T, lower, upper, T, path, name,
              poly_lower=None, poly_upper=None):
    x = np.asarray(C_T, dtype=float) - mean_T
    lo, hi = lower - mean_T, upper - mean_T
    q_lo, q_hi = np.quantile(x, [0.025, 0.975])
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.hist(x, bins=70, weights=np.full(len(x), 1 / len(x)), zorder=1, **HIST)
    top = 0.74
    ax.axvline(0.0, ymax=top, color="#2e3440", lw=2.0, label="exact mean", zorder=3)
    ax.axvline(q_lo, ymax=top, color="#2e3440", ls=":", lw=1.8, label="empirical 2.5% / 97.5%", zorder=4)
    ax.axvline(q_hi, ymax=top, color="#2e3440", ls=":", lw=1.8, zorder=4)
    ax.axvline(lo, ymax=top, color=MGF["color"], ls=MGF["ls"], lw=MGF["lw"], label="MGF bounds", zorder=5)
    ax.axvline(hi, ymax=top, color=MGF["color"], ls=MGF["ls"], lw=MGF["lw"], zorder=5)
    half = max(abs(lo), abs(hi), 1e-9)
    if poly_lower is not None and poly_upper is not None:
        plo, phi = poly_lower - mean_T, poly_upper - mean_T
        ax.axvline(plo, ymax=top, color=POLY["color"], ls=POLY["ls"], lw=POLY["lw"],
                   label="polynomial (d=16)", zorder=6)
        ax.axvline(phi, ymax=top, color=POLY["color"], ls=POLY["ls"], lw=POLY["lw"], zorder=6)
        half = max(half, abs(plo), abs(phi))
    lim = 1.12 * max(half, np.abs(x).max())
    ax.set_xlim(-lim, lim)
    ax.set_ylim(0, ax.get_ylim()[1] / (top - 0.03))
    ax.set_xlabel(rf"$C_{{{T}}}-\mathbb{{E}}[C_{{{T}}}]$")
    ax.set_ylabel("fraction of trajectories")
    _ = name  # kept for call-site compatibility; LaTeX captions name the plant
    ax.set_title(f"Pointwise 95% at $T={T}$ ({len(x):,} traj.)", fontsize=11, pad=8)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5, handlelength=2.6)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def noise_sensitivity(rows, path, title):
    """rows: list of dicts with keys multiplier, mean, q975, upper."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    mult = [r["multiplier"] for r in rows]
    ax.plot(mult, [r["mean"] for r in rows], "o-", color="#2e3440", lw=2.0, label="exact mean")
    ax.plot(mult, [r["q975"] for r in rows], "s--", color="#4c566a", lw=1.6, label="empirical 97.5%")
    ax.plot(mult, [r["upper"] for r in rows], "^-", color=MGF["color"], lw=2.0, label="MGF upper bound")
    ax.set_xscale("log")
    ax.set_xlabel("measurement-noise covariance multiplier")
    ax.set_ylabel(r"terminal cost $C_T$")
    ax.set_title(title, fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
