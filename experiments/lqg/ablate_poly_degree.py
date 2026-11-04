"""
Ablation: polynomial degree d on a fixed plant (reuse finite model + MGF + trajectories).

  python experiments/lqg/ablate_poly_degree.py mass_spring_damper cartpole_linear
  python experiments/lqg/ablate_poly_degree.py --degrees 4,8,16,32,64

Increases d until the terminal poly interval collapses onto the mean (numerical failure).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from returnbounds import io as save_artifacts
from returnbounds import moments as paper_moments
from returnbounds.profiling import StageTimer, peak_rss_kb

HERE = Path(__file__).resolve().parent
# Prefer catalogued runs under data/runs/, then local experiment outputs.
SRC_CANDIDATES = [
    _ROOT / "data" / "runs" / "paper_clamp_v1",
    _ROOT / "data" / "runs" / "local",
    HERE / "outputs",
]
ABL = _ROOT / "data" / "runs" / "ablation_poly_degree"

DEFAULT_PLANTS = ["mass_spring_damper", "cartpole_linear"]
DEFAULT_DEGREES = [4, 8, 16, 20, 24]
TIMES_LARGE = list(range(1, 101, 5)) + [100]
TIMES_SMALL = list(range(1, 101))

MGF_STYLE = {"color": "#2ca02c", "ls": (0, (4, 1.5)), "lw": 2.4}


def _src_dir(name: str) -> Path:
    for root in SRC_CANDIDATES:
        if (root / name / "arrays.npz").exists() and (root / name / "finite_model.npz").exists():
            return root / name
    raise FileNotFoundError(f"no saved run for {name}; looked in {SRC_CANDIDATES}")


def _style(ax):
    ax.grid(True, alpha=0.28, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _poly_colors(degrees):
    cmap = plt.get_cmap("plasma")
    xs = np.linspace(0.2, 0.85, max(len(degrees), 1))
    return {d: cmap(x) for d, x in zip(degrees, xs)}


def _collapsed(row, emp_width):
    """True if poly band has numerically collapsed (useless / invalid for reporting)."""
    if not np.isfinite(row["width"]) or row["width"] < 1e-6:
        return True
    if emp_width > 0 and row["width"] < 0.02 * emp_width:
        return True
    if row["frac_below"] + row["frac_above"] > 0.4:
        return True
    return False


def run_plant(name: str, degrees: list[int], alpha: float = 0.025, stop_on_collapse: bool = True):
    out = _src_dir(name)
    abl = ABL / name
    abl.mkdir(parents=True, exist_ok=True)
    cfg = json.load(open(out / "config.json"))
    summary = json.load(open(out / "summary.json"))
    data = save_artifacts.load_run_arrays(out)
    fm = save_artifacts.load_finite_model(out / "finite_model.npz")

    T = int(max(cfg["horizons"]))
    mean = data["mean"]
    mgf_L, mgf_U = data["mgf_lower"], data["mgf_upper"]
    C = data["C"]
    plant = summary.get("name", name)
    n_states = int(fm.n_states)
    times = TIMES_SMALL if n_states < 2000 else TIMES_LARGE
    times = sorted(set(t for t in times if 1 <= t <= T))

    print("=" * 64)
    print(f"{name}: src={out}")
    print(f"  states={n_states}  T={T}  degrees={degrees}  n_times={len(times)}")

    rows = []
    curves = {}
    used_degrees = []
    emp_w = float(np.quantile(C[:, T], 0.975) - np.quantile(C[:, T], 0.025))

    for d in degrees:
        timer = StageTimer()
        print(f"  degree d={d} …")
        with timer.stage("poly_intervals"):
            lo, hi, Lsup, Usup, _ = paper_moments.poly_intervals(
                fm, T, d, alpha=alpha, times=times, log=print,
            )
        dt = timer.stages["poly_intervals"]
        t_rep = T if T in times else times[-1]
        width = float(hi[t_rep] - lo[t_rep])
        frac_lo = float(np.mean(C[:, t_rep] < lo[t_rep]))
        frac_hi = float(np.mean(C[:, t_rep] > hi[t_rep]))
        row = {
            "degree": d,
            "t": t_rep,
            "poly_L": float(lo[t_rep]),
            "poly_U": float(hi[t_rep]),
            "width": width,
            "frac_below": frac_lo,
            "frac_above": frac_hi,
            "seconds": dt,
            "max_rss_kb": peak_rss_kb(),
            "n_times": len(times),
            "support_L": float(Lsup[t_rep]),
            "support_U": float(Usup[t_rep]),
            "mgf_L": float(mgf_L[t_rep]),
            "mgf_U": float(mgf_U[t_rep]),
            "emp_q025": float(np.quantile(C[:, t_rep], 0.025)),
            "emp_q975": float(np.quantile(C[:, t_rep], 0.975)),
            "exact_mean": float(mean[t_rep]),
            "collapsed": False,
        }
        row["collapsed"] = _collapsed(row, emp_w)
        rows.append(row)
        curves[d] = {"lo": lo, "hi": hi, "Lsup": Lsup, "Usup": Usup, "times": times}
        used_degrees.append(d)
        np.savez_compressed(
            abl / f"poly_d{d}.npz",
            lower=lo, upper=hi, support_L=Lsup, support_U=Usup,
            times=np.asarray(times, dtype=np.int64),
            degree=np.array([d]),
        )
        print(
            f"  d={d}: width={width:.4f}  exceed=({frac_lo:.4f},{frac_hi:.4f})  "
            f"[{lo[t_rep]:.4f},{hi[t_rep]:.4f}]  {dt:.1f}s"
            + ("  COLLAPSED" if row["collapsed"] else "")
        )
        if stop_on_collapse and row["collapsed"]:
            print(f"  stopping: collapse at d={d}")
            break

    with open(abl / "summary.json", "w") as f:
        json.dump(
            {
                "plant": name,
                "display_name": plant,
                "source": str(out),
                "degrees": used_degrees,
                "times": times,
                "alpha": alpha,
                "T": T,
                "finite_states": n_states,
                "rows": rows,
            },
            f,
            indent=2,
        )

    _plot_traj(plant, mean, mgf_L, mgf_U, C, curves, used_degrees, abl)
    _plot_hist(plant, mean, mgf_L, mgf_U, C, curves, used_degrees, T, abl)
    return rows


def _plot_traj(plant, mean, mgf_L, mgf_U, C, curves, degrees, abl):
    colors = _poly_colors(degrees)
    T = C.shape[1] - 1
    t = np.arange(T + 1)
    n_draw = min(600, C.shape[0])
    idx = np.linspace(0, C.shape[0] - 1, n_draw).astype(int)
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    ax.plot(t, (C[idx] - mean).T, color="#4c566a", alpha=0.9, lw=0.22, rasterized=True, zorder=1)
    ax.plot(t, np.zeros_like(t, dtype=float), color="#2e3440", lw=2.0, label="exact mean", zorder=3)
    q_lo = np.quantile(C, 0.025, axis=0) - mean
    q_hi = np.quantile(C, 0.975, axis=0) - mean
    ax.plot(t, q_lo, color="#2e3440", ls=":", lw=1.6, label="empirical 2.5%/97.5%", zorder=4)
    ax.plot(t, q_hi, color="#2e3440", ls=":", lw=1.6, zorder=4)
    ax.plot(t[1:], (mgf_L - mean)[1:], color=MGF_STYLE["color"], ls=MGF_STYLE["ls"], lw=MGF_STYLE["lw"],
            label="MGF", zorder=5)
    ax.plot(t[1:], (mgf_U - mean)[1:], color=MGF_STYLE["color"], ls=MGF_STYLE["ls"], lw=MGF_STYLE["lw"], zorder=5)
    for d in degrees:
        ts = np.asarray(curves[d]["times"], dtype=int)
        lo = curves[d]["lo"][ts] - mean[ts]
        hi = curves[d]["hi"][ts] - mean[ts]
        ax.plot(ts, lo, color=colors[d], lw=2.0, marker="o", ms=3.0, label=f"poly $d={d}$", zorder=6)
        ax.plot(ts, hi, color=colors[d], lw=2.0, marker="o", ms=3.0, zorder=6)
    span = max(
        float(np.nanmax(np.abs(mgf_L[1:] - mean[1:]))),
        float(np.nanmax(np.abs(mgf_U[1:] - mean[1:]))),
        float(np.abs(C[idx] - mean).max()),
    )
    ax.set_xlim(0, T)
    ax.set_ylim(-1.08 * span, 1.08 * span)
    ax.set_xlabel(r"time step $t$")
    ax.set_ylabel(r"$C_t-\mathbb{E}[C_t]$")
    ax.set_title("Pointwise 95% bounds vs polynomial degree", pad=8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(abl / "traj_degree_ablation.png", dpi=200, bbox_inches="tight")
    fig.savefig(abl / "traj_degree_ablation.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_hist(plant, mean, mgf_L, mgf_U, C, curves, degrees, T, abl):
    colors = _poly_colors(degrees)
    x = C[:, T] - mean[T]
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.hist(x, bins=70, weights=np.full(len(x), 1 / len(x)),
            color="#a3b1c6", edgecolor="#4c566a", linewidth=0.4, zorder=1)
    top = 0.74
    ax.axvline(0.0, ymax=top, color="#2e3440", lw=2.0, label="exact mean", zorder=3)
    q_lo, q_hi = np.quantile(x, [0.025, 0.975])
    ax.axvline(q_lo, ymax=top, color="#2e3440", ls=":", lw=1.6, label="empirical 2.5%/97.5%", zorder=4)
    ax.axvline(q_hi, ymax=top, color="#2e3440", ls=":", lw=1.6, zorder=4)
    ax.axvline(mgf_L[T] - mean[T], ymax=top, color=MGF_STYLE["color"], ls=MGF_STYLE["ls"],
               lw=MGF_STYLE["lw"], label="MGF", zorder=5)
    ax.axvline(mgf_U[T] - mean[T], ymax=top, color=MGF_STYLE["color"], ls=MGF_STYLE["ls"],
               lw=MGF_STYLE["lw"], zorder=5)
    for d in degrees:
        lo = curves[d]["lo"][T] - mean[T]
        hi = curves[d]["hi"][T] - mean[T]
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue
        ax.axvline(lo, ymax=top, color=colors[d], lw=2.0, label=f"poly $d={d}$", zorder=6)
        ax.axvline(hi, ymax=top, color=colors[d], lw=2.0, zorder=6)
    half = max(abs(mgf_L[T] - mean[T]), abs(mgf_U[T] - mean[T]), abs(q_lo), abs(q_hi), 1e-9)
    lim = 1.12 * max(half, float(np.abs(x).max()))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(0, ax.get_ylim()[1] / (top - 0.03))
    ax.set_xlabel(rf"$C_{{{T}}}-\mathbb{{E}}[C_{{{T}}}]$")
    ax.set_ylabel("fraction of trajectories")
    ax.set_title(f"Terminal histogram at $T={T}$ vs polynomial degree", pad=8)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(abl / "hist_degree_ablation.png", dpi=200, bbox_inches="tight")
    fig.savefig(abl / "hist_degree_ablation.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_width(plant, mean, mgf_L, mgf_U, curves, degrees, rows, abl):
    colors = _poly_colors(degrees)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for d in degrees:
        ts = np.asarray(curves[d]["times"], dtype=int)
        w = curves[d]["hi"][ts] - curves[d]["lo"][ts]
        ax.plot(ts, w, color=colors[d], lw=2.0, marker="o", ms=3.5, label=f"poly $d={d}$")
    ts_m = np.arange(1, len(mean))
    ax.plot(ts_m, (mgf_U - mgf_L)[ts_m], color=MGF_STYLE["color"], ls=MGF_STYLE["ls"],
            lw=2.2, label="MGF width")
    ax.set_xlabel(r"time step $t$")
    ax.set_ylabel("95% interval width")
    ax.set_title("Bound width vs polynomial degree", pad=8)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(abl / "width_vs_degree.png", dpi=200, bbox_inches="tight")
    fig.savefig(abl / "width_vs_degree.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ds = [r["degree"] for r in rows]
    widths = [r["width"] for r in rows]
    ax.bar([str(d) for d in ds], widths, color=[colors[d] for d in ds], edgecolor="#2e3440", width=0.65)
    mgf_w = rows[0]["mgf_U"] - rows[0]["mgf_L"]
    emp_w = rows[0]["emp_q975"] - rows[0]["emp_q025"]
    ax.axhline(mgf_w, color=MGF_STYLE["color"], ls="--", lw=2.0, label="MGF width")
    ax.axhline(emp_w, color="#2e3440", ls=":", lw=1.8, label="empirical 95% width")
    ax.set_xlabel("polynomial degree $d$")
    ax.set_ylabel(f"terminal width at $T={rows[0]['t']}$")
    ax.set_title("Terminal width ablation", pad=8)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(abl / "terminal_width_bars.png", dpi=200, bbox_inches="tight")
    fig.savefig(abl / "terminal_width_bars.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plants", nargs="*", default=DEFAULT_PLANTS)
    ap.add_argument("--degrees", default=",".join(str(d) for d in DEFAULT_DEGREES))
    ap.add_argument("--no-stop", action="store_true", help="do not stop when collapse is detected")
    args = ap.parse_args()
    degrees = [int(x) for x in args.degrees.split(",") if x.strip()]
    ABL.mkdir(parents=True, exist_ok=True)
    all_rows = {}
    for name in args.plants:
        all_rows[name] = run_plant(name, degrees, stop_on_collapse=not args.no_stop)

    lines = [
        "plant,degree,t,poly_L,poly_U,width,frac_below,frac_above,seconds,"
        "mgf_width,emp_width,collapsed"
    ]
    for name, rows in all_rows.items():
        for r in rows:
            lines.append(",".join(str(x) for x in [
                name, r["degree"], r["t"], f"{r['poly_L']:.6f}", f"{r['poly_U']:.6f}",
                f"{r['width']:.6f}", f"{r['frac_below']:.6f}", f"{r['frac_above']:.6f}",
                f"{r['seconds']:.2f}",
                f"{r['mgf_U']-r['mgf_L']:.6f}", f"{r['emp_q975']-r['emp_q025']:.6f}",
                int(r["collapsed"]),
            ]))
    (ABL / "table.csv").write_text("\n".join(lines) + "\n")
    print("=" * 64)
    print(f"wrote {ABL / 'table.csv'}")
    print((ABL / "table.csv").read_text())


if __name__ == "__main__":
    main()
