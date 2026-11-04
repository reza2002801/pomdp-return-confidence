"""
Paper-algorithm experiments on quantized LQG plants:
  - finite reachable model
  - MGF via xi-recursion (TiltedOperator)
  - moments via paper moment recursion
  - polynomial bounds on exact [L_t, U_t]
  - Monte Carlo on the same finite model

  From repo root (with package on PYTHONPATH or pip install -e .):

    python -m experiments.lqg.run configs/mass_spring_damper.json
    python experiments/lqg/run.py configs/mass_spring_damper.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Allow running as a script without install.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from returnbounds import mgf
from returnbounds import continuous as gm
from returnbounds import io as save_artifacts
from returnbounds import moments as paper_moments
from returnbounds import plotting as figures
from returnbounds.build_quantized import build as finite_build_fn
from returnbounds.build_quantized import simulate as finite_simulate
from returnbounds.plant import design_report
from returnbounds.profiling import StageTimer
from returnbounds.quantize_setup import setup_from_config

HERE = Path(__file__).resolve().parent
# Prefer data/runs/<tag>/<plant> layout; fall back to local outputs/
DATA_ROOT = _ROOT / "data" / "runs"
OUT_DEFAULT = DATA_ROOT / "local"


def _load(path):
    with open(path) as f:
        return json.load(f)


def _dump(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def continuous_reference(plant, T, delta):
    """Closed-form continuous bounds for approximation reporting only."""
    an = gm.analyze_horizon(plant, T, delta=delta)
    return {
        "mean": an["mean"],
        "lower": an["lower"],
        "upper": an["upper"],
    }


def run_config(cfg, out_dir, log=print):
    timer = StageTimer()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup, plant = setup_from_config(cfg)
    design = design_report(plant)
    _dump(out_dir / "config.json", cfg)
    _dump(out_dir / "design.json", design)
    log(f"{plant.name} [paper recursion]: "
        f"rho(A-BK)={design['spectral_radius_A_minus_BK']:.4f}")

    T = int(max(cfg["horizons"]))
    delta = float(cfg.get("delta", 0.025))
    degree = int(cfg.get("poly_degree", 16))
    max_states = int(cfg.get("max_states", 25000))

    log("  building finite reachable model")
    with timer.stage("build_finite_model"):
        fm = finite_build_fn(setup, max_states=max_states, log=log)
    _dump(out_dir / "finite_pilot.json", fm.pilot)
    save_artifacts.save_finite_model(fm, out_dir / "finite_model.npz")
    log(f"  saved finite model → {out_dir / 'finite_model.npz'}  "
        f"(build={fm.build_seconds:.2f}s)")

    # --- MGF via xi-recursion ---
    with timer.stage("mgf"):
        op = mgf.TiltedOperator(fm)
        reach = mgf.reachable_sets(op, fm.x0, T)
        try:
            mgf.check_stochastic(op, reach, T)
        except AssertionError as e:
            log(f"  check_stochastic FAILED: {e}")
            raise SystemExit(1) from e
        horizons = list(cfg["horizons"])
        lam_cfg = cfg.get("lambda_grid", {
            "min": 0.01, "max": 100.0, "points_per_decade": 8, "max_expansions": 2,
        })
        log("  computing MGF via tilted xi-recursion")
        lams, log_m, valid, expansions, mgf_seconds = mgf.bounds_with_expansion(
            op, fm.x0, T, horizons, [delta], lam_cfg, reach, log=log)
        upper = np.full(T + 1, np.nan)
        for t in range(1, T + 1):
            upper[t], _ = mgf.chernoff(log_m[t], valid[t], lams, delta)
        lower, lams_neg = mgf.lower_bounds(
            op, fm.x0, T, reach, lams, delta, horizons,
            per_decade=lam_cfg.get("points_per_decade", 8),
            max_expansions=lam_cfg.get("max_expansions", 2),
        )
        mean, var, occ = mgf.exact_moments(op, fm.x0, T)

    # --- Polynomial via paper moments + exact support ---
    log(f"  computing polynomial bounds (d={degree}) via moment recursion")
    with timer.stage("poly"):
        poly_times = list(range(1, T + 1))
        poly_lower, poly_upper, Lsup, Usup, moments = paper_moments.poly_intervals(
            fm, T, degree, alpha=delta, times=poly_times, log=log)

    # --- Monte Carlo on finite model ---
    sim_cfg = cfg["simulation"]
    log(f"  simulating {sim_cfg['n_traj']} finite-model trajectories")
    with timer.stage("simulate"):
        C = finite_simulate(fm, sim_cfg["n_traj"], T, sim_cfg["seed"])

    with timer.stage("continuous_reference"):
        cont = continuous_reference(plant, T, delta)

    n = C.shape[0]
    below = int(np.sum(C[:, T] < lower[T]))
    above = int(np.sum(C[:, T] > upper[T]))
    p_below = int(np.sum(C[:, T] < poly_lower[T]))
    p_above = int(np.sum(C[:, T] > poly_upper[T]))

    summary = {
        "method": "paper xi-recursion MGF + paper moment recursion polynomial on finite model",
        "name": plant.name,
        "T": T,
        "dt": plant.dt,
        "physical_time_T": T * plant.dt,
        "finite_states": fm.n_states,
        "finite_edges": fm.n_edges,
        "poly_degree": degree,
        "exact_mean": float(mean[T]),
        "exact_var": float(var[T]),
        "sim_mean": float(C[:, T].mean()),
        "lower": float(lower[T]),
        "upper": float(upper[T]),
        "poly_lower": float(poly_lower[T]),
        "poly_upper": float(poly_upper[T]),
        "support_L": float(Lsup[T]),
        "support_U": float(Usup[T]),
        "empirical_q025": float(np.quantile(C[:, T], 0.025)),
        "empirical_q975": float(np.quantile(C[:, T], 0.975)),
        "frac_below_mgf": below / n,
        "frac_above_mgf": above / n,
        "frac_below_poly": p_below / n,
        "frac_above_poly": p_above / n,
        "continuous_reference": cont,
        "approx_vs_continuous": {
            "mean_abs_diff": abs(float(mean[T]) - cont["mean"]),
            "mean_rel_diff": abs(float(mean[T]) - cont["mean"]) / max(abs(cont["mean"]), 1e-12),
            "mgf_upper_abs_diff": abs(float(upper[T]) - cont["upper"]),
            "mgf_upper_rel_diff": abs(float(upper[T]) - cont["upper"]) / max(abs(cont["upper"]), 1e-12),
            "mgf_lower_abs_diff": abs(float(lower[T]) - cont["lower"]),
            "mgf_width_finite": float(upper[T] - lower[T]),
            "mgf_width_continuous": cont["upper"] - cont["lower"],
        },
        "mgf_seconds": mgf_seconds,
        "build_seconds": float(fm.build_seconds),
        "lambda_expansions": expansions,
        "profiling": timer.as_dict(),
        "wall_seconds": timer.wall_seconds(),
        "max_rss_kb": timer.as_dict()["max_rss_kb"],
        "guarantee": (
            "Bounds are for the finite quantized model with finite-support noise. "
            "MGF from xi-recursion; polynomial from moment recursion on exact [L_t,U_t]. "
            "They are not certificates for the continuous Gaussian plant."
        ),
    }
    _dump(out_dir / "summary.json", summary)
    _dump(out_dir / "profiling.json", summary["profiling"])

    rows = ["t,mean,var,mgf_L,mgf_U,poly_L,poly_U,support_L,support_U"]
    for t in range(1, T + 1):
        rows.append(",".join(str(x) for x in [
            t, mean[t], var[t], lower[t], upper[t],
            poly_lower[t], poly_upper[t], Lsup[t], Usup[t],
        ]))
    (out_dir / "results.csv").write_text("\n".join(rows) + "\n")

    save_artifacts.save_run_arrays(
        out_dir,
        mean=mean, var=var, lower=lower, upper=upper,
        poly_lower=poly_lower, poly_upper=poly_upper,
        Lsup=Lsup, Usup=Usup, moments=moments, C=C,
        lams=lams, log_m=log_m, valid=valid,
    )
    log(f"  saved arrays → {out_dir / 'arrays.npz'} (+ mgf_grid.npz)")

    figures.cost_trajectories(
        C, mean, lower, upper,
        out_dir / "cost_trajectories_95pct.png",
        f"{plant.name} (finite model, paper recursion)",
        n_draw=sim_cfg.get("n_draw", 800),
        poly_lower=poly_lower, poly_upper=poly_upper,
    )
    figures.histogram(
        C[:, T], mean[T], lower[T], upper[T], T,
        out_dir / f"histogram_T{T}.png", plant.name + " (finite)",
        poly_lower=float(poly_lower[T]), poly_upper=float(poly_upper[T]),
    )

    log(f"  T={T}: mean={mean[T]:.4f}  MGF=[{lower[T]:.4f},{upper[T]:.4f}]  "
        f"poly=[{poly_lower[T]:.4f},{poly_upper[T]:.4f}]")
    log(f"  wrote {out_dir} in {summary['wall_seconds']:.1f}s")
    return summary


def main(argv):
    path = argv[1] if len(argv) > 1 else str(HERE / "configs" / "mass_spring_damper.json")
    cfg = _load(path)
    out = OUT_DEFAULT / cfg["name"]
    run_config(cfg, out)


if __name__ == "__main__":
    main(sys.argv)
