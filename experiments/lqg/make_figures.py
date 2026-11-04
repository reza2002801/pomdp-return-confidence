"""
Regenerate traj/hist figures from saved arrays.npz — no MGF recompute.

  python experiments/lqg/make_figures.py [plant ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from returnbounds import io as save_artifacts
from returnbounds import plotting as figures

HERE = Path(__file__).resolve().parent
SRC_CANDIDATES = [
    _ROOT / "data" / "runs" / "local",
    _ROOT / "data" / "runs" / "paper_clamp_v1",
    HERE / "outputs",
]

PLANTS = [
    "mass_spring_damper", "two_room_thermal", "cartpole_linear",
    "coupled_masses", "quarter_car", "drone_hover",
]


def _out_dir(name: str) -> Path:
    for root in SRC_CANDIDATES:
        if (root / name / "arrays.npz").exists():
            return root / name
    raise FileNotFoundError(
        f"no arrays.npz for {name}; looked in {SRC_CANDIDATES}. "
        f"Run scripts/sync_heavy_artifacts.py or experiments/lqg/run.py first."
    )


def replot(name: str):
    out = _out_dir(name)
    data = save_artifacts.load_run_arrays(out)
    summary = json.load(open(out / "summary.json"))
    cfg = json.load(open(out / "config.json"))
    T = int(max(cfg["horizons"]))
    n_draw = cfg.get("simulation", {}).get("n_draw", 800)
    plant = summary.get("name", name)

    mean = data["mean"]
    lower, upper = data["mgf_lower"], data["mgf_upper"]
    poly_lower, poly_upper = data["poly_lower"], data["poly_upper"]
    Lsup, Usup = data["support_L"], data["support_U"]
    C = data["C"]

    figures.cost_trajectories(
        C, mean, lower, upper,
        out / "cost_trajectories_95pct.png",
        f"{plant} (finite model, paper recursion)",
        n_draw=n_draw,
        poly_lower=poly_lower, poly_upper=poly_upper,
        support_L=Lsup, support_U=Usup,
    )
    figures.histogram(
        C[:, T], mean[T], lower[T], upper[T], T,
        out / f"histogram_T{T}.png", plant + " (finite)",
        poly_lower=float(poly_lower[T]), poly_upper=float(poly_upper[T]),
    )
    print(f"  replotted {name} from {out}")


def main(argv):
    names = argv[1:] if len(argv) > 1 else PLANTS
    for name in names:
        replot(name)


if __name__ == "__main__":
    main(sys.argv)
