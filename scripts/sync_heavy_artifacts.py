#!/usr/bin/env python3
"""
Copy large .npz artifacts from a source tree into data/runs/<tag>/<plant>/.

Default source is the sibling exact_mgf_experiments clamp outputs (if present).

  python scripts/sync_heavy_artifacts.py
  python scripts/sync_heavy_artifacts.py --tag paper_clamp_v1 --src /path/to/outputs_paper_clamp
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = ROOT.parent / "exact_mgf_experiments" / "lqg_benchmark" / "outputs_paper_clamp"
HEAVY = ("arrays.npz", "finite_model.npz", "mgf_grid.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="paper_clamp_v1")
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    args = ap.parse_args()
    dst_root = ROOT / "data" / "runs" / args.tag
    if not args.src.exists():
        raise SystemExit(f"source not found: {args.src}")
    for plant_dir in sorted(p for p in args.src.iterdir() if p.is_dir()):
        dest = dst_root / plant_dir.name
        dest.mkdir(parents=True, exist_ok=True)
        for name in HEAVY:
            src = plant_dir / name
            if src.exists():
                shutil.copy2(src, dest / name)
                print(f"  {plant_dir.name}/{name} → {dest / name} ({src.stat().st_size/1e6:.1f} MB)")
    print("done — heavy artifacts are gitignored; keep them local for ablations/replots")


if __name__ == "__main__":
    main()
