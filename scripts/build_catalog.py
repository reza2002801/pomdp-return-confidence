#!/usr/bin/env python3
"""Build data/catalog/index.json from lightweight run summaries under data/runs/."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "data" / "runs"
CAT = ROOT / "data" / "catalog"


def main():
    CAT.mkdir(parents=True, exist_ok=True)
    index = {"runs": {}}
    for run_dir in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        plants = {}
        for plant_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            summary = plant_dir / "summary.json"
            if not summary.exists():
                # ablation plant folders also have summary.json
                continue
            s = json.loads(summary.read_text())
            plants[plant_dir.name] = {
                "path": str(plant_dir.relative_to(ROOT)),
                "name": s.get("name", plant_dir.name),
                "T": s.get("T"),
                "poly_degree": s.get("poly_degree"),
                "finite_states": s.get("finite_states"),
                "exact_mean": s.get("exact_mean"),
                "lower": s.get("lower"),
                "upper": s.get("upper"),
                "poly_lower": s.get("poly_lower"),
                "poly_upper": s.get("poly_upper"),
                "wall_seconds": s.get("wall_seconds"),
                "max_rss_kb": s.get("max_rss_kb"),
                "mgf_seconds": s.get("mgf_seconds"),
                "build_seconds": s.get("build_seconds"),
                "profiling": s.get("profiling"),
                "has_arrays": (plant_dir / "arrays.npz").exists(),
                "has_finite_model": (plant_dir / "finite_model.npz").exists(),
                "files": sorted(f.name for f in plant_dir.iterdir() if f.is_file()),
            }
        if plants:
            index["runs"][run_dir.name] = plants
    out = CAT / "index.json"
    out.write_text(json.dumps(index, indent=2) + "\n")
    print(f"wrote {out}  ({sum(len(v) for v in index['runs'].values())} plant entries)")


if __name__ == "__main__":
    main()
