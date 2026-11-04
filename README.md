# Confidence-Intervals

Finite-horizon **return confidence levels** for a fixed agent-state policy on a
**quantized finite model** (L4DC experiments).

Two bound methods:

1. **Chernoff / exact MGF** — tilted ξ-recursion on the finite MDP  
2. **Polynomial moment bounds** — moment recursion + TailLP on exact support \([L_t,U_t]\)

This repo is the cleaned, self-contained home for the paper code + experiment
artifacts (lightweight summaries/figures in git; heavy `.npz` local-only).

---

## Layout

```
returnbounds/              library (algorithms only)
  mgf.py                     tilted MGF + Chernoff
  moments.py                 moment recursion, support DP, poly intervals
  polybound.py               Chebyshev TailLP
  build_quantized.py         reachable finite model + simulate
  profiling.py               wall time + peak RSS by stage
  …

experiments/lqg/           paper LQG suite
  configs/*.json             six plants
  run.py / run_all.py        full pipeline
  ablate_poly_degree.py      moment-degree ablation
  make_figures.py            replot from arrays.npz

data/runs/<tag>/<plant>/   experiment store (see data/README.md)
data/catalog/index.json    index of summaries / profiling

scripts/
  build_catalog.py
  sync_heavy_artifacts.py    pull large npz from a sibling source tree
```

---

## Quick start

```bash
cd Confidence-Intervals
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# one plant → data/runs/local/<name>/
python experiments/lqg/run.py experiments/lqg/configs/mass_spring_damper.json

# all six (hours)
python experiments/lqg/run_all.py

# replot (needs arrays.npz — sync first if missing)
python scripts/sync_heavy_artifacts.py
python experiments/lqg/make_figures.py mass_spring_damper

# poly-degree ablation (needs finite_model.npz + arrays.npz)
python experiments/lqg/ablate_poly_degree.py mass_spring_damper --degrees 8,16,32
```

---

## Runtime / memory tracking

Full runs already record:

| Field | Meaning |
|-------|---------|
| `wall_seconds` | end-to-end wall clock |
| `max_rss_kb` | peak RSS (`resource.ru_maxrss`) |
| `mgf_seconds` / `build_seconds` | stage timers from MGF / builder |
| `profiling.stage_seconds` | `build_finite_model`, `mgf`, `poly`, `simulate`, … |
| `profiling.rss_after_kb` | RSS snapshot after each stage |

Ablations record per-degree `seconds` and `max_rss_kb` in their `summary.json`.

---

## Data retrieval

- **Numbers & figures for the paper clamp suite** live under  
  `data/runs/paper_clamp_v1/<plant>/` (`summary.json`, `results.csv`, PNGs, configs).
- **Heavy arrays** (`arrays.npz`, `finite_model.npz`) are gitignored. Restore with  
  `python scripts/sync_heavy_artifacts.py` when the sibling `exact_mgf_experiments`
  tree is available, or re-run `experiments/lqg/run.py`.
- Rebuild the catalog: `python scripts/build_catalog.py`.

---

## Tests

```bash
pytest -q
```
