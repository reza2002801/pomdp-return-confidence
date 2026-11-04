# Experiment data layout

```
data/
  runs/
    paper_clamp_v1/<plant>/     # paper suite snapshot (lightweight files in git)
    ablation_poly_degree/...    # poly-degree ablation tables/plots
    local/<plant>/              # fresh runs from experiments/lqg/run.py (gitignored)
  catalog/
    index.json                  # machine-readable index of all run summaries
```

## What is committed vs local-only

| Artifact | In git? | Purpose |
|----------|---------|---------|
| `config.json`, `design.json`, `summary.json`, `results.csv` | yes | reproduce settings + bound numbers |
| `*.png` figures | yes | paper / inspection |
| `profiling.json` (new runs) | yes | wall time + peak RSS by stage |
| `arrays.npz`, `finite_model.npz`, `mgf_grid.npz` | **no** (gitignored) | replot / ablate without recompute |

Pull heavy `.npz` files onto a machine with:

```bash
python scripts/sync_heavy_artifacts.py --tag paper_clamp_v1
```

Refresh the catalog after adding runs:

```bash
python scripts/build_catalog.py
```

## Profiling fields (for ablations)

New full runs write `summary.json` / `profiling.json` with:

- `wall_seconds`, `max_rss_kb`
- `stage_seconds`: `build_finite_model`, `mgf`, `poly`, `simulate`, …
- `rss_after_kb` per stage
- `mgf_seconds`, `build_seconds`

Poly-degree ablations also record per-degree `seconds` and `max_rss_kb`.
