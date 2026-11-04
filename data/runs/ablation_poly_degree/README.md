# Polynomial-degree ablation archive

All numerical artifacts are kept for later reuse.

Per plant directory (`<plant>/`):
- `poly_d{d}.npz` — poly lower/upper/support arrays for that degree
- `summary.json` — terminal stats for the reported degree set
- `traj_degree_ablation.png/.pdf`, `hist_degree_ablation.png/.pdf`

Reported figures use `d ∈ {4,8,16,20,24}`.

Also retained (not plotted):
- `mass_spring_damper/poly_d32.npz`, `poly_d64.npz`
- `cartpole_linear/poly_d32.npz`, `poly_d64.npz`
  (d=64 collapses onto the mean; kept for reference)

Combined table: `table.csv`.
Source finite models / MGF / trajectories: `lqg_benchmark/outputs_paper_clamp/<plant>/`.
