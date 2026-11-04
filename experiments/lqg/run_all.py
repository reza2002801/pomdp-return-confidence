"""Run all paper LQG plants: python experiments/lqg/run_all.py"""

from pathlib import Path

from run import run_config, _load, HERE, OUT_DEFAULT

CONFIGS = [
    "mass_spring_damper.json",
    "two_room_thermal.json",
    "cartpole_linear.json",
    "coupled_masses.json",
    "quarter_car.json",
    "drone_hover.json",
]


def main():
    summaries = []
    for name in CONFIGS:
        path = HERE / "configs" / name
        cfg = _load(path)
        print("=" * 60)
        try:
            s = run_config(cfg, OUT_DEFAULT / cfg["name"])
            summaries.append(s)
        except Exception as e:
            print(f"FAILED {name}: {e}")
            import traceback
            traceback.print_exc()
            summaries.append({"name": name, "ok": False, "error": str(e)})
    print("=" * 60)
    print("PAPER SUITE SUMMARY")
    for s in summaries:
        if s.get("ok") is False or ("error" in s and "exact_mean" not in s):
            print(s)
            continue
        print(
            f"{s['name']}: states={s.get('finite_states')}  "
            f"mean={s['exact_mean']:.4g}  "
            f"MGF=[{s['lower']:.4g},{s['upper']:.4g}]  "
            f"poly=[{s['poly_lower']:.4g},{s['poly_upper']:.4g}]"
        )


if __name__ == "__main__":
    main()
