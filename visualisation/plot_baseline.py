"""Plot baseline-policy training logs for the Franka thesis.

This entrypoint is for the standard baseline family. It reads all available TensorBoard history and
reports the highest last-epoch checkpoint found in each seed run.

Usage from the repo root:

    .\isaaclab.bat -p visualisation\plot_baseline.py

If auto-discovery does not find the runs, pass them explicitly:

    .\isaaclab.bat -p visualisation\plot_baseline.py ^
        --run_dirs logs\rl_games\baseline\baseline42 ^
                   logs\rl_games\baseline\baseline123 ^
                   logs\rl_games\baseline\baseline789
"""

from training_log_plotter import PROJECT_ROOT, PlotConfig, run


# Auto-discovers the three baseline seed runs. The franka_lift fallback is kept
# for runs produced before they were moved under logs/rl_games/baseline.
CONFIG = PlotConfig(
    family_name="Baseline",
    title_prefix="Baseline Training",
    output_name="baseline",
    log_roots=(
        PROJECT_ROOT / "logs" / "rl_games" / "baseline",
        PROJECT_ROOT / "logs" / "rl_games" / "franka_lift",
    ),
    run_patterns=(
        "baseline{seed}",
        "baseline_s{seed}",
        "baseline-s{seed}",
        "*BL*s{seed}*",
        "*Baseline*s{seed}*",
        "*baseline*s{seed}*",
        "*TRUE*s{seed}*",
        "*franka_lift*s{seed}*",
        "*lift*s{seed}*",
    ),
)


if __name__ == "__main__":
    run(CONFIG)
