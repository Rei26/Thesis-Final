"""Plot R16/main obstacle-policy training logs for the Franka thesis.

This entrypoint is for the obstacle-aware main condition. It reads the full TensorBoard history and
reports the highest last-epoch checkpoint found in each seed run.

Usage from the repo root:

    .\isaaclab.bat -p visualisation\plot_main_condition.py

If auto-discovery does not find the runs, pass them explicitly:

    .\isaaclab.bat -p visualisation\plot_main_condition.py ^
        --run_dirs logs\rl_games\obstacle_training\R16_final_s42 ^
                   logs\rl_games\obstacle_training\R16_final_s123 ^
                   logs\rl_games\obstacle_training\R16_final_s789
"""

from training_log_plotter import PROJECT_ROOT, PlotConfig, run


# Auto-discovers the R16/main-condition seed runs. The franka_grasping fallback
# supports logs created with the raw RL-Games experiment name.
CONFIG = PlotConfig(
    family_name="R16",
    title_prefix="R16 Main Condition Training",
    output_name="r16_main_condition",
    log_roots=(
        PROJECT_ROOT / "logs" / "rl_games" / "obstacle_training",
        PROJECT_ROOT / "logs" / "rl_games" / "franka_grasping",
    ),
    run_patterns=(
        "R16_final_s{seed}",
        "*R16*s{seed}*",
        "*r16*s{seed}*",
        "*main*s{seed}*",
        "*randobs*s{seed}*",
        "*obstacle*s{seed}*",
        "*grasping*s{seed}*",
    ),
)


if __name__ == "__main__":
    run(CONFIG)
