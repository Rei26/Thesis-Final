# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Generalization evaluation script for the Franka grasping thesis.

Tests each trained policy checkpoint in a specified test environment and records
per-episode metrics averaged over N evaluation episodes:

  grasp_success_rate    â€” fraction of episodes where object z > EVAL_GRASP_THRESHOLD
                          for at least CONSECUTIVE_STEPS_REQUIRED consecutive steps
                          after warmup (binary per episode, averaged to a rate).
  collision_frequency   â€” fraction of steps per episode where the EE is within
                          SURFACE_COLLISION_THRESHOLD of the nearest obstacle surface.
  near_obstacle_rate    â€” fraction of steps per episode where the EE is within
                          NEAR_OBSTACLE_THRESHOLD of the nearest obstacle surface.
  mean_clearance        â€” average per-step EE clearance to the nearest obstacle surface.
  min_clearance         â€” closest EE approach to any obstacle surface within the episode.
  max_object_z          â€” maximum object height reached during the episode.
  task_completion_time  â€” first step where the sustained-hold success criterion
                          is satisfied (250 = timeout / never achieved).

Observation-space mismatch between policy and test environment is resolved by
padding missing tail features with the checkpoint running-mean values (so they
normalize to zero inside rl_games) or truncating extra tail features when the
environment provides more than the policy expects.

Usage â€” run once per test environment from /home/rei/IsaacLab in WSL:

  ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/evaluate_generalization.py \\
      --task Isaac-Franka-Grasping-v0 \\
      --checkpoints \\
          logs/rl_games/franka_grasping/2026-03-13_16-40-40/nn/franka_grasping.pth \\
          logs/rl_games/franka_grasping/2026-03-30_19-05-34/nn/franka_grasping.pth \\
          logs/rl_games/franka_grasping/2026-03-30_23-17-16/nn/franka_grasping.pth \\
      --policy_labels A_s42 A_s123 A_s789 \\
      --num_episodes 100 --num_envs 16 \\
      --output_csv eval_results.csv

Use visualisation/run_evaluation.sh to run all 4 test envs automatically.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# CLI â€” must come before any other Isaac Lab imports
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Generalization evaluation: test each policy checkpoint in one test env."
)
parser.add_argument("--task", type=str, required=True,
                    help="Gym task id of the TEST environment (e.g. Isaac-Franka-Grasping-v0).")
parser.add_argument("--checkpoints", type=str, nargs="+", required=True,
                    help="Paths to .pth checkpoint files, one per policy.")
parser.add_argument("--policy_labels", type=str, nargs="+", required=True,
                    help="Short labels for each checkpoint (same order). E.g. A_s42 B_s123.")
parser.add_argument("--num_episodes", type=int, default=100,
                    help="Episodes to collect per (policy, env) pair.")
parser.add_argument("--num_envs", type=int, default=16,
                    help="Number of parallel environments.")
parser.add_argument("--seed", type=int, default=42,
                    help="Environment seed for eval runs.")
parser.add_argument("--output_csv", type=str, default="eval_results.csv",
                    help="CSV file to append results to (created if absent).")
parser.add_argument("--output_plot_dir", type=str, default=None,
                    help="Directory for evaluation PNG plots. Defaults to a folder derived from output_csv and task.")
parser.add_argument("--video", action="store_true", default=False,
                    help="Record one deterministic evaluation video per checkpoint after metrics are computed.")
parser.add_argument("--video_length", type=int, default=250,
                    help="Maximum length of each recorded evaluation video in steps.")
parser.add_argument("--output_video_dir", type=str, default=None,
                    help="Directory for evaluation videos. Defaults to a folder derived from output_csv and task.")
parser.add_argument("--video_seed", type=int, default=None,
                    help="Seed used for deterministic video rollouts. Defaults to --seed.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# All other imports AFTER AppLauncher
# ---------------------------------------------------------------------------
import copy
import csv
import math
import os
import shutil

import gymnasium as gym
import matplotlib
import numpy as np
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401 â€” registers all tasks including our grasping envs

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Evaluation constants
# ---------------------------------------------------------------------------

EVAL_OBJECT_REST_HEIGHT = 0.055
EVAL_GRASP_THRESHOLD = EVAL_OBJECT_REST_HEIGHT + 0.02  # align with env grasp-success gate
CONSECUTIVE_STEPS_REQUIRED = 50      # 50 steps ~= 1 second at 50 Hz RL control
SURFACE_COLLISION_THRESHOLD = 0.04   # surface-distance collision radius (m)
NEAR_OBSTACLE_THRESHOLD = 0.08       # wider proximity band for more sensitive eval
GRASP_WARMUP_STEPS = 15              # skip first N steps per episode

PLOT_SPECS = [
    ("grasp_success_mean", "grasp_success_std", "Grasp Success Rate", "Success Rate", (0.0, 1.0)),
    ("collision_freq_mean", "collision_freq_std", "Collision Frequency", "Fraction of Steps", (0.0, 1.0)),
    ("near_obstacle_mean", "near_obstacle_std", "Near-Obstacle Rate", "Fraction of Steps", (0.0, 1.0)),
    ("mean_clearance_mean", "mean_clearance_std", "Mean Clearance", "Meters", (0.0, None)),
    ("min_clearance_mean", "min_clearance_std", "Minimum Clearance", "Meters", (0.0, None)),
    ("max_object_z_mean", "max_object_z_std", "Maximum Object Height", "Meters", (0.0, None)),
    ("task_time_mean", "task_time_std", "Task Completion Time", "Steps", (0.0, None)),
]


# ---------------------------------------------------------------------------
# Per-task obstacle configuration (half-extents from HANDOFF2 Â§6)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fix_object_reset_to_default(env_cfg: ManagerBasedRLEnvCfg) -> None:
    """Force evaluation environments to keep the cube at its default spawn pose."""

    if not hasattr(env_cfg, "events") or not hasattr(env_cfg.events, "reset_object_position"):
        return
    env_cfg.events.reset_object_position.params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
    }


def get_obstacle_names(isaac_env: ManagerBasedRLEnv) -> list[str]:
    """List obstacle assets present in the current environment."""

    return [name for name in ("obstacle_1", "obstacle_2", "obstacle_3", "obstacle_4") if name in isaac_env.scene.keys()]


def get_runtime_obstacle_half_extents(
    isaac_env: ManagerBasedRLEnv,
    obstacle_name: str,
    device: torch.device,
) -> torch.Tensor:
    """Resolve obstacle half-extents from runtime env buffers or static spawn sizes."""

    if hasattr(isaac_env, "_obstacle_half_extents") and obstacle_name in isaac_env._obstacle_half_extents:
        return isaac_env._obstacle_half_extents[obstacle_name]

    obstacle = isaac_env.scene[obstacle_name]
    spawn_size = getattr(obstacle.cfg.spawn, "size", None)
    if spawn_size is None:
        raise ValueError(f"Obstacle '{obstacle_name}' does not expose a cuboid spawn size for evaluation.")
    full_size = torch.tensor(spawn_size, dtype=torch.float32, device=device)
    return 0.5 * full_size.unsqueeze(0).repeat(isaac_env.num_envs, 1)


def _slugify(text: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in text)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def resolve_plot_dir(csv_path: str, task_name: str, explicit_plot_dir: str | None) -> str:
    """Choose a stable output directory for evaluation plots."""

    if explicit_plot_dir:
        return explicit_plot_dir
    csv_abs = os.path.abspath(csv_path)
    csv_parent = os.path.dirname(csv_abs) or "."
    csv_stem = os.path.splitext(os.path.basename(csv_abs))[0]
    return os.path.join(csv_parent, f"{csv_stem}__{_slugify(task_name)}_plots")


def resolve_video_dir(csv_path: str, task_name: str, explicit_video_dir: str | None) -> str:
    """Choose a stable output directory for evaluation videos."""

    if explicit_video_dir:
        return explicit_video_dir
    csv_abs = os.path.abspath(csv_path)
    csv_parent = os.path.dirname(csv_abs) or "."
    csv_stem = os.path.splitext(os.path.basename(csv_abs))[0]
    return os.path.join(csv_parent, f"{csv_stem}__{_slugify(task_name)}_videos")


def _plot_metric_bars(
    labels: list[str],
    means: np.ndarray,
    stds: np.ndarray,
    title: str,
    ylabel: str,
    ylim: tuple[float | None, float | None],
    output_path: str,
) -> None:
    """Save one bar plot with error bars for a single evaluation metric."""

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 4.8))
    x = np.arange(len(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    y_min, y_max = ylim
    bottom = y_min if y_min is not None else np.nanmin(means - stds)
    top = y_max if y_max is not None else np.nanmax(means + stds)
    if np.isfinite(bottom) and np.isfinite(top) and top > bottom:
        margin = 0.05 * (top - bottom if top > bottom else 1.0)
        ax.set_ylim(bottom, top + margin)

    for bar, value in zip(bars, means):
        if not np.isfinite(value):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_evaluation_plots(all_results: list[dict], task_name: str, output_plot_dir: str) -> list[str]:
    """Create a dashboard plus one PNG per metric for the current evaluation batch."""

    os.makedirs(output_plot_dir, exist_ok=True)
    labels = [result["policy_label"] for result in all_results]
    saved_paths: list[str] = []
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

    valid_specs = []
    for mean_key, std_key, title, ylabel, ylim in PLOT_SPECS:
        means = np.array([result[mean_key] for result in all_results], dtype=float)
        stds = np.array([result[std_key] for result in all_results], dtype=float)
        if np.all(np.isnan(means)):
            continue
        valid_specs.append((mean_key, std_key, title, ylabel, ylim, means, stds))
        filename = f"{_slugify(mean_key)}.png"
        output_path = os.path.join(output_plot_dir, filename)
        _plot_metric_bars(labels, means, stds, title, ylabel, ylim, output_path)
        saved_paths.append(output_path)

    if not valid_specs:
        return saved_paths

    n_cols = 3
    n_rows = math.ceil(len(valid_specs) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    x = np.arange(len(labels))

    for ax, (_, _, title, ylabel, ylim, means, stds) in zip(axes, valid_specs):
        ax.bar(
            x,
            means,
            yerr=stds,
            capsize=4,
            color=colors,
            edgecolor="black",
            linewidth=0.8,
        )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        y_min, y_max = ylim
        if y_min is not None:
            ax.set_ylim(bottom=y_min)
        if y_max is not None:
            ax.set_ylim(top=y_max)

    for ax in axes[len(valid_specs):]:
        ax.axis("off")

    fig.suptitle(f"Evaluation Summary: {task_name}", fontsize=14)
    fig.tight_layout()
    dashboard_path = os.path.join(output_plot_dir, "evaluation_dashboard.png")
    fig.savefig(dashboard_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(dashboard_path)
    return saved_paths


def adapt_obs(obs: torch.Tensor, policy_obs_dim: int, running_mean: torch.Tensor | None = None) -> torch.Tensor:
    """Pad missing features with checkpoint running-mean values or truncate extras."""
    env_dim = obs.shape[-1]
    if env_dim == policy_obs_dim:
        return obs
    if env_dim < policy_obs_dim:
        pad_width = policy_obs_dim - env_dim
        if running_mean is not None and running_mean.shape[0] >= policy_obs_dim:
            tail = running_mean[env_dim:policy_obs_dim].to(device=obs.device, dtype=obs.dtype)
            view_shape = [1] * (obs.dim() - 1) + [pad_width]
            pad = tail.view(*view_shape).expand(*obs.shape[:-1], pad_width)
        else:
            pad = torch.zeros(*obs.shape[:-1], pad_width, device=obs.device, dtype=obs.dtype)
        return torch.cat([obs, pad], dim=-1)
    return obs[..., :policy_obs_dim]


def get_policy_obs_stats(checkpoint_path: str) -> tuple[int, torch.Tensor | None]:
    """Determine obs_dim and running_mean by inspecting checkpoint stats."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ckpt.get("model", ckpt)

    # Primary: running_mean_std.running_mean has shape (obs_dim,)
    for key, val in model.items():
        if (isinstance(val, torch.Tensor) and val.dim() == 1
                and "running_mean_std" in key and "running_mean" in key):
            return val.shape[0], val.detach().cpu().float()

    # Fallback: first 2-D weight is (hidden_dim, obs_dim)
    for key in sorted(model.keys()):
        val = model[key]
        if isinstance(val, torch.Tensor) and val.dim() == 2:
            return val.shape[1], None

    raise ValueError(f"Cannot determine obs_dim from checkpoint: {checkpoint_path}")


def load_agent_cfg(yaml_path: str) -> dict:
    import yaml
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def build_env_cfg(task_name: str, num_envs: int, seed: int):
    """Instantiate a task config with fixed object reset and requested env count."""

    cfg_entry = gym.spec(task_name).kwargs["env_cfg_entry_point"]
    if isinstance(cfg_entry, str):
        import importlib

        module_str, class_str = cfg_entry.rsplit(":", 1)
        env_cfg = getattr(importlib.import_module(module_str), class_str)()
    else:
        env_cfg = cfg_entry()
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    fix_object_reset_to_default(env_cfg)
    return env_cfg


def resolve_video_task_name(task_name: str) -> str:
    """Prefer the matching Play task for nicer camera framing when available."""

    if task_name.endswith("-v0"):
        candidate = task_name[:-3] + "-Play-v0"
        try:
            gym.spec(candidate)
            return candidate
        except Exception:
            pass
    return task_name


def _latest_recorded_video(video_folder: str) -> str | None:
    """Return the newest mp4 in a RecordVideo output folder."""

    if not os.path.isdir(video_folder):
        return None
    candidates = [
        os.path.join(video_folder, name)
        for name in os.listdir(video_folder)
        if name.lower().endswith(".mp4")
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def ee_near_obstacle_surface(
    ee_pos: torch.Tensor,           # (N, 3)
    obstacle_center: torch.Tensor,  # (N, 3)
    half_extents: torch.Tensor,     # (3,)
    threshold: float = SURFACE_COLLISION_THRESHOLD,
) -> torch.Tensor:                  # (N,) bool
    diff = torch.abs(ee_pos - obstacle_center) - half_extents.to(ee_pos.device)
    return torch.norm(diff.clamp(min=0.0), dim=-1) < threshold


def ee_surface_distance(
    ee_pos: torch.Tensor,           # (N, 3)
    obstacle_center: torch.Tensor,  # (N, 3)
    half_extents: torch.Tensor,     # (3,)
) -> torch.Tensor:                  # (N,)
    diff = torch.abs(ee_pos - obstacle_center) - half_extents.to(ee_pos.device)
    return torch.norm(diff.clamp(min=0.0), dim=-1)


def compute_ee_near_any_obstacle(
    isaac_env: ManagerBasedRLEnv,
    obstacle_names: list[str],
    device: torch.device,
) -> torch.Tensor:  # (num_envs,) bool
    if not obstacle_names:
        return torch.zeros(isaac_env.num_envs, dtype=torch.bool, device=device)
    ee_pos = isaac_env.scene["ee_frame"].data.target_pos_w[:, 0, :]  # (N, 3)
    near_any = torch.zeros(isaac_env.num_envs, dtype=torch.bool, device=device)
    for obstacle_name in obstacle_names:
        center = isaac_env.scene[obstacle_name].data.root_pos_w[:, :3]
        half = get_runtime_obstacle_half_extents(isaac_env, obstacle_name, device)
        near_any |= ee_near_obstacle_surface(ee_pos, center, half)
    return near_any


def compute_min_ee_obstacle_clearance(
    isaac_env: ManagerBasedRLEnv,
    obstacle_names: list[str],
    device: torch.device,
) -> torch.Tensor:  # (num_envs,)
    if not obstacle_names:
        return torch.full((isaac_env.num_envs,), float("inf"), dtype=torch.float32, device=device)
    ee_pos = isaac_env.scene["ee_frame"].data.target_pos_w[:, 0, :]  # (N, 3)
    clearances = []
    for obstacle_name in obstacle_names:
        center = isaac_env.scene[obstacle_name].data.root_pos_w[:, :3]
        half = get_runtime_obstacle_half_extents(isaac_env, obstacle_name, device)
        clearances.append(ee_surface_distance(ee_pos, center, half))
    return torch.stack(clearances, dim=0).amin(dim=0)


# ---------------------------------------------------------------------------
# Per-checkpoint evaluation
# ---------------------------------------------------------------------------

def evaluate_policy(
    checkpoint_path: str,
    policy_label: str,
    task_name: str,
    raw_env: gym.Env,
    isaac_env: ManagerBasedRLEnv,
    obstacle_names: list[str],
    agent_cfg_base: dict,
    num_episodes: int,
    device: torch.device,
) -> dict:
    """Run num_episodes episodes with one checkpoint and return a metrics dict."""
    print(f"\n[EVAL] ---- Policy: {policy_label} ----")
    print(f"[EVAL]   checkpoint : {checkpoint_path}")

    policy_obs_dim, policy_running_mean = get_policy_obs_stats(checkpoint_path)
    env_obs_dim = isaac_env.single_observation_space["policy"].shape[-1]
    print(f"[EVAL]   policy_obs_dim={policy_obs_dim}  env_obs_dim={env_obs_dim}", end="")
    if policy_obs_dim > env_obs_dim:
        print(f"  -> running-mean pad +{policy_obs_dim - env_obs_dim}")
    elif policy_obs_dim < env_obs_dim:
        print(f"  -> truncate -{env_obs_dim - policy_obs_dim}")
    else:
        print("  -> no adaptation needed")

    # ------------------------------------------------------------------
    # Patch obs space so RlGamesVecEnvWrapper builds the right network
    # ------------------------------------------------------------------
    original_policy_space = isaac_env.single_observation_space["policy"]
    isaac_env.single_observation_space["policy"] = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(policy_obs_dim,), dtype=np.float32
    )

    agent_cfg = copy.deepcopy(agent_cfg_base)
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    rl_env = RlGamesVecEnvWrapper(raw_env, str(device), clip_obs, clip_actions, None, True)

    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register(
        "rlgpu",
        {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: rl_env},
    )

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = checkpoint_path
    agent_cfg["params"]["config"]["num_actors"] = isaac_env.num_envs

    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(checkpoint_path)
    agent.reset()

    # ------------------------------------------------------------------
    # Per-env accumulators
    # ------------------------------------------------------------------
    num_envs = isaac_env.num_envs
    step_count         = torch.zeros(num_envs, dtype=torch.long,    device=device)
    collision_count    = torch.zeros(num_envs, dtype=torch.float32, device=device)
    near_obstacle_count = torch.zeros(num_envs, dtype=torch.float32, device=device)
    clearance_sum      = torch.zeros(num_envs, dtype=torch.float32, device=device)
    min_clearance      = torch.full((num_envs,), float("inf"), dtype=torch.float32, device=device)
    max_object_z       = torch.full((num_envs,), float("-inf"), dtype=torch.float32, device=device)
    consec_lift        = torch.zeros(num_envs, dtype=torch.long,    device=device)
    first_success_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    ep_success         = torch.zeros(num_envs, dtype=torch.bool,    device=device)

    results_grasp = []
    results_coll  = []
    results_near  = []
    results_mean_clear = []
    results_min_clear = []
    results_max_z = []
    results_time  = []

    # ------------------------------------------------------------------
    # Rollout loop
    # ------------------------------------------------------------------
    obs = rl_env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    with torch.no_grad():
        while len(results_grasp) < num_episodes:
            # Adapt obs to match policy input dim, THEN pass to agent
            obs_adapted = adapt_obs(obs, policy_obs_dim, policy_running_mean)
            actions = agent.get_action(agent.obs_to_torch(obs_adapted), is_deterministic=True)
            obs, _, dones, _ = rl_env.step(actions)
            if isinstance(obs, dict):
                obs = obs["obs"]

            step_count += 1

            # Grasp success metric
            past_warmup = step_count > GRASP_WARMUP_STEPS
            obj_z = isaac_env.scene["object"].data.root_pos_w[:, 2]
            max_object_z = torch.maximum(max_object_z, obj_z)
            above = (obj_z > EVAL_GRASP_THRESHOLD) & past_warmup
            consec_lift = torch.where(above, consec_lift + 1, torch.zeros_like(consec_lift))
            just_succeeded = (consec_lift >= CONSECUTIVE_STEPS_REQUIRED) & (~ep_success)
            first_success_step = torch.where(just_succeeded, step_count, first_success_step)
            ep_success |= just_succeeded

            # Obstacle interaction metrics
            clearance = compute_min_ee_obstacle_clearance(isaac_env, obstacle_names, device)
            collision_count += (clearance < SURFACE_COLLISION_THRESHOLD).float()
            near_obstacle_count += (clearance < NEAR_OBSTACLE_THRESHOLD).float()
            if obstacle_names:
                clearance_sum += clearance
                min_clearance = torch.minimum(min_clearance, clearance)

            # Reset RNN hidden state for finished envs
            if agent.is_rnn and agent.states is not None:
                for s in agent.states:
                    s[:, dones, :] = 0.0

            # Harvest completed episodes
            done_idx = dones.nonzero(as_tuple=False).squeeze(-1)
            if done_idx.numel() > 0:
                for i in done_idx.tolist():
                    if len(results_grasp) >= num_episodes:
                        break
                    results_grasp.append(float(ep_success[i].item()))
                    steps = max(int(step_count[i].item()), 1)
                    results_coll.append(float(collision_count[i].item()) / steps)
                    results_near.append(float(near_obstacle_count[i].item()) / steps)
                    if obstacle_names:
                        results_mean_clear.append(float(clearance_sum[i].item()) / steps)
                        results_min_clear.append(float(min_clearance[i].item()))
                    else:
                        results_mean_clear.append(float("nan"))
                        results_min_clear.append(float("nan"))
                    results_max_z.append(float(max_object_z[i].item()))
                    completion_step = int(first_success_step[i].item())
                    results_time.append(float(completion_step if completion_step >= 0 else steps))
                    # Reset counters for this env
                    step_count[i]          = 0
                    collision_count[i]     = 0.0
                    near_obstacle_count[i] = 0.0
                    clearance_sum[i]       = 0.0
                    min_clearance[i]       = float("inf")
                    max_object_z[i]        = float("-inf")
                    consec_lift[i]         = 0
                    first_success_step[i]  = -1
                    ep_success[i]          = False

    # ------------------------------------------------------------------
    # Restore obs space and clean up rl_games objects
    # (do NOT call rl_env.close() â€” that would close raw_env)
    # ------------------------------------------------------------------
    isaac_env.single_observation_space["policy"] = original_policy_space
    del runner, agent, rl_env

    # ------------------------------------------------------------------
    # Summarise
    # ------------------------------------------------------------------
    grasp = np.array(results_grasp[:num_episodes])
    coll  = np.array(results_coll[:num_episodes])
    near  = np.array(results_near[:num_episodes])
    mean_clear = np.array(results_mean_clear[:num_episodes])
    min_clear = np.array(results_min_clear[:num_episodes])
    max_z = np.array(results_max_z[:num_episodes])
    time_ = np.array(results_time[:num_episodes])

    summary = {
        "test_env":            task_name,
        "policy_label":        policy_label,
        "checkpoint":          checkpoint_path,
        "grasp_success_mean":  float(grasp.mean()),
        "grasp_success_std":   float(grasp.std()),
        "collision_freq_mean": float(coll.mean()),
        "collision_freq_std":  float(coll.std()),
        "near_obstacle_mean":  float(near.mean()),
        "near_obstacle_std":   float(near.std()),
        "mean_clearance_mean": float(np.nanmean(mean_clear)),
        "mean_clearance_std":  float(np.nanstd(mean_clear)),
        "min_clearance_mean":  float(np.nanmean(min_clear)),
        "min_clearance_std":   float(np.nanstd(min_clear)),
        "max_object_z_mean":   float(max_z.mean()),
        "max_object_z_std":    float(max_z.std()),
        "task_time_mean":      float(time_.mean()),
        "task_time_std":       float(time_.std()),
        "n_episodes":          len(grasp),
    }
    print(
        f"[EVAL]   grasp_success  = {summary['grasp_success_mean']:.3f} +/- {summary['grasp_success_std']:.3f}\n"
        f"[EVAL]   collision_freq = {summary['collision_freq_mean']:.4f} +/- {summary['collision_freq_std']:.4f}\n"
        f"[EVAL]   near_obstacle  = {summary['near_obstacle_mean']:.4f} +/- {summary['near_obstacle_std']:.4f}\n"
        f"[EVAL]   mean_clearance = {summary['mean_clearance_mean']:.4f} +/- {summary['mean_clearance_std']:.4f} m\n"
        f"[EVAL]   min_clearance  = {summary['min_clearance_mean']:.4f} +/- {summary['min_clearance_std']:.4f} m\n"
        f"[EVAL]   max_object_z   = {summary['max_object_z_mean']:.4f} +/- {summary['max_object_z_std']:.4f} m\n"
        f"[EVAL]   task_time      = {summary['task_time_mean']:.1f} +/- {summary['task_time_std']:.1f} steps"
    )
    return summary


def record_policy_video(
    checkpoint_path: str,
    policy_label: str,
    task_name: str,
    agent_cfg_base: dict,
    device: torch.device,
    output_video_dir: str,
    video_length: int,
    video_seed: int,
) -> str | None:
    """Record one deterministic rollout for a checkpoint in a dedicated 1-env video pass."""

    video_task_name = resolve_video_task_name(task_name)
    print(f"[VIDEO] ---- Policy: {policy_label} ----")
    print(f"[VIDEO]   task        : {video_task_name}")
    print(f"[VIDEO]   checkpoint  : {checkpoint_path}")
    if "ep_1500" not in os.path.basename(checkpoint_path):
        print("[VIDEO][WARN] Checkpoint name does not contain 'ep_1500'; verify this is the intended last-epoch file.")

    env_cfg = build_env_cfg(video_task_name, num_envs=1, seed=video_seed)
    raw_env = gym.make(video_task_name, cfg=env_cfg, render_mode="rgb_array")
    isaac_env: ManagerBasedRLEnv = raw_env.unwrapped

    policy_obs_dim, policy_running_mean = get_policy_obs_stats(checkpoint_path)
    original_policy_space = isaac_env.single_observation_space["policy"]
    isaac_env.single_observation_space["policy"] = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(policy_obs_dim,), dtype=np.float32
    )

    safe_policy = _slugify(policy_label)
    safe_task = _slugify(task_name)
    raw_video_folder = os.path.join(output_video_dir, "_raw", safe_task, safe_policy)
    shutil.rmtree(raw_video_folder, ignore_errors=True)
    os.makedirs(raw_video_folder, exist_ok=True)
    video_env = gym.wrappers.RecordVideo(
        raw_env,
        video_folder=raw_video_folder,
        step_trigger=lambda step: step == 0,
        video_length=video_length,
        disable_logger=True,
    )

    agent_cfg = copy.deepcopy(agent_cfg_base)
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    rl_env = RlGamesVecEnvWrapper(video_env, str(device), clip_obs, clip_actions, None, True)

    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register(
        "rlgpu",
        {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: rl_env},
    )

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = checkpoint_path
    agent_cfg["params"]["config"]["num_actors"] = 1
    agent_cfg["params"]["seed"] = video_seed

    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(checkpoint_path)
    agent.reset()

    obs = rl_env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    consec_lift = 0
    stop_reason = "max_length"

    with torch.no_grad():
        for step_idx in range(video_length):
            obs_adapted = adapt_obs(obs, policy_obs_dim, policy_running_mean)
            actions = agent.get_action(agent.obs_to_torch(obs_adapted), is_deterministic=True)
            obs, _, dones, _ = rl_env.step(actions)
            if isinstance(obs, dict):
                obs = obs["obs"]

            obj_z = float(isaac_env.scene["object"].data.root_pos_w[0, 2].item())
            above = (step_idx + 1 > GRASP_WARMUP_STEPS) and (obj_z > EVAL_GRASP_THRESHOLD)
            consec_lift = consec_lift + 1 if above else 0
            if consec_lift >= CONSECUTIVE_STEPS_REQUIRED:
                stop_reason = "success"
                break
            if bool(dones[0].item()):
                stop_reason = "terminated"
                break

    isaac_env.single_observation_space["policy"] = original_policy_space
    del runner, agent, rl_env
    video_env.close()

    recorded_video = _latest_recorded_video(raw_video_folder)
    if recorded_video is None:
        print(f"[VIDEO][WARN] No video file was produced for {policy_label} on {task_name}.")
        return None

    os.makedirs(output_video_dir, exist_ok=True)
    final_name = f"{safe_policy}__{safe_task}.mp4"
    final_path = os.path.join(output_video_dir, final_name)
    if os.path.exists(final_path):
        os.remove(final_path)
    shutil.move(recorded_video, final_path)
    print(f"[VIDEO]   saved       : {final_path}")
    print(f"[VIDEO]   stop_reason : {stop_reason}")
    return final_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(args_cli.checkpoints) != len(args_cli.policy_labels):
        raise ValueError(
            f"--checkpoints ({len(args_cli.checkpoints)}) and "
            f"--policy_labels ({len(args_cli.policy_labels)}) must have the same length."
        )

    device = torch.device(args_cli.device if args_cli.device else "cuda:0")
    video_seed = args_cli.video_seed if args_cli.video_seed is not None else args_cli.seed

    # Load agent config YAML
    try:
        from isaaclab_tasks.direct.franka_grasping import agents as _agents
        agents_dir = os.path.dirname(_agents.__file__)
        yaml_path = os.path.join(agents_dir, "rl_games_ppo_cfg.yaml")
    except (ImportError, AttributeError):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(
            script_dir,
            "../../../source/isaaclab_tasks/isaaclab_tasks/direct/franka_grasping/agents/rl_games_ppo_cfg.yaml",
        )
    agent_cfg_base = load_agent_cfg(yaml_path)
    agent_cfg_base["params"]["seed"] = args_cli.seed

    # Create the test environment ONCE â€” reused across all checkpoints
    print(f"\n[EVAL] Creating test environment: {args_cli.task}  ({args_cli.num_envs} envs)")
    env_cfg = build_env_cfg(args_cli.task, num_envs=args_cli.num_envs, seed=args_cli.seed)
    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    isaac_env: ManagerBasedRLEnv = raw_env.unwrapped
    obstacle_names = get_obstacle_names(isaac_env)

    # Evaluate each policy checkpoint
    all_results = []
    for checkpoint_path, policy_label in zip(args_cli.checkpoints, args_cli.policy_labels):
        result = evaluate_policy(
            checkpoint_path=os.path.abspath(checkpoint_path),
            policy_label=policy_label,
            task_name=args_cli.task,
            raw_env=raw_env,
            isaac_env=isaac_env,
            obstacle_names=obstacle_names,
            agent_cfg_base=agent_cfg_base,
            num_episodes=args_cli.num_episodes,
            device=device,
        )
        all_results.append(result)

    raw_env.close()

    if args_cli.video:
        video_dir = resolve_video_dir(args_cli.output_csv, args_cli.task, args_cli.output_video_dir)
        saved_videos = []
        for checkpoint_path, policy_label in zip(args_cli.checkpoints, args_cli.policy_labels):
            saved_path = record_policy_video(
                checkpoint_path=os.path.abspath(checkpoint_path),
                policy_label=policy_label,
                task_name=args_cli.task,
                agent_cfg_base=agent_cfg_base,
                device=device,
                output_video_dir=video_dir,
                video_length=args_cli.video_length,
                video_seed=video_seed,
            )
            if saved_path:
                saved_videos.append(saved_path)
        if saved_videos:
            print(f"[VIDEO] Videos saved to {video_dir}")

    # Append results to CSV
    fieldnames = [
        "test_env", "policy_label", "checkpoint",
        "grasp_success_mean", "grasp_success_std",
        "collision_freq_mean", "collision_freq_std",
        "near_obstacle_mean", "near_obstacle_std",
        "mean_clearance_mean", "mean_clearance_std",
        "min_clearance_mean", "min_clearance_std",
        "max_object_z_mean", "max_object_z_std",
        "task_time_mean", "task_time_std",
        "n_episodes",
    ]
    csv_path = args_cli.output_csv
    csv_parent = os.path.dirname(os.path.abspath(csv_path))
    if csv_parent:
        os.makedirs(csv_parent, exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(all_results)

    plot_dir = resolve_plot_dir(csv_path, args_cli.task, args_cli.output_plot_dir)
    plot_paths = save_evaluation_plots(all_results, args_cli.task, plot_dir)

    print(f"\n[EVAL] Results appended to {csv_path}")
    if plot_paths:
        print(f"[EVAL] Plots saved to {plot_dir}")
    print(f"[EVAL] Done â€” {len(all_results)} policies evaluated in '{args_cli.task}'.\n")


if __name__ == "__main__":
    main()
    simulation_app.close()
