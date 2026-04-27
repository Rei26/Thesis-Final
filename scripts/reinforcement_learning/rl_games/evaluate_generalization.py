# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Generalization evaluation script for the Franka grasping thesis.

This script evaluates one or more trained RL-Games checkpoints in a single
benchmark environment. It is designed to be run once per evaluation condition
(BL, A, B, C), appending results into a shared output directory. After each run
it refreshes:

* episode_results.csv: one row per completed evaluation episode.
* summary_by_checkpoint.csv: one row per checkpoint x evaluation condition.
* summary_by_family.csv: Baseline/R16 mean +- std across trained seeds.
* plots/: one plot per metric, with seed-detail and mean/std variants.

The evaluation condition controls the environment. The policy family controls
which PPO YAML is used to reconstruct each checkpoint:

* Baseline policies: rl_games_ppo_cfg.yaml
* R16 policies:      rl_games_r16_ppo_cfg.yaml

Example:

  ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/evaluate_generalization.py \\
      --task Isaac-Franka-Grasping-A-v0 \\
      --checkpoint_selection last \\
      --checkpoints \\
          logs/rl_games/baseline/baseline42 \\
          logs/rl_games/baseline/baseline123 \\
          logs/rl_games/baseline/baseline789 \\
          logs/rl_games/obstacle_training/R16_final_s42 \\
          logs/rl_games/obstacle_training/R16_final_s123 \\
          logs/rl_games/obstacle_training/R16_final_s789 \\
      --policy_labels BL_s42 BL_s123 BL_s789 R16_s42 R16_s123 R16_s789 \\
      --policy_families Baseline Baseline Baseline R16 R16 R16 \\
      --train_seeds 42 123 789 42 123 789 \\
      --agent_cfgs rl_games_ppo_cfg.yaml rl_games_ppo_cfg.yaml rl_games_ppo_cfg.yaml \\
                   rl_games_r16_ppo_cfg.yaml rl_games_r16_ppo_cfg.yaml rl_games_r16_ppo_cfg.yaml \\
      --num_episodes 100 --num_envs 16 \\
      --output_dir results_and_plots --headless

The default checkpoint selection is the final epoch checkpoint. For the thesis
runs this is expected to resolve to epoch 1500; the script warns if it resolves
to a different epoch.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

DEFAULT_OUTPUT_DIR = "results_and_plots"

# ---------------------------------------------------------------------------
# CLI - must come before any other Isaac Lab imports
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Generalization evaluation: evaluate checkpoints in one BL/A/B/C benchmark environment."
)
parser.add_argument(
    "--task",
    type=str,
    required=True,
    help="Gym task id of the TEST environment, e.g. Isaac-Franka-Grasping-A-v0.",
)
parser.add_argument(
    "--eval_condition",
    type=str,
    default=None,
    help="Short condition label for CSV/plots. Defaults from task id when possible: BL, A, B, C.",
)
parser.add_argument(
    "--checkpoints",
    type=str,
    nargs="+",
    required=True,
    help=(
        "Paths to checkpoint files, nn directories, or run directories, one per policy. "
        "By default each entry is resolved to the highest-epoch checkpoint in its nn folder."
    ),
)
parser.add_argument(
    "--checkpoint_selection",
    choices=("last", "explicit"),
    default="last",
    help=(
        "How to interpret --checkpoints. 'last' resolves each entry to the highest ep_#### checkpoint; "
        "'explicit' uses the exact .pth files provided."
    ),
)
parser.add_argument(
    "--policy_labels",
    type=str,
    nargs="+",
    required=True,
    help="Short labels for each checkpoint, same order as --checkpoints.",
)
parser.add_argument(
    "--policy_families",
    type=str,
    nargs="+",
    default=None,
    help="Policy family per checkpoint, e.g. Baseline Baseline Baseline R16 R16 R16. Inferred from labels if omitted.",
)
parser.add_argument(
    "--train_seeds",
    type=str,
    nargs="+",
    default=None,
    help="Training seed per checkpoint. Inferred from labels like *_s42 if omitted.",
)
parser.add_argument(
    "--agent_cfgs",
    type=str,
    nargs="+",
    default=None,
    help=(
        "RL-Games YAML per checkpoint. Relative names are resolved inside the Franka agents package. "
        "If omitted, Baseline uses rl_games_ppo_cfg.yaml and R16 uses rl_games_r16_ppo_cfg.yaml."
    ),
)
parser.add_argument("--num_episodes", type=int, default=100, help="Episodes per policy/environment pair.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments.")
parser.add_argument("--seed", type=int, default=42, help="Environment seed for eval runs.")
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help=f"Directory for all CSVs and plots. Defaults to {DEFAULT_OUTPUT_DIR}.",
)
parser.add_argument(
    "--output_csv",
    type=str,
    default=None,
    help="Checkpoint-summary CSV path. Defaults to <output_dir>/summary_by_checkpoint.csv.",
)
parser.add_argument("--episode_csv", type=str, default=None, help="Optional path for per-episode CSV.")
parser.add_argument("--family_csv", type=str, default=None, help="Optional path for family-aggregate CSV.")
parser.add_argument(
    "--output_plot_dir",
    type=str,
    default=None,
    help="Directory for evaluation PNG plots. Defaults to <output_dir>/plots.",
)
parser.add_argument(
    "--clear_output",
    action="store_true",
    default=False,
    help="Delete all CSV outputs before writing this run. Use only at the start of a fresh full sweep.",
)
parser.add_argument(
    "--video",
    action="store_true",
    default=False,
    help="Record one deterministic evaluation video per checkpoint after metrics are computed.",
)
parser.add_argument("--video_length", type=int, default=250, help="Maximum video rollout length in steps.")
parser.add_argument(
    "--output_video_dir",
    type=str,
    default=None,
    help="Directory for evaluation videos. Defaults to <output_dir>/videos.",
)
parser.add_argument("--video_seed", type=int, default=None, help="Seed for video rollouts. Defaults to --seed.")
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
import importlib
import math
import os
import re
import shutil
from collections import defaultdict

import gymnasium as gym
import matplotlib
import numpy as np
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401 - registers task packages

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Evaluation constants and metric specs
# ---------------------------------------------------------------------------

EVAL_OBJECT_REST_HEIGHT = 0.055
EVAL_GRASP_THRESHOLD = EVAL_OBJECT_REST_HEIGHT + 0.02
CONSECUTIVE_STEPS_REQUIRED = 50
SURFACE_COLLISION_THRESHOLD = 0.04
NEAR_OBSTACLE_THRESHOLD = 0.08
GRASP_WARMUP_STEPS = 15
EXPECTED_FINAL_EPOCH = 1500

# Maps registered Gym task IDs to the thesis condition labels used in CSVs and plots.
TASK_TO_CONDITION = {
    "Isaac-Franka-Grasping-Baseline-v0": "BL",
    "Isaac-Franka-Grasping-Baseline-Play-v0": "BL",
    "Isaac-Franka-Grasping-A-v0": "A",
    "Isaac-Franka-Grasping-A-Play-v0": "A",
    "Isaac-Franka-Grasping-B-v0": "B",
    "Isaac-Franka-Grasping-B-Play-v0": "B",
    "Isaac-Franka-Grasping-C-v0": "C",
    "Isaac-Franka-Grasping-C-Play-v0": "C",
}
CONDITION_ORDER = ["BL", "A", "B", "C"]
FAMILY_ORDER = ["Baseline", "R16"]
FAMILY_COLORS = {
    "Baseline": "#4c72b0",
    "R16": "#dd8452",
}

# Thesis metrics reported per checkpoint and aggregated across seeds. Grasp
# success is the primary generalization metric; the remaining values explain
# whether success came with collisions, slow completion, or unsafe clearances.
METRIC_SPECS = [
    {
        "key": "episode_return",
        "title": "Episode Return",
        "ylabel": "Return",
        "ylim": (None, None),
        "higher_is_better": True,
    },
    {
        "key": "grasp_success",
        "title": "Grasp Success Rate",
        "ylabel": "Success Rate",
        "ylim": (0.0, 1.0),
        "higher_is_better": True,
    },
    {
        "key": "task_completion_time",
        "title": "Task Completion Time",
        "ylabel": "Steps",
        "ylim": (0.0, None),
        "higher_is_better": False,
    },
    {
        "key": "episode_length",
        "title": "Episode Length",
        "ylabel": "Steps",
        "ylim": (245.0, 252.0),
        "higher_is_better": False,
    },
    {
        "key": "collision_frequency",
        "title": "Collision Frequency",
        "ylabel": "Fraction of Steps",
        "ylim": (0.0, 0.2),
        "higher_is_better": False,
    },
    {
        "key": "near_obstacle_rate",
        "title": "Near-Obstacle Rate",
        "ylabel": "Fraction of Steps",
        "ylim": (0.0, 1.0),
        "higher_is_better": False,
    },
    {
        "key": "mean_clearance",
        "title": "Mean Clearance",
        "ylabel": "Meters",
        "ylim": (0.0, None),
        "higher_is_better": True,
    },
    {
        "key": "min_clearance",
        "title": "Minimum Clearance",
        "ylabel": "Meters",
        "ylim": (0.0, None),
        "higher_is_better": True,
    },
    {
        "key": "max_object_z",
        "title": "Maximum Object Height",
        "ylabel": "Meters",
        "ylim": (0.0, None),
        "higher_is_better": True,
    },
]
CORE_METRIC_KEYS = [spec["key"] for spec in METRIC_SPECS]


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _slugify(text: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text))
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def _is_finite_number(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _as_float(value, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _mean_std(values: list[float], sample_std: bool = True) -> tuple[float, float, int]:
    finite = np.array([v for v in values if math.isfinite(float(v))], dtype=float)
    if finite.size == 0:
        return float("nan"), float("nan"), 0
    ddof = 1 if sample_std and finite.size > 1 else 0
    return float(np.mean(finite)), float(np.std(finite, ddof=ddof)), int(finite.size)


def _read_csv_rows(path: str) -> list[dict]:
    if not path or not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _write_csv_rows(path: str, rows: list[dict], preferred_fields: list[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    field_set = set(preferred_fields)
    for row in rows:
        field_set.update(row.keys())
    extra_fields = sorted(field_set.difference(preferred_fields))
    fieldnames = [field for field in preferred_fields if field in field_set] + extra_fields
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _expand_optional_list(values: list[str] | None, count: int, name: str, allow_single_repeat: bool = False):
    if values is None:
        return [None] * count
    if allow_single_repeat and len(values) == 1 and count > 1:
        return values * count
    if len(values) != count:
        raise ValueError(f"{name} has {len(values)} values, but --checkpoints has {count}.")
    return values


def normalize_policy_family(value: str | None, label: str) -> str:
    raw = (value or "").strip()
    if not raw:
        raw = infer_policy_family(label)
    low = raw.lower()
    if low in {"baseline", "base", "bl", "lift", "franka_lift"}:
        return "Baseline"
    if low in {"r16", "main", "grasping", "obstacle", "obstacle_aware", "condition_a", "a"}:
        return "R16"
    return raw


def infer_policy_family(label: str) -> str:
    low = label.lower()
    if re.search(r"(^|[_\-])bl($|[_\-])", low) or "baseline" in low or "franka_lift" in low:
        return "Baseline"
    return "R16"


def infer_train_seed(label: str, fallback_index: int) -> str:
    match = re.search(r"(?:^|[_\-])s(?:eed)?(\d+)(?:$|[_\-])", label.lower())
    if match:
        return match.group(1)
    match = re.search(r"(?:seed|s)(\d+)", label.lower())
    if match:
        return match.group(1)
    return str(fallback_index)


def normalize_condition_label(task_name: str, explicit_condition: str | None) -> str:
    if explicit_condition:
        label = explicit_condition.strip()
    else:
        label = TASK_TO_CONDITION.get(task_name, task_name)
    low = label.lower()
    if low in {"baseline", "base", "bl", "no_obstacles", "no_obstacle"}:
        return "BL"
    if low in {"condition_a", "cond_a", "a"}:
        return "A"
    if low in {"condition_b", "cond_b", "b"}:
        return "B"
    if low in {"condition_c", "cond_c", "c"}:
        return "C"
    return label


def sort_conditions(conditions: list[str]) -> list[str]:
    ordered = [item for item in CONDITION_ORDER if item in conditions]
    ordered.extend(sorted(cond for cond in conditions if cond not in CONDITION_ORDER))
    return ordered


def sort_families(families: list[str]) -> list[str]:
    ordered = [item for item in FAMILY_ORDER if item in families]
    ordered.extend(sorted(family for family in families if family not in FAMILY_ORDER))
    return ordered


def default_output_dir(output_csv: str | None, explicit_output_dir: str | None) -> str:
    if explicit_output_dir:
        return explicit_output_dir
    if output_csv:
        return os.path.dirname(os.path.abspath(output_csv)) or "."
    return DEFAULT_OUTPUT_DIR


def resolve_output_paths() -> dict[str, str]:
    output_dir = os.path.abspath(default_output_dir(args_cli.output_csv, args_cli.output_dir))
    summary_csv = (
        os.path.abspath(args_cli.output_csv)
        if args_cli.output_csv
        else os.path.join(output_dir, "summary_by_checkpoint.csv")
    )
    episode_csv = os.path.abspath(args_cli.episode_csv) if args_cli.episode_csv else os.path.join(output_dir, "episode_results.csv")
    family_csv = os.path.abspath(args_cli.family_csv) if args_cli.family_csv else os.path.join(output_dir, "summary_by_family.csv")
    plot_dir = os.path.abspath(args_cli.output_plot_dir) if args_cli.output_plot_dir else os.path.join(output_dir, "plots")
    video_dir = os.path.abspath(args_cli.output_video_dir) if args_cli.output_video_dir else os.path.join(output_dir, "videos")
    return {
        "output_dir": output_dir,
        "summary_csv": summary_csv,
        "episode_csv": episode_csv,
        "family_csv": family_csv,
        "plot_dir": plot_dir,
        "video_dir": video_dir,
    }


def find_agents_dir() -> str:
    # Prefer the manager-based Franka agents directory used by this thesis repo.
    # The direct path is retained only as a legacy fallback for older run copies.
    candidate_modules = [
        "isaaclab_tasks.manager_based.manipulation.lift.config.franka.agents",
        "isaaclab_tasks.direct.franka_grasping.agents",
    ]
    for module_name in candidate_modules:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        module_file = getattr(module, "__file__", None)
        if module_file:
            return os.path.dirname(module_file)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    fallback = os.path.abspath(
        os.path.join(
            script_dir,
            "../../../source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/agents",
        )
    )
    if os.path.isdir(fallback):
        return fallback
    return os.path.abspath(
        os.path.join(script_dir, "../../../source/isaaclab_tasks/isaaclab_tasks/direct/franka_grasping/agents")
    )


def resolve_agent_cfg_path(agent_cfg: str | None, policy_family: str) -> str:
    # Accept either a full path or a YAML filename. Filenames resolve inside the
    # Franka agents package so the PowerShell sweep can stay short/readable.
    filename = agent_cfg
    if filename is None:
        filename = "rl_games_ppo_cfg.yaml" if policy_family == "Baseline" else "rl_games_r16_ppo_cfg.yaml"

    candidates = []
    if os.path.isabs(filename):
        candidates.append(filename)
    else:
        candidates.append(os.path.abspath(filename))
        candidates.append(os.path.join(find_agents_dir(), filename))

    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)

    raise FileNotFoundError(
        f"Could not resolve RL-Games config '{filename}' for policy family '{policy_family}'. "
        f"Tried: {', '.join(candidates)}"
    )


def _checkpoint_epoch(path: str) -> int | None:
    match = re.search(r"(?:^|[_\-])ep[_\-]?(\d+)(?:[_\-.]|$)", os.path.basename(path).lower())
    if match:
        return int(match.group(1))
    return None


def _find_last_epoch_checkpoint(directory: str) -> str | None:
    if not os.path.isdir(directory):
        return None
    candidates = []
    for name in os.listdir(directory):
        if not name.lower().endswith(".pth"):
            continue
        path = os.path.join(directory, name)
        epoch = _checkpoint_epoch(path)
        if epoch is not None:
            candidates.append((epoch, os.path.getmtime(path), path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return os.path.abspath(candidates[-1][2])


def _warn_if_not_expected_final_epoch(path: str, source: str) -> None:
    epoch = _checkpoint_epoch(path)
    if epoch is None:
        print(
            f"[EVAL][WARN] Resolved checkpoint for '{source}' does not include an ep_#### marker: {path}. "
            f"Expected final epoch ep_{EXPECTED_FINAL_EPOCH} for thesis evaluation."
        )
    elif epoch != EXPECTED_FINAL_EPOCH:
        print(
            f"[EVAL][WARN] Resolved checkpoint for '{source}' is epoch {epoch}, "
            f"not the expected final epoch {EXPECTED_FINAL_EPOCH}: {path}"
        )


def resolve_checkpoint_path(checkpoint: str, selection: str) -> str:
    """Resolve checkpoint arguments, preferring final epoch checkpoints for thesis eval."""

    path = os.path.abspath(checkpoint)
    if selection == "explicit":
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint file does not exist: {path}")
        return path

    search_dirs = []
    if os.path.isdir(path):
        search_dirs.append(path)
        search_dirs.append(os.path.join(path, "nn"))
    elif os.path.isfile(path):
        parent = os.path.dirname(path)
        search_dirs.append(parent)
        # If a run-level file is ever passed, also check a sibling nn directory.
        search_dirs.append(os.path.join(parent, "nn"))
    else:
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    for directory in search_dirs:
        resolved = _find_last_epoch_checkpoint(directory)
        if resolved is not None:
            if os.path.abspath(path) != resolved:
                print(f"[EVAL] Resolved checkpoint '{checkpoint}' -> final epoch '{resolved}'")
            _warn_if_not_expected_final_epoch(resolved, checkpoint)
            return resolved

    if os.path.isfile(path):
        epoch = _checkpoint_epoch(path)
        if epoch is None:
            print(
                f"[EVAL][WARN] No ep_#### checkpoint found near '{checkpoint}'. "
                f"Using the provided file; expected final epoch ep_{EXPECTED_FINAL_EPOCH} for thesis evaluation."
            )
        else:
            _warn_if_not_expected_final_epoch(path, checkpoint)
        return path

    raise FileNotFoundError(f"No .pth checkpoint could be resolved from: {path}")


def load_agent_cfg(yaml_path: str) -> dict:
    import yaml

    with open(yaml_path) as f:
        return yaml.safe_load(f)


def _nested_get(data: dict, keys: tuple[str, ...]):
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _run_root_from_checkpoint(checkpoint_path: str) -> str:
    parent = os.path.dirname(os.path.abspath(checkpoint_path))
    if os.path.basename(parent).lower() == "nn":
        return os.path.dirname(parent)
    return parent


def _load_saved_agent_yaml(checkpoint_path: str) -> tuple[str | None, dict | None]:
    import yaml

    agent_yaml = os.path.join(_run_root_from_checkpoint(checkpoint_path), "params", "agent.yaml")
    if not os.path.isfile(agent_yaml):
        return None, None
    with open(agent_yaml) as f:
        return agent_yaml, yaml.safe_load(f)


def warn_if_saved_metadata_mismatch(
    checkpoint_path: str,
    policy_label: str,
    policy_family: str,
    train_seed: str,
) -> None:
    """Warn when CLI labels disagree with the saved run snapshot."""

    agent_yaml_path, saved_cfg = _load_saved_agent_yaml(checkpoint_path)
    if not saved_cfg:
        print(
            f"[EVAL][WARN] No saved params/agent.yaml found for {policy_label}. "
            "Cannot verify the saved training seed or run name."
        )
        return

    saved_seed = _nested_get(saved_cfg, ("params", "seed"))
    if saved_seed is not None and str(saved_seed) != str(train_seed):
        print(
            f"[EVAL][WARN] Seed label mismatch for {policy_label}: CLI train_seed={train_seed}, "
            f"but {agent_yaml_path} has params.seed={saved_seed}."
        )

    saved_name = _nested_get(saved_cfg, ("params", "config", "name"))
    expected_names = {"Baseline": "franka_lift", "R16": "franka_grasping"}
    expected_name = expected_names.get(policy_family)
    if expected_name and saved_name and saved_name != expected_name:
        print(
            f"[EVAL][WARN] Policy family mismatch for {policy_label}: CLI family={policy_family} "
            f"expects config.name={expected_name}, but {agent_yaml_path} has config.name={saved_name}."
        )


def build_policy_specs() -> list[dict]:
    count = len(args_cli.checkpoints)
    if len(args_cli.policy_labels) != count:
        raise ValueError(
            f"--checkpoints ({count}) and --policy_labels ({len(args_cli.policy_labels)}) must have the same length."
        )

    families = _expand_optional_list(args_cli.policy_families, count, "--policy_families")
    seeds = _expand_optional_list(args_cli.train_seeds, count, "--train_seeds")
    agent_cfgs = _expand_optional_list(args_cli.agent_cfgs, count, "--agent_cfgs", allow_single_repeat=True)

    specs = []
    for idx, (checkpoint, label, family_raw, seed_raw, cfg_raw) in enumerate(
        zip(args_cli.checkpoints, args_cli.policy_labels, families, seeds, agent_cfgs)
    ):
        # Each CLI slot describes one trained policy. The saved params/agent.yaml
        # is checked below so seed/family labels cannot silently drift from reality.
        family = normalize_policy_family(family_raw, label)
        train_seed = str(seed_raw) if seed_raw is not None else infer_train_seed(label, idx)
        agent_cfg_path = resolve_agent_cfg_path(cfg_raw, family)
        resolved_checkpoint = resolve_checkpoint_path(checkpoint, args_cli.checkpoint_selection)
        warn_if_saved_metadata_mismatch(resolved_checkpoint, label, family, train_seed)
        specs.append(
            {
                "checkpoint": resolved_checkpoint,
                "policy_label": label,
                "policy_family": family,
                "train_seed": train_seed,
                "agent_cfg_path": agent_cfg_path,
            }
        )
    return specs


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


def fix_object_reset_to_default(env_cfg: ManagerBasedRLEnvCfg) -> None:
    """Force evaluation environments to keep the cube at its default spawn pose."""

    if not hasattr(env_cfg, "events") or not hasattr(env_cfg.events, "reset_object_position"):
        return
    # The thesis fixed benchmarks isolate obstacle-condition transfer, so object
    # pose randomization is disabled during evaluation.
    env_cfg.events.reset_object_position.params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
    }


def build_env_cfg(task_name: str, num_envs: int, seed: int):
    """Instantiate a task config with fixed object reset and requested env count."""

    cfg_entry = gym.spec(task_name).kwargs["env_cfg_entry_point"]
    if isinstance(cfg_entry, str):
        module_str, class_str = cfg_entry.rsplit(":", 1)
        env_cfg = getattr(importlib.import_module(module_str), class_str)()
    else:
        env_cfg = cfg_entry()
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    fix_object_reset_to_default(env_cfg)
    return env_cfg


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


def ee_surface_distance(
    ee_pos: torch.Tensor,
    obstacle_center: torch.Tensor,
    half_extents: torch.Tensor,
) -> torch.Tensor:
    diff = torch.abs(ee_pos - obstacle_center) - half_extents.to(ee_pos.device)
    return torch.norm(diff.clamp(min=0.0), dim=-1)


def compute_min_ee_obstacle_clearance(
    isaac_env: ManagerBasedRLEnv,
    obstacle_names: list[str],
    device: torch.device,
) -> torch.Tensor:
    if not obstacle_names:
        return torch.full((isaac_env.num_envs,), float("inf"), dtype=torch.float32, device=device)
    ee_pos = isaac_env.scene["ee_frame"].data.target_pos_w[:, 0, :]
    clearances = []
    for obstacle_name in obstacle_names:
        center = isaac_env.scene[obstacle_name].data.root_pos_w[:, :3]
        half = get_runtime_obstacle_half_extents(isaac_env, obstacle_name, device)
        clearances.append(ee_surface_distance(ee_pos, center, half))
    return torch.stack(clearances, dim=0).amin(dim=0)


_CONTACT_FALLBACK_WARNED = False


def compute_training_collision_mask(
    isaac_env: ManagerBasedRLEnv,
    obstacle_names: list[str],
    clearance: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Match the training collision metric by using the env contact sensors."""

    global _CONTACT_FALLBACK_WARNED

    if not obstacle_names:
        return torch.zeros(isaac_env.num_envs, dtype=torch.bool, device=device)

    contact_fn = getattr(isaac_env, "_ee_near_obstacle", None)
    if callable(contact_fn):
        try:
            contact_mask = contact_fn().to(device=device, dtype=torch.bool).flatten()
            if contact_mask.numel() == isaac_env.num_envs:
                return contact_mask
            raise ValueError(f"expected {isaac_env.num_envs} values, got {contact_mask.numel()}")
        except Exception as exc:
            if not _CONTACT_FALLBACK_WARNED:
                print(
                    "[EVAL][WARN] Could not read the training-style obstacle contact mask "
                    f"from _ee_near_obstacle(): {exc}. Falling back to clearance threshold."
                )
                _CONTACT_FALLBACK_WARNED = True

    if not _CONTACT_FALLBACK_WARNED:
        print(
            "[EVAL][WARN] Test environment does not expose _ee_near_obstacle(). "
            "Falling back to clearance-threshold collision estimates."
        )
        _CONTACT_FALLBACK_WARNED = True
    return clearance < SURFACE_COLLISION_THRESHOLD


def get_reward_terms(isaac_env: ManagerBasedRLEnv) -> list[str]:
    reward_manager = getattr(isaac_env, "reward_manager", None)
    if reward_manager is None:
        return []
    return list(getattr(reward_manager, "active_terms", []))


def get_step_reward_terms(isaac_env: ManagerBasedRLEnv, reward_terms: list[str], device: torch.device) -> torch.Tensor | None:
    reward_manager = getattr(isaac_env, "reward_manager", None)
    if reward_manager is None or not reward_terms:
        return None
    step_reward = getattr(reward_manager, "_step_reward", None)
    if step_reward is None:
        return None
    # _step_reward stores per-second term values; multiply by env step_dt to match returned reward scale.
    return step_reward[:, : len(reward_terms)].to(device=device, dtype=torch.float32) * float(isaac_env.step_dt)


# ---------------------------------------------------------------------------
# Observation / policy helpers
# ---------------------------------------------------------------------------


def adapt_obs(obs: torch.Tensor, policy_obs_dim: int, running_mean: torch.Tensor | None = None) -> torch.Tensor:
    """Pad missing tail features with checkpoint running-mean values or truncate extras."""

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

    for key, val in model.items():
        if isinstance(val, torch.Tensor) and val.dim() == 1 and "running_mean_std" in key and "running_mean" in key:
            return val.shape[0], val.detach().cpu().float()

    for key in sorted(model.keys()):
        val = model[key]
        if isinstance(val, torch.Tensor) and val.dim() == 2:
            return val.shape[1], None

    raise ValueError(f"Cannot determine obs_dim from checkpoint: {checkpoint_path}")


def build_agent(agent_cfg_base: dict, checkpoint_path: str, num_actors: int, seed: int) -> tuple[Runner, object]:
    agent_cfg = copy.deepcopy(agent_cfg_base)
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = checkpoint_path
    agent_cfg["params"]["seed"] = seed
    agent_cfg["params"]["config"]["num_actors"] = num_actors

    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(checkpoint_path)
    agent.reset()
    return runner, agent


# ---------------------------------------------------------------------------
# Summaries and plotting
# ---------------------------------------------------------------------------


EPISODE_PREFERRED_FIELDS = [
    "eval_condition",
    "test_env",
    "policy_family",
    "train_seed",
    "policy_label",
    "checkpoint",
    "agent_cfg_path",
    "eval_seed",
    "num_envs",
    "episode_index",
    "source_env_index",
    "has_obstacles",
    "n_obstacles",
    "episode_length",
    "episode_return",
    "grasp_success",
    "task_completion_time",
    "collision_frequency",
    "near_obstacle_rate",
    "mean_clearance",
    "min_clearance",
    "max_object_z",
]

SUMMARY_PREFERRED_FIELDS = [
    "eval_condition",
    "test_env",
    "policy_family",
    "train_seed",
    "policy_label",
    "checkpoint",
    "agent_cfg_path",
    "eval_seed",
    "num_envs",
    "n_episodes",
    "has_obstacles",
    "n_obstacles",
]

FAMILY_PREFERRED_FIELDS = [
    "eval_condition",
    "test_env",
    "policy_family",
    "n_train_seeds",
    "train_seeds",
    "n_checkpoints",
    "checkpoint_labels",
]


def metric_summary_fields(metric_keys: list[str]) -> list[str]:
    fields = []
    for key in metric_keys:
        fields.extend([f"{key}_mean", f"{key}_std", f"{key}_n"])
    return fields


def discover_metric_keys(rows: list[dict], include_raw_episode_metrics: bool = False) -> list[str]:
    keys = list(CORE_METRIC_KEYS)
    seen = set(keys)
    for row in rows:
        for key in row.keys():
            candidate = None
            if key.startswith("reward_term_"):
                candidate = key
            elif key.endswith("_mean"):
                base = key[: -len("_mean")]
                if base.startswith("reward_term_"):
                    candidate = base
            elif include_raw_episode_metrics and key in CORE_METRIC_KEYS:
                candidate = key
            if candidate and candidate not in seen:
                seen.add(candidate)
                keys.append(candidate)
    return keys


def summarize_episode_rows(episode_rows: list[dict], metadata: dict, metric_keys: list[str]) -> dict:
    summary = dict(metadata)
    summary["n_episodes"] = len(episode_rows)
    for key in metric_keys:
        values = [_as_float(row.get(key)) for row in episode_rows]
        mean, std, n = _mean_std(values, sample_std=True)
        summary[f"{key}_mean"] = mean
        summary[f"{key}_std"] = std
        summary[f"{key}_n"] = n
    return summary


def build_family_summary_rows(checkpoint_rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in checkpoint_rows:
        condition = normalize_condition_label(row.get("test_env", ""), row.get("eval_condition"))
        family = normalize_policy_family(row.get("policy_family"), row.get("policy_label", ""))
        groups[(condition, family)].append(row)

    metric_keys = discover_metric_keys(checkpoint_rows)
    family_rows = []
    for (condition, family), rows in sorted(groups.items(), key=lambda item: (sort_conditions([item[0][0]])[0], item[0][1])):
        test_envs = sorted({row.get("test_env", "") for row in rows if row.get("test_env")})
        seeds = sorted({str(row.get("train_seed", "")) for row in rows if str(row.get("train_seed", ""))})
        labels = sorted({row.get("policy_label", "") for row in rows if row.get("policy_label")})
        out = {
            "eval_condition": condition,
            "test_env": "|".join(test_envs),
            "policy_family": family,
            "n_train_seeds": len(seeds),
            "train_seeds": "|".join(seeds),
            "n_checkpoints": len(rows),
            "checkpoint_labels": "|".join(labels),
        }
        for key in metric_keys:
            values = [_as_float(row.get(f"{key}_mean")) for row in rows]
            mean, std, n = _mean_std(values, sample_std=True)
            out[f"{key}_mean"] = mean
            out[f"{key}_std"] = std
            out[f"{key}_n"] = n
        family_rows.append(out)
    return family_rows


def metric_spec_for(key: str) -> dict:
    for spec in METRIC_SPECS:
        if spec["key"] == key:
            return spec
    if key.startswith("reward_term_"):
        name = key[len("reward_term_") :].replace("_", " ").title()
        return {
            "key": key,
            "title": f"Reward Term: {name}",
            "ylabel": "Return Contribution",
            "ylim": (None, None),
            "higher_is_better": True,
        }
    return {"key": key, "title": key.replace("_", " ").title(), "ylabel": key, "ylim": (None, None)}


def _apply_metric_ylim(ax, values: list[float], spec: dict) -> None:
    y_min, y_max = spec.get("ylim", (None, None))
    fixed_min = y_min is not None
    fixed_max = y_max is not None
    finite = [v for v in values if math.isfinite(float(v))]
    if y_min is None and finite:
        y_min = min(finite)
        if y_min > 0:
            y_min = 0.0
    if y_max is None and finite:
        y_max = max(finite)
    if y_min is not None and y_max is not None and math.isfinite(y_min) and math.isfinite(y_max):
        if y_max <= y_min:
            y_max = y_min + 1.0
        if fixed_min and fixed_max:
            ax.set_ylim(y_min, y_max)
        else:
            margin = 0.08 * (y_max - y_min)
            ax.set_ylim(y_min, y_max + margin)
    elif y_min is not None:
        ax.set_ylim(bottom=y_min)
    elif y_max is not None:
        ax.set_ylim(top=y_max)


def _checkpoint_values_for_metric(checkpoint_rows: list[dict], condition: str, family: str, metric_key: str) -> list[tuple[str, float]]:
    values = []
    for row in checkpoint_rows:
        row_condition = normalize_condition_label(row.get("test_env", ""), row.get("eval_condition"))
        row_family = normalize_policy_family(row.get("policy_family"), row.get("policy_label", ""))
        if row_condition != condition or row_family != family:
            continue
        value = _as_float(row.get(f"{metric_key}_mean"))
        if math.isfinite(value):
            values.append((str(row.get("train_seed", row.get("policy_label", ""))), value))
    return values


def plot_metric_seed_detail(
    checkpoint_rows: list[dict],
    family_rows: list[dict],
    metric_key: str,
    output_path: str,
) -> bool:
    conditions = sort_conditions(sorted({row.get("eval_condition", "") for row in family_rows if row.get("eval_condition")}))
    families = sort_families(sorted({row.get("policy_family", "") for row in family_rows if row.get("policy_family")}))
    if not conditions or not families:
        return False

    spec = metric_spec_for(metric_key)
    x = np.arange(len(conditions), dtype=float)
    width = min(0.34, 0.75 / max(len(families), 1))
    fig, ax = plt.subplots(figsize=(max(9.5, len(conditions) * 1.9), 6.0))
    all_values = []

    for fam_idx, family in enumerate(families):
        offset = (fam_idx - (len(families) - 1) / 2.0) * width
        means, stds = [], []
        for condition in conditions:
            row = next(
                (
                    item
                    for item in family_rows
                    if item.get("eval_condition") == condition and item.get("policy_family") == family
                ),
                None,
            )
            means.append(_as_float(row.get(f"{metric_key}_mean")) if row else float("nan"))
            stds.append(_as_float(row.get(f"{metric_key}_std")) if row else float("nan"))

        color = FAMILY_COLORS.get(family, f"C{fam_idx}")
        bar_positions = x + offset
        ax.bar(
            bar_positions,
            means,
            width=width * 0.92,
            yerr=stds,
            capsize=5,
            color=color,
            alpha=0.72,
            edgecolor="black",
            linewidth=0.8,
            label=f"{family} mean +- std",
            zorder=2,
        )
        all_values.extend(means)
        all_values.extend([m + s for m, s in zip(means, stds) if math.isfinite(m) and math.isfinite(s)])

        for cond_idx, condition in enumerate(conditions):
            seed_values = _checkpoint_values_for_metric(checkpoint_rows, condition, family, metric_key)
            if not seed_values:
                continue
            if len(seed_values) == 1:
                jitters = [0.0]
            else:
                jitters = np.linspace(-width * 0.24, width * 0.24, len(seed_values))
            for (seed, value), jitter in zip(seed_values, jitters):
                ax.scatter(
                    x[cond_idx] + offset + jitter,
                    value,
                    s=48,
                    color="white",
                    edgecolor="black",
                    linewidth=0.9,
                    zorder=4,
                )
                ax.annotate(
                    str(seed),
                    (x[cond_idx] + offset + jitter, value),
                    textcoords="offset points",
                    xytext=(0, 7),
                    ha="center",
                    fontsize=8,
                    color="#333333",
                    zorder=5,
                )
                all_values.append(value)

    ax.set_title(f"{spec['title']} by Evaluation Condition", fontsize=18, weight="bold")
    ax.set_ylabel(spec["ylabel"], fontsize=13)
    ax.set_xlabel("Evaluation Condition", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=12)
    ax.grid(axis="y", alpha=0.28)
    ax.set_axisbelow(True)
    ax.legend(loc="best", frameon=True)
    _apply_metric_ylim(ax, all_values, spec)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_metric_mean_std(family_rows: list[dict], metric_key: str, output_path: str) -> bool:
    conditions = sort_conditions(sorted({row.get("eval_condition", "") for row in family_rows if row.get("eval_condition")}))
    families = sort_families(sorted({row.get("policy_family", "") for row in family_rows if row.get("policy_family")}))
    if not conditions or not families:
        return False

    spec = metric_spec_for(metric_key)
    x = np.arange(len(conditions), dtype=float)
    fig, ax = plt.subplots(figsize=(max(9.5, len(conditions) * 1.9), 6.0))
    all_values = []

    for fam_idx, family in enumerate(families):
        means, stds = [], []
        for condition in conditions:
            row = next(
                (
                    item
                    for item in family_rows
                    if item.get("eval_condition") == condition and item.get("policy_family") == family
                ),
                None,
            )
            means.append(_as_float(row.get(f"{metric_key}_mean")) if row else float("nan"))
            stds.append(_as_float(row.get(f"{metric_key}_std")) if row else float("nan"))

        means_np = np.array(means, dtype=float)
        stds_np = np.array(stds, dtype=float)
        color = FAMILY_COLORS.get(family, f"C{fam_idx}")
        ax.plot(x, means_np, marker="o", linewidth=2.8, markersize=7, color=color, label=family)
        lower = means_np - stds_np
        upper = means_np + stds_np
        finite = np.isfinite(means_np) & np.isfinite(stds_np)
        if finite.any():
            ax.fill_between(x, lower, upper, where=finite, color=color, alpha=0.18, interpolate=True)
        all_values.extend(means_np[np.isfinite(means_np)].tolist())
        all_values.extend(upper[np.isfinite(upper)].tolist())

    ax.set_title(f"{spec['title']} (Mean +- Std Across Seeds)", fontsize=18, weight="bold")
    ax.set_ylabel(spec["ylabel"], fontsize=13)
    ax.set_xlabel("Evaluation Condition", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=12)
    ax.grid(alpha=0.28)
    ax.set_axisbelow(True)
    ax.legend(loc="best", frameon=True)
    _apply_metric_ylim(ax, all_values, spec)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def save_dashboard(family_rows: list[dict], metric_keys: list[str], output_path: str) -> bool:
    metric_keys = [key for key in metric_keys if any(_is_finite_number(row.get(f"{key}_mean")) for row in family_rows)]
    if not metric_keys:
        return False
    conditions = sort_conditions(sorted({row.get("eval_condition", "") for row in family_rows if row.get("eval_condition")}))
    families = sort_families(sorted({row.get("policy_family", "") for row in family_rows if row.get("policy_family")}))
    if not conditions or not families:
        return False

    n_cols = 3
    n_rows = int(math.ceil(len(metric_keys) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.0 * n_cols, 4.3 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    x = np.arange(len(conditions), dtype=float)

    for ax, metric_key in zip(axes, metric_keys):
        spec = metric_spec_for(metric_key)
        all_values = []
        for fam_idx, family in enumerate(families):
            means, stds = [], []
            for condition in conditions:
                row = next(
                    (
                        item
                        for item in family_rows
                        if item.get("eval_condition") == condition and item.get("policy_family") == family
                    ),
                    None,
                )
                means.append(_as_float(row.get(f"{metric_key}_mean")) if row else float("nan"))
                stds.append(_as_float(row.get(f"{metric_key}_std")) if row else float("nan"))
            means_np = np.array(means, dtype=float)
            stds_np = np.array(stds, dtype=float)
            color = FAMILY_COLORS.get(family, f"C{fam_idx}")
            ax.plot(x, means_np, marker="o", linewidth=2.2, color=color, label=family)
            finite = np.isfinite(means_np) & np.isfinite(stds_np)
            if finite.any():
                ax.fill_between(x, means_np - stds_np, means_np + stds_np, where=finite, color=color, alpha=0.16)
            all_values.extend(means_np[np.isfinite(means_np)].tolist())
            upper = means_np + stds_np
            all_values.extend(upper[np.isfinite(upper)].tolist())

        ax.set_title(spec["title"], fontsize=12, weight="bold")
        ax.set_ylabel(spec["ylabel"], fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(conditions)
        ax.grid(alpha=0.25)
        _apply_metric_ylim(ax, all_values, spec)

    for ax in axes[len(metric_keys) :]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)), frameon=True)
    fig.suptitle("Evaluation Summary Across BL/A/B/C", fontsize=16, weight="bold", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def save_all_plots(checkpoint_rows: list[dict], family_rows: list[dict], output_plot_dir: str) -> list[str]:
    os.makedirs(output_plot_dir, exist_ok=True)
    metric_keys = discover_metric_keys(checkpoint_rows + family_rows)
    saved = []
    for metric_key in metric_keys:
        if not any(_is_finite_number(row.get(f"{metric_key}_mean")) for row in family_rows):
            continue
        seed_path = os.path.join(output_plot_dir, "seed_detail", f"{_slugify(metric_key)}_seed_detail.png")
        mean_path = os.path.join(output_plot_dir, "mean_std", f"{_slugify(metric_key)}_mean_std.png")
        if plot_metric_seed_detail(checkpoint_rows, family_rows, metric_key, seed_path):
            saved.append(seed_path)
        if plot_metric_mean_std(family_rows, metric_key, mean_path):
            saved.append(mean_path)

    dashboard_path = os.path.join(output_plot_dir, "evaluation_dashboard.png")
    if save_dashboard(family_rows, [key for key in metric_keys if key in CORE_METRIC_KEYS], dashboard_path):
        saved.append(dashboard_path)
    return saved


# ---------------------------------------------------------------------------
# Per-checkpoint evaluation
# ---------------------------------------------------------------------------


def evaluate_policy(
    policy_spec: dict,
    task_name: str,
    eval_condition: str,
    raw_env: gym.Env,
    isaac_env: ManagerBasedRLEnv,
    obstacle_names: list[str],
    num_episodes: int,
    device: torch.device,
) -> tuple[dict, list[dict]]:
    """Run num_episodes episodes with one checkpoint and return summary + episode rows."""

    checkpoint_path = policy_spec["checkpoint"]
    policy_label = policy_spec["policy_label"]
    print(f"\n[EVAL] ---- Policy: {policy_label} ({policy_spec['policy_family']}, seed {policy_spec['train_seed']}) ----")
    print(f"[EVAL]   checkpoint : {checkpoint_path}")
    print(f"[EVAL]   agent cfg  : {policy_spec['agent_cfg_path']}")

    agent_cfg_base = load_agent_cfg(policy_spec["agent_cfg_path"])
    # The evaluation seed controls environment rollout reproducibility. The
    # policy weights still come entirely from the loaded checkpoint.
    agent_cfg_base["params"]["seed"] = args_cli.seed

    policy_obs_dim, policy_running_mean = get_policy_obs_stats(checkpoint_path)
    env_obs_dim = isaac_env.single_observation_space["policy"].shape[-1]
    print(f"[EVAL]   policy_obs_dim={policy_obs_dim}  env_obs_dim={env_obs_dim}", end="")
    if policy_obs_dim > env_obs_dim:
        print(f"  -> running-mean pad +{policy_obs_dim - env_obs_dim}")
    elif policy_obs_dim < env_obs_dim:
        print(f"  -> truncate -{env_obs_dim - policy_obs_dim}")
    else:
        print("  -> no adaptation needed")

    # Baseline and obstacle policies can have different observation widths across
    # older checkpoints. Adaptation lets the same evaluator handle both safely.
    original_policy_space = isaac_env.single_observation_space["policy"]
    isaac_env.single_observation_space["policy"] = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(policy_obs_dim,), dtype=np.float32
    )

    clip_obs = agent_cfg_base["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg_base["params"]["env"].get("clip_actions", math.inf)
    rl_env = RlGamesVecEnvWrapper(raw_env, str(device), clip_obs, clip_actions, None, True)

    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register(
        "rlgpu",
        {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: rl_env},
    )

    runner, agent = build_agent(agent_cfg_base, checkpoint_path, isaac_env.num_envs, args_cli.seed)

    num_envs = isaac_env.num_envs
    reward_terms = get_reward_terms(isaac_env)
    reward_term_sums = {
        term_name: torch.zeros(num_envs, dtype=torch.float32, device=device) for term_name in reward_terms
    }

    step_count = torch.zeros(num_envs, dtype=torch.long, device=device)
    episode_return = torch.zeros(num_envs, dtype=torch.float32, device=device)
    collision_count = torch.zeros(num_envs, dtype=torch.float32, device=device)
    near_obstacle_count = torch.zeros(num_envs, dtype=torch.float32, device=device)
    clearance_sum = torch.zeros(num_envs, dtype=torch.float32, device=device)
    min_clearance = torch.full((num_envs,), float("inf"), dtype=torch.float32, device=device)
    max_object_z = torch.full((num_envs,), float("-inf"), dtype=torch.float32, device=device)
    consec_lift = torch.zeros(num_envs, dtype=torch.long, device=device)
    first_success_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    ep_success = torch.zeros(num_envs, dtype=torch.bool, device=device)

    episode_rows = []

    obs = rl_env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    with torch.no_grad():
        while len(episode_rows) < num_episodes:
            obs_adapted = adapt_obs(obs, policy_obs_dim, policy_running_mean)
            actions = agent.get_action(agent.obs_to_torch(obs_adapted), is_deterministic=True)
            obs, rewards, dones, _ = rl_env.step(actions)
            if isinstance(obs, dict):
                obs = obs["obs"]

            rewards = rewards.to(device=device, dtype=torch.float32)
            dones = dones.to(device=device)
            step_count += 1
            episode_return += rewards

            step_reward_terms = get_step_reward_terms(isaac_env, reward_terms, device)
            if step_reward_terms is not None:
                for term_idx, term_name in enumerate(reward_terms):
                    reward_term_sums[term_name] += step_reward_terms[:, term_idx]

            past_warmup = step_count > GRASP_WARMUP_STEPS
            obj_z = isaac_env.scene["object"].data.root_pos_w[:, 2]
            max_object_z = torch.maximum(max_object_z, obj_z)
            above = (obj_z > EVAL_GRASP_THRESHOLD) & past_warmup
            consec_lift = torch.where(above, consec_lift + 1, torch.zeros_like(consec_lift))
            just_succeeded = (consec_lift >= CONSECUTIVE_STEPS_REQUIRED) & (~ep_success)
            first_success_step = torch.where(just_succeeded, step_count, first_success_step)
            ep_success |= just_succeeded

            clearance = compute_min_ee_obstacle_clearance(isaac_env, obstacle_names, device)
            # Collision frequency follows the same contact-sensor signal used
            # during training when available, with clearance fallback otherwise.
            collision_count += compute_training_collision_mask(isaac_env, obstacle_names, clearance, device).float()
            near_obstacle_count += (clearance < NEAR_OBSTACLE_THRESHOLD).float()
            if obstacle_names:
                clearance_sum += clearance
                min_clearance = torch.minimum(min_clearance, clearance)

            if agent.is_rnn and agent.states is not None:
                for state in agent.states:
                    state[:, dones, :] = 0.0

            done_idx = dones.nonzero(as_tuple=False).squeeze(-1)
            if done_idx.numel() > 0:
                for env_index in done_idx.tolist():
                    if len(episode_rows) >= num_episodes:
                        break
                    steps = max(int(step_count[env_index].item()), 1)
                    success = bool(ep_success[env_index].item())
                    completion_step = int(first_success_step[env_index].item())
                    row = {
                        "eval_condition": eval_condition,
                        "test_env": task_name,
                        "policy_family": policy_spec["policy_family"],
                        "train_seed": policy_spec["train_seed"],
                        "policy_label": policy_label,
                        "checkpoint": checkpoint_path,
                        "agent_cfg_path": policy_spec["agent_cfg_path"],
                        "eval_seed": args_cli.seed,
                        "num_envs": num_envs,
                        "episode_index": len(episode_rows),
                        "source_env_index": env_index,
                        "has_obstacles": bool(obstacle_names),
                        "n_obstacles": len(obstacle_names),
                        "episode_length": float(steps),
                        "episode_return": float(episode_return[env_index].item()),
                        "grasp_success": float(success),
                        "task_completion_time": float(completion_step if completion_step >= 0 else steps),
                        "collision_frequency": float(collision_count[env_index].item()) / steps,
                        "near_obstacle_rate": float(near_obstacle_count[env_index].item()) / steps,
                        "max_object_z": float(max_object_z[env_index].item()),
                    }
                    if obstacle_names:
                        row["mean_clearance"] = float(clearance_sum[env_index].item()) / steps
                        row["min_clearance"] = float(min_clearance[env_index].item())
                    else:
                        row["mean_clearance"] = float("nan")
                        row["min_clearance"] = float("nan")
                    for term_name in reward_terms:
                        row[f"reward_term_{_slugify(term_name)}"] = float(reward_term_sums[term_name][env_index].item())

                    episode_rows.append(row)

                    step_count[env_index] = 0
                    episode_return[env_index] = 0.0
                    collision_count[env_index] = 0.0
                    near_obstacle_count[env_index] = 0.0
                    clearance_sum[env_index] = 0.0
                    min_clearance[env_index] = float("inf")
                    max_object_z[env_index] = float("-inf")
                    consec_lift[env_index] = 0
                    first_success_step[env_index] = -1
                    ep_success[env_index] = False
                    for term_name in reward_terms:
                        reward_term_sums[term_name][env_index] = 0.0

    isaac_env.single_observation_space["policy"] = original_policy_space
    del runner, agent, rl_env

    metadata = {
        "eval_condition": eval_condition,
        "test_env": task_name,
        "policy_family": policy_spec["policy_family"],
        "train_seed": policy_spec["train_seed"],
        "policy_label": policy_label,
        "checkpoint": checkpoint_path,
        "agent_cfg_path": policy_spec["agent_cfg_path"],
        "eval_seed": args_cli.seed,
        "num_envs": num_envs,
        "has_obstacles": bool(obstacle_names),
        "n_obstacles": len(obstacle_names),
    }
    metric_keys = discover_metric_keys(episode_rows, include_raw_episode_metrics=True)
    summary = summarize_episode_rows(episode_rows, metadata, metric_keys)

    print(
        f"[EVAL]   episode_return = {summary.get('episode_return_mean', float('nan')):.3f} "
        f"+/- {summary.get('episode_return_std', float('nan')):.3f}\n"
        f"[EVAL]   grasp_success  = {summary.get('grasp_success_mean', float('nan')):.3f} "
        f"+/- {summary.get('grasp_success_std', float('nan')):.3f}\n"
        f"[EVAL]   collision_freq = {summary.get('collision_frequency_mean', float('nan')):.4f} "
        f"+/- {summary.get('collision_frequency_std', float('nan')):.4f}\n"
        f"[EVAL]   near_obstacle  = {summary.get('near_obstacle_rate_mean', float('nan')):.4f} "
        f"+/- {summary.get('near_obstacle_rate_std', float('nan')):.4f}\n"
        f"[EVAL]   mean_clearance = {summary.get('mean_clearance_mean', float('nan')):.4f} "
        f"+/- {summary.get('mean_clearance_std', float('nan')):.4f} m\n"
        f"[EVAL]   min_clearance  = {summary.get('min_clearance_mean', float('nan')):.4f} "
        f"+/- {summary.get('min_clearance_std', float('nan')):.4f} m\n"
        f"[EVAL]   max_object_z   = {summary.get('max_object_z_mean', float('nan')):.4f} "
        f"+/- {summary.get('max_object_z_std', float('nan')):.4f} m\n"
        f"[EVAL]   task_time      = {summary.get('task_completion_time_mean', float('nan')):.1f} "
        f"+/- {summary.get('task_completion_time_std', float('nan')):.1f} steps"
    )
    return summary, episode_rows


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------


def resolve_video_task_name(task_name: str) -> str:
    if task_name.endswith("-v0"):
        candidate = task_name[:-3] + "-Play-v0"
        try:
            gym.spec(candidate)
            return candidate
        except Exception:
            pass
    return task_name


def _latest_recorded_video(video_folder: str) -> str | None:
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


def record_policy_video(
    policy_spec: dict,
    task_name: str,
    device: torch.device,
    output_video_dir: str,
    video_length: int,
    video_seed: int,
) -> str | None:
    video_task_name = resolve_video_task_name(task_name)
    checkpoint_path = policy_spec["checkpoint"]
    policy_label = policy_spec["policy_label"]
    print(f"[VIDEO] ---- Policy: {policy_label} ----")
    print(f"[VIDEO]   task        : {video_task_name}")
    print(f"[VIDEO]   checkpoint  : {checkpoint_path}")

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

    agent_cfg_base = load_agent_cfg(policy_spec["agent_cfg_path"])
    clip_obs = agent_cfg_base["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg_base["params"]["env"].get("clip_actions", math.inf)
    rl_env = RlGamesVecEnvWrapper(video_env, str(device), clip_obs, clip_actions, None, True)

    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register(
        "rlgpu",
        {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: rl_env},
    )

    runner, agent = build_agent(agent_cfg_base, checkpoint_path, 1, video_seed)

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
    paths = resolve_output_paths()
    for directory in (paths["output_dir"], paths["plot_dir"]):
        os.makedirs(directory, exist_ok=True)

    eval_condition = normalize_condition_label(args_cli.task, args_cli.eval_condition)

    if args_cli.clear_output:
        # Used once at the beginning of a full BL/A/B/C sweep to avoid mixing old
        # CSV rows with the current evaluation run.
        for path in (paths["summary_csv"], paths["episode_csv"], paths["family_csv"]):
            if os.path.exists(path):
                os.remove(path)

    policy_specs = build_policy_specs()
    device = torch.device(args_cli.device if args_cli.device else "cuda:0")
    video_seed = args_cli.video_seed if args_cli.video_seed is not None else args_cli.seed

    print(f"\n[EVAL] Creating test environment: {args_cli.task} ({args_cli.num_envs} envs)")
    print(f"[EVAL] Evaluation condition: {eval_condition}")
    env_cfg = build_env_cfg(args_cli.task, num_envs=args_cli.num_envs, seed=args_cli.seed)
    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    isaac_env: ManagerBasedRLEnv = raw_env.unwrapped
    obstacle_names = get_obstacle_names(isaac_env)
    print(f"[EVAL] Obstacles present: {obstacle_names if obstacle_names else 'none'}")

    new_summary_rows = []
    new_episode_rows = []
    for policy_spec in policy_specs:
        summary, episode_rows = evaluate_policy(
            policy_spec=policy_spec,
            task_name=args_cli.task,
            eval_condition=eval_condition,
            raw_env=raw_env,
            isaac_env=isaac_env,
            obstacle_names=obstacle_names,
            num_episodes=args_cli.num_episodes,
            device=device,
        )
        new_summary_rows.append(summary)
        new_episode_rows.extend(episode_rows)

    raw_env.close()

    if args_cli.video:
        saved_videos = []
        for policy_spec in policy_specs:
            saved_path = record_policy_video(
                policy_spec=policy_spec,
                task_name=args_cli.task,
                device=device,
                output_video_dir=paths["video_dir"],
                video_length=args_cli.video_length,
                video_seed=video_seed,
            )
            if saved_path:
                saved_videos.append(saved_path)
        if saved_videos:
            print(f"[VIDEO] Videos saved to {paths['video_dir']}")

    existing_episode_rows = _read_csv_rows(paths["episode_csv"])
    existing_summary_rows = _read_csv_rows(paths["summary_csv"])
    episode_rows_all = existing_episode_rows + new_episode_rows
    summary_rows_all = existing_summary_rows + new_summary_rows
    family_rows = build_family_summary_rows(summary_rows_all)

    summary_metric_keys = discover_metric_keys(summary_rows_all)
    episode_metric_keys = discover_metric_keys(episode_rows_all, include_raw_episode_metrics=True)
    _write_csv_rows(
        paths["episode_csv"],
        episode_rows_all,
        EPISODE_PREFERRED_FIELDS + [key for key in episode_metric_keys if key.startswith("reward_term_")],
    )
    _write_csv_rows(
        paths["summary_csv"],
        summary_rows_all,
        SUMMARY_PREFERRED_FIELDS + metric_summary_fields(summary_metric_keys),
    )
    _write_csv_rows(
        paths["family_csv"],
        family_rows,
        FAMILY_PREFERRED_FIELDS + metric_summary_fields(discover_metric_keys(family_rows)),
    )

    plot_paths = save_all_plots(summary_rows_all, family_rows, paths["plot_dir"])

    print(f"\n[EVAL] Episode rows written to:          {paths['episode_csv']}")
    print(f"[EVAL] Checkpoint summary written to:   {paths['summary_csv']}")
    print(f"[EVAL] Family summary written to:       {paths['family_csv']}")
    if plot_paths:
        print(f"[EVAL] Plots saved to:                  {paths['plot_dir']}")
    print(f"[EVAL] Done - {len(new_summary_rows)} policies evaluated in '{args_cli.task}'.\n")


if __name__ == "__main__":
    main()
    simulation_app.close()
