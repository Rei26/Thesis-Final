"""Shared TensorBoard training-log plotter for the Franka thesis runs.

The script reads RL-Games TensorBoard event files, writes clean CSV summaries,
and saves one seed-detail plot plus one mean/std plot for every discovered
training metric.

It intentionally uses the full TensorBoard history that is available in each
run. For checkpoint reporting it selects the highest ``last_*_ep_*.pth`` file
in the run's ``nn`` directory, not the best-reward checkpoint alias.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def configure_matplotlib_cache() -> None:
    """Use a writable Matplotlib cache even when Isaac Lab runs from a locked environment."""

    if os.environ.get("MPLCONFIGDIR"):
        return

    candidates = []
    for env_name in ("TEMP", "TMP"):
        env_value = os.environ.get(env_name)
        if env_value:
            candidates.append(Path(env_value) / "isaaclab_thesis_matplotlib")
    candidates.append(PROJECT_ROOT / ".matplotlib")

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        os.environ["MPLCONFIGDIR"] = str(candidate)
        return


configure_matplotlib_cache()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


DEFAULT_SEEDS = ("42", "123", "789")
COLORS = ("#4C72B0", "#DD8452", "#55A868", "#8172B3", "#64B5CD")
SMOOTH_WEIGHT = 0.85

# Preferred thesis metrics are plotted first when they exist. Baseline logs may
# not contain obstacle-specific metrics, so missing tags are skipped gracefully.
PREFERRED_METRICS = [
    ("rewards/iter", "Total Reward per Iteration", "Reward"),
    ("Episode/metrics/grasp_success_rate", "Grasp Success Rate", "Success Rate"),
    ("Episode/metrics/collision_frequency", "Collision Frequency", "Fraction of Steps"),
    ("Episode/metrics/task_completion_time", "Task Completion Time", "Steps"),
    ("Episode/Episode_Reward/lifting_object", "Lifting Object Reward", "Reward"),
    ("Episode/Episode_Reward/reaching_object", "Reaching Object Reward", "Reward"),
    ("Episode/Episode_Reward/action_rate", "Action Rate Penalty", "Penalty"),
    ("Episode/Episode_Reward/joint_vel", "Joint Velocity Penalty", "Penalty"),
    ("Episode/Episode_Reward/obstacle_1_penalty", "Obstacle 1 Penalty", "Penalty"),
    ("Episode/Episode_Reward/obstacle_2_penalty", "Obstacle 2 Penalty", "Penalty"),
    ("Episode/Episode_Reward/obstacle_3_penalty", "Obstacle 3 Penalty", "Penalty"),
    ("Episode/Episode_Reward/obstacle_4_penalty", "Obstacle 4 Penalty", "Penalty"),
    ("Episode/Curriculum/action_rate", "Curriculum: Action Rate Weight", "Weight"),
    ("Episode/Curriculum/joint_vel", "Curriculum: Joint Velocity Weight", "Weight"),
]

FIXED_YLIM_TAGS = {
    "Episode/metrics/collision_frequency": (0.0, None),
    "Episode/metrics/grasp_success_rate": (0.0, 1.0),
    "Episode/metrics/task_completion_time": (0.0, None),
}

LOWER_IS_BETTER_TAG_PARTS = (
    "collision_frequency",
    "near_obstacle_rate",
    "task_completion_time",
    "episode_length",
    "loss",
)

LAST_CHECKPOINT_RE = re.compile(
    r"last_.*?_ep_(?P<epoch>\d+)_rew_+?(?P<reward>-?\d+(?:\.\d+)?)_?\.pth$"
)


@dataclass(frozen=True)
class PlotConfig:
    family_name: str
    title_prefix: str
    output_name: str
    log_roots: tuple[Path, ...]
    run_patterns: tuple[str, ...]
    default_seeds: tuple[str, ...] = DEFAULT_SEEDS


@dataclass
class RunInfo:
    seed: str
    label: str
    run_dir: Path
    event_dir: Path
    scalar_tags: list[str]


@dataclass
class ScalarSeries:
    steps: np.ndarray
    values: np.ndarray


def repo_path(path_text: str | Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def slugify(text: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text))
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def metric_title_from_tag(tag: str) -> str:
    if tag == "rewards/iter":
        return "Total Reward per Iteration"
    name = tag.split("/")[-1].replace("_", " ").strip()
    return name.title()


def metric_ylabel_from_tag(tag: str) -> str:
    if "collision_frequency" in tag or "success_rate" in tag:
        return "Rate"
    if "task_completion_time" in tag:
        return "Steps"
    if "Curriculum" in tag:
        return "Weight"
    if "Reward" in tag or tag.startswith("rewards/"):
        return "Reward"
    return "Value"


def metric_higher_is_better(tag: str) -> bool:
    """Return whether larger scalar values should be treated as better."""

    tag_lower = tag.lower()
    return not any(part in tag_lower for part in LOWER_IS_BETTER_TAG_PARTS)


def metric_best_value(tag: str, values: np.ndarray) -> float:
    """Select the best value using the metric's natural optimization direction."""

    if metric_higher_is_better(tag):
        return float(np.max(values))
    return float(np.min(values))


def smooth(values: np.ndarray, weight: float = SMOOTH_WEIGHT) -> np.ndarray:
    if len(values) == 0:
        return values
    smoothed = []
    last = float(values[0])
    for value in values:
        last = last * weight + (1.0 - weight) * float(value)
        smoothed.append(last)
    return np.asarray(smoothed, dtype=float)


def load_event_accumulator(event_dir: Path) -> EventAccumulator:
    accumulator = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    return accumulator


def resolve_event_dir(run_dir: Path) -> Path:
    summaries = run_dir / "summaries"
    return summaries if summaries.is_dir() else run_dir


def list_scalar_tags(event_dir: Path) -> list[str]:
    accumulator = load_event_accumulator(event_dir)
    return list(accumulator.Tags().get("scalars", []))


def load_scalar(event_dir: Path, tag: str) -> ScalarSeries | None:
    accumulator = load_event_accumulator(event_dir)
    if tag not in accumulator.Tags().get("scalars", []):
        return None
    events = accumulator.Scalars(tag)
    if not events:
        return None
    steps = np.asarray([event.step for event in events], dtype=float)
    values = np.asarray([event.value for event in events], dtype=float)
    finite = np.isfinite(steps) & np.isfinite(values)
    if not finite.any():
        return None
    return ScalarSeries(steps=steps[finite], values=values[finite])


def parse_run_dirs(values: list[str] | None, seeds: tuple[str, ...]) -> dict[str, Path]:
    if not values:
        return {}
    if len(values) != len(seeds):
        raise ValueError(f"--run_dirs must contain {len(seeds)} paths, one per seed.")
    return {seed: repo_path(value) for seed, value in zip(seeds, values)}


def find_run_for_seed(log_roots: tuple[Path, ...], patterns: tuple[str, ...], seed: str) -> Path | None:
    candidates: list[Path] = []
    for root in log_roots:
        if not root.is_dir():
            continue
        for pattern in patterns:
            candidates.extend(path for path in root.glob(pattern.format(seed=seed)) if path.is_dir())
    if not candidates:
        return None
    candidates = sorted(set(candidates), key=lambda path: (path.stat().st_mtime, str(path)))
    return candidates[-1]


def discover_runs(config: PlotConfig, explicit_run_dirs: dict[str, Path], seeds: tuple[str, ...]) -> list[RunInfo]:
    runs: list[RunInfo] = []
    for seed in seeds:
        # Explicit paths are used when provided; otherwise each family entrypoint
        # searches its known log roots for the latest matching seed folder.
        run_dir = explicit_run_dirs.get(seed) or find_run_for_seed(config.log_roots, config.run_patterns, seed)
        if run_dir is None:
            print(f"[WARN] No run folder found for seed {seed}.")
            continue
        event_dir = resolve_event_dir(run_dir)
        if not event_dir.exists():
            print(f"[WARN] Run for seed {seed} has no TensorBoard event directory: {event_dir}")
            continue
        try:
            tags = list_scalar_tags(event_dir)
        except Exception as exc:
            print(f"[WARN] Could not read TensorBoard scalars for seed {seed}: {event_dir} ({exc})")
            continue
        if not tags:
            print(f"[WARN] No scalar tags found for seed {seed}: {event_dir}")
            continue
        runs.append(
            RunInfo(
                seed=seed,
                label=f"Seed {seed}",
                run_dir=run_dir,
                event_dir=event_dir,
                scalar_tags=tags,
            )
        )
    return runs


def discover_metrics(runs: list[RunInfo], include_all_scalars: bool) -> list[tuple[str, str, str]]:
    # Metrics are discovered from TensorBoard tags rather than hard-coded per
    # family, which keeps baseline plots valid even without custom R16 scalars.
    available = set()
    for run in runs:
        available.update(run.scalar_tags)

    metrics: list[tuple[str, str, str]] = []
    seen = set()
    for tag, title, ylabel in PREFERRED_METRICS:
        if tag in available:
            metrics.append((tag, title, ylabel))
            seen.add(tag)

    prefixes = ("Episode/Episode_Reward/", "Episode/Curriculum/", "Episode/metrics/", "rewards/")
    for tag in sorted(available):
        if tag in seen:
            continue
        if include_all_scalars or tag.startswith(prefixes):
            metrics.append((tag, metric_title_from_tag(tag), metric_ylabel_from_tag(tag)))
            seen.add(tag)

    return metrics


def align_smoothed_series(series_by_run: dict[str, ScalarSeries]) -> tuple[np.ndarray, np.ndarray] | None:
    valid = [series for series in series_by_run.values() if series is not None and len(series.steps) > 1]
    if len(valid) < 2:
        return None

    # Seeds may have slightly different event steps. Interpolating onto the
    # shortest common timeline makes the mean/std plots comparable.
    reference = min(valid, key=lambda item: len(item.steps))
    max_common_step = min(float(series.steps[-1]) for series in valid)
    common_steps = reference.steps[reference.steps <= max_common_step]
    if len(common_steps) < 2:
        return None

    aligned = []
    for series in valid:
        smoothed_values = smooth(series.values)
        aligned.append(np.interp(common_steps, series.steps, smoothed_values))
    return common_steps, np.asarray(aligned, dtype=float)


def apply_metric_limits(ax, tag: str, values: list[float]) -> None:
    ymin, ymax = FIXED_YLIM_TAGS.get(tag, (None, None))
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if ymin is None and finite and min(finite) >= 0.0:
        ymin = 0.0
    if ymax is None and tag == "Episode/metrics/grasp_success_rate":
        ymax = 1.0
    if ymin is not None:
        ax.set_ylim(bottom=ymin)
    if ymax is not None:
        ax.set_ylim(top=ymax)


def save_seed_detail_plot(
    config: PlotConfig,
    tag: str,
    title: str,
    ylabel: str,
    series_by_run: dict[str, ScalarSeries],
    output_dir: Path,
) -> Path | None:
    if not any(series is not None for series in series_by_run.values()):
        return None

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    all_values: list[float] = []
    aligned_input: dict[str, ScalarSeries] = {}

    for index, (label, series) in enumerate(series_by_run.items()):
        if series is None:
            continue
        color = COLORS[index % len(COLORS)]
        smoothed_values = smooth(series.values)
        ax.plot(series.steps, series.values, color=color, alpha=0.12, linewidth=0.8)
        ax.plot(series.steps, smoothed_values, color=color, linewidth=2.1, label=label)
        aligned_input[label] = series
        all_values.extend(smoothed_values.tolist())

    aligned = align_smoothed_series(aligned_input)
    if aligned is not None:
        common_steps, aligned_values = aligned
        mean = np.mean(aligned_values, axis=0)
        ax.plot(common_steps, mean, color="black", linewidth=2.5, linestyle="--", label="Mean", zorder=5)
        all_values.extend(mean.tolist())

    apply_metric_limits(ax, tag, all_values)
    ax.set_title(f"{config.title_prefix} - {title}", fontsize=15, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.28)
    ax.legend(fontsize=10.5)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()

    output_path = output_dir / "seed_detail" / f"{slugify(tag)}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_mean_std_plot(
    config: PlotConfig,
    tag: str,
    title: str,
    ylabel: str,
    series_by_run: dict[str, ScalarSeries],
    output_dir: Path,
) -> Path | None:
    aligned = align_smoothed_series({label: series for label, series in series_by_run.items() if series is not None})
    if aligned is None:
        return None

    common_steps, aligned_values = aligned
    mean = np.mean(aligned_values, axis=0)
    std = np.std(aligned_values, axis=0, ddof=1 if aligned_values.shape[0] > 1 else 0)

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    color = COLORS[0]
    ax.plot(common_steps, mean, color=color, linewidth=2.7, label=f"Mean ({aligned_values.shape[0]} seeds)")
    ax.fill_between(common_steps, mean - std, mean + std, color=color, alpha=0.24, label="+- 1 std dev")

    apply_metric_limits(ax, tag, (mean + std).tolist() + (mean - std).tolist())
    ax.set_title(f"{config.title_prefix} - {title} (Mean +- Std)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.28)
    ax.legend(fontsize=10.5)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()

    output_path = output_dir / "mean_std" / f"{slugify(tag)}_mean_std_band.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    if tag == "rewards/iter":
        fig.savefig(output_dir / "rewards_mean_std_band.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def find_last_checkpoint(run_dir: Path) -> tuple[Path | None, int | None, float | None]:
    checkpoint_dir = run_dir / "nn"
    if not checkpoint_dir.is_dir():
        return None, None, None

    matches: list[tuple[int, float | None, Path]] = []
    for path in checkpoint_dir.glob("*.pth"):
        match = LAST_CHECKPOINT_RE.match(path.name)
        if not match:
            continue
        epoch = int(match.group("epoch"))
        try:
            reward = float(match.group("reward"))
        except ValueError:
            reward = None
        matches.append((epoch, reward, path))

    if not matches:
        return None, None, None
    epoch, reward, path = sorted(matches, key=lambda item: (item[0], item[2].stat().st_mtime))[-1]
    return path, epoch, reward


def write_scalar_history_csv(
    path: Path,
    config: PlotConfig,
    runs: list[RunInfo],
    metrics: list[tuple[str, str, str]],
    loaded: dict[tuple[str, str], ScalarSeries | None],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["policy_family", "seed", "run_name", "run_dir", "tag", "step", "value"],
        )
        writer.writeheader()
        for run in runs:
            for tag, _, _ in metrics:
                series = loaded.get((run.seed, tag))
                if series is None:
                    continue
                for step, value in zip(series.steps, series.values):
                    writer.writerow(
                        {
                            "policy_family": config.family_name,
                            "seed": run.seed,
                            "run_name": run.run_dir.name,
                            "run_dir": str(run.run_dir),
                            "tag": tag,
                            "step": int(step),
                            "value": float(value),
                        }
                    )


def write_summary_csvs(
    output_dir: Path,
    config: PlotConfig,
    runs: list[RunInfo],
    metrics: list[tuple[str, str, str]],
    loaded: dict[tuple[str, str], ScalarSeries | None],
) -> None:
    by_seed_path = output_dir / "training_summary_by_seed.csv"
    by_metric_path = output_dir / "training_summary_mean_std.csv"

    metric_slugs = [slugify(tag) for tag, _, _ in metrics]
    by_seed_fields = [
        "policy_family",
        "seed",
        "run_name",
        "run_dir",
        "last_checkpoint",
        "last_checkpoint_epoch",
        "last_checkpoint_reward_from_name",
    ]
    for metric_slug in metric_slugs:
        by_seed_fields.extend(
            [
                f"{metric_slug}_last_step",
                f"{metric_slug}_last_value",
                f"{metric_slug}_best_value",
                f"{metric_slug}_n_points",
            ]
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    final_values_by_metric: dict[str, list[float]] = {tag: [] for tag, _, _ in metrics}
    with by_seed_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=by_seed_fields)
        writer.writeheader()
        for run in runs:
            checkpoint_path, checkpoint_epoch, checkpoint_reward = find_last_checkpoint(run.run_dir)
            row = {
                "policy_family": config.family_name,
                "seed": run.seed,
                "run_name": run.run_dir.name,
                "run_dir": str(run.run_dir),
                "last_checkpoint": str(checkpoint_path) if checkpoint_path else "",
                "last_checkpoint_epoch": checkpoint_epoch if checkpoint_epoch is not None else "",
                "last_checkpoint_reward_from_name": checkpoint_reward if checkpoint_reward is not None else "",
            }
            for tag, _, _ in metrics:
                metric_slug = slugify(tag)
                series = loaded.get((run.seed, tag))
                if series is None or len(series.values) == 0:
                    row[f"{metric_slug}_last_step"] = ""
                    row[f"{metric_slug}_last_value"] = ""
                    row[f"{metric_slug}_best_value"] = ""
                    row[f"{metric_slug}_n_points"] = 0
                    continue
                final_value = float(series.values[-1])
                final_values_by_metric[tag].append(final_value)
                row[f"{metric_slug}_last_step"] = int(series.steps[-1])
                row[f"{metric_slug}_last_value"] = final_value
                row[f"{metric_slug}_best_value"] = metric_best_value(tag, series.values)
                row[f"{metric_slug}_n_points"] = int(len(series.values))
            writer.writerow(row)

    with by_metric_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "policy_family",
                "tag",
                "title",
                "n_seeds",
                "last_value_mean",
                "last_value_std",
                "last_value_min",
                "last_value_max",
            ],
        )
        writer.writeheader()
        for tag, title, _ in metrics:
            values = np.asarray(final_values_by_metric[tag], dtype=float)
            if len(values) == 0:
                continue
            writer.writerow(
                {
                    "policy_family": config.family_name,
                    "tag": tag,
                    "title": title,
                    "n_seeds": int(len(values)),
                    "last_value_mean": float(np.mean(values)),
                    "last_value_std": float(np.std(values, ddof=1 if len(values) > 1 else 0)),
                    "last_value_min": float(np.min(values)),
                    "last_value_max": float(np.max(values)),
                }
            )


def parse_args(config: PlotConfig) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Plot training logs for {config.family_name}.")
    parser.add_argument(
        "--run_dirs",
        nargs="+",
        default=None,
        help="Explicit run directories, one per seed. Relative paths are resolved from the repo root.",
    )
    parser.add_argument("--seeds", nargs="+", default=list(config.default_seeds), help="Seed labels for --run_dirs/discovery.")
    parser.add_argument(
        "--log_roots",
        nargs="+",
        default=None,
        help="Override log roots used for auto-discovery. Relative paths are resolved from the repo root.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Override output directory. Relative paths are resolved from the repo root.",
    )
    parser.add_argument(
        "--include_all_scalars",
        action="store_true",
        help="Plot every TensorBoard scalar instead of only rewards, curriculum, and episode metrics.",
    )
    return parser.parse_args()


def run(config: PlotConfig) -> None:
    args = parse_args(config)
    seeds = tuple(str(seed) for seed in args.seeds)
    log_roots = tuple(repo_path(path) for path in args.log_roots) if args.log_roots else config.log_roots
    config = PlotConfig(
        family_name=config.family_name,
        title_prefix=config.title_prefix,
        output_name=config.output_name,
        log_roots=log_roots,
        run_patterns=config.run_patterns,
        default_seeds=seeds,
    )
    output_dir = repo_path(args.output_dir) if args.output_dir else PROJECT_ROOT / "thesis_plots" / "training" / config.output_name
    explicit_run_dirs = parse_run_dirs(args.run_dirs, seeds)

    print(f"\n[TRAINING PLOTS] Policy family: {config.family_name}")
    print("[TRAINING PLOTS] Searching log roots:")
    for root in config.log_roots:
        print(f"  - {root}")
    print(f"[TRAINING PLOTS] Saving to: {output_dir}\n")

    runs = discover_runs(config, explicit_run_dirs, seeds)
    if not runs:
        raise RuntimeError(
            "No readable training runs were found. Pass explicit paths with --run_dirs, "
            "or update the run patterns in the entrypoint script."
        )

    print("[TRAINING PLOTS] Using runs:")
    for run in runs:
        checkpoint_path, checkpoint_epoch, _ = find_last_checkpoint(run.run_dir)
        checkpoint_note = f"last epoch {checkpoint_epoch}" if checkpoint_epoch is not None else "no last checkpoint found"
        print(f"  - {run.label}: {run.run_dir} ({checkpoint_note})")
        if checkpoint_path:
            print(f"      {checkpoint_path.name}")

    metrics = discover_metrics(runs, include_all_scalars=args.include_all_scalars)
    if not metrics:
        raise RuntimeError("No matching scalar metrics were found in the selected runs.")

    loaded: dict[tuple[str, str], ScalarSeries | None] = {}
    saved_plots: list[Path] = []
    for tag, title, ylabel in metrics:
        series_by_run: dict[str, ScalarSeries | None] = {}
        for run in runs:
            series = load_scalar(run.event_dir, tag) if tag in run.scalar_tags else None
            loaded[(run.seed, tag)] = series
            series_by_run[run.label] = series

        seed_plot = save_seed_detail_plot(config, tag, title, ylabel, series_by_run, output_dir)
        mean_plot = save_mean_std_plot(config, tag, title, ylabel, series_by_run, output_dir)
        if seed_plot:
            saved_plots.append(seed_plot)
        if mean_plot:
            saved_plots.append(mean_plot)

    write_scalar_history_csv(output_dir / "training_scalar_history.csv", config, runs, metrics, loaded)
    write_summary_csvs(output_dir, config, runs, metrics, loaded)

    print(f"\n[TRAINING PLOTS] Wrote CSVs:")
    print(f"  - {output_dir / 'training_scalar_history.csv'}")
    print(f"  - {output_dir / 'training_summary_by_seed.csv'}")
    print(f"  - {output_dir / 'training_summary_mean_std.csv'}")
    print(f"[TRAINING PLOTS] Saved {len(saved_plots)} plots.")
    print("[TRAINING PLOTS] Done.\n")
