# Franka Obstacle-Grasping Thesis Reproducibility Guide

This repository is a thesis-specific Isaac Lab workspace for training and evaluating Franka Panda reinforcement-learning policies. It compares a standard no-obstacle cube-lifting baseline against an obstacle-aware R16 condition trained with randomized obstacle geometry.

The important experimental control is that the saved baseline and R16 runs use the same RL-Games PPO architecture and hyperparameters. The main experimental difference is the training environment:

- Baseline: standard Franka cube lifting without obstacles.
- R16 main condition: randomized obstacle-aware grasping.
- Evaluation: both policy families are tested on fixed benchmark environments BL, A, B, and C.

The repository keeps Isaac Lab's original structure and adds thesis-specific environment code, PPO YAMLs, plotting scripts, and evaluation utilities.

## Project Layout

Most thesis-specific code lives in these files:

| Path | Purpose |
| --- | --- |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/franka_grasping_env.py` | Defines the obstacle-aware grasping environments, fixed BL/A/B/C benchmark configs, obstacle observations, obstacle rewards, curricula, contact metrics, and wrapper classes. |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/__init__.py` | Registers the Gymnasium task IDs used for training and evaluation. |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/agents/rl_games_ppo_cfg.yaml` | Baseline RL-Games PPO config. It uses the same PPO hyperparameters as R16, with `name: franka_lift`. |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/agents/rl_games_r16_ppo_cfg.yaml` | R16 RL-Games PPO config. It matches the baseline PPO hyperparameters, with `name: franka_grasping`. |
| `scripts/reinforcement_learning/rl_games/evaluate_generalization.py` | Evaluates saved checkpoints across BL/A/B/C and writes CSV summaries plus plots. |
| `visualisation/training_log_plotter.py` | Shared TensorBoard log parser and plotter for training curves. |
| `visualisation/plot_baseline.py` | Entrypoint for baseline training plots. |
| `visualisation/plot_main_condition.py` | Entrypoint for R16 training plots. |

Saved runs live under `logs/rl_games/`. Each run should contain:

- `params/agent.yaml`: exact RL-Games runtime config.
- `params/env.yaml`: exact environment runtime config.
- `nn/*.pth`: saved checkpoints.
- `summaries/events.out.tfevents.*`: TensorBoard event logs.

For reproducibility, the saved `params/*.yaml` files are the source of truth for what actually ran.

## Task IDs

Training tasks:

| Task ID | Meaning | PPO config |
| --- | --- | --- |
| `Isaac-Lift-Cube-Franka-v0` | Baseline no-obstacle cube lift | `rl_games_ppo_cfg.yaml` |
| `Isaac-Franka-Grasping-v0` | R16 randomized obstacle-aware grasping | `rl_games_r16_ppo_cfg.yaml` |

Evaluation tasks:

| Task ID | Evaluation condition |
| --- | --- |
| `Isaac-Franka-Grasping-Baseline-v0` | BL, no obstacles |
| `Isaac-Franka-Grasping-A-v0` | A, fixed two-obstacle benchmark |
| `Isaac-Franka-Grasping-B-v0` | B, larger fixed two-obstacle benchmark |
| `Isaac-Franka-Grasping-C-v0` | C, four-obstacle clutter benchmark |

Play/video variants also exist with `-Play-v0`.

Evaluation benchmark geometry is defined in `franka_grasping_env.py`. The fixed benchmark layouts are:

| Condition | Obstacles | Obstacle size `(x, y, z)` | Offsets from object center `(x, y)` |
| --- | ---: | --- | --- |
| BL | 0 | none | none |
| A | 2 | `(0.06, 0.06, 0.18)` | `(-0.10, 0.075)`, `(-0.10, -0.075)` |
| B | 2 | `(0.12, 0.12, 0.25)` | `(-0.10, 0.13)`, `(-0.10, -0.13)` |
| C | 4 | `(0.04, 0.04, 0.10)` | `(-0.10, 0.07)`, `(-0.10, -0.07)`, `(-0.02, 0.11)`, `(-0.02, -0.11)` |

## PPO Configuration

Baseline and R16 intentionally use matched PPO hyperparameters. The two YAML files differ only in `params.config.name`, which controls logging/checkpoint naming:

| File | `params.config.name` |
| --- | --- |
| `rl_games_ppo_cfg.yaml` | `franka_lift` |
| `rl_games_r16_ppo_cfg.yaml` | `franka_grasping` |

Key shared PPO values:

| Setting | Value |
| --- | ---: |
| `reward_shaper.scale_value` | `1.0` |
| `learning_rate` | `5e-4` |
| `kl_threshold` | `0.016` |
| `mini_epochs` | `4` |
| `max_epochs` | `1500` |
| `horizon_length` | `24` |
| `minibatch_size` | `24576` |
| `entropy_coef` | `0.001` |
| `critic_coef` | `4` |
| network MLP | `[256, 128, 64]` |

This means the thesis comparison is not a comparison of different PPO optimizers. It is a comparison of training conditions under matched PPO settings.

## Setup

This workspace is based on Isaac Lab and is normally run from PowerShell on Windows using `isaaclab.bat`.

This thesis code was developed and tested with Isaac Lab `2.3.2` / `isaaclab` extension `0.54.3` and Isaac Sim `4.5.0.0`. Other Isaac Sim or Isaac Lab version combinations may work, but are not guaranteed to reproduce the reported results.

Minimum practical requirements:

- NVIDIA GPU with CUDA support.
- Isaac Sim/Isaac Lab installation compatible with this repo.
- RL-Games dependencies installed through Isaac Lab.
- Enough GPU memory for `4096` environments, or reduce `--num_envs` for debugging.

From the repository root:

```powershell
.\isaaclab.bat --install
```

If only RL-Games dependencies are needed:

```powershell
.\isaaclab.bat --install rl_games
```

Quick launch smoke test:

```powershell
.\isaaclab.bat -p scripts\tutorials\00_sim\create_empty.py
```

## Training

Run commands from the repository root.

### Baseline

Train the five baseline seeds `42`, `123`, `456`, `789`, and `999`. The command below shows seed `999`; repeat it with the other seed values and matching experiment names.

```powershell
.\isaaclab.bat -p scripts\reinforcement_learning\rl_games\train.py `
  --task Isaac-Lift-Cube-Franka-v0 `
  --num_envs 4096 `
  --seed 999 `
  --headless `
  +agent.params.config.full_experiment_name=baseline999
```

The expected output folder is:

```text
logs/rl_games/franka_lift/baseline999
```

For the thesis evaluation layout, this run can be moved or copied to:

```text
logs/rl_games/baseline/baseline999
```

The saved `params/agent.yaml` should contain:

```yaml
seed: 999
config:
  name: franka_lift
```

### R16 Main Condition

Train the five R16 main-condition seeds `42`, `123`, `456`, `789`, and `999`. The command below shows seed `999`; repeat it with the other seed values and matching `R16_final_s<seed>` experiment names.

```powershell
.\isaaclab.bat -p scripts\reinforcement_learning\rl_games\train.py `
  --task Isaac-Franka-Grasping-v0 `
  --num_envs 4096 `
  --seed 999 `
  --headless `
  +agent.params.config.full_experiment_name=R16_final_s999
```

The `+` is intentional. Hydra adds `full_experiment_name` at launch time.

## Monitoring Training

Open TensorBoard for all RL-Games logs:

```powershell
.\isaaclab.bat -p -m tensorboard.main --logdir logs\rl_games
```

Then open the printed URL, usually:

```text
http://localhost:6006
```

## Training Plots

After training is complete, generate TensorBoard-derived training plots:

```powershell
.\isaaclab.bat -p visualisation\plot_baseline.py `
  --seeds 42 123 456 789 999
```

```powershell
.\isaaclab.bat -p visualisation\plot_main_condition.py `
  --seeds 42 123 456 789 999
```

The plot scripts auto-discover the expected saved runs:

```text
logs/rl_games/baseline/baseline42
logs/rl_games/baseline/baseline123
logs/rl_games/baseline/baseline456
logs/rl_games/baseline/baseline789
logs/rl_games/baseline/baseline999
logs/rl_games/obstacle_training/R16_final_s42
logs/rl_games/obstacle_training/R16_final_s123
logs/rl_games/obstacle_training/R16_final_s456
logs/rl_games/obstacle_training/R16_final_s789
logs/rl_games/obstacle_training/R16_final_s999
```

Outputs are written under:

```text
thesis_plots/training/
```

The full TensorBoard-derived training plot set is available in this folder, split by policy family under `baseline/` and `r16_main_condition/`. The summary CSVs in each subfolder provide the plotted scalar histories and seed-level summaries.

Baseline TensorBoard logs do not contain all custom R16 metrics, such as obstacle collision frequency during training. The plotter skips missing metrics and continues.

## Generalization Evaluation

Generalization is evaluated by testing each trained policy on fixed BL/A/B/C benchmark environments. The main thesis metric is grasp success rate. Supporting metrics include episode return, task completion time, collision frequency, near-obstacle rate, mean clearance, minimum clearance, and maximum object height.

The script does not invent a separate scalar "generalization score". Generalization is operationalized as the ability to maintain performance across the fixed benchmark conditions.

Run a clean full evaluation sweep:

```powershell
$OutputDir = "results_and_plots"

$Checkpoints = @(
  "logs\rl_games\baseline\baseline42",
  "logs\rl_games\baseline\baseline123",
  "logs\rl_games\baseline\baseline456",
  "logs\rl_games\baseline\baseline789",
  "logs\rl_games\baseline\baseline999",
  "logs\rl_games\obstacle_training\R16_final_s42",
  "logs\rl_games\obstacle_training\R16_final_s123",
  "logs\rl_games\obstacle_training\R16_final_s456",
  "logs\rl_games\obstacle_training\R16_final_s789",
  "logs\rl_games\obstacle_training\R16_final_s999"
)

$PolicyLabels = @(
  "BL_s42", "BL_s123", "BL_s456", "BL_s789", "BL_s999",
  "R16_s42", "R16_s123", "R16_s456", "R16_s789", "R16_s999"
)

$PolicyFamilies = @(
  "Baseline", "Baseline", "Baseline", "Baseline", "Baseline",
  "R16", "R16", "R16", "R16", "R16"
)

$TrainSeeds = @(
  "42", "123", "456", "789", "999",
  "42", "123", "456", "789", "999"
)

$AgentCfgs = @(
  "rl_games_ppo_cfg.yaml",
  "rl_games_ppo_cfg.yaml",
  "rl_games_ppo_cfg.yaml",
  "rl_games_ppo_cfg.yaml",
  "rl_games_ppo_cfg.yaml",
  "rl_games_r16_ppo_cfg.yaml",
  "rl_games_r16_ppo_cfg.yaml",
  "rl_games_r16_ppo_cfg.yaml",
  "rl_games_r16_ppo_cfg.yaml",
  "rl_games_r16_ppo_cfg.yaml"
)

$EvalTasks = @(
  "Isaac-Franka-Grasping-Baseline-v0",
  "Isaac-Franka-Grasping-A-v0",
  "Isaac-Franka-Grasping-B-v0",
  "Isaac-Franka-Grasping-C-v0"
)

$FirstRun = $true

foreach ($Task in $EvalTasks) {
  $ClearArg = @()
  if ($FirstRun) {
    $ClearArg = @("--clear_output")
    $FirstRun = $false
  }

  .\isaaclab.bat -p scripts\reinforcement_learning\rl_games\evaluate_generalization.py `
    --task $Task `
    --checkpoint_selection last `
    --checkpoints $Checkpoints `
    --policy_labels $PolicyLabels `
    --policy_families $PolicyFamilies `
    --train_seeds $TrainSeeds `
    --agent_cfgs $AgentCfgs `
    --num_episodes 100 `
    --num_envs 16 `
    --output_dir $OutputDir `
    --headless `
    @ClearArg
}
```

The first condition uses `--clear_output` to start fresh. Later conditions append into the same CSVs. The expected checkpoint summary has 40 rows:

```text
4 evaluation conditions x 2 policy families x 5 seeds = 40 rows
```

Outputs:

```text
results_and_plots/episode_results.csv
results_and_plots/summary_by_checkpoint.csv
results_and_plots/summary_by_family.csv
results_and_plots/plots/
```

The full evaluation plot set generated from the evaluation CSVs is available in `results_and_plots/plots/`, including family-level mean/std summaries and seed-detail plots.

The evaluation script resolves each run directory to the highest final epoch checkpoint, expected to be `ep_1500`. It warns if a checkpoint or saved seed does not match the command-line label.

## Final Evaluation Results

The final saved evaluation in `results_and_plots/` contains:

```text
episode_results.csv:       4000 rows
summary_by_checkpoint.csv: 40 rows
summary_by_family.csv:     8 rows
```

This corresponds to:

```text
4 evaluation conditions x 2 policy families x 5 seeds x 100 episodes = 4000 episodes
4 evaluation conditions x 2 policy families x 5 seeds = 40 checkpoint summaries
4 evaluation conditions x 2 policy families = 8 family summaries
```

The table below reports family-level means across the five seeds. Grasp success rate is the primary thesis metric. Episode return, task completion time, collision frequency, near-obstacle rate, and clearance are supporting metrics used to interpret whether success is safe, efficient, and robust.

| Condition | Policy | Grasp success | Episode return | Task time steps | Collision frequency | Near-obstacle rate | Mean clearance m |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BL | Baseline | `1.000 +/- 0.000` | `140.9 +/- 12.5` | `80.8 +/- 1.7` | `0.0000 +/- 0.0000` | `0.0000` | n/a |
| BL | R16 | `0.114 +/- 0.244` | `0.6 +/- 23.4` | `234.9 +/- 32.3` | `0.0000 +/- 0.0000` | `0.0000` | n/a |
| A | Baseline | `0.930 +/- 0.101` | `129.6 +/- 23.5` | `96.4 +/- 22.1` | `0.0029 +/- 0.0062` | `0.0544` | `0.214` |
| A | R16 | `1.000 +/- 0.000` | `133.4 +/- 4.8` | `81.4 +/- 1.5` | `0.0000 +/- 0.0000` | `0.0616` | `0.213` |
| B | Baseline | `0.000 +/- 0.000` | `1.7 +/- 0.5` | `250.0 +/- 0.0` | `0.0075 +/- 0.0147` | `0.3474` | `0.094` |
| B | R16 | `0.778 +/- 0.309` | `88.5 +/- 40.9` | `130.3 +/- 57.3` | `0.0120 +/- 0.0147` | `0.3796` | `0.130` |
| C | Baseline | `0.382 +/- 0.156` | `42.8 +/- 18.0` | `205.1 +/- 19.8` | `0.1152 +/- 0.0525` | `0.4433` | `0.138` |
| C | R16 | `0.892 +/- 0.150` | `105.9 +/- 22.7` | `110.4 +/- 27.7` | `0.0263 +/- 0.0410` | `0.1687` | `0.216` |

The final results show a clear tradeoff. The baseline policy is strongest in the clean BL environment, where it reaches perfect success across all five seeds. The R16 policy sacrifices this clean-environment performance, but gains substantial obstacle robustness. The clearest improvement appears in Condition B, where the baseline fails completely while R16 reaches `0.778` mean grasp success. Condition C shows the same pattern in a cluttered setting: R16 reaches `0.892` mean success versus `0.382` for the baseline, with lower collision frequency and greater mean clearance. Condition A is easier for both policies, but R16 is more consistent and reaches `1.000` success across all seeds.

## Expected Saved Runs

Before generating final plots, verify these ten runs exist and have final epoch checkpoints:

| Family | Seed | Expected path |
| --- | ---: | --- |
| Baseline | 42 | `logs/rl_games/baseline/baseline42` |
| Baseline | 123 | `logs/rl_games/baseline/baseline123` |
| Baseline | 456 | `logs/rl_games/baseline/baseline456` |
| Baseline | 789 | `logs/rl_games/baseline/baseline789` |
| Baseline | 999 | `logs/rl_games/baseline/baseline999` |
| R16 | 42 | `logs/rl_games/obstacle_training/R16_final_s42` |
| R16 | 123 | `logs/rl_games/obstacle_training/R16_final_s123` |
| R16 | 456 | `logs/rl_games/obstacle_training/R16_final_s456` |
| R16 | 789 | `logs/rl_games/obstacle_training/R16_final_s789` |
| R16 | 999 | `logs/rl_games/obstacle_training/R16_final_s999` |

Each run should contain an `nn/last_*_ep_1500_*.pth` checkpoint and a saved `params/agent.yaml`.

## Attribution And External Dependencies

This project is built on NVIDIA Isaac Lab and Isaac Sim. The upstream Isaac Lab code is retained under its original licenses. See:

- `LICENSE`
- `LICENSE-mimic`
- `CITATION.cff`
- `docs/licenses/`
- upstream documentation in `docs/`

The thesis-specific Python code, task configuration, plotting scripts, and evaluation scripts in this repository are original project work, but they rely on external open-source and NVIDIA software. These dependencies should be credited when reporting or publishing results:

| Dependency | Used For | Reference |
| --- | --- | --- |
| NVIDIA Isaac Lab | Manager-based task structure, vectorized robot learning environments, RL integration, assets, wrappers, and training scripts. | <https://developer.nvidia.com/isaac/lab> |
| NVIDIA Isaac Sim | Physics simulation, rendering, robot simulation backend, PhysX/Omniverse runtime. | <https://developer.nvidia.com/isaac/sim> |
| RL-Games | PPO implementation, actor-critic policy training, checkpointing, and inference player. | <https://github.com/Denys88/rl_games> |
| PyTorch | Neural-network tensors, checkpoint loading, and policy inference/training backend used by RL-Games. | <https://pytorch.org/> |
| Gymnasium | Gym-style task registration and environment construction. | <https://gymnasium.farama.org/> |
| Hydra / OmegaConf | Runtime configuration composition and command-line overrides used by Isaac Lab training scripts. | <https://hydra.cc/> |
| NumPy | Numerical analysis, array processing, interpolation, and CSV/plot summaries. | <https://numpy.org/> |
| Matplotlib | Training and evaluation figure generation. | <https://matplotlib.org/> |
| TensorBoard | Reading and visualizing RL-Games training summaries. | <https://www.tensorflow.org/tensorboard> |
| OpenUSD / Pixar USD Python APIs | USD scene inspection and local asset handling through Isaac Sim/Isaac Lab. | <https://www.pixar.com/openusd> |

The thesis-specific additions are the Franka obstacle-grasping environment, fixed BL/A/B/C benchmark configs, matched PPO YAML organization, saved experiment workflow, training-log plotting scripts, and generalization evaluation script.
