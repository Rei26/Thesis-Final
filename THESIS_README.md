# Thesis Reproducibility Guide

This repository contains an Isaac Lab based reinforcement learning project for
Franka cube lifting and obstacle-aware grasping. It keeps the original Isaac Lab
framework structure and adds thesis-specific task code, PPO configurations,
saved runs, and evaluation utilities for comparing a baseline lifting policy
against the R16 randomized-obstacle grasping condition.

The main goal of the project is reproducible training and evaluation. Each run
stores its environment and agent configuration under `logs/.../params/`, so the
exact runtime setup can be inspected after training.

## Project Structure

Important project-specific files:

- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/joint_pos_env_cfg.py`
  defines the baseline Franka cube lifting environment.
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/franka_grasping_env.py`
  defines the obstacle-aware Franka grasping environments, obstacle
  randomization, obstacle observations, reward terms, curricula, and thesis
  metrics.
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/__init__.py`
  registers the Gymnasium task IDs used by Isaac Lab.
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/agents/rl_games_ppo_cfg.yaml`
  is the default RL-Games PPO configuration used by the baseline lift task.
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/agents/rl_games_r16_ppo_cfg.yaml`
  is the R16 PPO configuration used by the main obstacle-grasping condition.
- `scripts/reinforcement_learning/rl_games/evaluate_generalization.py`
  evaluates trained policies on specified test environments and writes metrics
  to CSV, with optional plots and videos.
- `logs/rl_games/`
  contains saved training outputs, checkpoints, TensorBoard summaries, and the
  runtime `env.yaml` and `agent.yaml` files for completed runs.

## Conditions

### Baseline: Cube Lifting

Task ID:

```text
Isaac-Lift-Cube-Franka-v0
```

This condition uses the standard manager-based Franka cube lift environment. It
does not include obstacle geometry or obstacle-specific observations. It uses the
default PPO configuration:

```text
agents/rl_games_ppo_cfg.yaml
```

This configuration is named `franka_lift` and uses the default lift-task PPO
settings.

### R16 Main Condition: Randomized Obstacle Grasping

Task ID:

```text
Isaac-Franka-Grasping-v0
```

This condition extends the baseline lifting task with two randomized obstacles.
The obstacle condition includes:

- randomized obstacle size ranges;
- per-reset obstacle position randomization;
- curricula for obstacle position difficulty and obstacle penalties;
- obstacle position and size observations;
- proximity and contact penalty reward terms;
- logged thesis metrics such as grasp success rate, collision frequency, task
  completion time, and lift progress rate.

The R16 condition uses the task-specific PPO configuration:

```text
agents/rl_games_r16_ppo_cfg.yaml
```

The policy architecture is kept the same as the baseline, but selected PPO
optimization settings differ to stabilize learning in the harder obstacle-rich
task.

## PPO Configuration Differences

Both conditions use RL-Games PPO with:

- continuous actor-critic model;
- MLP policy/value network with hidden units `[256, 128, 64]`;
- ELU activation;
- `gamma = 0.99`;
- `tau = 0.95`;
- `horizon_length = 24`;
- `minibatch_size = 24576`;
- `entropy_coef = 0.001`;
- `e_clip = 0.2`;
- `critic_coef = 4`;
- `max_epochs = 1500`.

The main differences are:

| Setting | Baseline | R16 obstacle condition |
| --- | ---: | ---: |
| `config.name` | `franka_lift` | `franka_grasping` |
| `reward_shaper.scale_value` | `0.01` | `1.0` |
| `learning_rate` | `1e-4` | `5e-4` |
| `kl_threshold` | `0.01` | `0.016` |
| `mini_epochs` | `8` | `4` |

Interpretation for the thesis: the baseline and R16 conditions use the same PPO
implementation and policy architecture, but R16 uses a task-specific optimization
schedule. The R16 task has obstacle randomization and additional penalty terms,
so the reward scale and update schedule were adjusted to make learning stable in
that environment. This means the comparison should be described as the baseline
training setup versus the full obstacle-aware training setup, not as a pure
single-variable ablation of obstacles only.

## Setup

This repository is based on Isaac Lab `2.3.2`, which supports Isaac Sim 4.5,
5.0, and 5.1. The local workspace was used on Windows with PowerShell and the
`isaaclab.bat` helper script.

Minimum requirements:

- NVIDIA GPU with CUDA support;
- Isaac Sim compatible with Isaac Lab `2.3.x`;
- Python version required by the installed Isaac Sim version;
- RL-Games installed through Isaac Lab;
- enough GPU memory for `4096` parallel environments, or use a lower
  `--num_envs` value for debugging.

Install Isaac Lab extensions and learning-framework dependencies from the repo
root:

```powershell
.\isaaclab.bat --install
```

If only RL-Games is needed:

```powershell
.\isaaclab.bat --install rl_games
```

Verify that Isaac Lab can launch a simple script:

```powershell
.\isaaclab.bat -p scripts\tutorials\00_sim\create_empty.py
```

## Training

Run commands from the repository root.

### Baseline Training

Use the baseline task and the default PPO config:

```powershell
.\isaaclab.bat -p scripts\reinforcement_learning\rl_games\train.py `
  --task Isaac-Lift-Cube-Franka-v0 `
  --num_envs 4096 `
  --seed 789 `
  +agent.params.config.full_experiment_name=baseline789 `
  --headless
```

The training script writes logs under `logs/rl_games/franka_lift/<run-name>/`.
If a run is copied or renamed for reporting, always check
`params/agent.yaml` and `params/env.yaml` for the exact seed and runtime config.

### R16 Obstacle-Grasping Training

Use the R16 task. In this repository, `Isaac-Franka-Grasping-v0` is registered
to use `rl_games_r16_ppo_cfg.yaml`.

Seed 42:

```powershell
.\isaaclab.bat -p scripts\reinforcement_learning\rl_games\train.py `
  --task Isaac-Franka-Grasping-v0 `
  --num_envs 4096 `
  --seed 42 `
  +agent.params.config.full_experiment_name=R16_final_s42 `
  --headless
```

Seed 123:

```powershell
.\isaaclab.bat -p scripts\reinforcement_learning\rl_games\train.py `
  --task Isaac-Franka-Grasping-v0 `
  --num_envs 4096 `
  --seed 123 `
  +agent.params.config.full_experiment_name=R16_final_s123 `
  --headless
```

Seed 789:

```powershell
.\isaaclab.bat -p scripts\reinforcement_learning\rl_games\train.py `
  --task Isaac-Franka-Grasping-v0 `
  --num_envs 4096 `
  --seed 789 `
  +agent.params.config.full_experiment_name=R16_final_s789 `
  --headless
```

The `+` before `agent.params.config.full_experiment_name` is required because
the key is not present in the source YAML. Hydra adds it at launch time.

## Monitoring

Start TensorBoard from the repository root:

```powershell
.\isaaclab.bat -p -m tensorboard.main --logdir logs\rl_games
```

Open the URL printed by TensorBoard, usually:

```text
http://localhost:6006/
```

## Playing Policies and Recording Videos

Example command for a saved R16 checkpoint:

```powershell
.\isaaclab.bat -p scripts\reinforcement_learning\rl_games\play.py `
  --task Isaac-Franka-Grasping-Play-v0 `
  --checkpoint logs\rl_games\obstacle_training\R16_final_s42\nn\franka_grasping.pth `
  --num_envs 1 `
  --seed 42 `
  --headless `
  --video `
  --video_length 400
```

Use a baseline checkpoint with `Isaac-Lift-Cube-Franka-Play-v0` when visualizing
the baseline environment.

## Evaluation

The generalization evaluation script accepts arbitrary checkpoint lists and
policy labels. It computes per-policy metrics and writes a CSV file. This makes
the evaluation reusable for the saved thesis checkpoints and for future runs.

Example:

```powershell
.\isaaclab.bat -p scripts\reinforcement_learning\rl_games\evaluate_generalization.py `
  --task Isaac-Franka-Grasping-v0 `
  --checkpoints `
    logs\rl_games\obstacle_training\R16_final_s42\nn\franka_grasping.pth `
    logs\rl_games\obstacle_training\R16_final_s123\nn\franka_grasping.pth `
    logs\rl_games\obstacle_training\R16_final_s789\nn\franka_grasping.pth `
  --policy_labels R16_s42 R16_s123 R16_s789 `
  --num_episodes 100 `
  --num_envs 16 `
  --output_csv outputs\r16_generalization.csv `
  --headless
```

Recorded metrics include:

- `grasp_success_rate`;
- `collision_frequency`;
- `near_obstacle_rate`;
- `mean_clearance`;
- `min_clearance`;
- `max_object_z`;
- `task_completion_time`.

These metrics are intended to validate the thesis research question by checking
whether trained policies can lift the object while avoiding or reducing contact
with obstacles.

## Saved Runs and Reproducibility

Saved runs are organized under `logs/rl_games/`.

Typical run contents:

- `params/env.yaml`: full environment configuration at launch time;
- `params/agent.yaml`: full RL-Games agent configuration at launch time;
- `nn/*.pth`: checkpoints;
- `summaries/events.out.tfevents.*`: TensorBoard logs;
- `videos/`: optional recorded videos.

The saved `params/*.yaml` files are the most reliable source of truth for a
completed run. If a folder name and a runtime seed disagree, trust the seed in
`params/agent.yaml` and `params/env.yaml`.

Exact bitwise determinism is not guaranteed across different GPUs, driver
versions, Isaac Sim versions, or operating systems. For thesis reproducibility,
the important requirements are to keep the same task ID, seed, source YAML,
environment code, Isaac Lab version, and evaluation script.

## Code Quality and Scope

The code is designed around Isaac Lab's manager-based environment pattern. The
custom R16 environment reuses standard Isaac Lab managers for observations,
events, rewards, commands, curricula, and terminations. The additional obstacle
logic is placed in `franka_grasping_env.py` so that the baseline lift
environment remains available and unchanged.

The evaluation script is context-independent with respect to the specific
checkpoint list: it accepts any set of RL-Games checkpoints and labels through
command-line arguments, then writes structured CSV output. This supports
reproducible analysis for the thesis runs and for future policies trained with
the same task family.

## Attribution

This project is built on NVIDIA Isaac Lab and Isaac Sim. The upstream Isaac Lab
code is retained under its BSD-3-Clause license; see `LICENSE`, `CITATION.cff`,
and the upstream documentation in `README.md` and `docs/`.

Key reused packages and frameworks include:

- NVIDIA Isaac Lab and Isaac Sim for simulation, assets, vectorized
  environments, and task infrastructure;
- RL-Games for PPO training and checkpointing;
- PyTorch for neural network training;
- Gymnasium for environment registration and interfaces;
- Matplotlib and NumPy for evaluation outputs.

The thesis-specific additions are the Franka obstacle-grasping environment,
R16 PPO configuration, saved experiment organization, and evaluation workflow.
When reporting results, cite Isaac Lab/Isaac Sim and any learning framework or
library used according to their licenses and citation guidance.
