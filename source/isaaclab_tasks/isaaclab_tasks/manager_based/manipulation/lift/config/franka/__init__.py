# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

##
# Joint Position Control
##

# Thesis baseline training task. This remains the standard no-obstacle lift
# environment, but uses the matched RL-Games PPO values saved in rl_games_ppo_cfg.yaml.
gym.register(
    id="Isaac-Lift-Cube-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaCubeLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Cube-Franka-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaCubeLiftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Absolute Pose Control
##

gym.register(
    id="Isaac-Lift-Cube-Franka-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:FrankaCubeLiftEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Teddy-Bear-Franka-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:FrankaTeddyBearLiftEnvCfg",
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Lift-Cube-Franka-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_env_cfg:FrankaCubeLiftEnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc.json",
    },
    disable_env_checker=True,
)

##
# Randomized-obstacle grasping (thesis training condition)
##

# Thesis R16/main training task. The environment adds randomized obstacle
# geometry; the PPO hyperparameters intentionally match the baseline YAML.
gym.register(
    id="Isaac-Franka-Grasping-v0",
    entry_point="isaaclab_tasks.manager_based.manipulation.lift.config.franka.franka_grasping_env:FrankaGraspingEnvWithMetrics",
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.manager_based.manipulation.lift.config.franka.franka_grasping_env:FrankaGraspingEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_r16_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Franka-Grasping-Play-v0",
    entry_point="isaaclab_tasks.manager_based.manipulation.lift.config.franka.franka_grasping_env:FrankaGraspingEnvWithMetrics",
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.manager_based.manipulation.lift.config.franka.franka_grasping_env:FrankaGraspingEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_r16_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

##
# Fixed thesis evaluation benchmark conditions
##

# The checkpoint passed to evaluate_generalization.py controls the policy family.
# These task IDs only choose the fixed test geometry: BL, A, B, or C.
gym.register(
    id="Isaac-Franka-Grasping-Baseline-v0",
    entry_point="isaaclab_tasks.manager_based.manipulation.lift.config.franka.franka_grasping_env:FrankaGraspingBaselineEnvWithMetrics",
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.manager_based.manipulation.lift.config.franka.franka_grasping_env:FrankaGraspingBaselineEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Franka-Grasping-Baseline-Play-v0",
    entry_point="isaaclab_tasks.manager_based.manipulation.lift.config.franka.franka_grasping_env:FrankaGraspingBaselineEnvWithMetrics",
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.manager_based.manipulation.lift.config.franka.franka_grasping_env:FrankaGraspingBaselineEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Franka-Grasping-A-v0",
    entry_point="isaaclab_tasks.manager_based.manipulation.lift.config.franka.franka_grasping_env:FrankaGraspingEnvWithMetrics",
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.manager_based.manipulation.lift.config.franka.franka_grasping_env:FrankaGraspingFixedAEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_r16_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Franka-Grasping-A-Play-v0",
    entry_point="isaaclab_tasks.manager_based.manipulation.lift.config.franka.franka_grasping_env:FrankaGraspingEnvWithMetrics",
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.manager_based.manipulation.lift.config.franka.franka_grasping_env:FrankaGraspingFixedAEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_r16_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Franka-Grasping-B-v0",
    entry_point="isaaclab_tasks.manager_based.manipulation.lift.config.franka.franka_grasping_env:FrankaGraspingEnvWithMetrics",
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.manager_based.manipulation.lift.config.franka.franka_grasping_env:FrankaGraspingFixedBEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_r16_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Franka-Grasping-B-Play-v0",
    entry_point="isaaclab_tasks.manager_based.manipulation.lift.config.franka.franka_grasping_env:FrankaGraspingEnvWithMetrics",
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.manager_based.manipulation.lift.config.franka.franka_grasping_env:FrankaGraspingFixedBEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_r16_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Franka-Grasping-C-v0",
    entry_point="isaaclab_tasks.manager_based.manipulation.lift.config.franka.franka_grasping_env:FrankaGraspingEnvWithMetrics",
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.manager_based.manipulation.lift.config.franka.franka_grasping_env:FrankaGraspingFixedCEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_r16_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Franka-Grasping-C-Play-v0",
    entry_point="isaaclab_tasks.manager_based.manipulation.lift.config.franka.franka_grasping_env:FrankaGraspingEnvWithMetrics",
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.manager_based.manipulation.lift.config.franka.franka_grasping_env:FrankaGraspingFixedCEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_r16_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)
