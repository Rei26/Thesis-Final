# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg, JointPositionActionCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, check_file_path
from pxr import Gf, Sdf, UsdGeom, Vt

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip
import isaaclab_tasks.manager_based.manipulation.lift.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import (
    CurriculumCfg,
    EventCfg,
    LiftEnvCfg,
    ObjectTableSceneCfg,
    ObservationsCfg,
    RewardsCfg,
)


OBSTACLE_NAMES = ("obstacle_1", "obstacle_2", "obstacle_3", "obstacle_4")
OBSTACLE_CONTACT_FORCE_THRESHOLD = 1.0
OBSTACLE_SURFACE_PENALTY_STD = 0.05

DEFAULT_TWO_OBSTACLE_SIZE = (0.06, 0.06, 0.18)
BENCHMARK_B_OBSTACLE_SIZE = (0.12, 0.12, 0.25)
BENCHMARK_C_OBSTACLE_SIZE = (0.04, 0.04, 0.10)

TRAIN_OBSTACLE_SIZE_RANGE = {
    "x": (0.035, 0.13),
    "y": (0.035, 0.13),
    "z": (0.09, 0.27),
}
ZERO_JITTER = (0.0, 0.0)

TRAIN_START_OBSTACLE_1_OFFSET = (-0.05, 0.10)
TRAIN_START_OBSTACLE_2_OFFSET = (-0.05, -0.10)
TRAIN_START_OBSTACLE_JITTER = (0.10, 0.05)
TRAIN_OBSTACLE_1_OFFSET = (-0.08, 0.10)
TRAIN_OBSTACLE_2_OFFSET = (-0.08, -0.10)
TRAIN_OBSTACLE_JITTER = (0.04, 0.03)
TRAIN_START_MIN_OBJECT_Y_SURFACE_CLEARANCE = 0.04
TRAIN_START_MIN_PAIR_Y_SURFACE_GAP = 0.08
TRAIN_MIN_OBJECT_Y_SURFACE_CLEARANCE = 0.02
TRAIN_MIN_PAIR_Y_SURFACE_GAP = 0.02
OBSTACLE_RESET_MAX_RESAMPLES = 32
TRAIN_PROXIMITY_PENALTY_WEIGHT = -0.02
TRAIN_PROXIMITY_PENALTY_FINAL_WEIGHT = -0.25
TRAIN_CONTACT_PENALTY_WEIGHT = -0.05
TRAIN_CONTACT_PENALTY_FINAL_WEIGHT = -0.50
TRAIN_OBSTACLE_PENALTY_CURRICULUM_STEPS = 6000
TRAIN_POSITION_CURRICULUM_STEPS = 6000

A_OBSTACLE_1_OFFSET = (-0.10, 0.10)
A_OBSTACLE_2_OFFSET = (-0.10, -0.10)

B_OBSTACLE_1_OFFSET = (-0.10, 0.13)
B_OBSTACLE_2_OFFSET = (-0.10, -0.13)

C_OBSTACLE_1_OFFSET = (-0.10, 0.07)
C_OBSTACLE_2_OFFSET = (-0.10, -0.07)
C_OBSTACLE_3_OFFSET = (-0.02, 0.11)
C_OBSTACLE_4_OFFSET = (-0.02, -0.11)

OBJECT_REST_HEIGHT = 0.055
GRASP_WARMUP_STEPS = 15
GRASP_SUCCESS_HEIGHT_THRESHOLD = OBJECT_REST_HEIGHT + 0.02
GRASP_SUCCESS_CONSECUTIVE_STEPS = 50
LIFT_PROGRESS_HEIGHT_THRESHOLD = OBJECT_REST_HEIGHT + 0.01
LIFT_REWARD_HEIGHT_THRESHOLD = OBJECT_REST_HEIGHT + 0.02

REMOTE_FRANKA_USD_PATH = FRANKA_PANDA_CFG.spawn.usd_path
REMOTE_OBJECT_USD_PATH = f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd"
REMOTE_TABLE_USD_PATH = f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
REMOTE_GROUND_USD_PATH = f"{ISAAC_NUCLEUS_DIR}/Environments/Grid/default_environment.usd"
LOCAL_FRANKA_USD_PATH = Path(__file__).resolve().parent / "assets" / "local_franka" / "panda_arm_hand.usd"
LOCAL_TABLE_SIZE = (0.90, 1.20, 0.90)
LOCAL_TABLE_TOP_Z = 0.03
LOCAL_OBJECT_SIZE = (0.05, 0.05, 0.05)
LOCAL_GROUND_SIZE = (8.0, 8.0, 0.10)
LOCAL_GROUND_TOP_Z = -1.05
PLAY_VIEWER_EYE = (1.55, 0.85, 0.95)
PLAY_VIEWER_LOOKAT = (0.50, 0.00, 0.18)
PLAY_VIEWER_RESOLUTION = (1920, 1080)
FORCE_LOCAL_ASSETS_ENV = "ISAACLAB_FRANKA_FORCE_LOCAL_ASSETS"


def _prefer_local_assets() -> bool:
    """Prefer local scene assets on Windows to avoid slow/failing remote fetches during playback."""

    default_value = "1" if sys.platform.startswith("win") else "0"
    return os.environ.get(FORCE_LOCAL_ASSETS_ENV, default_value) == "1"


def _has_accessible_usd(path: str) -> bool:
    """Return whether a USD path is reachable either locally or through omniverse storage."""

    if _prefer_local_assets() and path.startswith(("http://", "https://", "omniverse://")):
        return False
    try:
        return check_file_path(path) != 0
    except Exception:
        return False


def _resolve_franka_robot_cfg():
    """Use the standard Franka asset when available, otherwise fall back to a locally bootstrapped USD."""

    robot_cfg = FRANKA_PANDA_CFG.copy()
    if _has_accessible_usd(robot_cfg.spawn.usd_path):
        return robot_cfg
    if LOCAL_FRANKA_USD_PATH.is_file():
        robot_cfg.spawn.usd_path = str(LOCAL_FRANKA_USD_PATH)
    return robot_cfg


def _make_object_cfg() -> RigidObjectCfg:
    """Use the shared DexCube when available, otherwise spawn a local cuboid fallback for Windows playback."""

    common_rigid_props = sim_utils.RigidBodyPropertiesCfg(
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=1,
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=5.0,
        disable_gravity=False,
    )

    if _has_accessible_usd(REMOTE_OBJECT_USD_PATH):
        spawn_cfg = UsdFileCfg(
            usd_path=REMOTE_OBJECT_USD_PATH,
            scale=(0.8, 0.8, 0.8),
            rigid_props=common_rigid_props,
        )
    else:
        spawn_cfg = sim_utils.CuboidCfg(
            size=LOCAL_OBJECT_SIZE,
            rigid_props=common_rigid_props,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.16, 0.56, 0.84)),
        )

    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.055), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=spawn_cfg,
    )


def _make_table_cfg() -> AssetBaseCfg:
    """Use the SeattleLab table when reachable, otherwise spawn a simple local tabletop cuboid."""

    if _has_accessible_usd(REMOTE_TABLE_USD_PATH):
        return AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
            spawn=UsdFileCfg(usd_path=REMOTE_TABLE_USD_PATH),
        )

    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.5, 0.0, LOCAL_TABLE_TOP_Z - 0.5 * LOCAL_TABLE_SIZE[2]],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=LOCAL_TABLE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.69, 0.62)),
        ),
    )


def _make_ground_cfg() -> AssetBaseCfg:
    """Use the default Isaac ground environment when reachable, otherwise spawn a local floor slab."""

    if _has_accessible_usd(REMOTE_GROUND_USD_PATH):
        return AssetBaseCfg(
            prim_path="/World/GroundPlane",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -1.05]),
            spawn=sim_utils.GroundPlaneCfg(),
        )

    return AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, LOCAL_GROUND_TOP_Z - 0.5 * LOCAL_GROUND_SIZE[2]],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=LOCAL_GROUND_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.12, 0.12)),
        ),
    )


def _configure_play_viewer(env_cfg) -> None:
    """Set a tighter camera framing for recorded videos."""

    env_cfg.viewer.eye = PLAY_VIEWER_EYE
    env_cfg.viewer.lookat = PLAY_VIEWER_LOOKAT
    env_cfg.viewer.resolution = PLAY_VIEWER_RESOLUTION
    env_cfg.viewer.origin_type = "world"
    env_cfg.viewer.env_index = 0


def _resolve_env_ids(env: ManagerBasedRLEnv, env_ids: torch.Tensor | slice | None) -> torch.Tensor:
    """Convert reset indices into a dense device tensor."""

    if env_ids is None or env_ids == slice(None):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=env.device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


def _spawn_size_tensor(asset: RigidObject, device: torch.device | str) -> torch.Tensor:
    """Return the asset's configured cuboid size as a tensor."""

    spawn_size = getattr(asset.cfg.spawn, "size", None)
    if spawn_size is None:
        raise ValueError(f"Obstacle '{asset.cfg.prim_path}' does not expose a cuboid spawn size.")
    return torch.tensor(spawn_size, dtype=torch.float32, device=device)


def _current_obstacle_sizes(env: ManagerBasedRLEnv, obstacle_cfg: SceneEntityCfg) -> torch.Tensor:
    """Resolve the current full obstacle size for every environment."""

    if hasattr(env, "_obstacle_sizes") and obstacle_cfg.name in env._obstacle_sizes:
        return env._obstacle_sizes[obstacle_cfg.name]
    if hasattr(env, "_prestartup_obstacle_sizes") and obstacle_cfg.name in env._prestartup_obstacle_sizes:
        return env._prestartup_obstacle_sizes[obstacle_cfg.name]
    asset: RigidObject = env.scene[obstacle_cfg.name]
    spawn_size = _spawn_size_tensor(asset, env.device)
    return spawn_size.unsqueeze(0).repeat(env.num_envs, 1)


def _current_obstacle_half_extents(env: ManagerBasedRLEnv, obstacle_cfg: SceneEntityCfg) -> torch.Tensor:
    """Resolve current obstacle half-extents for every environment."""

    if hasattr(env, "_obstacle_half_extents") and obstacle_cfg.name in env._obstacle_half_extents:
        return env._obstacle_half_extents[obstacle_cfg.name]
    return 0.5 * _current_obstacle_sizes(env, obstacle_cfg)


def _box_surface_clearance(
    point_w: torch.Tensor,
    cuboid_center_w: torch.Tensor,
    half_extents: torch.Tensor | tuple[float, float, float],
) -> torch.Tensor:
    """Euclidean clearance from a point to the surface of an axis-aligned cuboid."""

    if not isinstance(half_extents, torch.Tensor):
        half_extents = point_w.new_tensor(half_extents)
    if half_extents.dim() == 1:
        half_extents = half_extents.unsqueeze(0).expand(point_w.shape[0], -1)
    diff = torch.abs(point_w - cuboid_center_w) - half_extents
    return torch.linalg.vector_norm(diff.clamp(min=0.0), dim=-1)


def obstacle_size_observation(
    env: ManagerBasedRLEnv,
    obstacle_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Expose obstacle size so randomized geometry is not hidden state."""

    return _current_obstacle_sizes(env, obstacle_cfg)


def zero_obstacle_position_observation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return a zero obstacle-position placeholder to keep baseline eval obs-compatible with the obstacle policy."""

    return torch.zeros((env.num_envs, 3), dtype=torch.float32, device=env.device)


def zero_obstacle_size_observation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return a zero obstacle-size placeholder to keep baseline eval obs-compatible with the obstacle policy."""

    return torch.zeros((env.num_envs, 3), dtype=torch.float32, device=env.device)


def no_obstacle_reward(env: ManagerBasedRLEnv, **_) -> torch.Tensor:
    """Return a zero reward term while preserving obstacle-term names for baseline parity."""

    return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)


def noop_obstacle_reset(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None = None, **_) -> None:
    """Expose baseline obstacle reset slots without moving any physical assets."""

    return None


def _rank_obstacles_by_object_clearance(
    env: ManagerBasedRLEnv,
    obstacle_names: tuple[str, ...],
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Sort obstacles per-environment by clearance to the object center.

    This lets the four-obstacle benchmark expose the two most relevant obstacles while
    preserving the same 48D observation schema used during randomized training.
    """

    object_pos = env.scene[object_cfg.name].data.root_pos_w[:, :3]
    obstacle_centers = []
    obstacle_half_extents = []
    for obstacle_name in obstacle_names:
        obstacle_centers.append(env.scene[obstacle_name].data.root_pos_w[:, :3])
        obstacle_half_extents.append(_current_obstacle_half_extents(env, SceneEntityCfg(obstacle_name)))
    centers = torch.stack(obstacle_centers, dim=1)
    half_extents = torch.stack(obstacle_half_extents, dim=1)
    diff = torch.abs(object_pos.unsqueeze(1) - centers) - half_extents
    clearances = torch.linalg.vector_norm(diff.clamp(min=0.0), dim=-1)
    return clearances.argsort(dim=1)


def ranked_obstacle_position_observation(
    env: ManagerBasedRLEnv,
    rank: int,
    obstacle_names: tuple[str, ...],
) -> torch.Tensor:
    """Expose the position of the nth closest obstacle to the object in robot-root coordinates."""

    positions = torch.stack(
        [
            mdp.object_position_in_robot_root_frame(env, object_cfg=SceneEntityCfg(obstacle_name))
            for obstacle_name in obstacle_names
        ],
        dim=1,
    )
    order = _rank_obstacles_by_object_clearance(env, obstacle_names)
    selected = order[:, rank].unsqueeze(-1).unsqueeze(-1).expand(-1, 1, positions.shape[-1])
    return positions.gather(1, selected).squeeze(1)


def ranked_obstacle_size_observation(
    env: ManagerBasedRLEnv,
    rank: int,
    obstacle_names: tuple[str, ...],
) -> torch.Tensor:
    """Expose the full size of the nth closest obstacle to the object."""

    sizes = torch.stack(
        [_current_obstacle_sizes(env, SceneEntityCfg(obstacle_name)) for obstacle_name in obstacle_names],
        dim=1,
    )
    order = _rank_obstacles_by_object_clearance(env, obstacle_names)
    selected = order[:, rank].unsqueeze(-1).unsqueeze(-1).expand(-1, 1, sizes.shape[-1])
    return sizes.gather(1, selected).squeeze(1)


def obstacle_surface_distance(
    env: ManagerBasedRLEnv,
    obstacle_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Per-environment end-effector clearance to an obstacle surface."""

    ee_pos = env.scene[ee_frame_cfg.name].data.target_pos_w[:, 0, :]
    obstacle_pos = env.scene[obstacle_cfg.name].data.root_pos_w[:, :3]
    return _box_surface_clearance(ee_pos, obstacle_pos, _current_obstacle_half_extents(env, obstacle_cfg))


def obstacle_surface_proximity(
    env: ManagerBasedRLEnv,
    obstacle_cfg: SceneEntityCfg,
    std: float = OBSTACLE_SURFACE_PENALTY_STD,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Smooth proximity penalty based on distance to obstacle surface."""

    clearance = obstacle_surface_distance(env, obstacle_cfg=obstacle_cfg, ee_frame_cfg=ee_frame_cfg)
    return 1.0 - torch.tanh(clearance / std)


def obstacle_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_names: tuple[str, ...] = ("left_finger_obstacle_contact", "right_finger_obstacle_contact"),
    force_threshold: float = OBSTACLE_CONTACT_FORCE_THRESHOLD,
) -> torch.Tensor:
    """Binary penalty when either finger contacts any filtered obstacle."""

    contact_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for sensor_name in sensor_names:
        if sensor_name not in env.scene.keys():
            continue
        force_matrix = env.scene[sensor_name].data.force_matrix_w
        if force_matrix is None:
            continue
        force_norm = torch.linalg.vector_norm(force_matrix, dim=-1)
        contact_mask |= (force_norm > force_threshold).flatten(start_dim=1).any(dim=1)
    return contact_mask.float()


def _sample_obstacle_sizes(
    device: torch.device | str,
    count: int,
    size_range: dict[str, tuple[float, float]] | None,
    fixed_size: tuple[float, float, float] | None,
) -> torch.Tensor:
    """Sample full obstacle sizes or broadcast a fixed size."""

    if fixed_size is not None:
        return torch.tensor(fixed_size, dtype=torch.float32, device=device).unsqueeze(0).repeat(count, 1)
    if size_range is None:
        raise ValueError("Either 'fixed_size' or 'size_range' must be provided for obstacle resets.")
    range_tensor = torch.tensor(
        [size_range[axis] for axis in ("x", "y", "z")],
        dtype=torch.float32,
        device=device,
    )
    return math_utils.sample_uniform(range_tensor[:, 0], range_tensor[:, 1], (count, 3), device=device)


def _curriculum_alpha(env: ManagerBasedRLEnv, num_steps: int | None) -> float:
    """Return a 0-1 curriculum progress value based on the shared environment step counter."""

    if num_steps is None or num_steps <= 0:
        return 1.0
    return float(min(max(env.common_step_counter / float(num_steps), 0.0), 1.0))


def _lerp_scalar(start: float | None, end: float | None, alpha: float) -> float | None:
    """Linearly interpolate optional scalar curriculum values."""

    if start is None:
        return end
    if end is None:
        return start
    return (1.0 - alpha) * start + alpha * end


def _lerp_pair(
    start: tuple[float, float] | None,
    end: tuple[float, float],
    alpha: float,
) -> tuple[float, float]:
    """Linearly interpolate 2D tuples for obstacle reset curricula."""

    if start is None:
        return end
    return tuple((1.0 - alpha) * s + alpha * e for s, e in zip(start, end, strict=True))


def _sample_constrained_obstacle_y_positions(
    *,
    obj_center_y: torch.Tensor,
    position_offset_y: float,
    position_jitter_y: float,
    obstacle_half_y: torch.Tensor,
    object_half_y: float,
    min_object_y_surface_clearance: float | None = None,
    paired_center_y: torch.Tensor | None = None,
    paired_half_y: torch.Tensor | None = None,
    min_pair_y_surface_gap: float | None = None,
    max_resamples: int = OBSTACLE_RESET_MAX_RESAMPLES,
) -> torch.Tensor:
    """Sample obstacle y positions while preserving a minimum corridor around the cube."""

    side_sign = 1.0 if position_offset_y >= 0.0 else -1.0
    remaining_mask = torch.ones_like(obj_center_y, dtype=torch.bool)
    chosen_y = torch.empty_like(obj_center_y)

    for _ in range(max_resamples):
        if not remaining_mask.any():
            break

        sample_count = int(remaining_mask.sum().item())
        sampled_jitter_y = math_utils.sample_uniform(
            obj_center_y.new_full((sample_count,), -position_jitter_y),
            obj_center_y.new_full((sample_count,), position_jitter_y),
            (sample_count,),
            device=obj_center_y.device,
        )
        candidate_y = obj_center_y[remaining_mask] + position_offset_y + sampled_jitter_y
        candidate_half_y = obstacle_half_y[remaining_mask]
        valid_mask = torch.ones(sample_count, dtype=torch.bool, device=obj_center_y.device)

        if min_object_y_surface_clearance is not None:
            if side_sign > 0.0:
                valid_mask &= (
                    candidate_y - candidate_half_y
                    >= obj_center_y[remaining_mask] + object_half_y + min_object_y_surface_clearance
                )
            else:
                valid_mask &= (
                    candidate_y + candidate_half_y
                    <= obj_center_y[remaining_mask] - object_half_y - min_object_y_surface_clearance
                )

        if paired_center_y is not None and paired_half_y is not None and min_pair_y_surface_gap is not None:
            paired_candidate_y = paired_center_y[remaining_mask]
            paired_candidate_half_y = paired_half_y[remaining_mask]
            if side_sign > 0.0:
                valid_mask &= (
                    candidate_y - candidate_half_y
                    >= paired_candidate_y + paired_candidate_half_y + min_pair_y_surface_gap
                )
            else:
                valid_mask &= (
                    candidate_y + candidate_half_y
                    <= paired_candidate_y - paired_candidate_half_y - min_pair_y_surface_gap
                )

        remaining_indices = remaining_mask.nonzero(as_tuple=False).flatten()
        chosen_y[remaining_indices[valid_mask]] = candidate_y[valid_mask]
        remaining_mask[remaining_indices[valid_mask]] = False

    if remaining_mask.any():
        remaining_obj_y = obj_center_y[remaining_mask]
        remaining_half_y = obstacle_half_y[remaining_mask]
        lower_bound = remaining_obj_y + position_offset_y - position_jitter_y
        upper_bound = remaining_obj_y + position_offset_y + position_jitter_y

        if side_sign > 0.0:
            min_allowed_y = lower_bound.clone()
            if min_object_y_surface_clearance is not None:
                min_allowed_y = torch.maximum(
                    min_allowed_y,
                    remaining_obj_y + object_half_y + remaining_half_y + min_object_y_surface_clearance,
                )
            if paired_center_y is not None and paired_half_y is not None and min_pair_y_surface_gap is not None:
                min_allowed_y = torch.maximum(
                    min_allowed_y,
                    paired_center_y[remaining_mask] + paired_half_y[remaining_mask] + remaining_half_y + min_pair_y_surface_gap,
                )
            chosen_y[remaining_mask] = torch.minimum(min_allowed_y, upper_bound)
        else:
            max_allowed_y = upper_bound.clone()
            if min_object_y_surface_clearance is not None:
                max_allowed_y = torch.minimum(
                    max_allowed_y,
                    remaining_obj_y - object_half_y - remaining_half_y - min_object_y_surface_clearance,
                )
            if paired_center_y is not None and paired_half_y is not None and min_pair_y_surface_gap is not None:
                max_allowed_y = torch.minimum(
                    max_allowed_y,
                    paired_center_y[remaining_mask] - paired_half_y[remaining_mask] - remaining_half_y - min_pair_y_surface_gap,
                )
            chosen_y[remaining_mask] = torch.maximum(max_allowed_y, lower_bound)

    return chosen_y


def randomize_obstacle_size_prestartup(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("obstacle_1"),
    size_range: dict[str, tuple[float, float]] | None = None,
    fixed_size: tuple[float, float, float] | None = None,
):
    """Randomize obstacle size once before sim startup and store the sampled geometry state."""

    asset: RigidObject = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids_cpu = torch.arange(env.scene.num_envs, device="cpu", dtype=torch.long)
    else:
        env_ids_cpu = torch.as_tensor(env_ids, device="cpu", dtype=torch.long)

    full_sizes = _sample_obstacle_sizes(
        device=env.device,
        count=env_ids_cpu.numel(),
        size_range=size_range,
        fixed_size=fixed_size,
    )
    base_size = _spawn_size_tensor(asset, env.device)
    scales = (full_sizes / base_size.unsqueeze(0)).cpu().tolist()

    stage = get_current_stage()
    prim_paths = sim_utils.find_matching_prim_paths(asset.cfg.prim_path)
    with Sdf.ChangeBlock():
        for sample_id, env_id in enumerate(env_ids_cpu.tolist()):
            prim_path = prim_paths[env_id]
            prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)
            scale_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOp:scale")
            has_scale_attr = scale_spec is not None
            if not has_scale_attr:
                scale_spec = Sdf.AttributeSpec(prim_spec, prim_path + ".xformOp:scale", Sdf.ValueTypeNames.Double3)
            scale_spec.default = Gf.Vec3f(*scales[sample_id])
            if not has_scale_attr:
                op_order_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOpOrder")
                if op_order_spec is None:
                    op_order_spec = Sdf.AttributeSpec(
                        prim_spec, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray
                    )
                op_order_spec.default = Vt.TokenArray(["xformOp:translate", "xformOp:orient", "xformOp:scale"])

    if not hasattr(env, "_prestartup_obstacle_sizes"):
        env._prestartup_obstacle_sizes = {}
    if asset_cfg.name not in env._prestartup_obstacle_sizes:
        env._prestartup_obstacle_sizes[asset_cfg.name] = base_size.unsqueeze(0).repeat(env.scene.num_envs, 1)
    env._prestartup_obstacle_sizes[asset_cfg.name][env_ids_cpu.to(device=env.device)] = full_sizes


def reset_obstacle_geometry_relative_to_object(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | slice | None,
    position_offset: tuple[float, float],
    position_jitter: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("obstacle_1"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    size_range: dict[str, tuple[float, float]] | None = None,
    fixed_size: tuple[float, float, float] | None = None,
    paired_obstacle_cfg: SceneEntityCfg | None = None,
    min_object_y_surface_clearance: float | None = None,
    min_pair_y_surface_gap: float | None = None,
    curriculum_steps: int | None = None,
    curriculum_start_position_offset: tuple[float, float] | None = None,
    curriculum_start_position_jitter: tuple[float, float] | None = None,
    curriculum_start_min_object_y_surface_clearance: float | None = None,
    curriculum_start_min_pair_y_surface_gap: float | None = None,
    max_resamples: int = OBSTACLE_RESET_MAX_RESAMPLES,
):
    """Reset one obstacle pose relative to the cube using the currently assigned obstacle size."""

    env_ids_tensor = _resolve_env_ids(env, env_ids)
    asset: RigidObject = env.scene[asset_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    curriculum_alpha = _curriculum_alpha(env, curriculum_steps)
    effective_position_offset = _lerp_pair(curriculum_start_position_offset, position_offset, curriculum_alpha)
    effective_position_jitter = _lerp_pair(curriculum_start_position_jitter, position_jitter, curriculum_alpha)
    effective_min_object_y_surface_clearance = _lerp_scalar(
        curriculum_start_min_object_y_surface_clearance,
        min_object_y_surface_clearance,
        curriculum_alpha,
    )
    effective_min_pair_y_surface_gap = _lerp_scalar(
        curriculum_start_min_pair_y_surface_gap,
        min_pair_y_surface_gap,
        curriculum_alpha,
    )

    root_states = asset.data.default_root_state[env_ids_tensor].clone()
    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids_tensor]
    orientations = root_states[:, 3:7]
    velocities = root_states[:, 7:13]

    obstacle_sizes = _current_obstacle_sizes(env, asset_cfg)[env_ids_tensor]
    object_positions_xy = obj.data.root_pos_w[env_ids_tensor, :2]

    sampled_jitter_x = math_utils.sample_uniform(
        positions.new_full((root_states.shape[0],), -effective_position_jitter[0]),
        positions.new_full((root_states.shape[0],), effective_position_jitter[0]),
        (root_states.shape[0],),
        device=asset.device,
    )
    positions[:, 0] = object_positions_xy[:, 0] + effective_position_offset[0] + sampled_jitter_x

    paired_center_y = None
    paired_half_y = None
    if paired_obstacle_cfg is not None:
        if hasattr(env, "_current_reset_obstacle_positions") and paired_obstacle_cfg.name in env._current_reset_obstacle_positions:
            paired_center_y = env._current_reset_obstacle_positions[paired_obstacle_cfg.name][env_ids_tensor, 1]
        else:
            paired_center_y = env.scene[paired_obstacle_cfg.name].data.root_pos_w[env_ids_tensor, 1]
        paired_half_y = 0.5 * _current_obstacle_sizes(env, paired_obstacle_cfg)[env_ids_tensor, 1]

    positions[:, 1] = _sample_constrained_obstacle_y_positions(
        obj_center_y=object_positions_xy[:, 1],
        position_offset_y=effective_position_offset[1],
        position_jitter_y=effective_position_jitter[1],
        obstacle_half_y=0.5 * obstacle_sizes[:, 1],
        object_half_y=0.5 * LOCAL_OBJECT_SIZE[1],
        min_object_y_surface_clearance=effective_min_object_y_surface_clearance,
        paired_center_y=paired_center_y,
        paired_half_y=paired_half_y,
        min_pair_y_surface_gap=effective_min_pair_y_surface_gap,
        max_resamples=max_resamples,
    )

    base_size = _spawn_size_tensor(asset, positions.device)
    positions[:, 2] = positions[:, 2] - 0.5 * base_size[2] + 0.5 * obstacle_sizes[:, 2]

    if not hasattr(env, "_current_reset_obstacle_positions"):
        env._current_reset_obstacle_positions = {}
    if asset_cfg.name not in env._current_reset_obstacle_positions:
        env._current_reset_obstacle_positions[asset_cfg.name] = asset.data.default_root_state[:, 0:3].clone()
    env._current_reset_obstacle_positions[asset_cfg.name][env_ids_tensor] = positions

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids_tensor)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids_tensor)


def _fixed_object_pose_range() -> dict[str, tuple[float, float]]:
    """Zero object pose offsets so the cube stays at its default spawn every reset."""

    return {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}


def _configure_obstacle_contact_sensors(env_cfg) -> None:
    """Attach filtered finger contact sensors for any obstacle scene."""

    obstacle_paths = []
    for obstacle_name in OBSTACLE_NAMES:
        if hasattr(env_cfg.scene, obstacle_name):
            obstacle_paths.append(getattr(env_cfg.scene, obstacle_name).prim_path)

    if not obstacle_paths:
        return

    env_cfg.scene.robot.spawn.activate_contact_sensors = True
    env_cfg.scene.left_finger_obstacle_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=obstacle_paths,
    )
    env_cfg.scene.right_finger_obstacle_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=obstacle_paths,
    )


def _configure_franka_grasping_components(env_cfg) -> None:
    """Populate shared Franka/object/action config for all thesis grasping tasks."""

    env_cfg.scene.robot = _resolve_franka_robot_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")
    env_cfg.scene.table = _make_table_cfg()
    env_cfg.scene.plane = _make_ground_cfg()

    env_cfg.scene.ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.1034)),
            )
        ],
    )

    env_cfg.scene.object = _make_object_cfg()

    env_cfg.actions.arm_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=0.5,
        use_default_offset=True,
    )
    env_cfg.actions.gripper_action = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger_joint.*"],
        open_command_expr={"panda_finger_joint.*": 0.04},
        close_command_expr={"panda_finger_joint.*": 0.0},
    )

    env_cfg.commands.object_pose.body_name = "panda_hand"
    env_cfg.commands.object_pose.debug_vis = False
    _configure_obstacle_contact_sensors(env_cfg)


def _make_obstacle_cfg(
    prim_path: str,
    init_pos: tuple[float, float, float],
    size: tuple[float, float, float],
    color: tuple[float, float, float],
) -> RigidObjectCfg:
    """Create a cuboid obstacle config."""

    return RigidObjectCfg(
        prim_path=prim_path,
        init_state=RigidObjectCfg.InitialStateCfg(pos=init_pos),
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
    )


@configclass
class GraspingSceneCfg(ObjectTableSceneCfg):
    """Two-obstacle scene used by the randomized training task and fixed A benchmark."""

    obstacle_1: RigidObjectCfg = _make_obstacle_cfg(
        prim_path="{ENV_REGEX_NS}/Obstacle1",
        init_pos=(0.45, 0.08, 0.075),
        size=DEFAULT_TWO_OBSTACLE_SIZE,
        color=(0.8, 0.2, 0.2),
    )
    obstacle_2: RigidObjectCfg = _make_obstacle_cfg(
        prim_path="{ENV_REGEX_NS}/Obstacle2",
        init_pos=(0.45, -0.08, 0.075),
        size=DEFAULT_TWO_OBSTACLE_SIZE,
        color=(0.2, 0.2, 0.8),
    )


@configclass
class GraspingBenchmarkBSceneCfg(GraspingSceneCfg):
    """Fixed Condition B geometry with two larger obstacles."""

    obstacle_1: RigidObjectCfg = _make_obstacle_cfg(
        prim_path="{ENV_REGEX_NS}/Obstacle1",
        init_pos=(0.45, 0.08, 0.125),
        size=BENCHMARK_B_OBSTACLE_SIZE,
        color=(0.8, 0.2, 0.2),
    )
    obstacle_2: RigidObjectCfg = _make_obstacle_cfg(
        prim_path="{ENV_REGEX_NS}/Obstacle2",
        init_pos=(0.45, -0.08, 0.125),
        size=BENCHMARK_B_OBSTACLE_SIZE,
        color=(0.2, 0.2, 0.8),
    )


@configclass
class GraspingBenchmarkCSceneCfg(ObjectTableSceneCfg):
    """Fixed Condition C geometry with four smaller clutter obstacles."""

    obstacle_1: RigidObjectCfg = _make_obstacle_cfg(
        prim_path="{ENV_REGEX_NS}/Obstacle1",
        init_pos=(0.40, 0.10, 0.05),
        size=BENCHMARK_C_OBSTACLE_SIZE,
        color=(0.8, 0.2, 0.2),
    )
    obstacle_2: RigidObjectCfg = _make_obstacle_cfg(
        prim_path="{ENV_REGEX_NS}/Obstacle2",
        init_pos=(0.50, 0.10, 0.05),
        size=BENCHMARK_C_OBSTACLE_SIZE,
        color=(0.2, 0.8, 0.2),
    )
    obstacle_3: RigidObjectCfg = _make_obstacle_cfg(
        prim_path="{ENV_REGEX_NS}/Obstacle3",
        init_pos=(0.40, -0.10, 0.05),
        size=BENCHMARK_C_OBSTACLE_SIZE,
        color=(0.2, 0.2, 0.8),
    )
    obstacle_4: RigidObjectCfg = _make_obstacle_cfg(
        prim_path="{ENV_REGEX_NS}/Obstacle4",
        init_pos=(0.50, -0.10, 0.05),
        size=BENCHMARK_C_OBSTACLE_SIZE,
        color=(0.8, 0.8, 0.2),
    )


@configclass
class GraspingObservationsCfg(ObservationsCfg):
    """Two-obstacle policy observation with positions and explicit sizes."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        obstacle_1_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("obstacle_1")},
        )
        obstacle_1_size = ObsTerm(
            func=obstacle_size_observation,
            params={"obstacle_cfg": SceneEntityCfg("obstacle_1")},
        )
        obstacle_2_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("obstacle_2")},
        )
        obstacle_2_size = ObsTerm(
            func=obstacle_size_observation,
            params={"obstacle_cfg": SceneEntityCfg("obstacle_2")},
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class BaselineMatchedObservationsCfg(ObservationsCfg):
    """Baseline observation schema padded with explicit zero obstacle features to match the 48D obstacle policy."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        obstacle_1_position = ObsTerm(func=zero_obstacle_position_observation)
        obstacle_1_size = ObsTerm(func=zero_obstacle_size_observation)
        obstacle_2_position = ObsTerm(func=zero_obstacle_position_observation)
        obstacle_2_size = ObsTerm(func=zero_obstacle_size_observation)

    policy: PolicyCfg = PolicyCfg()


@configclass
class GraspingBenchmarkCMatchedObservationsCfg(ObservationsCfg):
    """Four-obstacle benchmark observation compressed into the same two-obstacle schema used during training."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        obstacle_1_position = ObsTerm(
            func=ranked_obstacle_position_observation,
            params={"rank": 0, "obstacle_names": ("obstacle_1", "obstacle_2", "obstacle_3", "obstacle_4")},
        )
        obstacle_1_size = ObsTerm(
            func=ranked_obstacle_size_observation,
            params={"rank": 0, "obstacle_names": ("obstacle_1", "obstacle_2", "obstacle_3", "obstacle_4")},
        )
        obstacle_2_position = ObsTerm(
            func=ranked_obstacle_position_observation,
            params={"rank": 1, "obstacle_names": ("obstacle_1", "obstacle_2", "obstacle_3", "obstacle_4")},
        )
        obstacle_2_size = ObsTerm(
            func=ranked_obstacle_size_observation,
            params={"rank": 1, "obstacle_names": ("obstacle_1", "obstacle_2", "obstacle_3", "obstacle_4")},
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class GraspingRewardsCfg(RewardsCfg):
    """Geometry-aware obstacle penalties for two-obstacle environments."""

    obstacle_1_penalty = RewTerm(
        func=obstacle_surface_proximity,
        params={"obstacle_cfg": SceneEntityCfg("obstacle_1"), "std": OBSTACLE_SURFACE_PENALTY_STD},
        weight=TRAIN_PROXIMITY_PENALTY_WEIGHT,
    )
    obstacle_2_penalty = RewTerm(
        func=obstacle_surface_proximity,
        params={"obstacle_cfg": SceneEntityCfg("obstacle_2"), "std": OBSTACLE_SURFACE_PENALTY_STD},
        weight=TRAIN_PROXIMITY_PENALTY_WEIGHT,
    )
    obstacle_contact_penalty = RewTerm(func=obstacle_contact_penalty, weight=TRAIN_CONTACT_PENALTY_WEIGHT)


@configclass
class GraspingBenchmarkCRewardsCfg(RewardsCfg):
    """Geometry-aware obstacle penalties for the cluttered benchmark."""

    obstacle_1_penalty = RewTerm(
        func=obstacle_surface_proximity,
        params={"obstacle_cfg": SceneEntityCfg("obstacle_1"), "std": OBSTACLE_SURFACE_PENALTY_STD},
        weight=-1.0,
    )
    obstacle_2_penalty = RewTerm(
        func=obstacle_surface_proximity,
        params={"obstacle_cfg": SceneEntityCfg("obstacle_2"), "std": OBSTACLE_SURFACE_PENALTY_STD},
        weight=-1.0,
    )
    obstacle_3_penalty = RewTerm(
        func=obstacle_surface_proximity,
        params={"obstacle_cfg": SceneEntityCfg("obstacle_3"), "std": OBSTACLE_SURFACE_PENALTY_STD},
        weight=-1.0,
    )
    obstacle_4_penalty = RewTerm(
        func=obstacle_surface_proximity,
        params={"obstacle_cfg": SceneEntityCfg("obstacle_4"), "std": OBSTACLE_SURFACE_PENALTY_STD},
        weight=-1.0,
    )
    obstacle_contact_penalty = RewTerm(func=obstacle_contact_penalty, weight=-1.0)


@configclass
class GraspingBaselineRewardsCfg(RewardsCfg):
    """Baseline reward layout that mirrors the grasping family without physical obstacles."""

    obstacle_1_penalty = RewTerm(func=no_obstacle_reward, weight=TRAIN_PROXIMITY_PENALTY_WEIGHT)
    obstacle_2_penalty = RewTerm(func=no_obstacle_reward, weight=TRAIN_PROXIMITY_PENALTY_WEIGHT)
    obstacle_contact_penalty = RewTerm(func=no_obstacle_reward, weight=TRAIN_CONTACT_PENALTY_WEIGHT)


@configclass
class GraspingCurriculumCfg(CurriculumCfg):
    """Same curriculum timing as the base lift task."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000},
    )
    obstacle_1_penalty = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "obstacle_1_penalty",
            "weight": TRAIN_PROXIMITY_PENALTY_FINAL_WEIGHT,
            "num_steps": TRAIN_OBSTACLE_PENALTY_CURRICULUM_STEPS,
        },
    )
    obstacle_2_penalty = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "obstacle_2_penalty",
            "weight": TRAIN_PROXIMITY_PENALTY_FINAL_WEIGHT,
            "num_steps": TRAIN_OBSTACLE_PENALTY_CURRICULUM_STEPS,
        },
    )
    obstacle_contact_penalty = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "obstacle_contact_penalty",
            "weight": TRAIN_CONTACT_PENALTY_FINAL_WEIGHT,
            "num_steps": TRAIN_OBSTACLE_PENALTY_CURRICULUM_STEPS,
        },
    )


@configclass
class RandomizedObstacleEventCfg(EventCfg):
    """Current training task: fixed object, per-env obstacle size randomization plus per-reset position randomization."""

    randomize_obstacle_1_size = EventTerm(
        func=randomize_obstacle_size_prestartup,
        mode="prestartup",
        params={
            "size_range": TRAIN_OBSTACLE_SIZE_RANGE,
            "asset_cfg": SceneEntityCfg("obstacle_1"),
        },
    )
    randomize_obstacle_2_size = EventTerm(
        func=randomize_obstacle_size_prestartup,
        mode="prestartup",
        params={
            "size_range": TRAIN_OBSTACLE_SIZE_RANGE,
            "asset_cfg": SceneEntityCfg("obstacle_2"),
        },
    )

    reset_obstacle_1 = EventTerm(
        func=reset_obstacle_geometry_relative_to_object,
        mode="reset",
        params={
            "position_offset": TRAIN_OBSTACLE_1_OFFSET,
            "position_jitter": TRAIN_OBSTACLE_JITTER,
            "curriculum_steps": TRAIN_POSITION_CURRICULUM_STEPS,
            "curriculum_start_position_offset": TRAIN_START_OBSTACLE_1_OFFSET,
            "curriculum_start_position_jitter": TRAIN_START_OBSTACLE_JITTER,
            "asset_cfg": SceneEntityCfg("obstacle_1"),
            "min_object_y_surface_clearance": TRAIN_MIN_OBJECT_Y_SURFACE_CLEARANCE,
            "curriculum_start_min_object_y_surface_clearance": TRAIN_START_MIN_OBJECT_Y_SURFACE_CLEARANCE,
        },
    )
    reset_obstacle_2 = EventTerm(
        func=reset_obstacle_geometry_relative_to_object,
        mode="reset",
        params={
            "position_offset": TRAIN_OBSTACLE_2_OFFSET,
            "position_jitter": TRAIN_OBSTACLE_JITTER,
            "curriculum_steps": TRAIN_POSITION_CURRICULUM_STEPS,
            "curriculum_start_position_offset": TRAIN_START_OBSTACLE_2_OFFSET,
            "curriculum_start_position_jitter": TRAIN_START_OBSTACLE_JITTER,
            "asset_cfg": SceneEntityCfg("obstacle_2"),
            "paired_obstacle_cfg": SceneEntityCfg("obstacle_1"),
            "min_object_y_surface_clearance": TRAIN_MIN_OBJECT_Y_SURFACE_CLEARANCE,
            "min_pair_y_surface_gap": TRAIN_MIN_PAIR_Y_SURFACE_GAP,
            "curriculum_start_min_object_y_surface_clearance": TRAIN_START_MIN_OBJECT_Y_SURFACE_CLEARANCE,
            "curriculum_start_min_pair_y_surface_gap": TRAIN_START_MIN_PAIR_Y_SURFACE_GAP,
        },
    )


@configclass
class FixedAEventCfg(EventCfg):
    """Evaluation-only fixed Condition A."""

    reset_obstacle_1 = EventTerm(
        func=reset_obstacle_geometry_relative_to_object,
        mode="reset",
        params={
            "position_offset": A_OBSTACLE_1_OFFSET,
            "position_jitter": ZERO_JITTER,
            "asset_cfg": SceneEntityCfg("obstacle_1"),
        },
    )
    reset_obstacle_2 = EventTerm(
        func=reset_obstacle_geometry_relative_to_object,
        mode="reset",
        params={
            "position_offset": A_OBSTACLE_2_OFFSET,
            "position_jitter": ZERO_JITTER,
            "asset_cfg": SceneEntityCfg("obstacle_2"),
        },
    )


@configclass
class BaselineGraspingEventCfg(EventCfg):
    """Baseline reset layout that mirrors fixed-condition obstacle event names."""

    reset_obstacle_1 = EventTerm(func=noop_obstacle_reset, mode="reset")
    reset_obstacle_2 = EventTerm(func=noop_obstacle_reset, mode="reset")


@configclass
class FixedBEventCfg(EventCfg):
    """Evaluation-only fixed Condition B."""

    reset_obstacle_1 = EventTerm(
        func=reset_obstacle_geometry_relative_to_object,
        mode="reset",
        params={
            "position_offset": B_OBSTACLE_1_OFFSET,
            "position_jitter": ZERO_JITTER,
            "asset_cfg": SceneEntityCfg("obstacle_1"),
        },
    )
    reset_obstacle_2 = EventTerm(
        func=reset_obstacle_geometry_relative_to_object,
        mode="reset",
        params={
            "position_offset": B_OBSTACLE_2_OFFSET,
            "position_jitter": ZERO_JITTER,
            "asset_cfg": SceneEntityCfg("obstacle_2"),
        },
    )


@configclass
class FixedCEventCfg(EventCfg):
    """Evaluation-only fixed Condition C."""

    reset_obstacle_1 = EventTerm(
        func=reset_obstacle_geometry_relative_to_object,
        mode="reset",
        params={
            "position_offset": C_OBSTACLE_1_OFFSET,
            "position_jitter": ZERO_JITTER,
            "asset_cfg": SceneEntityCfg("obstacle_1"),
        },
    )
    reset_obstacle_2 = EventTerm(
        func=reset_obstacle_geometry_relative_to_object,
        mode="reset",
        params={
            "position_offset": C_OBSTACLE_2_OFFSET,
            "position_jitter": ZERO_JITTER,
            "asset_cfg": SceneEntityCfg("obstacle_2"),
        },
    )
    reset_obstacle_3 = EventTerm(
        func=reset_obstacle_geometry_relative_to_object,
        mode="reset",
        params={
            "position_offset": C_OBSTACLE_3_OFFSET,
            "position_jitter": ZERO_JITTER,
            "asset_cfg": SceneEntityCfg("obstacle_3"),
        },
    )
    reset_obstacle_4 = EventTerm(
        func=reset_obstacle_geometry_relative_to_object,
        mode="reset",
        params={
            "position_offset": C_OBSTACLE_4_OFFSET,
            "position_jitter": ZERO_JITTER,
            "asset_cfg": SceneEntityCfg("obstacle_4"),
        },
    )


@configclass
class FrankaGraspingBaselineEnvCfg(LiftEnvCfg):
    """True baseline with no obstacles."""

    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    observations: BaselineMatchedObservationsCfg = BaselineMatchedObservationsCfg()
    rewards: GraspingBaselineRewardsCfg = GraspingBaselineRewardsCfg()
    events: BaselineGraspingEventCfg = BaselineGraspingEventCfg()
    curriculum: GraspingCurriculumCfg = GraspingCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        _configure_franka_grasping_components(self)


@configclass
class FrankaGraspingBaselineEnvCfg_PLAY(FrankaGraspingBaselineEnvCfg):
    """Smaller no-obstacle baseline for visualisation/play."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        _configure_play_viewer(self)


@configclass
class FrankaGraspingEnvCfg(FrankaGraspingBaselineEnvCfg):
    """Current training task: baseline behavior plus randomized obstacle geometry."""

    scene: GraspingSceneCfg = GraspingSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    observations: GraspingObservationsCfg = GraspingObservationsCfg()
    rewards: GraspingRewardsCfg = GraspingRewardsCfg()
    events: RandomizedObstacleEventCfg = RandomizedObstacleEventCfg()
    curriculum: GraspingCurriculumCfg = GraspingCurriculumCfg()


@configclass
class FrankaGraspingEnvCfg_PLAY(FrankaGraspingEnvCfg):
    """Smaller randomized version for playback and video recording."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        _configure_play_viewer(self)


@configclass
class FrankaGraspingFixedAEnvCfg(FrankaGraspingBaselineEnvCfg):
    """Evaluation-only fixed Condition A."""

    scene: GraspingSceneCfg = GraspingSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: GraspingObservationsCfg = GraspingObservationsCfg()
    rewards: GraspingRewardsCfg = GraspingRewardsCfg()
    events: FixedAEventCfg = FixedAEventCfg()
    curriculum: GraspingCurriculumCfg = GraspingCurriculumCfg()


@configclass
class FrankaGraspingFixedAEnvCfg_PLAY(FrankaGraspingFixedAEnvCfg):
    """Smaller fixed A version for playback."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        _configure_play_viewer(self)


@configclass
class FrankaGraspingFixedBEnvCfg(FrankaGraspingBaselineEnvCfg):
    """Evaluation-only fixed Condition B."""

    scene: GraspingBenchmarkBSceneCfg = GraspingBenchmarkBSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: GraspingObservationsCfg = GraspingObservationsCfg()
    rewards: GraspingRewardsCfg = GraspingRewardsCfg()
    events: FixedBEventCfg = FixedBEventCfg()
    curriculum: GraspingCurriculumCfg = GraspingCurriculumCfg()


@configclass
class FrankaGraspingFixedBEnvCfg_PLAY(FrankaGraspingFixedBEnvCfg):
    """Smaller fixed B version for playback."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        _configure_play_viewer(self)


@configclass
class FrankaGraspingFixedCEnvCfg(FrankaGraspingBaselineEnvCfg):
    """Evaluation-only fixed Condition C."""

    scene: GraspingBenchmarkCSceneCfg = GraspingBenchmarkCSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: GraspingBenchmarkCMatchedObservationsCfg = GraspingBenchmarkCMatchedObservationsCfg()
    rewards: GraspingBenchmarkCRewardsCfg = GraspingBenchmarkCRewardsCfg()
    events: FixedCEventCfg = FixedCEventCfg()
    curriculum: GraspingCurriculumCfg = GraspingCurriculumCfg()


@configclass
class FrankaGraspingFixedCEnvCfg_PLAY(FrankaGraspingFixedCEnvCfg):
    """Smaller fixed C version for playback."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        _configure_play_viewer(self)


class FrankaGraspingEnvWithMetrics(ManagerBasedRLEnv):
    """Thin wrapper that tracks thesis metrics and owns runtime obstacle geometry state."""

    def __init__(self, cfg: LiftEnvCfg, **kwargs):
        # Prestartup events attach per-env obstacle sizes before `super().__init__` returns.
        self._prestartup_obstacle_sizes: dict[str, torch.Tensor] = {}
        self._obstacle_names: list[str] = []
        self._obstacle_base_sizes: dict[str, torch.Tensor] = {}
        self._obstacle_sizes: dict[str, torch.Tensor] = {}
        self._obstacle_half_extents: dict[str, torch.Tensor] = {}

        super().__init__(cfg, **kwargs)

        num = self.num_envs
        device = self.device
        self._step_count = torch.zeros(num, dtype=torch.long, device=device)
        self._lift_count = torch.zeros(num, dtype=torch.float32, device=device)
        self._consec_lift_steps = torch.zeros(num, dtype=torch.long, device=device)
        self._success_achieved = torch.zeros(num, dtype=torch.bool, device=device)
        self._first_success_step = torch.full((num,), self.max_episode_length, dtype=torch.long, device=device)
        self._collision_count = torch.zeros(num, dtype=torch.float32, device=device)

        self._obstacle_names = [name for name in OBSTACLE_NAMES if name in self.scene.keys()]
        for obstacle_name in self._obstacle_names:
            asset: RigidObject = self.scene[obstacle_name]
            base_size = _spawn_size_tensor(asset, self.device)
            full_sizes = self._prestartup_obstacle_sizes.get(obstacle_name)
            if full_sizes is None:
                full_sizes = base_size.unsqueeze(0).repeat(self.num_envs, 1)
            else:
                full_sizes = full_sizes.to(device=self.device, dtype=torch.float32)

            self._obstacle_base_sizes[obstacle_name] = base_size
            self._obstacle_sizes[obstacle_name] = full_sizes.clone()
            self._obstacle_half_extents[obstacle_name] = 0.5 * full_sizes.clone()

    def _sensor_contacts_obstacle(self, sensor_name: str) -> torch.Tensor:
        if sensor_name not in self.scene.keys():
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        force_matrix = self.scene[sensor_name].data.force_matrix_w
        if force_matrix is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        force_norm = torch.linalg.vector_norm(force_matrix, dim=-1)
        return (force_norm > OBSTACLE_CONTACT_FORCE_THRESHOLD).flatten(start_dim=1).any(dim=1)

    def _ee_near_obstacle(self) -> torch.Tensor:
        return self._sensor_contacts_obstacle("left_finger_obstacle_contact") | self._sensor_contacts_obstacle(
            "right_finger_obstacle_contact"
        )

    def step(self, action: torch.Tensor):
        obs, reward, terminated, truncated, info = super().step(action)

        self._step_count += 1

        past_warmup = self._step_count > GRASP_WARMUP_STEPS
        obj_z = self.scene["object"].data.root_pos_w[:, 2]
        above_progress_threshold = (obj_z > LIFT_PROGRESS_HEIGHT_THRESHOLD) & past_warmup
        above_success_threshold = (obj_z > GRASP_SUCCESS_HEIGHT_THRESHOLD) & past_warmup

        self._lift_count += above_progress_threshold.float()
        self._consec_lift_steps = torch.where(
            above_success_threshold,
            self._consec_lift_steps + 1,
            torch.zeros_like(self._consec_lift_steps),
        )
        just_succeeded = (self._consec_lift_steps >= GRASP_SUCCESS_CONSECUTIVE_STEPS) & (~self._success_achieved)
        self._first_success_step = torch.where(just_succeeded, self._step_count, self._first_success_step)
        self._success_achieved |= just_succeeded

        self._collision_count += self._ee_near_obstacle().float()

        done = terminated | truncated
        if done.any():
            done_idx = done.nonzero(as_tuple=False).squeeze(-1)

            post_warmup = (self._step_count[done_idx] - GRASP_WARMUP_STEPS).float().clamp(min=1.0)
            lift_progress_rate = (self._lift_count[done_idx] / post_warmup).mean().item()
            grasp_rate = self._success_achieved[done_idx].float().mean().item()

            coll_freq = (self._collision_count[done_idx] / self._step_count[done_idx].float()).mean().item()
            completion_steps = torch.where(
                self._success_achieved[done_idx],
                self._first_success_step[done_idx],
                torch.full_like(self._first_success_step[done_idx], self.max_episode_length),
            )
            completion_time = float(completion_steps.float().mean().item())

            if "log" not in info:
                info["log"] = {}
            info["log"]["metrics/grasp_success_rate"] = grasp_rate
            info["log"]["metrics/collision_frequency"] = coll_freq
            info["log"]["metrics/task_completion_time"] = completion_time
            info["log"]["metrics/lift_progress_rate"] = lift_progress_rate

            self._lift_count[done_idx] = 0.0
            self._consec_lift_steps[done_idx] = 0
            self._success_achieved[done_idx] = False
            self._first_success_step[done_idx] = self.max_episode_length
            self._collision_count[done_idx] = 0.0
            self._step_count[done_idx] = 0

        return obs, reward, terminated, truncated, info


class FrankaGraspingBaselineEnvWithMetrics(FrankaGraspingEnvWithMetrics):
    """No-obstacle baseline wrapper: collision frequency is always zero."""

    def _ee_near_obstacle(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)