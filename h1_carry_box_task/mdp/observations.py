from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


# Patterns for finding hand/wrist/elbow links (fallback chain)
# For H1_MINIMAL_CFG which lacks hand/wrist links, falls back to elbow
LEFT_HAND_PATTERNS = ("left_.*hand.*", "left_.*wrist_yaw.*", "left_.*wrist_roll.*", "left_.*elbow.*")
RIGHT_HAND_PATTERNS = ("right_.*hand.*", "right_.*wrist_yaw.*", "right_.*wrist_roll.*", "right_.*elbow.*")
TORSO_PATTERNS = ("torso_link", "torso")


def _get_body_index(env: ManagerBasedRLEnv, cache_key: str, patterns: tuple[str, ...]) -> int:
    if not hasattr(env, "_carry_body_index_cache"):
        env._carry_body_index_cache = {}
    if cache_key not in env._carry_body_index_cache:
        robot = env.scene["robot"]
        for pattern in patterns:
            try:
                body_ids, _ = robot.find_bodies(pattern)
            except ValueError:
                continue
            if body_ids:
                env._carry_body_index_cache[cache_key] = body_ids[0]
                break
        else:
            raise ValueError(f"Could not resolve robot body for patterns: {patterns}")
    return env._carry_body_index_cache[cache_key]


def _robot_body_pose_w(env: ManagerBasedRLEnv, cache_key: str, patterns: tuple[str, ...]) -> torch.Tensor:
    robot: RigidObject = env.scene["robot"]
    body_idx = _get_body_index(env, cache_key, patterns)
    return robot.data.body_state_w.torch[:, body_idx, :7]


def _robot_body_position_w(env: ManagerBasedRLEnv, cache_key: str, patterns: tuple[str, ...]) -> torch.Tensor:
    return _robot_body_pose_w(env, cache_key, patterns)[:, :3]


def _support_offset(env: ManagerBasedRLEnv, side: str) -> torch.Tensor:
    attr_name = f"carry_box_{side}_hand_target_offset_b"
    offset = torch.tensor(getattr(env.cfg, attr_name), device=env.device, dtype=torch.float32)
    return offset.unsqueeze(0).repeat(env.num_envs, 1)


def _hand_patterns(side: str) -> tuple[str, ...]:
    if side == "left":
        return LEFT_HAND_PATTERNS
    if side == "right":
        return RIGHT_HAND_PATTERNS
    raise ValueError(f"Unsupported side: {side}")


def _hand_position_w(env: ManagerBasedRLEnv, side: str) -> torch.Tensor:
    """Get hand position. Falls back to elbow if no hand/wrist links exist (H1_MINIMAL_CFG)."""
    return _robot_body_position_w(env, f"{side}_hand", _hand_patterns(side))


def _box_support_point_w(env: ManagerBasedRLEnv, side: str, object_cfg: SceneEntityCfg) -> torch.Tensor:
    carry_box: RigidObject = env.scene[object_cfg.name]
    return carry_box.data.root_pos_w.torch[:, :3] + quat_apply(
        carry_box.data.root_quat_w.torch,
        _support_offset(env, side),
    )


def _desired_box_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.tensor(env.cfg.carry_box_target_pos_b, device=env.device, dtype=torch.float32).unsqueeze(0)


def box_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("carry_box"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    carry_box: RigidObject = env.scene[object_cfg.name]
    box_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w.torch,
        robot.data.root_quat_w.torch,
        carry_box.data.root_pos_w.torch[:, :3],
    )
    return box_pos_b


def desired_box_position_in_robot_root_frame(env: ManagerBasedRLEnv) -> torch.Tensor:
    return _desired_box_position(env).repeat(env.num_envs, 1)


def hand_position_in_robot_root_frame(env: ManagerBasedRLEnv, side: str) -> torch.Tensor:
    robot: RigidObject = env.scene["robot"]
    hand_pos_w = _hand_position_w(env, side)
    hand_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w.torch,
        robot.data.root_quat_w.torch,
        hand_pos_w,
    )
    return hand_pos_b


def hand_to_box_vector_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    side: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("carry_box"),
) -> torch.Tensor:
    robot: RigidObject = env.scene["robot"]
    carry_box: RigidObject = env.scene[object_cfg.name]
    return quat_apply_inverse(robot.data.root_quat_w.torch, carry_box.data.root_pos_w.torch[:, :3] - _hand_position_w(env, side))


def hand_to_box_support_vector_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    side: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("carry_box"),
) -> torch.Tensor:
    robot: RigidObject = env.scene["robot"]
    support_point_w = _box_support_point_w(env, side, object_cfg)
    return quat_apply_inverse(robot.data.root_quat_w.torch, support_point_w - _hand_position_w(env, side))


def box_position_in_torso_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("carry_box"),
) -> torch.Tensor:
    torso_pose_w = _robot_body_pose_w(env, "torso", TORSO_PATTERNS)
    carry_box: RigidObject = env.scene[object_cfg.name]
    box_pos_t, _ = subtract_frame_transforms(
        torso_pose_w[:, :3],
        torso_pose_w[:, 3:7],
        carry_box.data.root_pos_w.torch[:, :3],
    )
    return box_pos_t


def box_linear_velocity_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("carry_box"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    carry_box: RigidObject = env.scene[object_cfg.name]
    return quat_apply_inverse(robot.data.root_quat_w.torch, carry_box.data.root_lin_vel_w.torch)


def box_angular_velocity_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("carry_box"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    carry_box: RigidObject = env.scene[object_cfg.name]
    return quat_apply_inverse(robot.data.root_quat_w.torch, carry_box.data.root_ang_vel_w.torch)


def box_height(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("carry_box"),
) -> torch.Tensor:
    carry_box: RigidObject = env.scene[object_cfg.name]
    return carry_box.data.root_pos_w.torch[:, 2:3]


def box_upright_projection(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("carry_box"),
) -> torch.Tensor:
    carry_box: RigidObject = env.scene[object_cfg.name]
    up_axis = torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)
    up_axis[:, 2] = 1.0
    world_up = quat_apply(carry_box.data.root_quat_w.torch, up_axis)
    return world_up[:, 2:3]
