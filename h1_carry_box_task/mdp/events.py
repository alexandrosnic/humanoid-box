from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedEnv


def _sample_range_tensor(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    value_range: dict[str, tuple[float, float]],
) -> torch.Tensor:
    range_tensor = torch.tensor(
        [value_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]],
        device=env.device,
    )
    return math_utils.sample_uniform(range_tensor[:, 0], range_tensor[:, 1], (len(env_ids), 6), device=env.device)


def reset_robot_root_for_carry(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    robot: Articulation = env.scene[asset_cfg.name]
    default_root_pose = robot.data.default_root_pose.torch[env_ids].clone()
    default_root_vel = robot.data.default_root_vel.torch[env_ids].clone()

    pose_samples = _sample_range_tensor(env, env_ids, pose_range)
    positions = default_root_pose[:, :3] + env.scene.env_origins[env_ids] + pose_samples[:, :3]
    orientation_delta = math_utils.quat_from_euler_xyz(
        pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
    )
    orientations = math_utils.quat_mul(default_root_pose[:, 3:7], orientation_delta)

    velocity_samples = _sample_range_tensor(env, env_ids, velocity_range)
    velocities = default_root_vel + velocity_samples

    root_pose = torch.cat([positions, orientations], dim=-1)
    robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim_index(root_velocity=velocities, env_ids=env_ids)

    if not hasattr(env, "_carry_robot_reset_pose"):
        env._carry_robot_reset_pose = robot.data.default_root_pose.torch.clone()
        env._carry_robot_reset_velocity = robot.data.default_root_vel.torch.clone()
    env._carry_robot_reset_pose[env_ids] = root_pose
    env._carry_robot_reset_velocity[env_ids] = velocities


def _get_robot_reset_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    robot: Articulation = env.scene[robot_cfg.name]
    if hasattr(env, "_carry_robot_reset_pose") and hasattr(env, "_carry_robot_reset_velocity"):
        return env._carry_robot_reset_pose[env_ids], env._carry_robot_reset_velocity[env_ids]

    default_root_pose = robot.data.default_root_pose.torch[env_ids].clone()
    default_root_vel = robot.data.default_root_vel.torch[env_ids].clone()
    default_root_pose[:, :3] += env.scene.env_origins[env_ids]
    return default_root_pose, default_root_vel


def reset_box_in_carry_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("carry_box"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    box_pos_in_robot_frame: tuple[float, float, float] = (0.52, 0.0, 0.92),
):
    carry_box: RigidObject = env.scene[asset_cfg.name]
    robot_root_pose, _ = _get_robot_reset_pose(env, env_ids, robot_cfg)

    pose_samples = _sample_range_tensor(env, env_ids, pose_range)
    box_offset = torch.tensor(box_pos_in_robot_frame, device=carry_box.device, dtype=torch.float32).unsqueeze(0).repeat(
        len(env_ids), 1
    )

    local_positions = box_offset + pose_samples[:, :3]
    positions = robot_root_pose[:, :3] + math_utils.quat_apply(robot_root_pose[:, 3:7], local_positions)
    orientation_delta = math_utils.quat_from_euler_xyz(
        pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
    )
    orientations = math_utils.quat_mul(robot_root_pose[:, 3:7], orientation_delta)

    velocity_samples = _sample_range_tensor(env, env_ids, velocity_range)
    linear_velocity = math_utils.quat_apply(robot_root_pose[:, 3:7], velocity_samples[:, :3])
    velocities = torch.cat([linear_velocity, velocity_samples[:, 3:]], dim=-1)

    carry_box.write_root_pose_to_sim_index(root_pose=torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    carry_box.write_root_velocity_to_sim_index(root_velocity=velocities, env_ids=env_ids)


def reset_box_on_table(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("carry_box"),
    table_pos: tuple[float, float, float] = (0.55, 0.0, 0.89),
):
    carry_box: RigidObject = env.scene[asset_cfg.name]
    origins = env.scene.env_origins[env_ids]

    pose_samples = _sample_range_tensor(env, env_ids, pose_range)
    box_offset = torch.tensor(table_pos, device=carry_box.device, dtype=torch.float32).unsqueeze(0).repeat(
        len(env_ids), 1
    )

    positions = origins + box_offset + pose_samples[:, :3]
    orientations = math_utils.quat_from_euler_xyz(
        pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
    )

    velocity_samples = _sample_range_tensor(env, env_ids, velocity_range)

    carry_box.write_root_pose_to_sim_index(root_pose=torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    carry_box.write_root_velocity_to_sim_index(root_velocity=velocity_samples, env_ids=env_ids)
