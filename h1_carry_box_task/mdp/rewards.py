from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, subtract_frame_transforms

from . import observations as carry_observations

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def _box_pos_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    carry_box: RigidObject = env.scene[object_cfg.name]
    box_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w.torch,
        robot.data.root_quat_w.torch,
        carry_box.data.root_pos_w.torch[:, :3],
    )
    return box_pos_b


def box_height_above_minimum(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("carry_box"),
) -> torch.Tensor:
    carry_box: RigidObject = env.scene[object_cfg.name]
    return torch.where(carry_box.data.root_pos_w.torch[:, 2] > minimum_height, 1.0, 0.0)


def box_carry_position_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("carry_box"),
) -> torch.Tensor:
    target_pos_b = torch.tensor(env.cfg.carry_box_target_pos_b, device=env.device, dtype=torch.float32).unsqueeze(0)
    position_error = torch.linalg.norm(_box_pos_b(env, robot_cfg, object_cfg) - target_pos_b, dim=1)
    return 1.0 - torch.tanh(position_error / std)


def box_upright_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("carry_box"),
) -> torch.Tensor:
    carry_box: RigidObject = env.scene[object_cfg.name]
    up_axis = torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)
    up_axis[:, 2] = 1.0
    world_up = quat_apply(carry_box.data.root_quat_w.torch, up_axis)
    upright_error = 1.0 - world_up[:, 2].clamp(-1.0, 1.0)
    return 1.0 - torch.tanh(upright_error / std)


def box_velocity_l2(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("carry_box"),
) -> torch.Tensor:
    carry_box: RigidObject = env.scene[object_cfg.name]
    return torch.sum(torch.square(carry_box.data.root_lin_vel_w.torch), dim=1) + 0.25 * torch.sum(
        torch.square(carry_box.data.root_ang_vel_w.torch), dim=1
    )


def hand_support_proximity_tanh(
    env: ManagerBasedRLEnv,
    side: str,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("carry_box"),
) -> torch.Tensor:
    support_error = carry_observations.hand_to_box_support_vector_in_robot_root_frame(env, side, object_cfg)
    return 1.0 - torch.tanh(torch.linalg.norm(support_error, dim=1) / std)


def box_centered_between_hands_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("carry_box"),
) -> torch.Tensor:
    left_hand_pos_b = carry_observations.hand_position_in_robot_root_frame(env, side="left")
    right_hand_pos_b = carry_observations.hand_position_in_robot_root_frame(env, side="right")
    box_pos_b = carry_observations.box_position_in_robot_root_frame(env, object_cfg=object_cfg)
    hand_midpoint_b = 0.5 * (left_hand_pos_b + right_hand_pos_b)
    return 1.0 - torch.tanh(torch.linalg.norm(hand_midpoint_b - box_pos_b, dim=1) / std)


def box_close_to_torso_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("carry_box"),
) -> torch.Tensor:
    target_pos_t = torch.tensor(env.cfg.carry_box_torso_target_pos_t, device=env.device, dtype=torch.float32).unsqueeze(0)
    box_pos_t = carry_observations.box_position_in_torso_frame(env, object_cfg=object_cfg)
    return 1.0 - torch.tanh(torch.linalg.norm(box_pos_t - target_pos_t, dim=1) / std)


def hand_to_box_side_tanh(
    env: ManagerBasedRLEnv,
    side: str,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("carry_box"),
) -> torch.Tensor:
    """Reward hand for being at the correct side of the box.

    Left hand should be at box center + left offset (+y side).
    Right hand should be at box center + right offset (-y side).
    """
    support_error = carry_observations.hand_to_box_support_vector_in_robot_root_frame(env, side, object_cfg)
    return 1.0 - torch.tanh(torch.linalg.norm(support_error, dim=1) / std)


def symmetric_grasp_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("carry_box"),
) -> torch.Tensor:
    """Reward symmetric hand placement on opposite sides of the box.

    Computes the difference in distance from each hand to the box center.
    A symmetric grasp has equal distances on both sides.
    """
    left_hand_pos_b = carry_observations.hand_position_in_robot_root_frame(env, side="left")
    right_hand_pos_b = carry_observations.hand_position_in_robot_root_frame(env, side="right")
    box_pos_b = carry_observations.box_position_in_robot_root_frame(env, object_cfg=object_cfg)

    left_dist = torch.linalg.norm(left_hand_pos_b - box_pos_b, dim=1)
    right_dist = torch.linalg.norm(right_hand_pos_b - box_pos_b, dim=1)

    # Symmetry error: difference in distances
    symmetry_error = torch.abs(left_dist - right_dist)
    return 1.0 - torch.tanh(symmetry_error / std)


def box_carry_velocity_tracking_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "base_velocity",
    object_cfg: SceneEntityCfg = SceneEntityCfg("carry_box"),
) -> torch.Tensor:
    """Reward the box for moving in the direction of the velocity command.

    This encourages the robot to carry the box forward while walking.
    """
    carry_box: RigidObject = env.scene[object_cfg.name]
    command_manager = env.command_manager

    # Get the velocity command in the robot's base frame
    vel_cmd_b = command_manager.get_command(command_name)[:, :2]  # (vx, vy)

    # Transform box velocity to robot base frame
    box_vel_b = carry_observations.box_linear_velocity_in_robot_root_frame(env, object_cfg=object_cfg)[:, :2]

    # Error between box velocity and commanded velocity
    vel_error = torch.linalg.norm(box_vel_b - vel_cmd_b, dim=1)
    return torch.exp(-vel_error / std)


def hands_in_front_of_robot(
    env: ManagerBasedRLEnv,
    std: float,
) -> torch.Tensor:
    """Reward for keeping both hands in front of the robot (positive x in robot frame).

    This prevents the arms from swinging behind the body.
    Returns 1.0 when both hands are in front, decays as hands move behind.
    """
    left_hand_pos_b = carry_observations.hand_position_in_robot_root_frame(env, side="left")
    right_hand_pos_b = carry_observations.hand_position_in_robot_root_frame(env, side="right")

    # x-coordinate in robot frame: positive = in front, negative = behind
    left_x = left_hand_pos_b[:, 0]
    right_x = right_hand_pos_b[:, 0]

    # Penalize hands that are behind the robot (negative x)
    # Use a smooth penalty: 0 when x >= 0, increases as x becomes more negative
    left_behind = torch.clamp(-left_x, min=0.0)
    right_behind = torch.clamp(-right_x, min=0.0)

    total_behind = left_behind + right_behind
    return torch.exp(-total_behind / std)


def hands_under_box_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("carry_box"),
) -> torch.Tensor:
    """Reward both hands for being UNDER the box (below box center in z).

    For bottom-carry: hands should be below the box to cradle it.
    Returns 1.0 when both hands are below the box center, decays as they move above.
    """
    left_hand_pos_b = carry_observations.hand_position_in_robot_root_frame(env, side="left")
    right_hand_pos_b = carry_observations.hand_position_in_robot_root_frame(env, side="right")
    box_pos_b = carry_observations.box_position_in_robot_root_frame(env, object_cfg=object_cfg)

    # z-coordinate: hands should be BELOW box center (lower z)
    left_z_diff = box_pos_b[:, 2] - left_hand_pos_b[:, 2]  # positive = hand below box
    right_z_diff = box_pos_b[:, 2] - right_hand_pos_b[:, 2]

    # Penalize hands that are ABOVE the box (negative diff)
    left_above = torch.clamp(-left_z_diff, min=0.0)
    right_above = torch.clamp(-right_z_diff, min=0.0)

    total_above = left_above + right_above
    return torch.exp(-total_above / std)


def box_resting_on_arms_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("carry_box"),
) -> torch.Tensor:
    """Reward the box for resting on top of the arms (cradle position).

    The box should be above the hand midpoint and centered between the hands.
    """
    left_hand_pos_b = carry_observations.hand_position_in_robot_root_frame(env, side="left")
    right_hand_pos_b = carry_observations.hand_position_in_robot_root_frame(env, side="right")
    box_pos_b = carry_observations.box_position_in_robot_root_frame(env, object_cfg=object_cfg)

    # Hand midpoint
    hand_midpoint_b = 0.5 * (left_hand_pos_b + right_hand_pos_b)

    # Box should be above the hand midpoint (positive z difference)
    z_above = box_pos_b[:, 2] - hand_midpoint_b[:, 2]
    z_above = torch.clamp(z_above, min=0.0)  # Only reward when box is above

    # Box should be centered between hands in y
    y_error = torch.abs(box_pos_b[:, 1] - hand_midpoint_b[:, 1])

    # Combined error: box should be above AND centered
    error = y_error - 0.5 * z_above  # Reward z_above, penalize y_error
    error = torch.clamp(error, min=0.0)

    return torch.exp(-error / std)
