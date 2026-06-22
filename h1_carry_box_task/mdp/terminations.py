from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def box_too_far_from_robot(
    env: ManagerBasedRLEnv,
    max_distance: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("carry_box"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    carry_box: RigidObject = env.scene[object_cfg.name]
    distance = torch.linalg.norm(robot.data.root_pos_w.torch[:, :3] - carry_box.data.root_pos_w.torch[:, :3], dim=1)
    return distance > max_distance


def box_not_lifted_timeout(
    env: ManagerBasedRLEnv,
    timeout_time: float,
    minimum_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("carry_box"),
) -> torch.Tensor:
    """Terminates the episode if the box is not lifted above the minimum height after a timeout period."""
    carry_box: RigidObject = env.scene[object_cfg.name]
    is_below = carry_box.data.root_pos_w.torch[:, 2] < minimum_height
    # Compute elapsed time in seconds
    elapsed_time = env.episode_length_buf.float() * env.step_dt
    has_timed_out = elapsed_time > timeout_time
    return is_below & has_timed_out


def arms_below_table_limit(
    env: ManagerBasedRLEnv,
    min_height: float = 1.16,
    start_step: int = 15,
) -> torch.Tensor:
    """Terminates the episode if either hand/elbow goes below the table height after the initial steps."""
    from . import observations as carry_observations
    # Check if the episode step is past the initialization grace period
    grace_period_over = env.episode_length_buf > start_step
    
    # Get hand positions in world frame
    left_hand_pos_w = carry_observations._hand_position_w(env, "left")
    right_hand_pos_w = carry_observations._hand_position_w(env, "right")
    
    # Check if either hand is below min_height
    left_below = left_hand_pos_w[:, 2] < min_height
    right_below = right_hand_pos_w[:, 2] < min_height
    
    return grace_period_over & (left_below | right_below)

