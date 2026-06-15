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
    # Use arccos to get the actual tilt angle in radians (linear gradient at zero tilt)
    tilt_angle = torch.arccos(world_up[:, 2].clamp(-1.0 + 1e-6, 1.0 - 1e-6))
    return 1.0 - torch.tanh(tilt_angle / std)


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
    """Reward hands for being close to the handles, penalizing them if they go above the handles."""
    support_error = carry_observations.hand_to_box_support_vector_in_robot_root_frame(env, side, object_cfg)
    
    # 3D distance error
    dist = torch.linalg.norm(support_error, dim=1)
    
    # Z component of support_error is (handle_z - hand_z).
    # If hand_z > handle_z, then support_error[:, 2] is negative.
    above_penalty = torch.clamp(-support_error[:, 2], min=0.0)
    
    total_error = dist + 4.0 * above_penalty
    return 1.0 - torch.tanh(total_error / std)


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
    """Reward the box handles for resting on top of the wrists/forearms.
    
    This penalizes the robot if its hands go above the handles, and rewards
    holding the handles from directly below.
    """
    # Get support vectors in robot root frame (handle_pos - hand_pos)
    left_vec = carry_observations.hand_to_box_support_vector_in_robot_root_frame(env, "left", object_cfg)
    right_vec = carry_observations.hand_to_box_support_vector_in_robot_root_frame(env, "right", object_cfg)
    
    # Horizontal error (XY deviation)
    left_xy_err = torch.linalg.norm(left_vec[:, :2], dim=1)
    right_xy_err = torch.linalg.norm(right_vec[:, :2], dim=1)
    
    # Z differences (positive = hand is below handle)
    left_z_diff = left_vec[:, 2]
    right_z_diff = right_vec[:, 2]
    
    # Penalize hands that go above the handles (negative diff)
    left_above = torch.clamp(-left_z_diff, min=0.0)
    right_above = torch.clamp(-right_z_diff, min=0.0)
    
    # Target vertical support offset (e.g. 2cm below handle)
    left_z_err = torch.abs(left_z_diff - 0.02)
    right_z_err = torch.abs(right_z_diff - 0.02)
    
    total_error = left_xy_err + right_xy_err + left_z_err + right_z_err + 5.0 * (left_above + right_above)
    return torch.exp(-total_error / std)


def upright_posture_penalty(
    env: ManagerBasedRLEnv,
    std: float = 0.15,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize the robot torso from leaning forward (pitch) to maintain balance."""
    robot = env.scene[robot_cfg.name]
    root_quat = robot.data.root_quat_w.torch
    from isaaclab.utils.math import yaw_pitch_roll_from_quat
    _, pitch, _ = yaw_pitch_roll_from_quat(root_quat)
    forward_lean = torch.clamp(pitch, min=0.0)  # Only penalize forward lean
    return torch.exp(-forward_lean / std)


def robot_to_box_distance_target_tanh(
    env: ManagerBasedRLEnv,
    target_dist: float,
    std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("carry_box"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    carry_box: RigidObject = env.scene[object_cfg.name]
    dist = torch.linalg.norm(robot.data.root_pos_w.torch[:, :2] - carry_box.data.root_pos_w.torch[:, :2], dim=1)
    error = torch.abs(dist - target_dist)
    return 1.0 - torch.tanh(error / std)


def box_lift_above_table_tanh(
    env: ManagerBasedRLEnv,
    table_height: float,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("carry_box"),
) -> torch.Tensor:
    carry_box: RigidObject = env.scene[object_cfg.name]
    box_z = carry_box.data.root_pos_w.torch[:, 2]
    lift_height = torch.clamp(box_z - table_height, min=0.0)
    return torch.tanh(lift_height / std)


def track_gait_mocap(
    env: ManagerBasedRLEnv,
    std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    import numpy as np
    robot = env.scene[robot_cfg.name]
    
    # Initialize the mocap cache on the environment if not present
    if not hasattr(env, "_mocap_cache"):
        mocap_path = r"C:\Users\alexa\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\humanoid_amp\motions\humanoid_walk.npz"
        mocap_data = np.load(mocap_path)
        ref_dof_names = list(mocap_data["dof_names"])
        ref_dof_positions = torch.tensor(mocap_data["dof_positions"], device=env.device, dtype=torch.float32)
        
        # Build index mapping for robot's joint order
        robot_joint_names = robot.data.joint_names
        
        H1_TO_REF_MAP = {
            "left_hip_yaw": "left_hip_z",
            "left_hip_roll": "left_hip_x",
            "left_hip_pitch": "left_hip_y",
            "left_knee": "left_knee",
            "left_ankle": "left_ankle_y",
            "right_hip_yaw": "right_hip_z",
            "right_hip_roll": "right_hip_x",
            "right_hip_pitch": "right_hip_y",
            "right_knee": "right_knee",
            "right_ankle": "right_ankle_y",
            "torso": "abdomen_z",
            "left_shoulder_pitch": "left_shoulder_y",
            "left_shoulder_roll": "left_shoulder_x",
            "left_shoulder_yaw": "left_shoulder_z",
            "left_elbow": "left_elbow",
            "right_shoulder_pitch": "right_shoulder_y",
            "right_shoulder_roll": "right_shoulder_x",
            "right_shoulder_yaw": "right_shoulder_z",
            "right_elbow": "right_elbow",
        }
        
        # Create mapping indices
        mapping_indices = []
        ref_indices = []
        
        for robot_idx, name in enumerate(robot_joint_names):
            matched = False
            for h1_key, ref_key in H1_TO_REF_MAP.items():
                if h1_key in name:
                    if ref_key in ref_dof_names:
                        mapping_indices.append(robot_idx)
                        ref_indices.append(ref_dof_names.index(ref_key))
                        matched = True
                        break
            
        env._mocap_cache = {
            "robot_indices": torch.tensor(mapping_indices, device=env.device),
            "ref_indices": torch.tensor(ref_indices, device=env.device),
            "ref_positions": ref_dof_positions,
            "fps": float(mocap_data["fps"]),
            "num_frames": ref_dof_positions.shape[0]
        }
    
    cache = env._mocap_cache
    
    # Calculate current frame index for each environment
    fps = cache["fps"]
    num_frames = cache["num_frames"]
    time_elapsed = env.episode_length_buf.float() * env.dt
    frame_indices = (time_elapsed * fps).long() % num_frames
    
    # Get reference joint positions
    ref_pos_mapped = cache["ref_positions"][frame_indices][:, cache["ref_indices"]]
    
    # Get robot's current joint positions (rel)
    robot_pos_mapped = robot.data.joint_pos[:, cache["robot_indices"]]
    
    # Compute squared error
    pos_error = torch.sum(torch.square(robot_pos_mapped - ref_pos_mapped), dim=1)
    return torch.exp(-pos_error / std)


def box_moving_away_from_table(
    env: ManagerBasedRLEnv,
    table_pos: tuple[float, float],
    std: float = 1.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("carry_box"),
) -> torch.Tensor:
    carry_box = env.scene[object_cfg.name]
    origins = env.scene.env_origins[:, :2]
    
    # Absolute table position in world coordinates for each environment
    table_offset = torch.tensor(table_pos, device=env.device, dtype=torch.float32).unsqueeze(0)
    table_pos_w = origins + table_offset
    
    # Distance in XY plane between box and table center
    dist = torch.linalg.norm(carry_box.data.root_pos_w.torch[:, :2] - table_pos_w, dim=1)
    
    # Reward distance from table, clamping at 2.0 meters to prevent unbounded rewards
    return torch.clamp(dist, max=2.0)

