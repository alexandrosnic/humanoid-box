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


def upright_posture_penalty(
    env: ManagerBasedRLEnv,
    std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize the robot for leaning forward (pitch).

    When carrying a box, the weight can pull the robot forward.
    This reward encourages the robot to stay upright.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    
    # Get the robot's orientation (quaternion)
    quat = robot.data.root_quat_w.torch
    
    # Convert to euler angles to get pitch
    # Isaac Lab uses (w, x, y, z) quaternion format
    # Pitch is rotation around y-axis
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    # Pitch calculation: sin(pitch) = 2(wy - xz)
    sin_pitch = 2.0 * (w * y - x * z)
    sin_pitch = torch.clamp(sin_pitch, -1.0, 1.0)
    pitch = torch.asin(sin_pitch)
    
    # Penalize forward lean (positive pitch = leaning forward)
    # We want pitch close to 0 (upright)
    forward_lean = torch.clamp(pitch, min=0.0)  # Only penalize forward lean
    
    return torch.exp(-forward_lean / std)


def torso_lean_back_reward(
    env: ManagerBasedRLEnv,
    target_lean: float = -0.15,  # Negative = lean back (radians)
    std: float = 0.1,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the torso for leaning back to counterbalance the box.
    
    The H1 torso joint rotates the upper body. Negative values lean back.
    This helps counterbalance the forward moment from carrying a heavy box.
    
    Args:
        target_lean: Target torso angle in radians (negative = lean back)
        std: Standard deviation for the exponential reward
    """
    robot = env.scene[robot_cfg.name]
    
    # Get torso joint position
    torso_joint_id = robot.find_joints("torso")[0][0]
    torso_pos = robot.data.joint_pos[:, torso_joint_id]
    
    # Reward for being close to target lean-back angle
    error = torch.abs(torso_pos - target_lean)
    return torch.exp(-error / std)


def torso_forward_penalty(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize the torso for leaning forward (positive angle).
    
    When carrying a heavy box, forward lean causes the robot to fall.
    This penalty directly fights forward torso lean.
    
    Args:
        std: Standard deviation for the exponential penalty
    """
    robot = env.scene[robot_cfg.name]
    
    # Get torso joint position
    torso_joint_id = robot.find_joints("torso")[0][0]
    torso_pos = robot.data.joint_pos[:, torso_joint_id]
    
    # Penalize positive (forward) lean - return 0 when leaning back, penalty when forward
    forward_lean = torch.clamp(torso_pos, min=0.0)  # Only penalize positive values
    return torch.exp(-forward_lean / std)  # Returns 1.0 when no forward lean, decays as lean increases


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



