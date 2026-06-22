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
    """Reward box for being upright — ONLY active once the box is lifted."""
    gate = _box_is_lifted_gate(env, object_cfg=object_cfg)
    carry_box: RigidObject = env.scene[object_cfg.name]
    up_axis = torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)
    up_axis[:, 2] = 1.0
    world_up = quat_apply(carry_box.data.root_quat_w.torch, up_axis)
    # Use arccos to get the actual tilt angle in radians (linear gradient at zero tilt)
    tilt_angle = torch.arccos(world_up[:, 2].clamp(-1.0 + 1e-6, 1.0 - 1e-6))
    raw = 1.0 - torch.tanh(tilt_angle / std)
    return gate * raw



def box_z_velocity_on_table_l2(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("carry_box"),
) -> torch.Tensor:
    """Penalize box velocity in the Z direction — ONLY active when the box is on the table."""
    not_lifted_gate = 1.0 - _box_is_lifted_gate(env, object_cfg=object_cfg)
    carry_box: RigidObject = env.scene[object_cfg.name]
    z_vel_sq = torch.square(carry_box.data.root_lin_vel_w[:, 2])
    return not_lifted_gate * z_vel_sq


def box_velocity_l2(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("carry_box"),
) -> torch.Tensor:
    carry_box: RigidObject = env.scene[object_cfg.name]
    return torch.sum(torch.square(carry_box.data.root_lin_vel_w.torch), dim=1) + 0.25 * torch.sum(
        torch.square(carry_box.data.root_ang_vel_w.torch), dim=1
    )



def hand_support_proximity_exp(
    env: ManagerBasedRLEnv,
    side: str,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("carry_box"),
) -> torch.Tensor:
    """Reward hands for being close to the handle target in 3D, constrained by a Z-height envelope."""
    support_error = carry_observations.hand_to_box_support_vector_in_robot_root_frame(env, side, object_cfg)
    dist = torch.linalg.norm(support_error, dim=1)
    
    # 1. 3D distance reward
    raw_dist_reward = torch.exp(-dist / std)
    
    # 2. Z-height envelope constraints (world frame)
    hand_pos_w = carry_observations._robot_body_position_w(env, f"{side}_hand", carry_observations._hand_patterns(side))
    hand_z = hand_pos_w[:, 2]
    
    # Flat penalty of -5.0 if hand height goes below 1.18m, and linear penalty above 1.20m
    below_penalty = torch.where(hand_z < 1.18, -5.0, 0.0)
    above_penalty = torch.clamp(hand_z - 1.20, min=0.0)
    total_penalty = below_penalty - 5.0 * above_penalty
    
    # Combine the 3D proximity reward with the height penalty
    return raw_dist_reward + total_penalty


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


def _box_is_lifted_gate(
    env: ManagerBasedRLEnv,
    table_height: float = 1.24,
    lift_threshold: float = 0.03,
    object_cfg: SceneEntityCfg = SceneEntityCfg("carry_box"),
) -> torch.Tensor:
    """Returns 1.0 per env where the box is lifted >=lift_threshold above the table, else 0.0."""
    carry_box: RigidObject = env.scene[object_cfg.name]
    box_z = carry_box.data.root_pos_w[:, 2]
    return (box_z > table_height + lift_threshold).float()


def box_close_to_torso_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("carry_box"),
) -> torch.Tensor:
    """Reward box being close to the torso — ONLY active once the box is lifted off the table.

    Gated to prevent the exploit of leaning the torso against the table/box to earn
    this reward without actually lifting.
    """
    gate = _box_is_lifted_gate(env, object_cfg=object_cfg)
    target_pos_t = torch.tensor(env.cfg.carry_box_torso_target_pos_t, device=env.device, dtype=torch.float32).unsqueeze(0)
    box_pos_t = carry_observations.box_position_in_torso_frame(env, object_cfg=object_cfg)
    raw = 1.0 - torch.tanh(torch.linalg.norm(box_pos_t - target_pos_t, dim=1) / std)
    return gate * raw



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

    Computes the difference in distance from each hand to the box center,
    and also penalizes height differences (Z-asymmetry) in the box frame.
    """
    left_hand_pos_b = carry_observations.hand_position_in_robot_root_frame(env, side="left")
    right_hand_pos_b = carry_observations.hand_position_in_robot_root_frame(env, side="right")
    box_pos_b = carry_observations.box_position_in_robot_root_frame(env, object_cfg=object_cfg)

    left_dist = torch.linalg.norm(left_hand_pos_b - box_pos_b, dim=1)
    right_dist = torch.linalg.norm(right_hand_pos_b - box_pos_b, dim=1)

    # Symmetry error: difference in distances
    dist_symmetry_error = torch.abs(left_dist - right_dist)

    # Height symmetry error: difference in Z coordinates relative to the box center
    left_z_b = left_hand_pos_b[:, 2] - box_pos_b[:, 2]
    right_z_b = right_hand_pos_b[:, 2] - box_pos_b[:, 2]
    z_symmetry_error = torch.abs(left_z_b - right_z_b)

    total_error = dist_symmetry_error + z_symmetry_error
    return 1.0 - torch.tanh(total_error / std)


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


def both_feet_grounded_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    table_height: float = 1.24,
    min_contact_force: float = 1.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("carry_box"),
) -> torch.Tensor:
    """Penalize the robot for raising a foot during the pre-lift stationary phase.

    Only active before the box is lifted — once walking with the box, feet naturally
    leave the ground as part of the gait, so the penalty is gated to zero.
    Returns 0.0 when both feet are grounded OR when the box is already lifted.
    Returns -1.0 when a foot is in the air during the reaching/grasping phase.
    """
    from isaaclab.sensors import ContactSensor
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # Use net_forces_w (always available) rather than net_forces_w_history
    # Shape: (num_envs, num_tracked_bodies, 3)
    net_forces = contact_sensor.data.net_forces_w[:, :, 2]  # z-component only

    # Check if each foot has contact (force > threshold)
    foot_contact = net_forces > min_contact_force
    both_feet_contact = foot_contact.all(dim=1)  # (num_envs,)

    # Gate by lift: once the box is lifted, penalty vanishes so walking is unrestricted
    carry_box = env.scene[object_cfg.name]
    box_height = carry_box.data.root_pos_w[:, 2]
    lift_mult = torch.clamp((box_height - table_height) / 0.10, 0.0, 1.0)
    pre_lift_gate = 1.0 - lift_mult  # 1 before lift, 0 after lift

    # Penalty of -1.0 if either foot is in the air, gated to 0 once box is lifted
    return (both_feet_contact.float() - 1.0) * pre_lift_gate


def pre_lift_locomotion_penalty(
    env: ManagerBasedRLEnv,
    table_height: float = 1.24,
    object_cfg: SceneEntityCfg = SceneEntityCfg("carry_box"),
) -> torch.Tensor:
    """Penalize non-zero locomotion actions during the pre-lift stationary phase.

    The high-level policy outputs locomotion velocity commands (actions[:, :3]) that
    drive the frozen low-level policy. Before the box is lifted, the robot should
    stand still. If it deliberately outputs non-zero locomotion commands to shift its
    weight/raise a leg to reach handles, this penalty discourages that exploit.

    Returns 0.0 when locomotion actions are zero OR box is already lifted.
    Returns negative value proportional to locomotion action magnitude before lift.
    """
    # Locomotion actions are the first 3 outputs of the high-level policy
    locomotion_actions = env.action_manager.action[:, :3]  # (num_envs, 3)
    locomotion_magnitude = torch.linalg.norm(locomotion_actions, dim=1)  # (num_envs,)

    # Gate by lift: no penalty once box is lifted (walking is expected)
    carry_box = env.scene[object_cfg.name]
    box_height = carry_box.data.root_pos_w[:, 2]
    lift_mult = torch.clamp((box_height - table_height) / 0.10, 0.0, 1.0)
    pre_lift_gate = 1.0 - lift_mult

    return -locomotion_magnitude * pre_lift_gate



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


def box_resting_on_arms_with_handles_tanh(
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


def upright_posture_pitch_penalty(
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
    """Reward robot being at target_dist from the box — ONLY active once the box is lifted.

    Gated to prevent the exploit of walking up to / leaning against the table to earn
    proximity reward without lifting the box.
    """
    gate = _box_is_lifted_gate(env, object_cfg=object_cfg)
    robot: RigidObject = env.scene[robot_cfg.name]
    carry_box: RigidObject = env.scene[object_cfg.name]
    dist = torch.linalg.norm(robot.data.root_pos_w[:, :2] - carry_box.data.root_pos_w[:, :2], dim=1)
    error = torch.abs(dist - target_dist)
    raw = 1.0 - torch.tanh(error / std)
    return gate * raw


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


def box_upright_raw_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("carry_box"),
) -> torch.Tensor:
    """Raw reward for the box being upright (no lift gating)."""
    carry_box: RigidObject = env.scene[object_cfg.name]
    up_axis = torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)
    up_axis[:, 2] = 1.0
    world_up = quat_apply(carry_box.data.root_quat_w.torch, up_axis)
    tilt_angle = torch.arccos(world_up[:, 2].clamp(-1.0 + 1e-6, 1.0 - 1e-6))
    return 1.0 - torch.tanh(tilt_angle / std)


def box_lift_above_table_gated(
    env: ManagerBasedRLEnv,
    table_height: float,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("carry_box"),
) -> torch.Tensor:
    """Reward for lifting the box above the table, gated by the quality of the handle grasp AND box uprightness."""
    raw_lift = box_lift_above_table_tanh(env, table_height, std, object_cfg)
    multiplier = handle_grasp_multiplier(env, object_cfg)
    
    # Gated by uprightness: if the box is tilted on the table, lift reward drops to 0
    upright_mult = box_upright_raw_tanh(env, std=0.15, object_cfg=object_cfg)
    
    # 0.4 bonus (equivalent to +2.0 to +3.2 reward depending on weight) when box is lifted > 5mm
    carry_box: RigidObject = env.scene[object_cfg.name]
    box_z = carry_box.data.root_pos_w.torch[:, 2]
    lift_height = box_z - table_height
    bonus = torch.where(lift_height > 0.005, 0.4, 0.0)
    
    return (raw_lift + bonus) * multiplier * upright_mult



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
    time_elapsed = env.episode_length_buf.float() * env.step_dt
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


def handle_grasp_multiplier(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("carry_box"),
) -> torch.Tensor:
    """Computes a multiplier in [0.0, 1.0] representing the quality of the handle grasp."""
    left_prox = hand_support_proximity_exp(env, "left", std=0.25, object_cfg=object_cfg)
    right_prox = hand_support_proximity_exp(env, "right", std=0.25, object_cfg=object_cfg)
    return left_prox * right_prox


def robot_turn_multiplier(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Computes a multiplier in [0.0, 1.0] based on whether the robot is facing away from the table.

    The starting heading is 0.0 (facing the table). The target heading is pi (facing away).
    Returns 1.0 when facing away, smoothly decaying to 0.0 when facing the table.
    """
    from isaaclab.utils.math import wrap_to_pi
    robot = env.scene[robot_cfg.name]
    # Compute yaw manually from the robot pelvis quaternion (w, x, y, z)
    quat = robot.data.root_quat_w.torch
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    # Heading error relative to target heading of pi
    heading_error = wrap_to_pi(3.1415926 - yaw)
    # Cosine multiplier: 1.0 when error is 0, 0.0 when error >= 90 degrees (pi/2)
    return torch.clamp(torch.cos(heading_error), min=0.0)


def box_moving_away_from_table_gated(
    env: ManagerBasedRLEnv,
    table_pos: tuple[float, float],
    std: float = 1.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("carry_box"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    raw_reward = box_moving_away_from_table(env, table_pos, std, object_cfg)
    grasp_mult = handle_grasp_multiplier(env, object_cfg)
    lift_mult = box_lift_above_table_tanh(env, table_height=1.24, std=0.05, object_cfg=object_cfg)
    turn_mult = robot_turn_multiplier(env, robot_cfg)
    return raw_reward * grasp_mult * lift_mult * turn_mult


def track_lin_vel_xy_exp_gated(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "base_velocity",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("carry_box"),
) -> torch.Tensor:
    from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import mdp as locomotion_mdp
    raw_reward = locomotion_mdp.track_lin_vel_xy_exp(env, std, command_name, robot_cfg)
    lift_mult = box_lift_above_table_gated(env, table_height=1.24, std=0.05, object_cfg=object_cfg)
    turn_mult = robot_turn_multiplier(env, robot_cfg)
    
    # locomotion gate: 0.0 before lift, turn_mult after lift
    locomotion_gate = lift_mult * turn_mult
    return raw_reward * locomotion_gate


def track_ang_vel_z_exp_gated(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "base_velocity",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("carry_box"),
) -> torch.Tensor:
    from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import mdp as locomotion_mdp
    raw_reward = locomotion_mdp.track_ang_vel_z_exp(env, std, command_name, robot_cfg)
    lift_mult = box_lift_above_table_gated(env, table_height=1.24, std=0.05, object_cfg=object_cfg)
    return raw_reward * lift_mult


def hand_height_asymmetry_penalty(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Penalize the difference in Z height of the two hands in the robot root frame."""
    left_hand_pos_b = carry_observations.hand_position_in_robot_root_frame(env, side="left")
    right_hand_pos_b = carry_observations.hand_position_in_robot_root_frame(env, side="right")
    
    # Absolute height difference (Z is index 2)
    z_diff = torch.abs(left_hand_pos_b[:, 2] - right_hand_pos_b[:, 2])
    return z_diff
