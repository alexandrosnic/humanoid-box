from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.flat_env_cfg import H1FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.rough_env_cfg import H1Rewards
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    CurriculumCfg as VelocityCurriculumCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import EventsCfg as VelocityEventsCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    ObservationsCfg as VelocityObservationsCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import TerminationsCfg as VelocityTerminationsCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import MySceneCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import mdp as locomotion_mdp

from isaaclab_assets import H1_CFG  # Use full config with complete collision geometry

from .low_level_policy_action import FrozenLocomotionPolicyActionCfg
from .mdp import curriculums as carry_curriculums
from .mdp import events as carry_events
from .mdp import observations as carry_observations
from .mdp import rewards as carry_rewards
from .mdp import terminations as carry_terminations


# Box dimensions: (x, y, z) - swap x and y so long side is perpendicular to arms
BOX_SIZE = (0.22, 0.32, 0.18)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCOMOTION_POLICY = str(PROJECT_ROOT / "artifacts" / "h1_flat_policy.pt")


def _resolve_locomotion_policy_path() -> str:
    return os.environ.get("H1_LOCOMOTION_POLICY_PATH", DEFAULT_LOCOMOTION_POLICY)


# ---------------------------------------------------------------------------
# Scene: box starts at waist height, right in front of the robot
# The H1's arm reach from standing is ~0.85m minimum, so 0.75m is reachable
# ---------------------------------------------------------------------------
@configclass
class H1CarryBoxSceneCfg(MySceneCfg):
    # Note: Ground plane is inherited from MySceneCfg
    # To customize ground appearance, modify the parent class or use terrain config
    
    carry_box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CarryBox",
        # Box long side (0.32m) is now in y-direction (perpendicular to arms)
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.35, 0.0, 1.3), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=sim_utils.CuboidCfg(
            size=BOX_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                enable_gyroscopic_forces=True,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),  # Increased to 5kg as requested
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,  # Explicitly enable collision
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.25, 0.15)),
        ),
    )


@configclass
class LowLevelLocomotionObservationsCfg(ObsGroup):
    base_lin_vel = ObsTerm(func=locomotion_mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=locomotion_mdp.base_ang_vel)
    projected_gravity = ObsTerm(func=locomotion_mdp.projected_gravity)
    velocity_commands = ObsTerm(func=locomotion_mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos = ObsTerm(func=locomotion_mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=locomotion_mdp.joint_vel_rel)
    actions = ObsTerm(func=locomotion_mdp.last_action, params={"action_name": "locomotion"})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


# ---------------------------------------------------------------------------
# Actions: arms controlled by high-level policy, torso by locomotion
# ---------------------------------------------------------------------------
@configclass
class ActionsCfg:
    locomotion = FrozenLocomotionPolicyActionCfg(
        asset_name="robot",
        policy_path=_resolve_locomotion_policy_path(),
        low_level_observations=LowLevelLocomotionObservationsCfg(),
        full_joint_names=[".*"],
        controlled_joint_names=[
            ".*_hip_yaw",
            ".*_hip_roll",
            ".*_hip_pitch",
            ".*_knee",
            ".*_ankle",
            "torso",  # Added back - locomotion policy controls torso yaw for stability
        ],
        low_level_decimation=4,
        policy_output_scale=0.5,
    )
    arms = locomotion_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            ".*_shoulder_pitch",
            ".*_shoulder_roll",
            ".*_shoulder_yaw",
            ".*_elbow",
        ],
        scale=0.15,
        use_default_offset=False,  # Use explicit offsets below
        # Explicit offsets: arms forward, elbows bent to cradle box
        # Order matches joint_names: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow (×2 sides)
        offset={
            "left_shoulder_pitch": -0.4,   # upper arm slightly forward-down
            "right_shoulder_pitch": -0.4,  # upper arm slightly forward-down
            "left_shoulder_roll": -0.25,   # rolled inward to narrow arm gap
            "right_shoulder_roll": 0.25,   # rolled inward to narrow arm gap
            "left_shoulder_yaw": 0.0,      # elbows pointing down/forward
            "right_shoulder_yaw": 0.0,     # elbows pointing down/forward
            "left_elbow": -0.1,             # bent ~70 degrees to form a tray
            "right_elbow": -0.1,            # bent ~70 degrees to form a tray
        },
    )


# ---------------------------------------------------------------------------
# Observations: unchanged (all still useful for the policy)
# ---------------------------------------------------------------------------
@configclass
class H1CarryBoxObservationsCfg(VelocityObservationsCfg):
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=locomotion_mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=locomotion_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=locomotion_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=locomotion_mdp.generated_commands, params={"command_name": "base_velocity"})
        arm_joint_pos = ObsTerm(
            func=locomotion_mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*", ".*_elbow"])},
        )
        arm_joint_vel = ObsTerm(
            func=locomotion_mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*", ".*_elbow"])},
        )
        locomotion_actions = ObsTerm(func=locomotion_mdp.last_action, params={"action_name": "locomotion"})
        arm_actions = ObsTerm(func=locomotion_mdp.last_action, params={"action_name": "arms"})
        box_position = ObsTerm(func=carry_observations.box_position_in_robot_root_frame)
        box_position_torso = ObsTerm(func=carry_observations.box_position_in_torso_frame)
        desired_box_position = ObsTerm(func=carry_observations.desired_box_position_in_robot_root_frame)
        left_hand_position = ObsTerm(func=carry_observations.hand_position_in_robot_root_frame, params={"side": "left"})
        right_hand_position = ObsTerm(func=carry_observations.hand_position_in_robot_root_frame, params={"side": "right"})
        left_hand_to_box = ObsTerm(func=carry_observations.hand_to_box_vector_in_robot_root_frame, params={"side": "left"})
        right_hand_to_box = ObsTerm(func=carry_observations.hand_to_box_vector_in_robot_root_frame, params={"side": "right"})
        left_hand_to_support = ObsTerm(
            func=carry_observations.hand_to_box_support_vector_in_robot_root_frame,
            params={"side": "left"},
        )
        right_hand_to_support = ObsTerm(
            func=carry_observations.hand_to_box_support_vector_in_robot_root_frame,
            params={"side": "right"},
        )
        box_lin_vel = ObsTerm(
            func=carry_observations.box_linear_velocity_in_robot_root_frame, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        box_ang_vel = ObsTerm(
            func=carry_observations.box_angular_velocity_in_robot_root_frame, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        box_height = ObsTerm(func=carry_observations.box_height)
        box_upright = ObsTerm(func=carry_observations.box_upright_projection)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ---------------------------------------------------------------------------
# Events: box resets on the ground in front of the robot
# ---------------------------------------------------------------------------
@configclass
class H1CarryBoxEventsCfg(VelocityEventsCfg):
    carry_box_material = EventTerm(
        func=locomotion_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("carry_box"),
            "static_friction_range": (0.7, 1.2),
            "dynamic_friction_range": (0.6, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 32,
        },
    )
    carry_box_mass = EventTerm(
        func=locomotion_mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("carry_box"),
            "mass_distribution_params": (0.85, 1.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    reset_base = EventTerm(
        func=carry_events.reset_robot_root_for_carry,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.03, 0.03),
                "z": (-0.01, 0.01),
                "yaw": (-0.15, 0.15),
            },
            "velocity_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.02, 0.02),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.1, 0.1),
            },
        },
    )
    reset_carry_box = EventTerm(
        func=carry_events.reset_box_in_carry_pose,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("carry_box"),
            # Box moved closer (0.25m) and lower (0.32m) to rest on forearms
            "box_pos_in_robot_frame": (0.25, 0.0, 0.32),
            "pose_range": {
                "x": (-0.03, 0.03),
                "y": (-0.03, 0.03),
                "z": (-0.01, 0.01),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.05, 0.05),
            },
            "velocity_range": {
                "x": (-0.02, 0.02),
                "y": (-0.02, 0.02),
                "z": (-0.02, 0.02),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.05, 0.05),
            },
        },
    )


# ---------------------------------------------------------------------------
# Rewards: two-phase design (BOTTOM-CARRY approach)
#   Phase 1 (always active): Get arms under the box to cradle it
#   Phase 2 (ramped via curriculum): Lift and Carry
# ---------------------------------------------------------------------------
@configclass
class H1CarryBoxRewards(H1Rewards):
    # -- Regularization --
    joint_deviation_arms = RewTerm(
        func=locomotion_mdp.joint_deviation_l1,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_pitch", ".*_shoulder_roll", ".*_elbow"])},
    )
    # Penalize fast arm movements to make carrying smooth and stable
    joint_vel_arms = RewTerm(
        func=locomotion_mdp.joint_vel_l2,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*", ".*_elbow"])},
    )
    # Strong penalty to prevent arms from swinging behind the body
    shoulder_yaw_penalty = RewTerm(
        func=locomotion_mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_yaw"])},
    )
    # Reward for keeping hands in front of the robot (positive x in robot frame)
    hands_in_front = RewTerm(
        func=carry_rewards.hands_in_front_of_robot,
        params={"std": 0.2},
        weight=1.0,
    )

    # -- Phase 1: Bottom-carry grasp (always active) --
    # Reward hands for being UNDER the box (cradle position)
    hands_under_box = RewTerm(
        func=carry_rewards.hands_under_box_tanh,
        params={"std": 0.2},
        weight=1.5,
    )
    # Reward box for resting on top of the arms
    box_on_arms = RewTerm(
        func=carry_rewards.box_resting_on_arms_tanh,
        params={"std": 0.2},
        weight=1.0,
    )
    # Keep box close to torso (reduces tipping moment)
    box_close_to_torso = RewTerm(
        func=carry_rewards.box_close_to_torso_tanh,
        params={"std": 0.15},
        weight=1.0,
    )
    # Keep hands symmetric (centered under box)
    symmetric_grasp = RewTerm(
        func=carry_rewards.symmetric_grasp_tanh,
        params={"std": 0.15},
        weight=1.5,
    )
    # Penalize box motion during grasp phase
    box_still_during_grasp = RewTerm(
        func=carry_rewards.box_velocity_l2,
        weight=-0.02,
    )

    # -- Phase 2: Lift and Carry (ramped up via curriculum) --
    box_lift = RewTerm(
        func=carry_rewards.box_height_above_minimum,
        params={"minimum_height": 0.35},
        weight=0.0,
    )
    box_upright = RewTerm(
        func=carry_rewards.box_upright_tanh,
        params={"std": 0.05},  # Tighten to ~3 degrees for precise horizontal alignment
        weight=3.0,            # Increase weight to strongly enforce keeping the box flat
    )
    box_carry_velocity = RewTerm(
        func=carry_rewards.box_carry_velocity_tracking_exp,
        params={"std": 0.25},
        weight=0.0,
    )
    # Upright posture: penalize robot from leaning forward (pitch)
    # This prevents the box weight from pulling the robot forward
    upright_posture = RewTerm(
        func=carry_rewards.upright_posture_penalty,
        params={"std": 0.10},  # Tightened from 0.15 to be more sensitive to forward lean
        weight=3.0,            # Increased from 1.5 to prioritize balance under load
    )
    # Note: Removed torso_lean_back and torso_forward_penalty
    # The H1's torso joint is YAW (left/right rotation), not PITCH (forward/back)
    # It cannot help with forward/backward balance when carrying a box
    # Hold duration: reward for each timestep the box is held above minimum height
    # This explicitly incentivizes longer holds
    box_hold_duration = RewTerm(
        func=carry_rewards.box_height_above_minimum,
        params={"minimum_height": 0.3},  # Box must be above 0.3m to count as "held"
        weight=0.1,  # Reduced to prevent survival/standing-still bias
    )


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------
@configclass
class H1CarryBoxTerminationsCfg(VelocityTerminationsCfg):
    carry_box_dropped = DoneTerm(
        func=locomotion_mdp.root_height_below_minimum,
        params={"minimum_height": 0.03, "asset_cfg": SceneEntityCfg("carry_box")},
    )
    carry_box_too_far = DoneTerm(func=carry_terminations.box_too_far_from_robot, params={"max_distance": 1.25})
    # Terminate episode when robot falls (root height below threshold)
    robot_fallen = DoneTerm(
        func=locomotion_mdp.root_height_below_minimum,
        params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("robot")},  # Robot falls if height < 0.5m
    )


# ---------------------------------------------------------------------------
# Curriculum: phased introduction of lift and carry rewards
# ---------------------------------------------------------------------------
@configclass
class H1CarryBoxCurriculumCfg(VelocityCurriculumCfg):
    box_lift = CurrTerm(
        func=carry_curriculums.ramp_reward_weight,
        params={"term_name": "box_lift", "start_step": 3000, "end_step": 6000, "target_weight": 0.5},
    )
    box_carry_velocity = CurrTerm(
        func=carry_curriculums.ramp_reward_weight,
        params={"term_name": "box_carry_velocity", "start_step": 3000, "end_step": 8000, "target_weight": 2.5}, # Reduced from 4.0 to prevent sprinting/tipping forward
    )
    # Locomotion rewards - moderate weights to encourage walking without causing "dancing"
    track_lin_vel_xy = CurrTerm(
        func=carry_curriculums.ramp_reward_weight,
        params={"term_name": "track_lin_vel_xy_exp", "start_step": 3000, "end_step": 8000, "target_weight": 2.5}, # Reduced from 3.5 to balance velocity tracking vs upright posture
    )
    track_ang_vel_z = CurrTerm(
        func=carry_curriculums.ramp_reward_weight,
        params={"term_name": "track_ang_vel_z_exp", "start_step": 3000, "end_step": 8000, "target_weight": 0.75},
    )
    feet_air_time = CurrTerm(
        func=carry_curriculums.ramp_reward_weight,
        params={"term_name": "feet_air_time", "start_step": 3000, "end_step": 8000, "target_weight": 0.75},
    )
    carry_box_mass = CurrTerm(
        func=carry_curriculums.increase_box_mass_range,
        params={
            "term_name": "carry_box_mass",
            "stages": (
                (3000, (0.9, 1.05)),
                (10000, (1.0, 1.15)),
                (20000, (1.0, 1.25)),
            ),
        },
    )


# ---------------------------------------------------------------------------
# Main environment config
# ---------------------------------------------------------------------------
@configclass
class H1CarryBoxEnvCfg(H1FlatEnvCfg):
    scene: H1CarryBoxSceneCfg = H1CarryBoxSceneCfg(num_envs=128, env_spacing=2.5)
    actions: ActionsCfg = ActionsCfg()
    observations: H1CarryBoxObservationsCfg = H1CarryBoxObservationsCfg()
    rewards: H1CarryBoxRewards = H1CarryBoxRewards()
    terminations: H1CarryBoxTerminationsCfg = H1CarryBoxTerminationsCfg()
    events: H1CarryBoxEventsCfg = H1CarryBoxEventsCfg()
    curriculum: H1CarryBoxCurriculumCfg = H1CarryBoxCurriculumCfg()

    # Target positions (in robot base frame)
    # Box moved closer (0.25m) and lower (0.32m) to rest on forearms
    carry_box_target_pos_b = (0.25, 0.0, 0.32)
    # Hand offsets match new inward box spacing (0.292m in y-direction)
    carry_box_left_hand_target_offset_b = (0.0, 0.146, 0.0)
    carry_box_right_hand_target_offset_b = (0.0, -0.146, 0.0)
    carry_box_torso_target_pos_t = (0.2, 0.0, -0.1)

    def __post_init__(self):
        super().__post_init__()

        # Disable base_contact termination to prevent resets when the box touches the torso
        self.terminations.base_contact = None

        # Create robot config with custom initial joint positions
        # Use deepcopy to ensure we're not modifying a shared reference
        import copy
        
        # Deep copy the full H1_CFG (has complete collision geometry on arms)
        robot_cfg = copy.deepcopy(H1_CFG)
        robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
        
        # Enable collision on arm links so the box doesn't pass through
        # Ensure the articulation has proper collision properties
        if robot_cfg.spawn.articulation_props is None:
            robot_cfg.spawn.articulation_props = sim_utils.ArticulationRootPropertiesCfg()
        robot_cfg.spawn.articulation_props.self_collisions = True
        
        # Clear existing joint positions and set our own
        # Use EXPLICIT joint names instead of regex to ensure they match
        # Keep the crucial leg and torso joint positions so the locomotion policy operates correctly,
        # and set the arms to a forward cradle position.
        robot_cfg.init_state.joint_pos = {
            # Legs & Torso (locomotion defaults)
            ".*_hip_yaw": 0.0,
            ".*_hip_roll": 0.0,
            ".*_hip_pitch": -0.28,
            ".*_knee": 0.79,
            ".*_ankle": -0.52,
            "torso": 0.0,
            # Arms (custom carriage posture)
            "left_shoulder_pitch": -0.4,
            "right_shoulder_pitch": -0.4,
            "left_shoulder_yaw": 0.0,
            "right_shoulder_yaw": 0.0,
            "left_elbow": -0.1,
            "right_elbow": -0.1,
            "left_shoulder_roll": -0.25,  # rolled inward
            "right_shoulder_roll": 0.25,   # rolled inward
        }
        
        self.scene.robot = robot_cfg
        
        # Debug: print to verify config is being set (use stderr for visibility)
        import sys
        print("=" * 60, file=sys.stderr)
        print("[DEBUG] Robot init_state.joint_pos:", file=sys.stderr)
        for k, v in self.scene.robot.init_state.joint_pos.items():
            print(f"  {k}: {v}", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        sys.stderr.flush()

        self.scene.num_envs = 128
        self.episode_length_s = 15.0

        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0  # Force moving during training
        self.commands.base_velocity.ranges.lin_vel_x = (0.15, 0.35)  # Require forward walking speed
        self.commands.base_velocity.ranges.lin_vel_y = (-0.05, 0.05)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.2, 0.2)

        # Locomotion rewards start at 0, ramped up via curriculum (Phase 2)
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.track_lin_vel_xy_exp.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0
        self.rewards.action_rate_l2.weight = -0.0025
        self.rewards.dof_acc_l2.weight = -1.25e-7
        
        # Override termination penalty to encourage exploration of walking motions
        self.rewards.termination_penalty.weight = -50.0

        self.curriculum.terrain_levels = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None


# ---------------------------------------------------------------------------
# Hold variant: standing still, no velocity commands
# ---------------------------------------------------------------------------
@configclass
class H1CarryBoxHoldEnvCfg(H1CarryBoxEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 12.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # No carry velocity reward for hold variant
        self.rewards.box_carry_velocity.weight = 0.0

        self.events.carry_box_mass = None
        self.curriculum.carry_box_mass = None
        self.curriculum.track_lin_vel_xy = None
        self.curriculum.track_ang_vel_z = None
        self.curriculum.feet_air_time = None
        self.curriculum.box_carry_velocity = None
        self.terminations.carry_box_too_far = None


# ---------------------------------------------------------------------------
# Play variants: deterministic, no randomization
# ---------------------------------------------------------------------------
@configclass
class H1CarryBoxEnvCfg_PLAY(H1CarryBoxEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 4
        self.scene.env_spacing = 3.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.2)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
        self.events.carry_box_mass = None
        self.curriculum.carry_box_mass = None
        self.events.reset_base.params["pose_range"] = {}
        self.events.reset_base.params["velocity_range"] = {}
        self.events.reset_carry_box.params["pose_range"] = {}
        self.events.reset_carry_box.params["velocity_range"] = {}


@configclass
class H1CarryBoxHoldEnvCfg_PLAY(H1CarryBoxHoldEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 4
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False
        self.events.reset_base.params["pose_range"] = {}
        self.events.reset_base.params["velocity_range"] = {}
        self.events.reset_carry_box.params["pose_range"] = {}
        self.events.reset_carry_box.params["velocity_range"] = {}