from __future__ import annotations

import os
from pathlib import Path

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import CommandTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from .h1_carry_box_env_cfg import H1CarryBoxEnvCfg, H1CarryBoxSceneCfg, H1CarryBoxRewards, H1CarryBoxEventsCfg, H1CarryBoxTerminationsCfg, H1CarryBoxCurriculumCfg
from .mdp import rewards as carry_rewards
from .mdp import events as carry_events
from .mdp import terminations as carry_terminations
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import mdp as locomotion_mdp


PROJECT_ROOT = Path(__file__).resolve().parents[1]


_fsm_command_class = None

def get_fsm_command_class():
    """Lazy-loader factory to define the FSM command class after AppLauncher has initialized Isaac Sim.
    This prevents duplicate pxr python converter registrations which cause fatal crashes.
    """
    global _fsm_command_class
    if _fsm_command_class is None:
        from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
        import isaaclab.utils.math as math_utils

        class FsmTableLiftVelocityCommand(UniformVelocityCommand):
            """Custom command generator implementing Finite State Machine logic for table lift and carry.
            
            - Phase 0 (Box not lifted): Commands a slow, straight walk forward to approach the table.
            - Phase 1 (Box lifted): Commands a 180-degree pivot turn away from the table.
            - Phase 2 (Turn complete): Commands a forward carry walk away from the table.
            """
            def _update_command(self):
                # Call standard controller updates (resolves heading errors, etc.)
                super()._update_command()
                
                # Get carry box states
                carry_box = self._env.scene["carry_box"]
                box_z = carry_box.data.root_pos_w.torch[:, 2]
                
                # Check if the box is lifted above its resting height on the table (1.24m)
                is_lifted = box_z >= 1.26
                
                # Phase 0: Approach
                not_lifted_ids = (~is_lifted).nonzero(as_tuple=False).flatten()
                if len(not_lifted_ids) > 0:
                    # Calculate XY distance between robot root and box
                    robot_pos_w = self.robot.data.root_pos_w.torch[not_lifted_ids]
                    box_pos_w = carry_box.data.root_pos_w.torch[not_lifted_ids]
                    dist_xy = torch.linalg.norm(robot_pos_w[:, :2] - box_pos_w[:, :2], dim=1)
                    
                    # Stop forward walking once within 0.38m of the box center
                    vel_x = torch.where(dist_xy < 0.38, 0.0, 0.12)
                    
                    self.vel_command_b[not_lifted_ids, 0] = vel_x
                    self.vel_command_b[not_lifted_ids, 1] = 0.0
                    self.vel_command_b[not_lifted_ids, 2] = 0.0   # Walk straight
                    
                # Phase 1 & 2: Turn and Carry Away
                lifted_ids = is_lifted.nonzero(as_tuple=False).flatten()
                if len(lifted_ids) > 0:
                    # Set target heading to pi (180 degrees opposite to starting heading)
                    self.heading_target[lifted_ids] = 3.1415926
                    
                    # Recompute angular velocity wz towards target heading
                    heading_error = math_utils.wrap_to_pi(
                        self.heading_target[lifted_ids] - self.robot.data.heading_w.torch[lifted_ids]
                    )
                    self.vel_command_b[lifted_ids, 2] = torch.clip(
                        self.cfg.heading_control_stiffness * heading_error,
                        min=self.cfg.ranges.ang_vel_z[0],
                        max=self.cfg.ranges.ang_vel_z[1],
                    )
                    
                    # Check if the pivot turn is mostly complete (error < 45 degrees / 0.8 rad)
                    is_turned = torch.abs(heading_error) < 0.8
                    forward_ids = lifted_ids[is_turned]
                    turning_ids = lifted_ids[~is_turned]
                    
                    if len(forward_ids) > 0:
                        self.vel_command_b[forward_ids, 0] = 0.20  # Walk forward carrying box
                        self.vel_command_b[forward_ids, 1] = 0.0
                    if len(turning_ids) > 0:
                        self.vel_command_b[turning_ids, 0] = 0.0   # Pivot in place
                        self.vel_command_b[turning_ids, 1] = 0.0
        
        _fsm_command_class = FsmTableLiftVelocityCommand
    return _fsm_command_class


class LazyFsmCommandClassWrapper(CommandTerm):
    """Lazy class wrapper to prevent importing velocity_command (which loads pxr) 
    before SimulationApp is started by the launcher.
    """
    def __new__(cls, cfg, env):
        real_class = get_fsm_command_class()
        return real_class(cfg, env)




@configclass
class H1LiftCarrySceneCfg(H1CarryBoxSceneCfg):
    # Table setup (static obstacle)
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.575), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 0.6, 1.15),
            rigid_props=None,  # None means static collider
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
        ),
    )

    # Carry box with handles USD asset
    carry_box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CarryBox",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, 0.0, 1.24), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(PROJECT_ROOT, "artifacts", "box_with_handles.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                enable_gyroscopic_forces=True,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
        ),
    )


@configclass
class H1LiftCarryRewards(H1CarryBoxRewards):
    # Reward for approaching the table (peaks at target distance 0.35m)
    robot_to_box_dist = RewTerm(
        func=carry_rewards.robot_to_box_distance_target_tanh,
        params={"target_dist": 0.35, "std": 0.25},
        weight=2.0000,
    )
    
    # Override box_lift to reward raising the box above table height (0.8m)
    box_lift = RewTerm(
        func=carry_rewards.box_lift_above_table_gated,
        params={"table_height": 1.24, "std": 0.15},
        weight=5.0000,  # Ramped via curriculum
    )

    # Reward for carrying the box away from the table (gated by handle grasp)
    box_carry_away = RewTerm(
        func=carry_rewards.box_moving_away_from_table_gated,
        params={"table_pos": (0.55, 0.0), "std": 1.0},
        weight=0.0,  # Ramped via curriculum
    )

    # Reward left hand for being close to the left handle target offset.
    # std=0.25: broad enough to provide gradient even when elbow starts 0.43m from handle.
    # above_penalty removed: the coefficient 15.0 was killing all reward signal before
    # the arm could learn to reach forward (any tiny z>target gave tanh->1 -> reward=0).
    left_hand_to_handle = RewTerm(
        func=carry_rewards.hand_support_proximity_exp,
        params={"side": "left", "std": 0.25},
        weight=3.0000,
    )
    
    # Reward right hand for being close to the right handle target offset.
    right_hand_to_handle = RewTerm(
        func=carry_rewards.hand_support_proximity_exp,
        params={"side": "right", "std": 0.25},
        weight=3.0000,
    )

    # Penalize if either foot leaves the ground (discourages one-leg balancing exploit)
    # Weight must be large enough to compete with hand proximity rewards (~6.0 max).
    # Force threshold lowered to 1.0N so it fires reliably even during quiet standing.
    both_feet_grounded = RewTerm(
        func=carry_rewards.both_feet_grounded_penalty,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
            "min_contact_force": 1.0,
        },
        weight=0.0,
    )

    # Penalize non-zero locomotion actions before the box is lifted.
    # Root cause fix: the high-level policy learns to output non-zero velocity commands
    # to make the low-level locomotion policy raise a leg and shift weight forward.
    # This penalty removes the incentive by directly penalising locomotion action magnitude.
    pre_lift_locomotion = RewTerm(
        func=carry_rewards.pre_lift_locomotion_penalty,
        params={},
        weight=0.0,
    )

    # Disable legacy bottom-carry grasp rewards to focus 100% on the handles
    hands_under_box = RewTerm(
        func=carry_rewards.hands_under_box_tanh,
        params={"std": 0.1},
        weight=0.0,
    )
    box_on_arms = RewTerm(
        func=carry_rewards.box_resting_on_arms_with_handles_tanh,
        params={"std": 0.1},
        weight=0.0,
    )
    
    # Disable hands_in_front as it is no longer needed for handles
    hands_in_front = RewTerm(
        func=carry_rewards.hands_in_front_of_robot,
        params={"std": 0.2},
        weight=0.0,
    )

    # Override box_still_during_grasp to only penalize z-velocity and only while on table
    box_still_during_grasp = RewTerm(
        func=carry_rewards.box_z_velocity_on_table_l2,
        weight=-0.02,
    )

    # Penalize hand height asymmetry (Z-axis difference) directly
    hand_height_asymmetry = RewTerm(
        func=carry_rewards.hand_height_asymmetry_penalty,
        weight=0.0,
    )

    # Gated locomotion velocity rewards (only active if holding both handles)
    track_lin_vel_xy_exp = RewTerm(
        func=carry_rewards.track_lin_vel_xy_exp_gated,
        params={"std": 0.5, "command_name": "base_velocity"},
        weight=0.0,  # Ramped via curriculum
    )
    track_ang_vel_z_exp = RewTerm(
        func=carry_rewards.track_ang_vel_z_exp_gated,
        params={"std": 0.5, "command_name": "base_velocity"},
        weight=0.0,  # Ramped via curriculum
    )

    # feet_air_time is disabled via post_init below


@configclass
class H1LiftCarryEventsCfg(H1CarryBoxEventsCfg):
    # Override robot reset to introduce counterfactual body tilts (extreme resets)
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
            "extreme_reset_prob": 0.2,  # 20% chance of severe body tilts
        },
    )

    # Override box reset to place it on the table with off-center offsets (extreme resets)
    reset_carry_box = EventTerm(
        func=carry_events.reset_box_on_table,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("carry_box"),
            "table_pos": (0.55, 0.0, 1.24),
            "pose_range": {
                "x": (-0.02, 0.02),
                "y": (-0.02, 0.02),
                "z": (0.0, 0.0),
                "roll": (-0.02, 0.02),
                "pitch": (-0.02, 0.02),
                "yaw": (-0.05, 0.05),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "extreme_reset_prob": 0.2,  # 20% chance of severe box offsets/tilts on table
        },
    )


@configclass
class H1LiftCarryTerminationsCfg(H1CarryBoxTerminationsCfg):
    # Terminate early if the box falls below 1.05m (dropped off the 1.15m table)
    carry_box_dropped = DoneTerm(
        func=locomotion_mdp.root_height_below_minimum,
        params={"minimum_height": 1.05, "asset_cfg": SceneEntityCfg("carry_box")},
    )
    # Terminate early if the robot's pelvis falls below 0.65m (robot fell down)
    robot_fallen = DoneTerm(
        func=locomotion_mdp.root_height_below_minimum,
        params={"minimum_height": 0.65, "asset_cfg": SceneEntityCfg("robot")},
    )
    # Terminate early if the robot tilts/leans too far forward or backward (exploit prevention)
    robot_bad_orientation = DoneTerm(
        func=locomotion_mdp.bad_orientation,
        params={"limit_angle": 0.52, "asset_cfg": SceneEntityCfg("robot")},
    )
    # Terminate if the robot's torso leans on the table (cheating exploit).
    # Knees and hips are intentionally excluded — they are naturally close to the table
    # when standing at lifting distance and should not trigger a reset.
    robot_body_table_contact = DoneTerm(
        func=locomotion_mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "torso_link",
                ],
            ),
            "threshold": 1.0,
        },
    )
    # Terminate early if the robot fails to lift the box within 5.0 seconds (250 steps)
    box_not_lifted_timeout = DoneTerm(
        func=carry_terminations.box_not_lifted_timeout,
        params={"timeout_time": 5.0, "minimum_height": 1.26},
    )
    
    # Terminate early if either hand/elbow drops below table height (1.15m) to prevent table leaning
    arms_below_table = DoneTerm(
        func=carry_terminations.arms_below_table_limit,
        params={"min_height": 1.16, "start_step": 15},
    )


@configclass
class H1LiftCarryCurriculumCfg(H1CarryBoxCurriculumCfg):
    # Ramp up the box lift reward to 5.0 over 2,000 steps (83 iterations)
    box_lift = CurrTerm(
        func=locomotion_mdp.modify_reward_weight,
        params={"term_name": "box_lift", "weight": 5.0000, "num_steps": 2000},
    )
    # Ramp up the gated carry-away reward to 3.0 over 2,000 steps
    box_carry_away = CurrTerm(
        func=locomotion_mdp.modify_reward_weight,
        params={"term_name": "box_carry_away", "weight": 3.0000, "num_steps": 2000},
    )
    # Ramp up the gated forward walking velocity reward to 2.5 over 2,000 steps
    track_lin_vel_xy = CurrTerm(
        func=locomotion_mdp.modify_reward_weight,
        params={"term_name": "track_lin_vel_xy_exp", "weight": 2.5, "num_steps": 2000},
    )
    # Ramp up the gated angular yaw velocity reward to 0.75 over 2,000 steps
    track_ang_vel_z = CurrTerm(
        func=locomotion_mdp.modify_reward_weight,
        params={"term_name": "track_ang_vel_z_exp", "weight": 0.75, "num_steps": 2000},
    )
    # Disable feet_air_time curriculum
    feet_air_time = None


@configclass
class H1LiftCarryEnvCfg(H1CarryBoxEnvCfg):
    scene: H1LiftCarrySceneCfg = H1LiftCarrySceneCfg(num_envs=128, env_spacing=2.5)
    rewards: H1LiftCarryRewards = H1LiftCarryRewards()
    terminations: H1LiftCarryTerminationsCfg = H1LiftCarryTerminationsCfg()
    events: H1LiftCarryEventsCfg = H1LiftCarryEventsCfg()
    curriculum: H1LiftCarryCurriculumCfg = H1LiftCarryCurriculumCfg()

    # The box handles are at y = +/-0.19m on the box sides.
    # z = -0.04 from box centre (1.24m) = 1.20m — hands must be BELOW this to grab from underneath.
    # The arm init pose below ensures hands start below handle height on spawn.
    carry_box_left_hand_target_offset_b = (0.0, 0.19, -0.04)
    carry_box_right_hand_target_offset_b = (0.0, -0.19, -0.04)

    # Box carry target after lifting: torso height (1.05m)
    carry_box_target_pos_b = (0.25, 0.0, 1.05)

    def __post_init__(self):
        super().__post_init__()

        # Spawn robot directly in front of the table (handles are at 0.55m, robot is at 0.12m, distance 0.43m)
        # Extra clearance so knees don't contact the table on spawn and during the pivot turn
        self.scene.robot.init_state.pos = (0.12, 0.0, 1.05)

        # Disable base_contact termination to prevent resets when touching the table
        self.terminations.base_contact = None

        # Disable feet air time for the high-level task policy (locomotion is frozen)
        self.rewards.feet_air_time.weight = 0.0
        self.curriculum.feet_air_time = None

        # Override the command manager to use our custom FSM velocity command generator
        self.commands.base_velocity.class_type = LazyFsmCommandClassWrapper
        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.heading_control_stiffness = 1.0

        # Arm init: shoulder_pitch=-0.5 + elbow=1.6.
        # From the forearm-direction scan:
        #   pitch=-1.0: forearm TIP reaches x=0.61m (env) -- INSIDE the box at 0.55m -> EXPLODES
        #   pitch=-0.5: forearm TIP at x=0.37m -- 10cm short of box front (0.475m) -> SAFE
        # Forearm direction at pitch=-0.5: fdir_x=0.638, fdir_z=-0.749
        #   -> upper arm goes forward-and-down, forearm then bends MORE downward
        #   -> creates a visible elbow 'L' shape, arms are NOT just straight
        # Arm COM: z_w=1.21m (at handle height), x_b=0.18m (reaching forward)
        # roll=0.25: spreads arms to y=0.24m (handles are at y=+/-0.19m, close match)
        self.scene.robot.init_state.joint_pos.update({
            "left_shoulder_pitch": -0.5,
            "right_shoulder_pitch": -0.5,
            "left_shoulder_roll": 0.25,
            "right_shoulder_roll": -0.25,
            "left_shoulder_yaw": 0.0,
            "right_shoulder_yaw": 0.0,
            "left_elbow": 1.6,
            "right_elbow": 1.6,
        })

        # CRITICAL: Override arm action offsets to match the new init pose.
        # With use_default_offset=False, arm target = action * scale + offset.
        # Offsets must match init pose so policy starts at zero action near the handles.
        self.actions.arms.offset = {
            "left_shoulder_pitch": -0.5,
            "right_shoulder_pitch": -0.5,
            "left_shoulder_roll": 0.25,
            "right_shoulder_roll": -0.25,
            "left_shoulder_yaw": 0.0,
            "right_shoulder_yaw": 0.0,
            "left_elbow": 1.6,
            "right_elbow": 1.6,
        }

        # Reduce joint_deviation_arms weight: the H1 USD default poses (shoulder_pitch≈0.28)
        # differ from our new init (shoulder_pitch=1.2). The deviation penalty is computed
        # relative to the articulation's USD defaults, so it would fight the new reaching
        # configuration. Keep it small (near zero) just for regularisation.
        self.rewards.joint_deviation_arms.weight = -0.001

        # Set standard range limits for the underlying sampler
        self.commands.base_velocity.ranges.lin_vel_x = (-0.25, 0.25)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)

        # 1. Increase arm action scale to 0.25 to provide stronger lifting targets/range
        self.actions.arms.scale = 0.25

        # 2. Box Mass Curriculum: Start at ~1.0kg - 1.5kg (20-30% of 5kg) and ramp to 5.0kg
        self.events.carry_box_mass.params["mass_distribution_params"] = [0.2, 0.3]
        self.curriculum.carry_box_mass.params["stages"] = [
            [3000, [0.2, 0.4]],    # up to 2.0 kg
            [10000, [0.4, 0.6]],   # up to 3.0 kg
            [20000, [0.6, 0.8]],   # up to 4.0 kg
            [30000, [0.8, 1.0]],   # up to 5.0 kg
        ]


@configclass
class H1LiftCarryEnvCfg_PLAY(H1LiftCarryEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 4
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False
        self.events.carry_box_mass = None
        self.curriculum.carry_box_mass = None
        self.events.reset_base.params["pose_range"] = {}
        self.events.reset_base.params["velocity_range"] = {}
        self.events.reset_carry_box.params["pose_range"] = {}
        self.events.reset_carry_box.params["velocity_range"] = {}
