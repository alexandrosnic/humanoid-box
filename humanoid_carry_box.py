from __future__ import annotations

"""Standalone Isaac Lab demo that spawns an H1 humanoid next to a 5 kg box."""

"""Launch Isaac Sim first."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Spawn an H1 humanoid with a carry box in Isaac Lab.")
parser.add_argument("--steps", type=int, default=600, help="Number of physics steps to simulate before exiting.")
parser.add_argument("--box-mass", type=float, default=5.0, help="Mass of the box in kilograms.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext
from isaaclab_assets import H1_CFG


BOX_SIZE = (0.2, 0.2, 0.2)
BOX_START_POS = (0.55, 0.0, BOX_SIZE[2] / 2.0)


def design_scene() -> tuple[Articulation, RigidObject]:
    """Create the humanoid, the carry box, and the default scene assets."""
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    humanoid = Articulation(H1_CFG.replace(prim_path="/World/H1"))

    carry_box_cfg = RigidObjectCfg(
        prim_path="/World/CarryBox",
        spawn=sim_utils.CuboidCfg(
            size=BOX_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=args_cli.box_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.25, 0.15)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=BOX_START_POS),
    )
    carry_box = RigidObject(cfg=carry_box_cfg)

    return humanoid, carry_box


def reset_scene(humanoid: Articulation, carry_box: RigidObject) -> None:
    """Reset the robot and box to their default states."""
    joint_pos = humanoid.data.default_joint_pos.torch.clone()
    joint_vel = humanoid.data.default_joint_vel.torch.clone()
    humanoid.write_joint_position_to_sim_index(position=joint_pos)
    humanoid.write_joint_velocity_to_sim_index(velocity=joint_vel)

    root_pose = humanoid.data.default_root_pose.torch.clone()
    humanoid.write_root_pose_to_sim_index(root_pose=root_pose)
    root_vel = humanoid.data.default_root_vel.torch.clone()
    humanoid.write_root_velocity_to_sim_index(root_velocity=root_vel)
    humanoid.reset()

    box_pose = carry_box.data.default_root_pose.torch.clone()
    carry_box.write_root_pose_to_sim_index(root_pose=box_pose)
    box_vel = carry_box.data.default_root_vel.torch.clone()
    carry_box.write_root_velocity_to_sim_index(root_velocity=box_vel)
    carry_box.reset()


def run_simulator(sim: SimulationContext, humanoid: Articulation, carry_box: RigidObject) -> None:
    """Run a short standalone simulation."""
    sim_dt = sim.get_physics_dt()
    default_joint_targets = humanoid.data.default_joint_pos.torch.clone()
    elbow_ids, elbow_names = humanoid.find_joints(".*_elbow")
    print(f"[INFO] Controlling elbow joints: {list(elbow_names)}")

    step_count = 0
    while simulation_app.is_running() and step_count < args_cli.steps:
        if step_count == 0:
            reset_scene(humanoid, carry_box)

        joint_targets = default_joint_targets.clone()
        if len(elbow_ids) > 0:
            elbow_offset = 0.35 * math.sin(step_count * sim_dt * 2.0)
            joint_targets[:, elbow_ids] += elbow_offset

        humanoid.set_joint_position_target_index(target=joint_targets)
        humanoid.write_data_to_sim()
        carry_box.write_data_to_sim()

        sim.step()

        humanoid.update(sim_dt)
        carry_box.update(sim_dt)

        if step_count % 120 == 0:
            box_position = carry_box.data.root_pos_w.torch[0].tolist()
            print(f"[INFO] Step {step_count:04d} | box position = {box_position}")

        step_count += 1

    print("[INFO] Simulation finished.")


def main() -> None:
    """Create the simulation and run the demo."""
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.8, 1.4, 1.8], target=[0.0, 0.0, 0.9])

    humanoid, carry_box = design_scene()

    sim.reset()
    print("[INFO] Scene initialized.")
    run_simulator(sim, humanoid, carry_box)


if __name__ == "__main__":
    main()
    simulation_app.close()
