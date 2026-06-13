import sys
import argparse
import time
from isaaclab.app import AppLauncher

# Set up the AppLauncher to support GUI launch
parser = argparse.ArgumentParser(description="Visualize H1 carriage pose without a policy.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Force the GUI to open by default
args_cli.headless = False
args_cli.visualizer = "kit"

# Launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# Make sure our custom environment is registered
sys.path.insert(0, r"C:\Users\alexa\projects\humanoid_training")
import h1_carry_box_task

task_name = "Isaac-H1-Carry-Box-Play-v0"

def main():
    print("\nLoading configuration...")
    env_cfg = parse_env_cfg(task_name, num_envs=1, device="cpu")
    # Disable observations noise to keep things clean
    env_cfg.observations.policy.enable_corruption = False
    
    print("Creating environment...")
    env = gym.make(task_name, cfg=env_cfg)
    
    print("\n" + "="*60)
    print("VISUALIZING POSE:")
    print("Press Ctrl+C in the terminal to close the simulation.")
    print("="*60 + "\n")
    
    # Reset the environment to spawn the robot and the box
    obs, _ = env.reset()
    unwrapped_env = env.unwrapped
    
    # Zero actions mean the high-level policy won't move the arms away from their offsets
    actions = torch.zeros((1, 11), device=unwrapped_env.device)
    
    try:
        with torch.no_grad():
            while simulation_app.is_running():
                # Step the simulation
                obs, reward, terminated, truncated, info = env.step(actions)
                
                # If the episode ends (e.g. box drops), reset it so we can keep visualizing
                if terminated or truncated:
                    env.reset()
                
    except KeyboardInterrupt:
        print("Closing environment...")
    finally:
        env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
