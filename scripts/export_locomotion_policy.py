from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import gymnasium as gym
from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import h1_carry_box_task  # noqa: F401

ISAACLAB_ROOT = Path(r"C:\Users\alexa\IsaacLab")
DEFAULT_TASK = "Isaac-Velocity-Flat-H1-v0"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "h1_flat_policy.pt"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the stage-1 H1 locomotion policy as TorchScript.")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--num_envs", type=int, default=1)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = _parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

    from isaaclab_tasks.utils import get_checkpoint_path
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = env_cfg.sim.device
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, "5.0.1")

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = args_cli.checkpoint or get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    output_path = Path(args_cli.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    runner.export_policy_to_jit(path=str(output_path.parent), filename=output_path.name)

    exported_path = output_path.parent / output_path.name
    if exported_path != output_path:
        shutil.copy2(exported_path, output_path)

    print(f"checkpoint={resume_path}")
    print(f"exported_policy={output_path}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
