from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ISAACLAB_ROOT = Path(r"C:\Users\alexa\IsaacLab")
RL_SCRIPT = ISAACLAB_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl" / "train_rsl_rl.py"


def main() -> None:
    if not RL_SCRIPT.exists():
        raise FileNotFoundError(f"Could not find Isaac Lab training script at: {RL_SCRIPT}")

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--locomotion_policy", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    known_args, argv = parser.parse_known_args(sys.argv[1:])
    if known_args.locomotion_policy:
        os.environ["H1_LOCOMOTION_POLICY_PATH"] = known_args.locomotion_policy
    
    # Handle --resume separately to avoid Hydra parsing issues
    resume_path = known_args.resume

    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(RL_SCRIPT.parent))
    sys.path.insert(0, str(RL_SCRIPT.parent.parent))

    from h1_carry_box_task import TRAIN_TASK_ID  # noqa: PLC0415

    if "--task" not in argv:
        argv = ["--task", TRAIN_TASK_ID, *argv]

    namespace = runpy.run_path(str(RL_SCRIPT))
    
    # Add resume path back to argv if provided (use + prefix for Hydra)
    if resume_path:
        argv = [*argv, f"+resume={resume_path}"]
    
    namespace["run"](argv)


if __name__ == "__main__":
    main()
