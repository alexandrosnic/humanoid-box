from __future__ import annotations

import runpy
import sys
from pathlib import Path


ISAACLAB_ROOT = Path(r"C:\Users\alexa\IsaacLab")
RL_SCRIPT = ISAACLAB_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl" / "train_rsl_rl.py"
DEFAULT_TASK = "Isaac-Velocity-Flat-H1-v0"


def main() -> None:
    if not RL_SCRIPT.exists():
        raise FileNotFoundError(f"Could not find Isaac Lab training script at: {RL_SCRIPT}")

    sys.path.insert(0, str(RL_SCRIPT.parent))
    sys.path.insert(0, str(RL_SCRIPT.parent.parent))

    argv = sys.argv[1:]
    if "--task" not in argv:
        argv = ["--task", DEFAULT_TASK, *argv]

    namespace = runpy.run_path(str(RL_SCRIPT))
    namespace["run"](argv)


if __name__ == "__main__":
    main()
