from __future__ import annotations

import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ISAACLAB_ROOT = Path(r"C:\Users\alexa\IsaacLab")
RL_SCRIPT = ISAACLAB_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl" / "train_rsl_rl.py"
DEFAULT_TASK = "Isaac-Velocity-Flat-H1-v0"


def main() -> None:
    if not RL_SCRIPT.exists():
        raise FileNotFoundError(f"Could not find Isaac Lab training script at: {RL_SCRIPT}")

    sys.path.insert(0, str(RL_SCRIPT.parent))
    sys.path.insert(0, str(RL_SCRIPT.parent.parent))
    sys.path.insert(0, str(PROJECT_ROOT))

    # Import package to register custom gym tasks
    import h1_carry_box_task  # noqa: F401

    argv = sys.argv[1:]
    if "--task" not in argv:
        argv = ["--task", DEFAULT_TASK, *argv]

    namespace = runpy.run_path(str(RL_SCRIPT))
    namespace["run"](argv)


if __name__ == "__main__":
    main()
