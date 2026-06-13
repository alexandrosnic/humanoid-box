# H1 carry-box staged locomanipulation project

This project now uses a **hierarchical / staged Isaac Lab workflow** designed for an **RTX 4060 with 8 GB VRAM**:

1. train a compact **stage-1 H1 locomotion prior**
2. export that locomotion policy as TorchScript
3. train a **stage-2 carry-box task** where:
   - the locomotion policy is frozen
   - the high-level RL policy outputs base velocity commands for the frozen lower body
   - the high-level RL policy also learns the H1 arm carry posture

That is much closer to Isaac Lab locomanipulation best practice than the previous single full-body PPO setup.

## Repository layout

```text
humanoid_training\
  h1_carry_box_task\
    __init__.py
    h1_carry_box_env_cfg.py
    low_level_policy_action.py
    agents\
      rsl_rl_ppo_cfg.py
    mdp\
      observations.py
      rewards.py
      terminations.py
      events.py
      curriculums.py
  scripts\
    train_locomotion.py
    export_locomotion_policy.py
    train.py
    play.py
  humanoid_carry_box.py
```

## RTX 4060 / 8 GB guidance

Use these settings first:

- **stage 1 locomotion training:** `--num_envs 128`
- **stage 2 carry training:** `--num_envs 64` or `128`
- **play / evaluation:** `--num_envs 4` to `8`
- use `--viz none` for training

The stage-2 environment defaults were reduced accordingly:

- train envs: `128`
- play envs: `8`
- smaller PPO MLP than before

## Stage 1: train the H1 locomotion prior (Standard or AMP)

### Option A: Standard H1 Locomotion Prior
This trains Isaac Lab's built-in `Isaac-Velocity-Flat-H1-v0` task through a local wrapper.

```powershell
Set-Location C:\Users\alexa\IsaacLab
.\isaaclab.bat -p C:\Users\alexa\projects\humanoid_training\scripts\train_locomotion.py --viz none --num_envs 128
```

### Option B: AMP-Regularized Locomotion Prior (Natural Gait)
This trains our custom task `Isaac-Velocity-Flat-H1-AMP-v0` which loads reference human walk trajectories from `humanoid_walk.npz` and maps them onto the H1 robot's 19 active joints, forcing the robot to learn a natural, human-like gait instead of standard shuffling.

```powershell
Set-Location C:\Users\alexa\IsaacLab
.\isaaclab.bat -p C:\Users\alexa\projects\humanoid_training\scripts\train_locomotion.py --task Isaac-Velocity-Flat-H1-AMP-v0 --viz none --num_envs 128
```

For a quick smoke run:

```powershell
Set-Location C:\Users\alexa\IsaacLab
.\isaaclab.bat -p C:\Users\alexa\projects\humanoid_training\scripts\train_locomotion.py --task Isaac-Velocity-Flat-H1-AMP-v0 --viz none --num_envs 32 --max_iterations 10
```

The locomotion checkpoints are written under:

```text
C:\Users\alexa\IsaacLab\logs\rsl_rl\h1_flat\ (Option A)
C:\Users\alexa\IsaacLab\logs\rsl_rl\h1_amp_locomotion\ (Option B)
```

## Stage 1.5: export the locomotion policy

Export the chosen locomotion checkpoint to a stable TorchScript file used by stage 2.

```powershell
Set-Location C:\Users\alexa\IsaacLab
.\isaaclab.bat -p C:\Users\alexa\projects\humanoid_training\scripts\export_locomotion_policy.py --checkpoint C:\Users\alexa\IsaacLab\logs\rsl_rl\h1_flat\<run>\model_XXXX.pt
```

By default this writes:

```text
C:\Users\alexa\projects\humanoid_training\artifacts\h1_flat_policy.pt
```

## Stage 2: Table Grasping, Lifting, and Carrying

This trains the active table-top box-lifting task `Isaac-H1-Table-Lift-v0`.

In this task:
*   The box spawns on a static **0.8m table** in front of the robot.
*   The box features **side handles** (`y = +/-0.19` spacing).
*   The robot must walk up to the table, align its arms, slide its wrists under the handles (form-closure), lift the box above the table top, and carry it forward.
*   The lower body remains controlled by the frozen Stage-1 locomotion prior (Standard or AMP).

### Train the Lift & Carry Policy:

```powershell
Set-Location C:\Users\alexa\IsaacLab
.\isaaclab.bat -p C:\Users\alexa\projects\humanoid_training\scripts\train.py --task Isaac-H1-Table-Lift-v0 --viz none --num_envs 64 --locomotion_policy C:\Users\alexa\projects\humanoid_training\artifacts\h1_flat_policy.pt
```

If you exported to the default path, the `--locomotion_policy` flag is optional:

```powershell
Set-Location C:\Users\alexa\IsaacLab
.\isaaclab.bat -p C:\Users\alexa\projects\humanoid_training\scripts\train.py --task Isaac-H1-Table-Lift-v0 --viz none --num_envs 64
```

### Run Automated Reward Tuning (Eureka via local Ollama):

To automatically search for optimal reward weights (`robot_to_box_dist` and `box_lift`), ensure your local Ollama server is running and run:

```powershell
Set-Location C:\Users\alexa\IsaacLab
.\isaaclab.bat -p C:\Users\alexa\projects\humanoid_training\scripts\eureka_tuner.py --model llama3:8b --iterations 5 --train-steps 150
```

### Play / Evaluate the Lift & Carry Policy:

```powershell
Set-Location C:\Users\alexa\IsaacLab
.\isaaclab.bat -p C:\Users\alexa\projects\humanoid_training\scripts\play.py --task Isaac-H1-Table-Lift-Play-v0 --num_envs 1 --real-time --visualizer kit --locomotion_policy C:\Users\alexa\projects\humanoid_training\artifacts\h1_flat_policy.pt
```

To load a specific carry checkpoint:

```powershell
Set-Location C:\Users\alexa\IsaacLab
.\isaaclab.bat -p C:\Users\alexa\projects\humanoid_training\scripts\play.py --task Isaac-H1-Table-Lift-Play-v0 --num_envs 4 --locomotion_policy C:\Users\alexa\projects\humanoid_training\artifacts\h1_flat_policy.pt --checkpoint C:\Users\alexa\IsaacLab\logs\rsl_rl\h1_carry_box\<run>\model_XXXX.pt
```

## Task IDs

*   **Stage 1 Locomotion:**
    *   Standard: `Isaac-Velocity-Flat-H1-v0`
    *   AMP-Regularized: `Isaac-Velocity-Flat-H1-AMP-v0`
*   **Stage 2 Active Lift & Carry:**
    *   Train: `Isaac-H1-Table-Lift-v0`
    *   Play: `Isaac-H1-Table-Lift-Play-v0`
*   **Stage 2 Hold / Bottom-Carry (Old):**
    *   Hold Train: `Isaac-H1-Carry-Box-Hold-v0`
    *   Carry Train: `Isaac-H1-Carry-Box-v0`


## What changed technically

- The project is no longer a one-shot full-body carry PPO task.
- The lower body is now controlled through a **frozen H1 locomotion TorchScript policy**.
- The learned high-level policy only handles:
  - locomotion commands for the frozen lower-body prior
  - upper-body arm control for stabilizing the box
- The repository now exposes a **stationary hold stage** and a **carry-walk stage**.
- The carry task keeps locomotion command tracking rewards from the H1 baseline, but now adds:
  - deterministic carry-aligned reset logic
  - torso/hand-relative box observations
  - hand-support shaping rewards
  - a gentler curriculum for mass and walking difficulty
- Defaults were reduced to fit **4060 / 8 GB** training more realistically.

## Standalone smoke test

The old standalone scene is still useful for simulator / asset sanity checks:

```powershell
Set-Location C:\Users\alexa\IsaacLab
.\isaaclab.bat -p C:\Users\alexa\projects\humanoid_training\humanoid_carry_box.py --steps 240 --viz none
```

## Future Work / TODOs

- [ ] Integrate **Docker** support to containerize training and simplify headless sim setup in cloud environments.
- [ ] Integrate **Weights & Biases (W&B)** for experiment tracking, hyperparameter sweeps, and training visualization.

