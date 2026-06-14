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

## Stage 1: train the H1 locomotion prior

This trains Isaac Lab's built-in `Isaac-Velocity-Flat-H1-v0` task through a local wrapper.

```powershell
Set-Location C:\Users\alexa\IsaacLab
.\isaaclab.bat -p C:\Users\alexa\projects\humanoid_training\scripts\train_locomotion.py --viz none --num_envs 128
```

For a quick smoke run:

```powershell
Set-Location C:\Users\alexa\IsaacLab
.\isaaclab.bat -p C:\Users\alexa\projects\humanoid_training\scripts\train_locomotion.py --viz none --num_envs 32 --max_iterations 10
```

The locomotion checkpoints are written under:

```text
C:\Users\alexa\IsaacLab\logs\rsl_rl\h1_flat\
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

## Stage 2A: train a stationary hold policy

Start with the easier **hold** stage so the robot learns to support the box before walking.

```powershell
Set-Location C:\Users\alexa\IsaacLab
.\isaaclab.bat -p C:\Users\alexa\projects\humanoid_training\scripts\train.py --task Isaac-H1-Carry-Box-Hold-v0 --viz none --num_envs 64 --locomotion_policy C:\Users\alexa\projects\humanoid_training\artifacts\h1_flat_policy.pt
```

## Stage 2B: train the carry-walk policy

This trains the project task `Isaac-H1-Carry-Box-v0`.

The high-level policy:

- sees the carry-box state and locomotion command
- outputs **3 locomotion commands** (`vx`, `vy`, `wz`) to the frozen lower-body policy
- outputs **8 arm joint commands** for the H1 shoulders and elbows
- gets extra observations for the box relative to the torso and the hands
- is rewarded for keeping both hands near useful support points on the box

Train with the exported locomotion policy:

```powershell
Set-Location C:\Users\alexa\IsaacLab
.\isaaclab.bat -p C:\Users\alexa\projects\humanoid_training\scripts\train.py --viz none --num_envs 64 --locomotion_policy C:\Users\alexa\projects\humanoid_training\artifacts\h1_flat_policy.pt
```

If you exported to the default path, the `--locomotion_policy` flag is optional:

```powershell
Set-Location C:\Users\alexa\IsaacLab
.\isaaclab.bat -p C:\Users\alexa\projects\humanoid_training\scripts\train.py --viz none --num_envs 64
```

Stage-2 checkpoints are written under:

```text
C:\Users\alexa\IsaacLab\logs\rsl_rl\h1_carry_box\
```

## Play / evaluate the hold policy

Use this first to verify deterministic reset alignment and a stable support posture:

```powershell
Set-Location C:\Users\alexa\IsaacLab
.\isaaclab.bat -p C:\Users\alexa\projects\humanoid_training\scripts\play.py --task Isaac-H1-Carry-Box-Hold-Play-v0 --num_envs 1 --real-time --visualizer kit --locomotion_policy C:\Users\alexa\projects\humanoid_training\artifacts\h1_flat_policy.pt
```

## Play / evaluate the carry policy

```powershell
Set-Location C:\Users\alexa\IsaacLab
.\isaaclab.bat -p C:\Users\alexa\projects\humanoid_training\scripts\play.py --num_envs 4 --real-time --visualizer kit --locomotion_policy C:\Users\alexa\projects\humanoid_training\artifacts\h1_flat_policy.pt
```

To load a specific carry checkpoint:

```powershell
Set-Location C:\Users\alexa\IsaacLab
.\isaaclab.bat -p C:\Users\alexa\projects\humanoid_training\scripts\play.py --num_envs 4 --locomotion_policy C:\Users\alexa\projects\humanoid_training\artifacts\h1_flat_policy.pt --checkpoint C:\Users\alexa\IsaacLab\logs\rsl_rl\h1_carry_box\<run>\model_XXXX.pt
```

## Task IDs

- stage 2 hold train: `Isaac-H1-Carry-Box-Hold-v0`
- stage 2 hold play: `Isaac-H1-Carry-Box-Hold-Play-v0`
- stage 2 train: `Isaac-H1-Carry-Box-v0`
- stage 2 play: `Isaac-H1-Carry-Box-Play-v0`

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
