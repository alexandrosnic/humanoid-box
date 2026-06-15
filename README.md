# H1 carry-box staged locomanipulation project

This project now uses a **hierarchical / staged Isaac Lab workflow** designed for an **RTX 4060 with 8 GB VRAM**:

1. train a compact **stage-1 H1 locomotion prior**
2. export that locomotion policy as TorchScript
3. train a **stage-2 carry-box task** where:
   - the locomotion policy is frozen
   - the high-level RL policy outputs base velocity commands for the frozen lower body
   - the high-level RL policy also learns the H1 arm carry posture

That is much closer to Isaac Lab locomanipulation best practice than the previous single full-body PPO setup.

## Policy Playback Demo

Here is the 10-second playback video recorded from Isaac Sim on the `main` branch showing the trained carrying policy:

<video src="artifacts/rl-video-step-0.mp4" controls width="100%"></video>

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

## First Stage of Learning: Mid-Air Carry (Legacy Task)

This task trains the H1 humanoid robot to carry a heavy (5 kg) box that spawns in mid-air (at shoulder height) and walk forward.

### Stage 2A: Train a Stationary Hold Policy
Start with the easier **hold** stage so the robot learns to support the box before walking:
```powershell
Set-Location C:\Users\alexa\IsaacLab
.\isaaclab.bat -p C:\Users\alexa\projects\humanoid_training\scripts\train.py --task Isaac-H1-Carry-Box-Hold-v0 --viz none --num_envs 64 --locomotion_policy C:\Users\alexa\projects\humanoid_training\artifacts\h1_flat_policy.pt
```

### Stage 2B: Train the Carry-Walk Policy
This trains the walking carry policy (`Isaac-H1-Carry-Box-v0`). The high-level policy outputs offsets for the frozen locomotion policy and arm joint coordinates.
```powershell
Set-Location C:\Users\alexa\IsaacLab
.\isaaclab.bat -p C:\Users\alexa\projects\humanoid_training\scripts\train.py --task Isaac-H1-Carry-Box-v0 --viz none --num_envs 64 --locomotion_policy C:\Users\alexa\projects\humanoid_training\artifacts\h1_flat_policy.pt
```

---

## Second Stage of Learning: Table Grasping, Lifting, and Carrying (Improved Task)

This task trains the active table-top box-lifting task `Isaac-H1-Table-Lift-v0` (which is registered as the default `TRAIN_TASK_ID` and `PLAY_TASK_ID`).

In this task:
- The box spawns on a static **1.24m table** in front of the robot.
- The box features **side handles** (`y = +/-0.19` spacing).
- The robot must approach the table, slide its wrists under the handles, lift the box above the table top, and carry it away (controlled by the Finite State Machine (FSM) command generator).
- The lower body remains controlled by the frozen Stage-1 locomotion prior (Standard or AMP).

### Improvements & Changes Implemented:
Based on the lessons from the first stage (where limited iterations on limited hardware caused gait instability and physical cheats), the following improvements were made:
- **Locomotion Prior for Human Walking (AMP):** Added Adversarial Motion Priors (AMP) to guide the lower body. Feeding motion capture walk data (`humanoid_walk.npz`) helps the policy learn a clean walking gait instead of dragging its feet.
- **LLM-in-the-loop Reward Generation (NVIDIA's Eureka):** Integrated a local Ollama feedback loop (`eureka_stage_tuner.py`) that tunes reward weights. It optimizes Phase 1 (approach and lift) first, locks the weights, and then resumes training automatically for Phase 2 (turn and carry), saving GPU training cycles.
- **Finite State Machine (FSM):** Implemented an FSM inside the environment's command generator to smoothly orchestrate transition states: walk to the box, stand still to lift, turn 180 degrees, and carry walk away.
- **Table-Body Collision Terminations:** To prevent the robot from leaning on the table (a common exploit), selective contact-sensor terminations are enabled on the pelvis, torso, hips, and knees. Touching the table with anything other than the arms/wrists triggers an instant reset.
- **Refined Hand-to-Handle Proximity Rewards:** Replaced generic cradle rewards with precise wrist-to-handle distance tracking, penalizing the robot if its hands go above the handles. This forces the arms to slide strictly below the handles to lift.

### Step 1: Run a Visual Sanity Check (Play Mode)
Before starting training, run the environment in play mode to verify the physical setup (table, box, robot spawn positions, and FSM approach commands):

```powershell
Set-Location C:\Users\alexa\IsaacLab
.\isaaclab.bat -p C:\Users\alexa\projects\humanoid_training\scripts\play.py --num_envs 1 --real-time --visualizer kit --locomotion_policy C:\Users\alexa\projects\humanoid_training\artifacts\h1_flat_policy.pt
```

### Step 2: Train the Lift & Carry Policy
To train the Stage-2 policy headless (fastest, fits in VRAM on RTX 4060):

```powershell
Set-Location C:\Users\alexa\IsaacLab
.\isaaclab.bat -p C:\Users\alexa\projects\humanoid_training\scripts\train.py --viz none --num_envs 128 --locomotion_policy C:\Users\alexa\projects\humanoid_training\artifacts\h1_flat_policy.pt
```

### Step 3: Run Automated Reward Tuning (Eureka via local Ollama)

You have two options for automated reward tuning depending on your sample-efficiency preference:

#### Option A: Sequential Stage-wise Tuning (Recommended)
This optimizes the rewards in two phases:
1. **Phase 1 (Grasp & Lift):** Tunes weights for table approach and box lifting from scratch, keeping carry weights at 0.0.
2. **Phase 2 (Turn & Carry):** Automatically resumes training from the best Phase 1 checkpoint and tunes the carry weight.

```powershell
Set-Location C:\Users\alexa\IsaacLab
.\isaaclab.bat -p C:\Users\alexa\projects\humanoid_training\scripts\eureka_stage_tuner.py --model qwen3-coder:latest --stage1-iterations 4 --stage1-steps 150 --stage2-iterations 4 --stage2-steps 300 --num-envs 64
```

#### Option B: Single-stage Tuning (Legacy)
Tunes all weights at the same time from scratch:

```powershell
Set-Location C:\Users\alexa\IsaacLab
.\isaaclab.bat -p C:\Users\alexa\projects\humanoid_training\scripts\eureka_tuner.py --model qwen3-coder:latest --iterations 5 --train-steps 150 --num-envs 64
```

### Step 4: Evaluate the Trained Checkpoint (Play Mode)
Once training is complete, visually evaluate the robot's performance by loading the learned checkpoint:

```powershell
Set-Location C:\Users\alexa\IsaacLab
.\isaaclab.bat -p C:\Users\alexa\projects\humanoid_training\scripts\play.py --num_envs 4 --real-time --visualizer kit --locomotion_policy C:\Users\alexa\projects\humanoid_training\artifacts\h1_flat_policy.pt --checkpoint C:\Users\alexa\IsaacLab\logs\rsl_rl\h1_carry_box\<run_folder>\model_XXXX.pt
```

#### Stage 2 Policy Playback Demo

Here is the 10-second playback video recorded from Isaac Sim on the `main` branch showing the trained table-lifting and carrying policy:

<video src="artifacts/rl-video-step-0-stage2.mp4" controls width="100%"></video>

## Task IDs on this Branch

- Stage 2 Default Train: `Isaac-H1-Table-Lift-v0`
- Stage 2 Default Play: `Isaac-H1-Table-Lift-Play-v0`
- Stage 1 AMP Locomotion: `Isaac-Velocity-Flat-H1-AMP-v0`
- Legacy Mid-Air Carry Train: `Isaac-H1-Carry-Box-v0`
- Legacy Mid-Air Carry Play: `Isaac-H1-Carry-Box-Play-v0`

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
