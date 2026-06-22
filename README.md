# H1 Humanoid Locomanipulation: Staged RL & Table-Top Carrying

This repository is a personal training ground to learn and put into practice Reinforcement Learning (RL) and the NVIDIA Isaac Sim simulator through a progressive, self-trained curriculum.

The project uses a **hierarchical / staged Isaac Lab workflow** designed for local consumer GPU hardware (RTX 4060 with 8 GB VRAM):
1. **Locomotion Prior:** A lower-body locomotion controller (standard PPO or motion-capture guided AMP).
2. **Task Policy:** A high-level policy controlling the arms and commanding the frozen locomotion prior to coordinate approach, lifting, and carrying.

---

## 🚀 Quick Start: Latest Advancement (Gated Rewards & Resets)

This is the **latest version** of our project (`Isaac-H1-Table-Lift-v0`). It adds transition-conditioned gated rewards and extreme roll/pitch/yaw balance and box offsets to make the humanoid robust to failures.

### 1. Train the Latest Lift & Carry Policy
To train the Stage-2 policy headlessly:
```powershell
Set-Location C:\Users\alexa\IsaacLab
.\isaaclab.bat -p C:\Users\alexa\projects\humanoid_training\scripts\train.py --task Isaac-H1-Table-Lift-v0 --viz none --num_envs 128 --locomotion_policy C:\Users\alexa\projects\humanoid_training\artifacts\h1_flat_policy.pt
```

### 2. Play / Evaluate the Latest Checkpoint
To visually evaluate the trained checkpoint:
```powershell
Set-Location C:\Users\alexa\IsaacLab
.\isaaclab.bat -p C:\Users\alexa\projects\humanoid_training\scripts\play.py --num_envs 4 --real-time --viz kit --checkpoint C:\Users\alexa\projects\humanoid_training\logs\rsl_rl\h1_carry_box\<run_folder>\model_XXXX.pt
```

---

## 🎓 Learning Journey: Self-Trained Curriculum

This project was built step-by-step to progress from basic locomotion to complex table-top manipulation.

### Phase 1: Stiff Locomotion & Mid-Air Cradle Carry
* **Goal:** Teach the H1 humanoid to balance a heavy (5 kg) box spawned in mid-air.
* **Locomotion Prior:** A standard PPO flat walking policy (without AMP), resulting in a stiff, robotic walk.
* **Task Policy:** Arm joints are trained to cradle/support the box from underneath (cradle grasp). Leg joints are frozen, and the task policy outputs velocity commands for the legs.
* **Results Video:**
  <video src="https://github.com/user-attachments/assets/d8a7ec16-ecd5-496a-ac25-f886643a5e59" controls></video>
  *(Local Preview: [artifacts/rl-video-step-0.mp4](artifacts/rl-video-step-0.mp4))*
* **Gait Video:** Stiff walking without AMP: [artifacts/locomotion_stiff_walk.mp4](artifacts/locomotion_stiff_walk.mp4)

#### How to run Phase 1:
* **Train Stiff Prior:**
  ```powershell
  Set-Location C:\Users\alexa\IsaacLab
  .\isaaclab.bat -p C:\Users\alexa\projects\humanoid_training\scripts\train_locomotion.py --viz none --num_envs 128
  ```
* **Train Mid-Air Carry:**
  ```powershell
  Set-Location C:\Users\alexa\IsaacLab
  .\isaaclab.bat -p C:\Users\alexa\projects\humanoid_training\scripts\train.py --task Isaac-H1-Carry-Box-v0 --viz none --num_envs 64 --locomotion_policy C:\Users\alexa\projects\humanoid_training\artifacts\h1_flat_policy.pt
  ```

---

### Phase 2: Natural Locomotion (AMP) & Table Grasp-to-Lift (FSM + Eureka)
* **Goal:** Transition from mid-air box spawning to a table-top scenario where the robot walks to a table, slides its wrists under side handles, and lifts the box.
* **Locomotion Prior:** Upgraded to Adversarial Motion Priors (AMP) trained using motion capture walk data (`humanoid_walk.npz`), producing a natural, human-like gait.
* **Task Policy:**
  * **FSM Command Generator:** Orchestrates stages (Approach Table -> Stand Still & Lift -> Turn -> Carry Walk).
  * **Eureka Reward Tuning:** Rewards tuned via local LLM feedback loops (`eureka_stage_tuner.py`) to balance approach speed, leveling, and lifting height.
  * **Table-Body Collisions:** Resets immediately if the torso or pelvis contacts the table, forcing the robot to rely on its arms.
* **Results Video:**
  <video src="https://github.com/user-attachments/assets/fd3b4ad8-6efd-4b7d-9177-31c9f3edcf46" controls></video>
  *(Local Preview: [artifacts/rl-video-step-0-stage2.mp4](artifacts/rl-video-step-0-stage2.mp4))*
* **Gait Video:** Natural walk with AMP: [artifacts/locomotion_amp_walk.mp4](artifacts/locomotion_amp_walk.mp4)

#### How to run Phase 2:
* **Train AMP Prior:**
  ```powershell
  Set-Location C:\Users\alexa\IsaacLab
  .\isaaclab.bat -p C:\Users\alexa\projects\humanoid_training\scripts\train_locomotion.py --task Isaac-Velocity-Flat-H1-AMP-v0 --viz none
  ```
* **Train Table Lift:**
  ```powershell
  Set-Location C:\Users\alexa\IsaacLab
  .\isaaclab.bat -p C:\Users\alexa\projects\humanoid_training\scripts\train.py --task Isaac-H1-Table-Lift-v0 --viz none --num_envs 128 --locomotion_policy C:\Users\alexa\projects\humanoid_training\artifacts\h1_flat_policy.pt
  ```

---

### Phase 3: Robust Grasping & Carrying (Gated Rewards & Failure-State Resets)
* **Goal:** Prevent the robot from exploiting locomotion rewards when dropping the box, and make the policy robust against severe disturbances.
* **Advancements Implemented:**
  1. **Transition-Gated Rewards:** Locomotion velocity and carrying rewards are strictly multiplied by a `handle_grasp_multiplier` (the product of hand-to-handle target proximities). If the robot lets go of a handle, its walking rewards drop to zero instantly.
  2. **Extreme Reset States:** During environment resets, there is a 20% chance of introducing severe roll/pitch body tilts (up to $\pm 8.6^\circ$) and off-center box offsets/tilts on the table to force the policy to recover from near-failure states.
  3. **Low-Level Policy Detachment:** Wrapped low-level policy inference in `no_grad()` to prevent gradient leakage.
* **Status:** ⏳ Currently training. Results video will be added soon!

---

## 🛠️ Repository Layout

```text
humanoid_training\
  h1_carry_box_task\
    __init__.py
    h1_carry_box_env_cfg.py
    h1_lift_carry_env_cfg.py
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

## 💻 Hardware Guidance (RTX 4060 / 8 GB VRAM)

* **Stage 1 Locomotion:** `--num_envs 128`
* **Stage 2 Carry / Lift:** `--num_envs 64` or `128`
* **Play / Evaluation:** `--num_envs 4` to `8`
* **Training Mode:** Use `--viz none` for headless simulation to save VRAM.

## 🎛️ Stage 1.5: Exporting the Locomotion Prior

Export the locomotion PPO checkpoint to a stable TorchScript JIT file to be loaded by the Stage-2 policy:

```powershell
Set-Location C:\Users\alexa\IsaacLab

# For standard locomotion prior:
.\isaaclab.bat -p C:\Users\alexa\projects\humanoid_training\scripts\export_locomotion_policy.py --checkpoint C:\Users\alexa\IsaacLab\logs\rsl_rl\h1_flat\<run>\model_XXXX.pt

# For AMP locomotion prior:
.\isaaclab.bat -p C:\Users\alexa\projects\humanoid_training\scripts\export_locomotion_policy.py --task Isaac-Velocity-Flat-H1-AMP-v0 --checkpoint C:\Users\alexa\projects\humanoid_training\logs\rsl_rl\h1_flat\<run>\model_XXXX.pt
```

By default, the JIT policy is saved to:
`C:\Users\alexa\projects\humanoid_training\artifacts\h1_flat_policy.pt`

## 🧪 Standalone Smoke Test

The standalone scene is useful for checking assets and simulator sanity:

```powershell
Set-Location C:\Users\alexa\IsaacLab
.\isaaclab.bat -p C:\Users\alexa\projects\humanoid_training\humanoid_carry_box.py --steps 240 --viz none
```
