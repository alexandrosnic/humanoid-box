# Humanoid Training

## The goal
Train a Humanoid carry a box.

## 1st task:

The goal was to train the H1 humanoid robot to walk forward while stably carrying a heavy (5 kg) box on its forearms.

### Method:
- I used **hierarchical RL** (RSL-RL PPO) on Isaac Sim/Lab -> action space dimensionality reduction: I trained the humanoid on a hierarchical control design.
    1. **Low-level locomotion policy**: I trained the complex low-level locomotion prior policy, exported it to Torchscript and froze it.
    2. **High-level loco-manipulation policy**: Then used the **frozen** pre-trained locomotion policy, while the active RL agent only had to output high-level command offsets (velocities) and control the 8 joints of the arms.
- **Simplified grasping**: grasping is perhaps the most challenging robotic task, so instead of hassling with grasping (fingers / end-effector), I preferred to design the task around a mechanical grasping (bottom-carry or form-closure) with the arms.

### Curriculum:
- **Phase 1 (Stabilizing the Grasp)**: Focused on properly grasping the box, symmetrically, horizontally, and keeping the robot's body upright.
- **Phase 2 (Walking and Carrying)**: Ramped up gradually to introduce walking while carrying the box.

### Termination conditions:
- Box dropped
- Box far from robot
- Robot falls
- Time out

### Rewards:
1. Phase 1
- Hands under box
- Symmetric grasp
- Upright posture
- Box lays on arms
- Box close to torso (to reduce tipping torque)
- Hands in front of the body
- Box hold duration

2. Phase 2
- Velocity of box carrying
- Robot's velocity tracking
- Feet air time (to take clean steps rather than dragging)

### Penalties:
- Robot falls on the ground
- Velocity of joints (penalizes fast, erratic arm movements)
- Shoulder yaw (prevents shoulder from twisting outwards)

* Used tensorboard to track some metrics that indicate early success or not, of the desired behavior.

### Result:

<video src="https://github.com/user-attachments/assets/d8a7ec16-ecd5-496a-ac25-f886643a5e59" controls></video>

*(Local Preview in VS Code: [artifacts/rl-video-step-0.mp4](artifacts/rl-video-step-0.mp4))*

### Outcome
I am not entirely satisfied with the result.

But I must also admit that my limited hardware (RTX 4060, 8GB VRAM, 32GB RAM) didn't allow me for fast iterations of training (until the end or up to a satisfied iteration) - observing the outcome - modifying environment/observations/rewards - repeat. And do this on only 128 parallel environments, headless, for 3000 learning iterations.

This training loop got me only that far, given the trade-off of letting the humanoid get sufficient number of training loops to learn the desired outcome, but also iterate fast for fast prototyping (reaching the desired outcome fast).

## 2nd task: Lift the box from a table, and walk away (branch:feature/box-lifting-grasping)

### Improvements
When I finished the first step of the learning path:
"Learn to grasp the box, as it spawns on shoulders' height, and carry it",
and because I was not satisfied neither with the walking outcome, nor with the time needed for a training iteration, I made the following improvements:
- **Adversarial Motion Priors (AMP)**: Fed the robot data of human walk (imitation learning), to give it a prior towards proper walking, rather than learning from scratch. Or even, given more time, I could make use of Isaac Lab Mimic.
- **NVIDIA's Eureka as a feedback loop**: I ran a custom-stage Eureka locally using Ollama to search for reward weights. Essentially, it is an LLM-in-the-loop that trains - observes - evaluates performance (via LLM) - adjusts (rewards, observations etc) accordingly - retrains without breaking the loop, and until the final desired behavior is reached. The tuner automatically optimized Phase 1 (grasp/lift) from scratch, identified the best intermediate checkpoint in the log directory, and resumed training automatically for Phase 2 (turn/carry) with the Stage 1 weights locked, which saved precious training iterations.
- **Integrate a FSM with a reward curriculum**: The robot must approach the box on the table, place the arms below the handles, lift it, and walk away from the table, via FSM inside the environment's command generator.

### Result:

<video src="https://github.com/user-attachments/assets/fd3b4ad8-6efd-4b7d-9177-31c9f3edcf46" controls></video>

*(Local Preview in VS Code: [artifacts/rl-video-step-0-stage2.mp4](artifacts/rl-video-step-0-stage2.mp4))*

### Other options:
- If I would do the training on an enterprise-level task, I would try to get access to GR00T foundational model to shortcut training and be more confident for the behavior.
- Given sufficient fund, I could make use of specialized AI clouds such as: RunPod, Lambda Labs, CoreWeave etc, which are more cost efficient than running on hyperscalers (AWS, Azure, GCP), for hardware use.
- The LLM could be replaced by a VLM in the loop.
- I would like to replace the physics engine with the newly announced [NVIDIA Newton](https://developer.nvidia.com/newton-physics)


* Notes: RL is notorious at exploitting local optima / physics engines loopholes, which made the training loop slow.
That was an example of the issues I faced and how I solved it:
The robot learned that leaning on the table, standing, would accumulate reward over time. I resolved this by disabling body-contact tolerance and implementing selective contact-sensor terminations (pelvis, torso, hips, knees) so that any contact with the table other than the arms/wrists instantly reset the environment.

