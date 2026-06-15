import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CFG_FILE = PROJECT_ROOT / "h1_carry_box_task" / "h1_lift_carry_env_cfg.py"
ISAACLAB_ROOT = Path(r"C:\Users\alexa\IsaacLab")
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train.py"

# Local Ollama defaults
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = "qwen3-coder:latest"


def get_current_weights():
    """Parse the current reward weights from the config file."""
    content = CFG_FILE.read_text()
    weights = {}
    
    # Extract robot_to_box_dist weight
    match = re.search(r"robot_to_box_dist\s*=\s*RewTerm\(.*?weight\s*=\s*([0-9.-]+)", content, re.DOTALL)
    if match:
        weights["robot_to_box_dist"] = float(match.group(1))
        
    # Extract box_lift weight
    match = re.search(r"box_lift\s*=\s*RewTerm\(.*?weight\s*=\s*([0-9.-]+)", content, re.DOTALL)
    if match:
        weights["box_lift"] = float(match.group(1))
        
    # Extract box_carry_away weight
    match = re.search(r"box_carry_away\s*=\s*RewTerm\(.*?weight\s*=\s*([0-9.-]+)", content, re.DOTALL)
    if match:
        weights["box_carry_away"] = float(match.group(1))
        
    return weights


def update_config_weights(new_weights):
    """Write the new reward weights back to the config file (both Rewards and Curriculum)."""
    content = CFG_FILE.read_text()
    
    # Update robot_to_box_dist
    if "robot_to_box_dist" in new_weights:
        val = new_weights["robot_to_box_dist"]
        content = re.sub(
            r"(robot_to_box_dist\s*=\s*RewTerm\(.*?weight\s*=\s*)([0-9.-]+)",
            f"\\g<1>{val:.4f}",
            content,
            flags=re.DOTALL
        )
        
    # Update box_lift
    if "box_lift" in new_weights:
        val = new_weights["box_lift"]
        content = re.sub(
            r"(box_lift\s*=\s*RewTerm\(.*?weight\s*=\s*)([0-9.-]+)",
            f"\\g<1>{val:.4f}",
            content,
            flags=re.DOTALL
        )
        content = re.sub(
            r"(box_lift\s*=\s*CurrTerm\(.*?\"weight\"\s*:\s*)([0-9.-]+)",
            f"\\g<1>{val:.4f}",
            content,
            flags=re.DOTALL
        )
        
    # Update box_carry_away
    if "box_carry_away" in new_weights:
        val = new_weights["box_carry_away"]
        content = re.sub(
            r"(box_carry_away\s*=\s*RewTerm\(.*?weight\s*=\s*)([0-9.-]+)",
            f"\\g<1>{val:.4f}",
            content,
            flags=re.DOTALL
        )
        content = re.sub(
            r"(box_carry_away\s*=\s*CurrTerm\(.*?\"weight\"\s*:\s*)([0-9.-]+)",
            f"\\g<1>{val:.4f}",
            content,
            flags=re.DOTALL
        )
        
    CFG_FILE.write_text(content)
    print(f"[EUREKA] Updated config weights in h1_lift_carry_env_cfg.py to: {new_weights}")


def query_ollama(prompt, model):
    """Query local Ollama instance using urllib."""
    import urllib.request
    import urllib.error
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are Eureka, an LLM-in-the-loop reward optimization agent for reinforcement learning. "
                    "Your task is to analyze the training feedback and suggest the next set of reward weights. "
                    "You MUST respond ONLY with a JSON object. "
                    "Do not include any explanation or additional text."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "options": {
            "temperature": 0.7
        },
        "stream": False
    }
    
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            message_content = res_data["message"]["content"]
            clean_content = re.sub(r"```json|```", "", message_content).strip()
            return json.loads(clean_content)
    except Exception as e:
        print("[ERROR] Failed to query or parse Ollama response:")
        if 'message_content' in locals():
            print(message_content)
        print(e)
        return None


def get_latest_run_dir():
    """Get the latest subdirectory in logs/rsl_rl/h1_carry_box."""
    logs_dir = Path(r"C:\Users\alexa\IsaacLab\logs\rsl_rl\h1_carry_box")
    if not logs_dir.exists():
        return None
    run_dirs = [d for d in logs_dir.glob("*") if d.is_dir()]
    if not run_dirs:
        return None
    run_dirs.sort(key=lambda d: d.stat().st_mtime)
    return run_dirs[-1]


def get_latest_checkpoint(run_dir):
    """Find the latest model_*.pt file in the run directory."""
    if not run_dir or not run_dir.exists():
        return None
    checkpoints = list(run_dir.glob("model_*.pt"))
    if not checkpoints:
        return None
    checkpoint_nums = []
    for cp in checkpoints:
        match = re.search(r"model_(\d+)\.pt", cp.name)
        if match:
            checkpoint_nums.append((int(match.group(1)), cp.name))
    if not checkpoint_nums:
        return None
    checkpoint_nums.sort()
    return checkpoint_nums[-1][1]


def run_training(max_iterations=150, num_envs=64, resume_run=None, checkpoint="model_150.pt"):
    """Run training subprocess and capture its stdout for metric extraction."""
    cmd = [
        str(ISAACLAB_ROOT / "isaaclab.bat"),
        "-p",
        str(TRAIN_SCRIPT),
        "--task", "Isaac-H1-Table-Lift-v0",
        "--viz", "none",
        "--num_envs", str(num_envs),
        "--max_iterations", str(max_iterations),
        "--locomotion_policy", str(PROJECT_ROOT / "artifacts" / "h1_flat_policy.pt")
    ]
    
    if resume_run:
        cmd.extend([
            "--resume",
            "--load_run", resume_run,
            "--checkpoint", checkpoint
        ])
        
    print(f"[EUREKA] Launching training run: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        cwd=str(ISAACLAB_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    max_reward = -float("inf")
    last_ep_length = 0.0
    
    while True:
        line = process.stdout.readline()
        if not line:
            break
        print(line, end="")
        sys.stdout.flush()
        
        # Parse RSL_RL metrics
        reward_match = re.search(r"Mean reward:\s*([0-9.-]+)", line)
        if reward_match:
            val = float(reward_match.group(1))
            if val > max_reward:
                max_reward = val
                
        len_match = re.search(r"Mean episode length:\s*([0-9.-]+)", line)
        if len_match:
            last_ep_length = float(len_match.group(1))
            
    process.wait()
    
    # Return the metrics and the run folder path
    latest_run = get_latest_run_dir()
    latest_run_name = latest_run.name if latest_run else None
    
    return max_reward, last_ep_length, latest_run_name


def main():
    parser = argparse.ArgumentParser(description="Eureka Stage-wise Reward Tuner via local Ollama.")
    parser.add_argument("--model", type=str, default=OLLAMA_MODEL, help="Local Ollama model name.")
    parser.add_argument("--stage1-iterations", type=int, default=4, help="Number of tuning iterations for Stage 1.")
    parser.add_argument("--stage1-steps", type=int, default=150, help="Training iterations for Stage 1.")
    parser.add_argument("--stage2-iterations", type=int, default=4, help="Number of tuning iterations for Stage 2.")
    parser.add_argument("--stage2-steps", type=int, default=300, help="Total training iterations for Stage 2 (resumed).")
    parser.add_argument("--num-envs", type=int, default=64, help="Number of environments for training.")
    args = parser.parse_args()
    
    print("=" * 60)
    print("           EUREKA STAGE-WISE OLLAMA REWARD TUNER")
    print("=" * 60)
    print(f"Target Config: {CFG_FILE}")
    print(f"Ollama Endpoint: {OLLAMA_URL} | Model: {args.model}")
    print(f"Stage 1 (Grasp & Lift) Tuning: {args.stage1_iterations} trials x {args.stage1_steps} steps")
    print(f"Stage 2 (Turn & Carry) Tuning: {args.stage2_iterations} trials x {args.stage2_steps} steps (resumed)")
    print(f"Envs: {args.num_envs}")
    print("=" * 60)
    
    # ----------------------------------------------------
    # PHASE 1: GRASP & LIFT
    # ----------------------------------------------------
    print("\n" + "#" * 60)
    print("  PHASE 1: TUNING GRASP & LIFT (Approaching & Lifting)")
    print("#" * 60)
    
    stage1_history = []
    best_stage1_reward = -float("inf")
    best_stage1_weights = {}
    best_stage1_run_name = None
    
    for i in range(args.stage1_iterations):
        print(f"\n--- STAGE 1 - TRIAL {i+1}/{args.stage1_iterations} ---")
        current_weights = get_current_weights()
        current_weights["box_carry_away"] = 0.0  # Force box_carry_away to 0 during Stage 1 tuning
        update_config_weights(current_weights)
        
        if not stage1_history:
            prompt = (
                "You are tuning the reward weights for Phase 1 (Grasp & Lift) of an H1 humanoid robot task. "
                "The goal of Phase 1 is to walk up to a static waist-height table and lift a box. "
                "The current config weights are:\n"
                f"{json.dumps(current_weights, indent=2)}\n\n"
                "Suggest a set of reward weights to start this task. "
                "We want to optimize 'robot_to_box_dist' (to approach the table) and 'box_lift' (to raise the box). "
                "Keep 'box_carry_away' as 0.0. "
                "Provide the response ONLY in JSON format: {'robot_to_box_dist': float, 'box_lift': float, 'box_carry_away': 0.0}."
            )
        else:
            prompt = (
                "Here is the history of Phase 1 weight trials and their performance:\n"
                f"{json.dumps(stage1_history, indent=2)}\n\n"
                "Based on this history, modify the weights for 'robot_to_box_dist' and 'box_lift' to improve "
                "both the max reward and the episode length (longer means it lifts the box stably without falling). "
                "Keep 'box_carry_away' as 0.0. "
                "Provide the response ONLY in JSON format: {'robot_to_box_dist': float, 'box_lift': float, 'box_carry_away': 0.0}."
            )
            
        print(f"[EUREKA] Querying Ollama for Stage 1 weights...")
        suggested = query_ollama(prompt, args.model)
        if not suggested:
            print("[EUREKA] Failed to get valid weights, using current weights.")
            suggested = current_weights
            
        suggested["box_carry_away"] = 0.0  # Double check it remains 0
        update_config_weights(suggested)
        
        max_reward, ep_length, run_name = run_training(
            max_iterations=args.stage1_steps,
            num_envs=args.num_envs
        )
        print(f"[EUREKA] Trial Complete. Run Name: {run_name} | Max Reward: {max_reward:.2f} | Ep Length: {ep_length:.1f}")
        
        trial_record = {
            "trial": i + 1,
            "weights": suggested,
            "performance": {
                "max_reward": max_reward if max_reward != -float("inf") else -9999.0,
                "ep_length": ep_length
            },
            "run_name": run_name
        }
        stage1_history.append(trial_record)
        
        if max_reward > best_stage1_reward and run_name:
            best_stage1_reward = max_reward
            best_stage1_weights = suggested
            best_stage1_run_name = run_name
            
    print("\n" + "=" * 60)
    print("  PHASE 1 COMPLETE")
    print("=" * 60)
    print(f"Best Stage 1 weights: {best_stage1_weights}")
    print(f"Best Stage 1 Max Reward: {best_stage1_reward:.2f}")
    print(f"Best Stage 1 Run Folder: {best_stage1_run_name}")
    print("=" * 60)
    
    if not best_stage1_run_name:
        print("[ERROR] No valid Stage 1 runs were completed. Cannot proceed to Phase 2.")
        sys.exit(1)
        
    # Write the best Stage 1 weights to the config
    update_config_weights(best_stage1_weights)
    
    # ----------------------------------------------------
    # PHASE 2: TURN & CARRY
    # ----------------------------------------------------
    print("\n" + "#" * 60)
    print("  PHASE 2: TUNING TURN & CARRY (Resuming from Phase 1 Checkpoint)")
    print("#" * 60)
    
    stage2_history = []
    best_stage2_reward = -float("inf")
    best_stage2_weights = {}
    
    for i in range(args.stage2_iterations):
        print(f"\n--- STAGE 2 - TRIAL {i+1}/{args.stage2_iterations} ---")
        
        # Read the config weights
        current_weights = get_current_weights()
        # Ensure Phase 1 weights are fixed at their best values
        current_weights["robot_to_box_dist"] = best_stage1_weights["robot_to_box_dist"]
        current_weights["box_lift"] = best_stage1_weights["box_lift"]
        update_config_weights(current_weights)
        
        if not stage2_history:
            prompt = (
                "You are tuning the reward weights for Phase 2 (Turn & Carry) of an H1 humanoid robot task. "
                "Phase 2 resumes training from a checkpoint where the robot has already learned to walk to the table and lift the box. "
                "The optimized Phase 1 weights are fixed:\n"
                f"robot_to_box_dist: {best_stage1_weights['robot_to_box_dist']}\n"
                f"box_lift: {best_stage1_weights['box_lift']}\n\n"
                "The current config weights are:\n"
                f"{json.dumps(current_weights, indent=2)}\n\n"
                "We want to optimize 'box_carry_away' (rewarding the robot for turning 180 degrees and walking away carrying the box). "
                "Suggest a value for 'box_carry_away'. "
                "Provide the response ONLY in JSON format: {'robot_to_box_dist': "
                f"{best_stage1_weights['robot_to_box_dist']}, 'box_lift': {best_stage1_weights['box_lift']}, 'box_carry_away': float}}."
            )
        else:
            prompt = (
                "Here is the history of Phase 2 trials and their performance:\n"
                f"{json.dumps(stage2_history, indent=2)}\n\n"
                f"The Phase 1 weights are fixed: robot_to_box_dist = {best_stage1_weights['robot_to_box_dist']}, box_lift = {best_stage1_weights['box_lift']}.\n"
                "Based on this history, modify 'box_carry_away' to maximize the resumed training reward and increase episode length. "
                "Provide the response ONLY in JSON format: {'robot_to_box_dist': "
                f"{best_stage1_weights['robot_to_box_dist']}, 'box_lift': {best_stage1_weights['box_lift']}, 'box_carry_away': float}}."
            )
            
        print(f"[EUREKA] Querying Ollama for Stage 2 weights...")
        suggested = query_ollama(prompt, args.model)
        if not suggested:
            print("[EUREKA] Failed to get valid weights, using current weights.")
            suggested = current_weights
            
        # Enforce that Phase 1 weights remain fixed
        suggested["robot_to_box_dist"] = best_stage1_weights["robot_to_box_dist"]
        suggested["box_lift"] = best_stage1_weights["box_lift"]
        update_config_weights(suggested)
        
        # Locate the best checkpoint from Stage 1 dynamically
        best_run_path = Path(r"C:\Users\alexa\IsaacLab\logs\rsl_rl\h1_carry_box") / best_stage1_run_name
        checkpoint_file = get_latest_checkpoint(best_run_path)
        if not checkpoint_file:
            print(f"[ERROR] Could not find any model checkpoint in {best_run_path}. Using fallback model_0.pt.")
            checkpoint_file = "model_0.pt"
            
        print(f"[EUREKA] Resolved checkpoint file: {checkpoint_file}")
        
        max_reward, ep_length, run_name = run_training(
            max_iterations=args.stage2_steps,
            num_envs=args.num_envs,
            resume_run=best_stage1_run_name,
            checkpoint=checkpoint_file
        )
        print(f"[EUREKA] Resumed Trial Complete. Run Name: {run_name} | Max Reward: {max_reward:.2f} | Ep Length: {ep_length:.1f}")
        
        trial_record = {
            "trial": i + 1,
            "weights": suggested,
            "performance": {
                "max_reward": max_reward if max_reward != -float("inf") else -9999.0,
                "ep_length": ep_length
            },
            "run_name": run_name
        }
        stage2_history.append(trial_record)
        
        if max_reward > best_stage2_reward:
            best_stage2_reward = max_reward
            best_stage2_weights = suggested
            
    print("\n" + "=" * 60)
    print("  PHASE 2 COMPLETE")
    print("=" * 60)
    print(f"Best Resumed Stage 2 weights: {best_stage2_weights}")
    print(f"Best Resumed Stage 2 Max Reward: {best_stage2_reward:.2f}")
    print("=" * 60)
    
    # Save the absolute best weights configuration to the config file
    update_config_weights(best_stage2_weights)
    
    print("\n" + "=" * 60)
    print("           STAGE-WISE REWARD TUNING SUCCESSFUL")
    print("=" * 60)
    print(f"Final Tuned Weights:")
    print(f"  - robot_to_box_dist: {best_stage2_weights.get('robot_to_box_dist')}")
    print(f"  - box_lift: {best_stage2_weights.get('box_lift')}")
    print(f"  - box_carry_away: {best_stage2_weights.get('box_carry_away')}")
    print("Config files and curriculum targets updated. Ready for final training!")
    print("=" * 60)


if __name__ == "__main__":
    main()
