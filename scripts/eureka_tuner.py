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
OLLAMA_MODEL = "qwen3-coder:latest"  # Recommended default, can be overridden


def get_current_weights():
    """Parse the current reward weights from the config file."""
    content = CFG_FILE.read_text()
    
    # Simple regex parsing of key reward terms in H1LiftCarryRewards
    weights = {}
    
    # Extract robot_to_box_dist weight
    match = re.search(r"robot_to_box_dist\s*=\s*RewTerm\([^)]*weight\s*=\s*([0-9.-]+)", content)
    if match:
        weights["robot_to_box_dist"] = float(match.group(1))
        
    # Extract box_lift weight
    match = re.search(r"box_lift\s*=\s*RewTerm\([^)]*weight\s*=\s*([0-9.-]+)", content)
    if match:
        weights["box_lift"] = float(match.group(1))
        
    # Extract box_carry_away weight
    match = re.search(r"box_carry_away\s*=\s*RewTerm\([^)]*weight\s*=\s*([0-9.-]+)", content)
    if match:
        weights["box_carry_away"] = float(match.group(1))
        
    return weights


def update_config_weights(new_weights):
    """Write the new reward weights back to the config file."""
    content = CFG_FILE.read_text()
    
    # Update robot_to_box_dist weight
    if "robot_to_box_dist" in new_weights:
        val = new_weights["robot_to_box_dist"]
        content = re.sub(
            r"(robot_to_box_dist\s*=\s*RewTerm\([^)]*weight\s*=\s*)([0-9.-]+)",
            f"\\g<1>{val:.4f}",
            content
        )
        
    # Update box_lift weight
    if "box_lift" in new_weights:
        val = new_weights["box_lift"]
        content = re.sub(
            r"(box_lift\s*=\s*RewTerm\([^)]*weight\s*=\s*)([0-9.-]+)",
            f"\\g<1>{val:.4f}",
            content
        )
        
    # Update box_carry_away weight
    if "box_carry_away" in new_weights:
        val = new_weights["box_carry_away"]
        content = re.sub(
            r"(box_carry_away\s*=\s*RewTerm\([^)]*weight\s*=\s*)([0-9.-]+)",
            f"\\g<1>{val:.4f}",
            content
        )
        
    CFG_FILE.write_text(content)
    print(f"[EUREKA] Updated config weights in h1_lift_carry_env_cfg.py to: {new_weights}")


def query_ollama(prompt, model):
    """Query local Ollama instance using python's urllib to avoid dependencies."""
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
                    "You MUST respond ONLY with a JSON object containing the keys: 'robot_to_box_dist', 'box_lift', and 'box_carry_away'. "
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
            
            # Clean LLM markdown wraps
            clean_content = re.sub(r"```json|```", "", message_content).strip()
            return json.loads(clean_content)
    except urllib.error.URLError as e:
        print(f"[ERROR] Could not connect to local Ollama at {OLLAMA_URL}. Ensure Ollama is running.")
        print(e)
        sys.exit(1)
    except Exception as e:
        print("[ERROR] Failed to parse Ollama response:")
        print(message_content if 'message_content' in locals() else "No response")
        print(e)
        return None


def run_training(max_iterations=150, num_envs=64):
    """Run training subprocess and capture its stdout for metric extraction."""
    cmd = [
        str(ISAACLAB_ROOT / "isaaclab.bat"),
        "-p",
        str(TRAIN_SCRIPT),
        "--task", "Isaac-H1-Table-Lift-v0",
        "--viz", "none",
        "--num_envs", str(num_envs),
        "--max_iterations", str(max_iterations)
    ]
    
    print(f"[EUREKA] Launching training run: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        cwd=str(ISAACLAB_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Parse metrics in real-time from output stream
    max_reward = -float("inf")
    last_ep_length = 0.0
    
    while True:
        line = process.stdout.readline()
        if not line:
            break
        print(line, end="") # Print training progress to console
        
        # Regex search for RSL_RL logging metrics
        reward_match = re.search(r"Mean reward:\s*([0-9.-]+)", line)
        if reward_match:
            val = float(reward_match.group(1))
            if val > max_reward:
                max_reward = val
                
        len_match = re.search(r"Mean episode length:\s*([0-9.-]+)", line)
        if len_match:
            last_ep_length = float(len_match.group(1))
            
    process.wait()
    return max_reward, last_ep_length


def main():
    parser = argparse.ArgumentParser(description="Eureka reward tuner via local Ollama.")
    parser.add_argument("--model", type=str, default=OLLAMA_MODEL, help="Local Ollama model name.")
    parser.add_argument("--iterations", type=int, default=5, help="Number of tuning iterations to run.")
    parser.add_argument("--train-steps", type=int, default=150, help="Number of training iterations per run.")
    parser.add_argument("--num-envs", type=int, default=64, help="Number of environments for training.")
    args = parser.parse_args()
    
    print("=" * 60)
    print("           EUREKA LOCAL OLLAMA REWARD TUNER")
    print("=" * 60)
    print(f"Target Config: {CFG_FILE}")
    print(f"Ollama Endpoint: {OLLAMA_URL} | Model: {args.model}")
    print(f"Max Tuning Iterations: {args.iterations}")
    print(f"Steps Per Run: {args.train_steps} | Envs: {args.num_envs}")
    print("=" * 60)
    
    best_reward = -float("inf")
    best_weights = {}
    
    history = []
    
    for i in range(args.iterations):
        print(f"\n--- TUNING ITERATION {i+1}/{args.iterations} ---")
        
        # 1. Get current weights
        current_weights = get_current_weights()
        print(f"[EUREKA] Current config weights: {current_weights}")
        
        # 2. Formulate optimization prompt
        if not history:
            prompt = (
                "You are tuning the reward weights for an H1 humanoid robot attempting to lift a box off a table. "
                "The current config weights are:\n"
                f"{json.dumps(current_weights, indent=2)}\n\n"
                "We want the robot to first walk up to the table (minimizing distance via 'robot_to_box_dist'), "
                "lift the box up (maximizing height via 'box_lift'), and then carry the box away from the table (maximizing distance via 'box_carry_away'). "
                "Suggest a set of reward weights to start this task. "
                "Provide the response ONLY in JSON format: {'robot_to_box_dist': float, 'box_lift': float, 'box_carry_away': float}."
            )
        else:
            last_run = history[-1]
            prompt = (
                "Here is the history of previous weight trials and their performance:\n"
                f"{json.dumps(history, indent=2)}\n\n"
                "Based on this history, modify the weights for 'robot_to_box_dist', 'box_lift', and 'box_carry_away' to improve "
                "both the max reward and the episode length (meaning it stays stable longer and does not drop the box). "
                "Provide the response ONLY in JSON format: {'robot_to_box_dist': float, 'box_lift': float, 'box_carry_away': float}."
            )
            
        # 3. Query Ollama for new weights
        print(f"[EUREKA] Querying local Ollama ({args.model})...")
        new_weights = query_ollama(prompt, args.model)
        
        if not new_weights:
            print("[EUREKA] Failed to get valid weights from Ollama, skipping iteration.")
            continue
            
        print(f"[EUREKA] Suggested weights: {new_weights}")
        
        # 4. Write weights to config
        update_config_weights(new_weights)
        
        # 5. Run training
        max_reward, ep_length = run_training(max_iterations=args.train_steps, num_envs=args.num_envs)
        print(f"[EUREKA] Run complete. Max Reward: {max_reward:.2f} | End Episode Length: {ep_length:.1f}")
        
        # 6. Save history
        run_record = {
            "iteration": i + 1,
            "weights": new_weights,
            "performance": {
                "max_reward": max_reward if max_reward != -float("inf") else -9999.0,
                "ep_length": ep_length
            }
        }
        history.append(run_record)
        
        # Track best weights
        if max_reward > best_reward:
            best_reward = max_reward
            best_weights = new_weights
            
    print("\n" + "=" * 60)
    print("                  TUNING COMPLETE")
    print("=" * 60)
    print(f"Best Max Reward achieved: {best_reward:.2f}")
    print(f"Best Weights configuration: {best_weights}")
    print("=" * 60)


if __name__ == "__main__":
    main()
