#!/usr/bin/env python3
"""
API-based sampling script for TextCraft environment.
使用 box format 提示词格式与环境交互，采样轨迹用于 SFT 训练。

适配 verl 框架，与 AgentGym 的 textcraft 环境服务器交互。
"""

import argparse
import json
import logging
import os
import re
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import requests
from tqdm import tqdm
from openai import OpenAI


# =============================================================================
# Box Format System Prompt
# =============================================================================

SYSTEM_PROMPT = '''You are a Minecraft Assistant. Your goal is to craft items by managing resources and recipes.

**CORE PROTOCOL (Strictly Follow):**
1. **THINK FIRST**: Before any action, analyze the current state using the "Reasoning Logic" below.
2. **ONE ACTION**: Output exactly ONE action per turn.
3. **BOX FORMAT**: Wrap your command in `[[ ]]`. Example: `Action: [[ inventory ]]`
4. **NO HALLUCINATION**: Do NOT simulate the Environment's response. Stop immediately after outputting the Action.

**REASONING LOGIC (The Algorithm):**
When trying to acquire an item [Target]:
1. **Check Inventory**: Do you already have [Target]?
   - If YES -> Task Complete / Proceed to next step.
   - If NO -> Go to step 2.
2. **Check Recipe**: Is there a crafting recipe for [Target]?
   - If YES -> Check if you have the [Ingredients].
     - If you have Ingredients -> `craft [Target] ...`
     - If you miss Ingredients -> **NEW SUB-GOAL**: Get the missing [Ingredient] (Repeat Logic from Step 1).
   - If NO (Base Material) -> `get [Target]` directly from environment.
3. **Handle Errors**:
   - If `get` fails -> The item might require crafting from a sub-ingredient (e.g., gold ingot needs gold nuggets). Check recipes again.
   - If `craft` fails -> Check if you have the exact items.

**CORE COMMAND SET (API):**
* `craft [target] using [ingredients]` (e.g., "craft 1 stick using 2 bamboo")
* `get [item]` (e.g., "get 3 log")
* `inventory` (Check what you have)

**INTERACTION EXAMPLES:**

[Environment]
Goal: craft powered rail
Recipes:
- craft 6 powered rail using 1 redstone, 1 stick, 6 gold ingot
- craft 1 gold ingot using 9 gold nugget
- craft 1 stick using 2 bamboo

[You]
Think: Recipe for powered rail needs redstone, stick, gold ingot. I need to check what I have.
Action: [[ inventory ]]

[Environment]
Inventory: [redstone] (1)

[You]
Think: I have redstone. I miss stick and gold ingot. Gold ingot has a recipe (needs gold nugget). I should get gold nugget first.
Action: [[ get 54 gold nugget ]]

[Environment]
Got 54 gold nugget.

[You]
Think: Now I have nuggets. I can craft gold ingots.
Action: [[ craft 6 gold ingot using 54 gold nugget ]]

[Environment]
Crafted 6 gold ingot.

[You]
Think: Now I need a stick. Recipe says stick needs bamboo. I need to get bamboo.
Action: [[ get 2 bamboo ]]

[Environment]
Got 2 bamboo.

[You]
Think: I have bamboo. I can craft the stick.
Action: [[ craft 1 stick using 2 bamboo ]]

[Environment]
Crafted 1 stick.

[You]
Think: I have redstone, gold ingots, and stick. I can finally craft the powered rail.
Action: [[ craft 6 powered rail using 1 redstone, 1 stick, 6 gold ingot ]]

[Environment]
Crafted 6 powered rail.

[You]
Think: Goal achieved.
Action: [[ Task Completed! ]]
'''


# =============================================================================
# API Agent Class
# =============================================================================

class APIAgent:
    """API-based agent using OpenAI-compatible API"""
    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        max_tokens: int = 16384,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = SYSTEM_PROMPT
    
    def generate(self, messages: List[Dict[str, str]]) -> str:
        """Generate response from API"""
        full_messages = [
            {"role": "system", "content": self.system_prompt},
            *messages
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"API call failed: {e}")
            raise


# =============================================================================
# TextCraft Environment Client
# =============================================================================

class TextCraftEnvClient:
    """Client for TextCraft environment server (AgentGym compatible)"""
    
    def __init__(self, server_base: str, timeout: int = 60):
        self.server_base = server_base.rstrip('/')
        self.timeout = timeout
        self.env_id = None
    
    def _post(self, endpoint: str, data: dict) -> dict:
        """Send POST request to environment server"""
        url = f"{self.server_base}/{endpoint}"
        if self.env_id is not None:
            data["id"] = self.env_id
        try:
            response = requests.post(url, json=data, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"Environment request failed: {e}")
            raise
    
    def create(self) -> str:
        """Create environment instance, return env_id"""
        result = self._post("create", {})
        self.env_id = result.get("id")
        return self.env_id
    
    def reset(self, task_idx: int) -> str:
        """Reset environment to specific task, return initial observation"""
        if self.env_id is None:
            self.create()
        
        result = self._post("reset", {"data_idx": task_idx})
        observation = result.get("observation", "")
        return observation
    
    def step(self, action: str) -> Tuple[str, float, bool]:
        """Execute action in environment, return (observation, reward, done)"""
        if self.env_id is None:
            raise RuntimeError("No active environment. Call create/reset first.")
        
        # Clean action string (remove non-alphanumeric except comma and space)
        clean_action = re.sub(r"[^A-Za-z0-9, ]+", "", action)
        clean_action = " ".join(clean_action.split()).strip()
        
        result = self._post("step", {"action": clean_action})
        
        observation = result.get("observation", "")
        reward = result.get("reward", 0.0)
        done = result.get("done", False)
        
        return observation, reward, done
    
    def close(self):
        """Close environment"""
        if self.env_id is not None:
            try:
                self._post("close", {})
            except:
                pass
            self.env_id = None


# =============================================================================
# Action Parser
# =============================================================================

def parse_box_action(response: str) -> Optional[str]:
    """
    Parse action from box format response.
    Expected format: Action: [[ command ]]
    """
    # 匹配 [[ ... ]] 格式
    pattern = r'\[\[\s*(.+?)\s*\]\]'
    matches = re.findall(pattern, response, re.DOTALL)
    
    if matches:
        # 返回最后一个匹配（通常模型只输出一个action）
        return matches[-1].strip()
    
    return None


def is_task_completed(action: str) -> bool:
    """Check if action indicates task completion"""
    if not action:
        return False
    action_lower = action.lower()
    return 'task completed' in action_lower or 'task complete' in action_lower


# =============================================================================
# Sampling Logic
# =============================================================================

def sample_one_episode(
    agent: APIAgent,
    env_client: TextCraftEnvClient,
    task_idx: int,
    max_rounds: int,
    logger: logging.Logger,
) -> Tuple[List[Dict], float, int]:
    """
    Sample one episode.
    
    Returns:
        conversations: List of conversation turns
        reward: Final reward
        success: 1 if successful, 0 otherwise
    """
    conversations = []
    
    # Reset environment to task and get initial observation
    observation = env_client.reset(task_idx)
    logger.debug(f"Reset to task {task_idx}, initial obs: {observation[:100]}...")
    
    # Add initial observation as user message
    conversations.append({
        "role": "user",
        "content": observation,
        "reasoning_content": None
    })
    
    total_reward = 0.0
    success = 0
    
    try:
        for round_idx in range(max_rounds):
            # Generate response
            messages = [{"role": c["role"], "content": c["content"]} for c in conversations]
            response = agent.generate(messages)
            
            # Add assistant response
            conversations.append({
                "role": "assistant",
                "content": response,
                "reasoning_content": None
            })
            
            # Parse action
            action = parse_box_action(response)
            
            if action is None:
                # No valid action found, try to continue
                logger.warning(f"Round {round_idx}: No valid action found in response")
                # Add error message
                conversations.append({
                    "role": "user",
                    "content": "Invalid action format. Please use: Action: [[ command ]]",
                    "reasoning_content": None
                })
                continue
            
            # Check for task completion
            if is_task_completed(action):
                logger.debug(f"Round {round_idx}: Task completed signal received")
                # 任务完成时，从环境获取最终状态
                obs, reward, done = env_client.step("inventory")
                total_reward = reward
                if reward == 1 or reward == 100:
                    success = 1
                break
            
            # Execute action in environment
            obs, reward, done = env_client.step(action)
            total_reward = reward
            
            logger.debug(f"Round {round_idx}: Action={action[:50]}, Reward={reward}, Done={done}")
            
            # Add environment response
            conversations.append({
                "role": "user",
                "content": obs,
                "reasoning_content": None
            })
            
            if done:
                if reward == 1 or reward == 100:
                    success = 1
                break
    
    except Exception as e:
        logger.error(f"Episode error: {e}")
    finally:
        env_client.close()
    
    return conversations, total_reward, success


# =============================================================================
# Main
# =============================================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="API-based sampling for TextCraft (box format)")
    
    # API config
    parser.add_argument("--api_key", type=str, required=True, help="API Key")
    parser.add_argument("--base_url", type=str, default="https://once.novai.su/v1", help="API base URL")
    parser.add_argument("--model", type=str, default="[福利]gemini-3-flash", help="Model name")
    parser.add_argument("--max_tokens", type=int, default=16384, help="Max tokens")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p")
    
    # Task config
    parser.add_argument("--task_name", type=str, default="textcraft", help="Task name")
    parser.add_argument("--inference_file", type=str, 
                        default="/Data/wyh/datasets/AgentGym-RL-Data/train/textcraft_train.json",
                        help="Training data file")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    
    # Environment config
    parser.add_argument("--env_server_base", type=str, default="http://127.0.0.1:36001",
                        help="Environment server URL")
    parser.add_argument("--max_round", type=int, default=20, help="Max interaction rounds")
    parser.add_argument("--data_len", type=int, default=400, help="Number of tasks to sample")
    parser.add_argument("--timeout", type=int, default=60, help="Request timeout")
    
    # Sampling config
    parser.add_argument("--num_samples", type=int, default=4, help="Samples per task")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--start_idx", type=int, default=0, 
                        help="Start from this task index (0-based, e.g., 192 for textcraft_192)")
    
    return vars(parser.parse_args())


def setup_logger(args) -> logging.Logger:
    """Setup logging"""
    log_dir = "/Data/wyh/verl/examples/sglang_multiturn/my_exp/sample/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = args['model'].replace('[', '').replace(']', '').replace('/', '_')
    log_file = os.path.join(log_dir, f"{args['task_name']}_{safe_model}_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Log file: {log_file}")
    return logger


def main(args):
    logger = setup_logger(args)
    logger.info("=" * 80)
    logger.info("TextCraft Sampling (Box Format)")
    logger.info("=" * 80)
    logger.info(f"Config:\n{json.dumps(args, indent=2, ensure_ascii=False)}")
    
    # Setup output directory
    if args["output_dir"] is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model = args['model'].replace('[', '').replace(']', '').replace('/', '_')
        args["output_dir"] = f"/Data/wyh/datasets/Sampling-Data/{args['task_name']}_{safe_model}_{timestamp}"
    os.makedirs(args["output_dir"], exist_ok=True)
    logger.info(f"Output dir: {args['output_dir']}")
    
    # Load task data
    with open(args["inference_file"], "r") as f:
        task_data = json.load(f)
    
    if args["data_len"] > 0:
        task_data = task_data[:args["data_len"]]
    
    # Apply start_idx to skip earlier tasks
    start_idx = args.get("start_idx", 0)
    if start_idx > 0:
        task_data = task_data[start_idx:]
        logger.info(f"Starting from index {start_idx}, skipped {start_idx} tasks")
    
    logger.info(f"Loaded {len(task_data)} tasks (start_idx={start_idx})")
    
    # Initialize agent
    agent = APIAgent(
        api_key=args["api_key"],
        base_url=args["base_url"],
        model=args["model"],
        max_tokens=args["max_tokens"],
        temperature=args["temperature"],
        top_p=args["top_p"],
    )
    logger.info(f"Agent initialized: model={args['model']}")
    
    # Output file
    output_file = os.path.join(args["output_dir"], f"{args['task_name']}_trajectories.jsonl")
    
    # Resume support: load existing trajectories
    completed_counts = defaultdict(int)
    total_score = 0.0
    total_success = 0.0
    completed_count = 0
    
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    traj = json.loads(line)
                    item_id = traj["item_id"]
                    completed_counts[item_id] += 1
                    total_score += traj["reward"]
                    total_success += traj["success"]
                    completed_count += 1
                except:
                    continue
        logger.info(f"Resumed: {completed_count} trajectories, {len(completed_counts)} tasks")
    
    target_samples = args["num_samples"]
    start_time = time.time()
    
    # Sampling loop
    logger.info("=" * 80)
    logger.info("Starting sampling...")
    
    for idx, task_item in enumerate(tqdm(task_data, desc="Sampling")):
        item_id = task_item["item_id"]
        # task_idx is the original index for environment reset (considering start_idx offset)
        task_idx = idx + start_idx
        
        current_count = completed_counts[item_id]
        if current_count >= target_samples:
            logger.debug(f"[{idx+1}] {item_id}: already {current_count}/{target_samples}, skip")
            continue
        
        needed = target_samples - current_count
        logger.info(f"[{idx+1}/{len(task_data)}] {item_id} (task_idx={task_idx}): need {needed} samples")
        
        for sample_idx in range(needed):
            actual_sample_idx = current_count + sample_idx
            
            retry_count = 0
            max_retries = 3
            success_flag = False
            
            while retry_count < max_retries:
                try:
                    logger.info(f"  -> Sample {actual_sample_idx+1}/{target_samples}")
                    
                    # Create environment client for each attempt
                    env_client = TextCraftEnvClient(
                        server_base=args["env_server_base"],
                        timeout=args["timeout"]
                    )
                    
                    conversations, reward, success = sample_one_episode(
                        agent=agent,
                        env_client=env_client,
                        task_idx=task_idx,
                        max_rounds=args["max_round"],
                        logger=logger,
                    )
                    
                    success_flag = True
                    break
                    
                except Exception as e:
                    retry_count += 1
                    error_msg = str(e)
                    
                    if "content" in error_msg.lower() and "filter" in error_msg.lower():
                        logger.error(f"  -> Content filter triggered, skip")
                        break
                    
                    logger.error(f"  -> Error (attempt {retry_count}/{max_retries}): {error_msg[:200]}")
                    if retry_count < max_retries:
                        time.sleep(5)
                    else:
                        logger.error(f"  -> Max retries reached, skip")
            
            if not success_flag:
                continue
            
            # Save trajectory
            result = {
                "conversations": conversations,
                "item_id": item_id,
                "sample_idx": actual_sample_idx,
                "reward": reward,
                "success": success,
                "task_name": args["task_name"],
                "model": args["model"],
            }
            
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            
            completed_counts[item_id] += 1
            total_score += reward
            total_success += success
            completed_count += 1
            
            logger.info(f"  -> Sample {actual_sample_idx+1}: reward={reward:.3f}, success={success}")
        
        # Periodic stats
        if completed_count % 20 == 0 and completed_count > 0:
            elapsed = time.time() - start_time
            logger.info(f"--- Progress: {completed_count} samples, "
                       f"avg_reward={total_score/completed_count:.4f}, "
                       f"success_rate={total_success/completed_count:.4f}, "
                       f"time={elapsed:.1f}s ---")
    
    # Final stats
    elapsed = time.time() - start_time
    final_score = total_score / completed_count if completed_count > 0 else 0
    final_success = total_success / completed_count if completed_count > 0 else 0
    
    logger.info("=" * 80)
    logger.info("Sampling Complete!")
    logger.info(f"Total samples: {completed_count}")
    logger.info(f"Average reward: {final_score:.4f}")
    logger.info(f"Success rate: {final_success:.4f}")
    logger.info(f"Total time: {elapsed:.2f}s ({elapsed/60:.2f}min)")
    logger.info(f"Trajectory file: {output_file}")
    
    # Save summary
    summary = {
        "task_name": args["task_name"],
        "model": args["model"],
        "total_tasks": len(task_data),
        "completed_samples": completed_count,
        "average_reward": final_score,
        "success_rate": final_success,
        "total_time_seconds": elapsed,
        "avg_time_per_sample": elapsed / completed_count if completed_count > 0 else 0,
        "trajectory_file": output_file,
        "action_format": "box_format",
        "config": args,
    }
    
    summary_file = os.path.join(args["output_dir"], "summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)
    logger.info(f"Summary saved: {summary_file}")


if __name__ == "__main__":
    args = parse_args()
    main(args)

