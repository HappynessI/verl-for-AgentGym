

#!/usr/bin/env python3
"""
TextCraft Evaluation Client - vLLM Service Mode
直接调用已部署的vLLM服务，无需本地加载模型
"""

import os
import sys
import json
import logging
import argparse
import uuid
import asyncio
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict

import pyarrow.parquet as pq
from tqdm import tqdm
import aiohttp
import fcntl

# Add verl to path
# Replace the old sys.path.insert line with this:

project_root = Path("/agent_distill")
if not (project_root / "verl").exists():
    raise RuntimeError(f"verl not found in {project_root}")
sys.path.insert(0, str(project_root))
# sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))



from verl.interactions.textcraft_interaction import TextCraftInteraction

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"/agent_distill/logs/eval_client_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger("TextCraftEval")

# =============================================================================
# Async Agent Class (HTTP Client)
# =============================================================================

class AsyncTextCraftAgent:
    """TextCraft Agent - Uses HTTP calls to vLLM service"""
    
    def __init__(self, server_url: str, max_new_tokens: int = 150, temperature: float = 0.0, top_p: float = 1.0):
        self.server_url = server_url.rstrip('/')
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = self._build_adapt_prompt()
    
    def _build_adapt_prompt(self) -> str:
        return '''You are a Minecraft Assistant. Your goal is to craft items efficiently. Minimize the number of turns. Do not fetch ingredients multiple times if you can fetch the total amount in one turn. Crafting commands are of the format "craft [target object] using [input ingredients]". You can either "fetch" an object (ingredients) from the inventory or the environment or "craft" (target) using any of the crafting commands. 
You can use ONLY these crafting commands provided, do not use your own crafting commands. However, if the crafting command uses a generic ingredient like "planks", you can use special types of the same ingredient e.g. "dark oak planks" in the command instead. 

**CORE PROTOCOL (Strictly Follow):**
1. **ONE ACTION**: Output exactly ONE action per turn.
2. **BOX FORMAT**: Wrap your command in `[[ ]]`. Example: `Action: [[ inventory ]]`

**CORE COMMAND SET (API):**
* `craft [target] using [ingredients]` (e.g., "craft 2 stick using 4 bamboo")
* `get [item]` (e.g., "get 10 log")
* `inventory` (Check what you have)
'''
    
    async def generate(self, messages: List[Dict[str, str]], session: aiohttp.ClientSession) -> str:
        """Generates response by calling vLLM service via HTTP"""
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": "qwen3",
            "messages": [
                {"role": "system", "content": self.system_prompt},
                *messages
            ],
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": False
        }
        
        try:
            async with session.post(
                f"{self.server_url}/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"HTTP {response.status} from vLLM: {error_text}")
                    return ""
                
                result = await response.json()
                if "choices" not in result or len(result["choices"]) == 0:
                    logger.error(f"Invalid response format: {result}")
                    return ""
                
                return result["choices"][0]["message"]["content"].strip()
                
        except asyncio.TimeoutError:
            logger.error("Request timed out after 120s")
            return ""
        except Exception as e:
            logger.error(f"HTTP request failed: {str(e)}")
            return ""


# =============================================================================
# Evaluation Logic
# =============================================================================

async def evaluate_one_episode(
    agent: AsyncTextCraftAgent,
    interaction: TextCraftInteraction,
    session_id: int,
    http_session: aiohttp.ClientSession,
    max_rounds: int = 50,
    initial_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """Evaluates a single episode asynchronously."""
    
    instance_id = f"eval_{session_id}_{uuid.uuid4().hex[:8]}"
    messages = []
    conversations = []
    done = False
    total_reward = 0.0
    
    # --- Initialization ---
    if initial_prompt is None:
        try:
            await interaction.start_interaction(instance_id, session_id=session_id)
            done, initial_obs, reward, extra = await interaction.generate_response(instance_id, messages)
            total_reward += reward
            messages.append({"role": "user", "content": initial_obs})
            conversations.append({"role": "user", "content": initial_obs})
            initial_prompt = initial_obs
        except Exception as e:
            logger.error(f"Start interaction failed for session {session_id}: {e}")
            return {
                "session_id": session_id,
                "reward": 0.0,
                "success": False,
                "num_turns": 0,
                "initial_prompt": None,
                "conversations": [],
                "error": str(e)
            }
    else:
        try:
            await interaction.start_interaction(instance_id, session_id=session_id)
            messages.append({"role": "user", "content": initial_prompt})
            conversations.append({"role": "user", "content": initial_prompt})
        except Exception as e:
            logger.error(f"Failed to start interaction for session {session_id}: {e}")
            return {
                "session_id": session_id,
                "reward": 0.0,
                "success": False,
                "num_turns": 0,
                "initial_prompt": initial_prompt,
                "conversations": [],
                "error": str(e)
            }

    # --- Interaction Loop ---
    for turn in range(max_rounds):
        if done:
            break
        
        try:
            response = await agent.generate(messages, http_session)
            response_lower = response.lower()
            if 'task completed' in response_lower or 'task failed' in response_lower:
                done = True
            messages.append({"role": "assistant", "content": response})
            conversations.append({"role": "assistant", "content": response})
        except Exception as e:
            logger.error(f"Generation error session {session_id} turn {turn}: {e}")
            break
        
        try:
            done, observation, step_reward, extra = await interaction.generate_response(instance_id, messages)
            total_reward += step_reward
            messages.append({"role": "user", "content": observation})
            conversations.append({"role": "user", "content": observation})
        except Exception as e:
            logger.error(f"Environment error session {session_id} turn {turn}: {e}")
            break
    
    try:
        await interaction.finalize_interaction(instance_id)
    except Exception as e:
        logger.warning(f"Finalization failed for session {session_id}: {e}")
    
    success = total_reward > 0.0
    num_turns = len([m for m in messages if m["role"] == "assistant"])
    
    return {
        "session_id": session_id,
        "reward": total_reward,
        "success": success,
        "num_turns": num_turns,
        "conversations": conversations,
        "initial_prompt": initial_prompt
    }


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """Estimates pass@k metric (k ≤ num_samples)"""
    if num_samples < k:
        return None
    if num_correct == 0:
        return 0.0
    if num_samples - num_correct < k:
        return 1.0
    
    prob_all_failure = 1.0
    for i in range(k):
        prob_all_failure *= (num_samples - num_correct - i) / (num_samples - i)
    return 1.0 - prob_all_failure


# =============================================================================
# Safe File Writing with Locking
# =============================================================================

def safe_write_record(output_file: str, record: Dict[str, Any]):
    """Safely appends a JSON record to a file using file locking."""
    try:
        with open(output_file, 'a') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(json.dumps(record) + '\n')
                f.flush()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except Exception as e:
        logger.error(f"Failed to write record to {output_file}: {e}")


# =============================================================================
# Main Evaluation Loop
# =============================================================================

async def run_evaluation(args: argparse.Namespace):
    """Main evaluation function with async HTTP client"""
    
    # Initialize components
    interaction = TextCraftInteraction({
        'env_server_base': args.textcraft_server,
        'timeout': 600,
        'max_retries': 3
    })
    
    agent = AsyncTextCraftAgent(
        server_url=args.vllm_server_url,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )
    
    # Load dataset
    logger.info(f"Loading dataset from: {args.data_path}")
    table = pq.read_table(args.data_path)
    df = table.to_pandas()
    df['original_index'] = df.index
    
    # Shuffle and sample
    if args.max_samples > 0 and args.max_samples < len(df):
        df = df.sample(n=args.max_samples, random_state=args.seed)
        logger.info(f"Using {args.max_samples} random samples")
    else:
        logger.info(f"Using full dataset ({len(df)} samples)")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"eval_results_{timestamp}.jsonl")
    summary_file = os.path.join(args.output_dir, f"eval_summary_{timestamp}.txt")
    
    logger.info(f"Output file: {output_file}")
    logger.info(f"Summary file: {summary_file}")
    
    # Clear output file
    open(output_file, 'w').close()
    
    # Setup HTTP client
    connector = aiohttp.TCPConnector(limit=args.concurrency, limit_per_host=args.concurrency)
    timeout = aiohttp.ClientTimeout(total=300)
    
    # Semaphore for concurrency control
    sem = asyncio.Semaphore(args.concurrency)
    
    async def worker(session_id: int, row: Dict, sample_idx: int):
        async with sem:
            try:
                result = await evaluate_one_episode(
                    agent=agent,
                    interaction=interaction,
                    session_id=session_id,
                    http_session=session,
                    max_rounds=args.max_rounds
                )
                
                # Prepare record
                record = {
                    "item_id": f"textcraft_{session_id}",
                    "session_id": session_id,
                    "sample_idx": sample_idx,
                    "reward": result['reward'],
                    "success": result['success'],
                    "num_turns": result['num_turns'],
                    "conversations": result['conversations'],
                    "initial_prompt": result['initial_prompt']
                }
                
                # Write to file
                await asyncio.to_thread(safe_write_record, output_file, record)
                return result
            except Exception as e:
                logger.error(f"Worker failed for session {session_id}: {str(e)}")
                return None
    
    # Run evaluation
    results = []
    total_tasks = len(df) * args.num_samples_per_task
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = []
        for _, row in df.iterrows():
            real_session_id = int(row['original_index'])
            for sample_idx in range(args.num_samples_per_task):
                tasks.append(worker(real_session_id, row, sample_idx))
        
        # Progress bar
        pbar = tqdm(total=total_tasks, desc="Evaluating", unit="sample")
        for f in asyncio.as_completed(tasks):
            res = await f
            if res:
                results.append(res)
            pbar.update(1)
        pbar.close()
    
    # Generate summary
    await generate_summary(output_file, summary_file, args)
    
    logger.info(f"✅ Evaluation completed! Results saved to:\n- {output_file}\n- {summary_file}")
    return output_file, summary_file


async def generate_summary(results_file: str, summary_file: str, args: argparse.Namespace):
    """Generates evaluation summary from results file"""
    # Load all results
    results = []
    try:
        with open(results_file, 'r') as f:
            for line in f:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON line: {e}")
    except Exception as e:
        logger.error(f"Failed to read results file: {e}")
        return
    
    if not results:
        logger.error("No valid results found to summarize!")
        return
    
    # Group by session_id
    grouped_results = defaultdict(list)
    for res in results:
        grouped_results[res['session_id']].append(res)
    
    total_tasks = len(grouped_results)
    total_samples = len(results)
    
    # Define k values for pass@k
    k_values = [1, 2, 4, 8]
    
    # Calculate metrics
    sum_avg_reward = 0.0
    sum_avg_success = 0.0
    pass_at_k_values = {k: [] for k in k_values}
    
    for session_id, task_runs in grouped_results.items():
        n = len(task_runs)
        c = sum(1 for r in task_runs if r.get('success', False))
        total_r = sum(r.get('reward', 0.0) for r in task_runs)
        
        if n > 0:
            avg_reward = total_r / n
            avg_success = c / n
            sum_avg_reward += avg_reward
            sum_avg_success += avg_success
        
        for k in k_values:
            if n >= k:
                pass_k = estimate_pass_at_k(n, c, k)
                pass_at_k_values[k].append(pass_k)
    
    global_avg_reward = sum_avg_reward / total_tasks if total_tasks > 0 else 0.0
    global_avg_success = sum_avg_success / total_tasks if total_tasks > 0 else 0.0
    
    # Prepare summary text
    metrics = [
        "=" * 60,
        "TextCraft Evaluation Summary (vLLM Service Mode)",
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        f"Model: {args.vllm_server_url}",
        f"Dataset: {args.data_path}",
        f"Total Tasks: {total_tasks}",
        f"Total Samples: {total_samples}",
        f"Samples Per Task: {args.num_samples_per_task}",
        "-" * 60,
        f"Average Reward: {global_avg_reward:.4f}",
        f"Average Success (Avg@1): {global_avg_success:.4f}",
        "-" * 60,
    ]
    
    for k in sorted(k_values):
        values = pass_at_k_values[k]
        if values:
            avg_pass_k = sum(values) / len(values)
            metrics.append(f"Pass@{k:<2}: {avg_pass_k:.4f} (tasks: {len(values)}/{total_tasks})")
        else:
            metrics.append(f"Pass@{k:<2}: N/A    (insufficient samples)")
    
    metrics.append("=" * 60)
    
    summary_text = "\n".join(metrics)
    print("\n" + summary_text + "\n")
    
    # Write to summary file
    try:
        with open(summary_file, 'w') as f:
            f.write(summary_text)
    except Exception as e:
        logger.error(f"Failed to write summary file: {e}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='TextCraft Evaluation Client (vLLM Service Mode)')
    
    # Paths
    parser.add_argument('--data_path', type=str, 
                        default='/agent_distill/Data/Verl-Data/eval/textcraft/test.parquet',
                        help='Path to test dataset (parquet format)')
    parser.add_argument('--output_dir', type=str,
                        default='/agent_distill/Data/Verl-Data/eval/textcraft/textcraft_eval',
                        help='Directory to save evaluation results')
    
    # Environment
    parser.add_argument('--textcraft_server', type=str,
                        default='http://127.0.0.1:3222',
                        help='TextCraft environment server URL')
    parser.add_argument('--vllm_server_url', type=str,
                        default='http://localhost:8000',
                        help='vLLM service URL (e.g., http://localhost:8000)')
    
    # Evaluation Config
    parser.add_argument('--max_rounds', type=int, default=40,
                        help='Maximum interaction rounds per episode')
    parser.add_argument('--max_samples', type=int, default=-1,
                        help='Maximum samples to evaluate (-1 for all)')
    parser.add_argument('--num_samples_per_task', type=int, default=8,
                        help='Number of samples per task (for pass@k)')
    parser.add_argument('--concurrency', type=int, default=256,
                        help='Maximum concurrent requests to vLLM service')
    
    # Generation Params
    parser.add_argument('--max_new_tokens', type=int, default=2000,
                        help='Maximum tokens to generate per response')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (0.0 for greedy)')
    parser.add_argument('--top_p', type=float, default=1.0,
                        help='Top-p sampling parameter')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Validate vLLM server URL
    if not args.vllm_server_url.startswith(('http://', 'https://')):
        args.vllm_server_url = f"http://{args.vllm_server_url}"
    
    # Safety check for temperature
    if args.num_samples_per_task > 1 and args.temperature == 0.0:
        logger.warning("⚠️ WARNING: num_samples_per_task > 1 but temperature=0.0. "
                      "Results will be identical for all samples. Consider increasing temperature.")
    
    logger.info("📋 Evaluation Configuration:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Run evaluation
    try:
        asyncio.run(run_evaluation(args))
    except KeyboardInterrupt:
        logger.info("🛑 Evaluation interrupted by user")
    except Exception as e:
        logger.exception(f"❌ Evaluation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
