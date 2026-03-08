#!/usr/bin/env python3
"""
SciWorld Evaluation Client - vLLM Service Mode
科学实验环境评估脚本
"""

import os
import sys
import json
import logging
import argparse
import uuid
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict

import pyarrow.parquet as pq
from tqdm import tqdm
import aiohttp
import fcntl

# Add verl to path
project_root = Path(__file__).parent.parent.parent.parent.parent
if not (project_root / "verl").exists():
    raise RuntimeError(f"verl not found in {project_root}")
sys.path.insert(0, str(project_root))

from verl.interactions.sciworld_interaction import SciWorldInteraction

# 日志配置
log_dir = Path("/Data/wyh/datasets/Verl-Data/outputs/sciworld_eval/logs")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / f"eval_client_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger("SciWorldEval")


class AsyncSciWorldAgent:
    """SciWorld Agent - Uses HTTP calls to vLLM service"""
    
    def __init__(self, server_url: str, model_name: str = "qwen3", max_new_tokens: int = 200, temperature: float = 1.0, top_p: float = 1.0):
        self.server_url = server_url.rstrip('/')
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = self._build_prompt()
    
    def _build_prompt(self) -> str:
        return '''You are a SciWorld Agent. Your goal is to complete science experiments in a text-based environment.

**CRITICAL: EXPLORE TO FIND ITEMS**
- Items you need are often in OTHER ROOMS, not the starting room!
- ALWAYS explore different rooms (kitchen, bathroom, workshop, etc.) to find required substances/tools
- NEVER give up just because an item isn't visible - GO FIND IT!

**NAVIGATION (Most Important):**
1. First: `open door to <room>` - Open a closed door
2. Then: `move to <room>` - Enter the room
3. Then: `look around` - See what's in the room

**CORE COMMAND SET:**
* `open door to <location>` - Open a door (MUST do before moving)
* `move to <location>` - Navigate to a location (door must be open)
* `look around` - Observe surroundings
* `take <object>` - Pick up an object
* `put <object> in <container>` - Place an object
* `use <object> on <target>` - Use an object
* `focus on <object>` - Focus on a substance/object
* `activate <device>` - Turn on a device (stove, sink, etc.)
* `deactivate <device>` - Turn off a device
* `inventory` - Check what you're carrying

**COMMON ITEM LOCATIONS:**
- Kitchen: thermometer, stove, freezer, fridge, sink, soap, food items
- Workshop/Foundry: heat sources, metals, tools
- Bathroom: water, sink, containers
- Greenhouse: plants, soil, water sources

**ACTION FORMAT:** Always wrap command in `[[ ]]`
Example: `Action: [[ open door to kitchen ]]`

**EXAMPLE WORKFLOW:**
Task: Freeze soap.
1. Soap is usually in the kitchen → `[[ open door to kitchen ]]`
2. After door opens → `[[ move to kitchen ]]`
3. Find soap → `[[ focus on soap ]]`
4. Find freezer → `[[ open freezer ]]`
5. Put soap in freezer → `[[ put soap in freezer ]]`

REMEMBER: If you don't see an item, EXPLORE other rooms! Don't give up!
'''
    
    async def generate(self, messages: List[Dict[str, str]], session: aiohttp.ClientSession) -> str:
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "messages": [{"role": "system", "content": self.system_prompt}, *messages],
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": False
        }
        
        try:
            async with session.post(
                f"{self.server_url}/v1/chat/completions",
                json=payload, headers=headers,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                if response.status != 200:
                    logger.error(f"HTTP {response.status}: {await response.text()}")
                    return ""
                result = await response.json()
                return result["choices"][0]["message"]["content"].strip() if result.get("choices") else ""
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return ""


async def evaluate_one_episode(
    agent: AsyncSciWorldAgent,
    interaction: SciWorldInteraction,
    session_id: int,
    http_session: aiohttp.ClientSession,
    max_rounds: int = 30
) -> Dict[str, Any]:
    instance_id = f"eval_{session_id}_{uuid.uuid4().hex[:8]}"
    messages, conversations = [], []
    done, total_reward = False, 0.0
    
    try:
        await interaction.start_interaction(instance_id, session_id=session_id)
        done, initial_obs, reward, _ = await interaction.generate_response(instance_id, messages)
        total_reward += reward
        messages.append({"role": "user", "content": initial_obs})
        conversations.append({"role": "user", "content": initial_obs})
    except Exception as e:
        logger.error(f"Start failed for session {session_id}: {e}")
        return {"session_id": session_id, "reward": 0.0, "success": False, "num_turns": 0, "conversations": [], "error": str(e)}

    for turn in range(max_rounds):
        if done:
            logger.info(f"Session {session_id} done at turn {turn}, total_reward={total_reward}")
            break
        try:
            response = await agent.generate(messages, http_session)
            if 'task completed' in response.lower():
                done = True
            messages.append({"role": "assistant", "content": response})
            conversations.append({"role": "assistant", "content": response})
            
            done, observation, step_reward, _ = await interaction.generate_response(instance_id, messages)
            total_reward += step_reward
            messages.append({"role": "user", "content": observation})
            conversations.append({"role": "user", "content": observation})
            
            # Debug: log when done becomes True
            if done:
                logger.warning(f"Session {session_id}: ENV returned done=True at turn {turn}, step_reward={step_reward}, obs='{observation[:100]}...'")
        except Exception as e:
            logger.error(f"Error session {session_id} turn {turn}: {e}")
            break
    
    try:
        await interaction.finalize_interaction(instance_id)
    except:
        pass
    
    return {
        "session_id": session_id,
        "reward": total_reward,
        "success": total_reward > 0.0,
        "num_turns": len([m for m in messages if m["role"] == "assistant"]),
        "conversations": conversations
    }


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    if n < k: return None
    if c == 0: return 0.0
    if n - c < k: return 1.0
    prob = 1.0
    for i in range(k):
        prob *= (n - c - i) / (n - i)
    return 1.0 - prob


def safe_write_record(output_file: str, record: Dict):
    with open(output_file, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(record) + '\n')
        fcntl.flock(f, fcntl.LOCK_UN)


async def fetch_model_name(server_url: str) -> str:
    """Fetches the actual model name from vLLM /v1/models endpoint"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{server_url.rstrip('/')}/v1/models",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get("data", [])
                    if models:
                        model_info = models[0]
                        model_id = model_info.get("id", "unknown")
                        model_root = model_info.get("root", "unknown")
                        return f"{model_id} ({model_root})"
                    return "unknown (no models found)"
                else:
                    logger.warning(f"Failed to query /v1/models: HTTP {response.status}")
                    return f"unknown (HTTP {response.status})"
    except Exception as e:
        logger.warning(f"Failed to fetch model name from vLLM: {e}")
        return f"unknown ({e})"


async def run_evaluation(args):
    interaction = SciWorldInteraction({
        'env_server_base': args.env_server,
        'timeout': 600, 'max_retries': 3
    })
    agent = AsyncSciWorldAgent(args.vllm_server_url, args.model_name, args.max_new_tokens, args.temperature, args.top_p)
    
    logger.info(f"Loading dataset: {args.data_path}")
    df = pq.read_table(args.data_path).to_pandas()
    df['original_index'] = df.index
    
    if 0 < args.max_samples < len(df):
        df = df.sample(n=args.max_samples, random_state=args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"eval_results_{timestamp}.jsonl")
    summary_file = os.path.join(args.output_dir, f"eval_summary_{timestamp}.txt")
    open(output_file, 'w').close()
    
    sem = asyncio.Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    
    async def worker(session_id, sample_idx):
        async with sem:
            result = await evaluate_one_episode(agent, interaction, session_id, session, args.max_rounds)
            record = {"item_id": f"sciworld_{session_id}", "session_id": session_id, "sample_idx": sample_idx,
                      "reward": result['reward'], "success": result['success'], "num_turns": result['num_turns']}
            if not args.no_save_trajectories:
                record["conversations"] = result.get('conversations', [])
            await asyncio.to_thread(safe_write_record, output_file, record)
            return result
    
    results = []
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [worker(int(row['original_index']), i) for _, row in df.iterrows() for i in range(args.num_samples_per_task)]
        pbar = tqdm(total=len(tasks), desc="Evaluating")
        for f in asyncio.as_completed(tasks):
            res = await f
            if res: results.append(res)
            pbar.update(1)
        pbar.close()
    
    # Summary
    grouped = defaultdict(list)
    for r in results: grouped[r['session_id']].append(r)
    
    total_tasks = len(grouped)
    avg_reward = sum(sum(r['reward'] for r in runs)/len(runs) for runs in grouped.values()) / total_tasks if total_tasks else 0
    avg_success = sum(sum(1 for r in runs if r['success'])/len(runs) for runs in grouped.values()) / total_tasks if total_tasks else 0
    
    model_name = await fetch_model_name(args.vllm_server_url)
    logger.info(f"Model name from vLLM: {model_name}")
    
    summary = f"""{'='*60}
SciWorld Evaluation Summary
{'='*60}
Model: {model_name}
Server: {args.vllm_server_url}
Dataset: {args.data_path}
Total Tasks: {total_tasks}
Average Reward: {avg_reward:.4f}
Average Success: {avg_success:.4f}
{'='*60}"""
    print(summary)
    with open(summary_file, 'w') as f: f.write(summary)
    logger.info(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='SciWorld Evaluation')
    parser.add_argument('--data_path', default='/Data/wyh/datasets/Verl-Data/eval/sciworld/test.parquet')
    parser.add_argument('--output_dir', default='/Data/wyh/datasets/Verl-Data/outputs/sciworld_eval')
    parser.add_argument('--env_server', default='http://127.0.0.1:36002')
    parser.add_argument('--vllm_server_url', default='http://localhost:8000')
    parser.add_argument('--model_name', default='qwen3')
    parser.add_argument('--max_rounds', type=int, default=30)
    parser.add_argument('--max_samples', type=int, default=-1)
    parser.add_argument('--num_samples_per_task', type=int, default=1)
    parser.add_argument('--concurrency', type=int, default=16)
    parser.add_argument('--max_new_tokens', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_save_trajectories', action='store_true',
                        help='Do not save conversation trajectories')
    
    asyncio.run(run_evaluation(args))


if __name__ == "__main__":
    main()

