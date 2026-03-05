#!/usr/bin/env python3
"""
SearchQA Evaluation Client - vLLM Service Mode
搜索问答环境评估脚本
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

from verl.interactions.searchqa_interaction import SearchQAInteraction

# 日志配置
log_dir = Path("/Data/wyh/datasets/Verl-Data/outputs/searchqa_eval/logs")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / f"eval_client_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger("SearchQAEval")


class AsyncSearchQAAgent:
    """SearchQA Agent - Uses HTTP calls to vLLM service"""
    
    def __init__(self, server_url: str, model_name: str = "qwen3", max_new_tokens: int = 512, temperature: float = 1.0, top_p: float = 1.0):
        self.server_url = server_url.rstrip('/')
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = self._build_prompt()
    
    def _build_prompt(self) -> str:
        return '''You are a SearchQA Agent. Your goal is to answer questions by searching and reading web pages.

**CORE PROTOCOL:**
1. **THINK FIRST**: Analyze the question and plan your search strategy.
2. **ONE ACTION**: Output exactly ONE action per turn.
3. **BOX FORMAT**: Wrap your command in `[[ ]]`. Example: `Action: [[ search[machine learning] ]]`

**CORE COMMAND SET:**
* `search[keywords]` - Search for information with keywords
* `click[number]` - Click on a search result by number
* `answer[your answer]` - Submit your final answer

**EXAMPLE:**
[Environment]
Question: What is the capital of France?

[You]
Think: I need to search for information about France's capital.
Action: [[ search[capital of France] ]]

[Environment]
Search results:
[1] Paris - Wikipedia: Paris is the capital and most populous city of France...
[2] France - Wikipedia: France is a country in Western Europe...

[You]
Think: The first result mentions Paris is the capital. Let me verify by clicking.
Action: [[ click[1] ]]

[Environment]
Paris is the capital and most populous city of France, with an estimated population of 2,102,650...

[You]
Think: I have confirmed that Paris is the capital of France.
Action: [[ answer[Paris] ]]
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
    agent: AsyncSearchQAAgent,
    interaction: SearchQAInteraction,
    session_id: int,
    http_session: aiohttp.ClientSession,
    max_rounds: int = 15
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
    interaction = SearchQAInteraction({
        'env_server_base': args.env_server,
        'timeout': 600, 'max_retries': 3
    })
    agent = AsyncSearchQAAgent(args.vllm_server_url, model_name=args.model_name, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p)
    
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
            record = {"item_id": f"searchqa_{session_id}", "session_id": session_id, "sample_idx": sample_idx, **result}
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
SearchQA Evaluation Summary
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
    parser = argparse.ArgumentParser(description='SearchQA Evaluation')
    parser.add_argument('--data_path', default='/Data/wyh/datasets/Verl-Data/eval/searchqa/test.parquet')
    parser.add_argument('--output_dir', default='/Data/wyh/datasets/Verl-Data/outputs/searchqa_eval')
    parser.add_argument('--env_server', default='http://127.0.0.1:36003')
    parser.add_argument('--vllm_server_url', default='http://localhost:8000')
    parser.add_argument('--model_name', type=str, default='qwen3',
                        help='Model name as registered in vLLM (default: qwen3)')
    parser.add_argument('--max_rounds', type=int, default=15)
    parser.add_argument('--max_samples', type=int, default=-1)
    parser.add_argument('--num_samples_per_task', type=int, default=1)
    parser.add_argument('--concurrency', type=int, default=16)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    asyncio.run(run_evaluation(args))


if __name__ == "__main__":
    main()

