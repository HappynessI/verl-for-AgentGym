#!/usr/bin/env python3
"""
TextCraft评估脚本 - vLLM版本（高性能推理）
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import asyncio
import random
import numpy as np

import pyarrow.parquet as pq
from tqdm import tqdm
import requests
import torch
from vllm import LLM, SamplingParams

# Add verl to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from verl.interactions.textcraft_interaction import TextCraftInteraction

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleTextCraftAgent:
    """TextCraft Agent - 使用vLLM高性能推理 + ADaPT风格"""
    
    def __init__(self, llm, max_new_tokens=150, temperature=0.0, top_p=1.0):
        self.llm = llm  # vLLM的LLM对象
        self.max_new_tokens = max_new_tokens  # ADaPT: 150
        self.temperature = temperature  # ADaPT: 0.0
        self.top_p = top_p  # ADaPT: 1.0
        
        # vLLM的SamplingParams
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            stop=['\n'],  # ADaPT: 单行输出
        )
        
        # === 原始简单prompt（已注释） ===
        # self.system_prompt = (
        #     'You are an agent in the TextCraft environment. Your goal is to craft items by gathering resources and following recipes.\n\n'
        #     'You can use actions like:\n'
        #     '- move [direction]\n'
        #     '- get [object]\n'
        #     '- craft [item]\n'
        #     '- inventory\n'
        #     '- look\n\n'
        #     'Instructions:\n'
        #     '- Read the task instruction carefully\n'
        #     '- Gather necessary resources\n'
        #     '- Follow crafting recipes\n'
        #     '- Complete the crafting goal\n\n'
        #     'Your response should contain only the action.'
        # )
        
        # === ADaPT风格的few-shot prompt ===
        self.system_prompt = self._build_adapt_prompt()
    
    def _build_adapt_prompt(self) -> str:
        """构建ADaPT风格的few-shot prompt（参考ADaPT L252-393）"""
        return '''You are given few useful crafting recipes to craft items in Minecraft. Crafting commands are of the format "craft [target object] using [input ingredients]". You can either "fetch" an object (ingredients) from the inventory or the environment or "craft" (target) using any of the crafting commands. 

You can use ONLY these crafting commands provided, do not use your own crafting commands. However, if the crafting command uses a generic ingredient like "planks", you can use special types of the same ingredient e.g. "dark oak planks" in the command instead. 

For any other natural language or thoughts, use prefix 'think: '.

Here is a demo of how to fetch and craft objects.

Goal: craft dark oak sign

> think: I should check if I can fetch dark oak sign directly from the environment or the inventory.
OK.

> inventory: 
Inventory: [stick] (1) [dark oak planks] (8)

> get dark oak sign
Could not find dark oak sign

> think: I cannot get dark oak sign directly, I need to craft it. From the crafting commands, I can use: craft dark oak sign using 6 dark oak planks, 1 stick. Ingredients needed: 6 dark oak planks, 1 stick. Input assumption: I have all the neccessary ingredients in my inventory. Let me verify this first.
OK.

> inventory
Inventory: [stick] (1) [dark oak planks] (8)

> think: I found my ingredients: 6 dark oak planks, 1 stick in my inventory. My assumption is true, I can proceed. I will use the crafting command: craft dark oak sign using 6 dark oak planks
OK.

> craft 1 dark oak sign using 6 dark oak planks, 1 stick
Crafted 1 minecraft:dark_oak_sign

> inventory 
Inventory: [dark oak sign] (1)

> think: I now have dark oak sign in my inventory. Task Completed!
OK.

Goal: fetch 2 dark oak logs.

> think: I should check my inventory first, to see if I already have dark oak sign. Otherwise, I will directly try to get it from the environment.
OK.

> inventory
Inventory: [stick] (1)

> get 2 dark oak logs.
Got 2 dark oak logs

> inventory
Inventory: [dark oak log] (2) [stick] (1)

> think: I have 2 dark oak logs in my inventory. Task Completed!
OK.

Now here is a different goal. You can use these crafting commands to accomplish the goal. When you have the desired item in your inventory, think: Task Completed! If you have tried your best but cannot proceed, think: task failed!'''
    
    def generate(self, messages: List[Dict[str, str]]) -> str:
        """生成模型响应（vLLM + ADaPT风格：stop at newline）"""
        prompt = self._build_prompt(messages)
        
        # vLLM推理：直接传入prompt字符串
        outputs = self.llm.generate([prompt], self.sampling_params)
        
        # 提取生成的文本
        response = outputs[0].outputs[0].text
        
        # Debug: 打印原始输出
        if len(response) < 50:
            logger.debug(f"Raw response (len={len(response)}): repr={repr(response)}")
        
        return response.strip()
    
    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        """构建prompt - ADaPT风格不使用chat template，直接拼接"""
        # ADaPT使用简单的字符串拼接，不需要chat template
        prompt_parts = []
        
        # 添加system prompt（ADaPT的few-shot examples）
        prompt_parts.append(self.system_prompt)
        prompt_parts.append("\n\n")
        
        # 添加对话历史（ADaPT格式：直接拼接，user的内容前加\n，assistant的内容后加\n）
        for msg in messages:
            if msg["role"] == "user":
                # User message (环境观察)
                prompt_parts.append(msg["content"])
                prompt_parts.append("\n>")  # ADaPT格式：每行以 > 开头
            elif msg["role"] == "assistant":
                # Assistant message (agent action/think)
                prompt_parts.append(" ")
                prompt_parts.append(msg["content"])
                prompt_parts.append("\n>")
        
        prompt = "".join(prompt_parts)
        return prompt


async def evaluate_one_episode(
    agent: SimpleTextCraftAgent,
    interaction: TextCraftInteraction,
    session_id: int,
    max_rounds: int = 50
) -> Dict[str, Any]:
    """评估一个episode"""
    instance_id = f"eval_{session_id}"
    messages = []
    conversations = []
    done = False
    total_reward = 0.0  # 累积reward
    
    try:
        await interaction.start_interaction(instance_id, session_id=session_id)
    except Exception as e:
        logger.error(f"Failed to start interaction for session {session_id}: {e}")
        return {
            "session_id": session_id,
            "reward": 0.0,
            "success": False,
            "num_turns": 0,
            "conversations": []
        }
    
    try:
        done, initial_obs, reward, extra = await interaction.generate_response(instance_id, messages)
        total_reward += reward  # 累积初始reward
        messages.append({"role": "user", "content": initial_obs})
        conversations.append({"role": "user", "content": initial_obs})
    except Exception as e:
        logger.error(f"Failed to get initial observation: {e}")
        await interaction.finalize_interaction(instance_id)
        return {
            "session_id": session_id,
            "reward": 0.0,
            "success": False,
            "num_turns": 0,
            "conversations": []
        }
    
    # === 验证：打印首轮完整prompt ===
    if session_id == 0:
        first_prompt = agent._build_prompt(messages)
        print("="*80)
        print("首轮完整Prompt (用于验证system prompt传递):")
        print("="*80)
        print(first_prompt[:3000] + "..." if len(first_prompt) > 3000 else first_prompt)
        print("="*80)
        print(f"Prompt total length: {len(first_prompt)} chars")
        print("="*80)
    
    for turn in range(max_rounds):
        if done:
            break
        
        try:
            response = agent.generate(messages)
            
            # === ADaPT风格处理：检查think:前缀 ===
            response_stripped = response.lstrip('> ').strip()
            
            # 如果是think:，处理终止条件
            if response_stripped.startswith('think:'):
                if 'task completed' in response_stripped.lower():
                    done = True
                    logger.info(f"Session {session_id} Turn {turn}: Task Completed (from think)")
                elif 'task failed' in response_stripped.lower():
                    done = True
                    logger.info(f"Session {session_id} Turn {turn}: Task Failed (from think)")
            
            messages.append({"role": "assistant", "content": response})
            conversations.append({"role": "assistant", "content": response})
        except Exception as e:
            logger.error(f"Generation error at turn {turn}: {e}")
            break
        
        try:
            done, observation, step_reward, extra = await interaction.generate_response(instance_id, messages)
            total_reward += step_reward  # 累积每步的reward
            messages.append({"role": "user", "content": observation})
            conversations.append({"role": "user", "content": observation})
        except Exception as e:
            logger.error(f"Environment error at turn {turn}: {e}")
            break
    
    # 不再调用calculate_score，直接使用累积的reward
    # try:
    #     total_reward = await interaction.calculate_score(instance_id)
    # except:
    #     total_reward = 0.0
    
    try:
        await interaction.finalize_interaction(instance_id)
    except Exception as e:
        logger.warning(f"Failed to finalize: {e}")
    
    success = total_reward > 0.0
    num_turns = len([m for m in messages if m["role"] == "assistant"])
    
    logger.info("=" * 80)
    logger.info(f"Task Completed - Session {session_id}")
    logger.info(f"Reward: {total_reward:.4f}")
    logger.info(f"Success: {success}")
    logger.info(f"Turns: {num_turns}")
    logger.info("=" * 80)
    
    return {
        "session_id": session_id,
        "reward": total_reward,
        "success": success,
        "num_turns": num_turns,
        "conversations": conversations
    }


async def main():
    parser = argparse.ArgumentParser(description='TextCraft Evaluation')
    parser.add_argument('--model_path', type=str, default='/Data/public/Qwen3-1.7B')
    parser.add_argument('--data_path', type=str, 
                       default='/Data/wyh/datasets/Verl-Data/textcraft/train.parquet')
    parser.add_argument('--output_dir', type=str,
                       default='/Data/wyh/datasets/Verl-Data/outputs/textcraft_eval')
    parser.add_argument('--textcraft_server', type=str,
                       default='http://127.0.0.1:36002')
    parser.add_argument('--max_rounds', type=int, default=40,
                       help='Max rounds per episode (ADaPT: 40)')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--max_length', type=int, default=8192)
    # === ADaPT风格参数（已修改默认值） ===
    parser.add_argument('--max_new_tokens', type=int, default=150,
                       help='Maximum new tokens per response (ADaPT: 150)')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Sampling temperature (ADaPT: 0.0)')
    parser.add_argument('--top_p', type=float, default=1.0,
                       help='Top-p sampling (ADaPT: 1.0)')
    parser.add_argument('--do_sample', action='store_true', default=False,
                       help='Whether to use sampling (ADaPT: False)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--num_samples_per_task', type=int, default=1,
                       help='Number of times to sample each task (default: 1, for multiple runs use 8)')
    # vLLM GPU配置
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9,
                       help='GPU memory utilization for vLLM (default: 0.9)')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"eval_results_{timestamp}.jsonl")
    summary_file = os.path.join(args.output_dir, f"eval_summary_{timestamp}.txt")
    
    logger.info("=" * 80)
    logger.info("TextCraft Evaluation - Qwen3-1.7B (vLLM)")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Data: {args.data_path}")
    logger.info(f"Output: {output_file}")
    logger.info(f"TextCraft Server: {args.textcraft_server}")
    logger.info(f"Random Seed: {args.seed}")
    logger.info(f"Samples per Task: {args.num_samples_per_task}")
    logger.info("=" * 80)
    
    try:
        response = requests.get(args.textcraft_server, timeout=5)
        logger.info("✓ TextCraft server is accessible")
    except Exception as e:
        logger.error(f"✗ Cannot connect to TextCraft server: {e}")
        return
    
    logger.info("Loading model with vLLM...")
    logger.info(f"  GPU memory utilization: {args.gpu_memory_utilization}")
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_length,
        tensor_parallel_size=1,  # 单GPU
    )
    logger.info("✓ vLLM model loaded")
    
    agent = SimpleTextCraftAgent(
        llm=llm,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    
    logger.info("=" * 80)
    logger.info("vLLM Configuration:")
    logger.info(f"  gpu_memory_utilization: {args.gpu_memory_utilization} (使用GPU显存比例)")
    logger.info(f"  tensor_parallel_size: 1 (单GPU)")
    logger.info(f"  dtype: bfloat16")
    logger.info("=" * 80)
    logger.info("Generation Parameters (ADaPT风格):")
    logger.info(f"  max_length: {args.max_length}")
    logger.info(f"  max_new_tokens: {args.max_new_tokens} (ADaPT: 150)")
    logger.info(f"  temperature: {args.temperature} (ADaPT: 0.0)")
    logger.info(f"  top_p: {args.top_p} (ADaPT: 1.0)")
    logger.info(f"  do_sample: {args.do_sample} (ADaPT: False)")
    logger.info(f"  stop_tokens: ['\\n'] (implemented as eos_token_id)")
    logger.info(f"  max_rounds: {args.max_rounds} (ADaPT: 40)")
    logger.info("=" * 80)
    
    logger.info("Initializing TextCraft interaction...")
    interaction = TextCraftInteraction({
        'env_server_base': args.textcraft_server,
        'timeout': 600,
        'max_retries': 3
    })
    logger.info("✓ Interaction initialized")
    
    logger.info("Loading evaluation data...")
    table = pq.read_table(args.data_path)
    df = table.to_pandas()
    
    # 使用固定随机种子进行 shuffle，确保每次测试的任务顺序相同
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    logger.info(f"Data shuffled with seed={args.seed}")
    
    if args.max_samples > 0:  # -1表示全部，不截取
        df = df.head(args.max_samples)
    
    logger.info(f"Total samples: {len(df)}")
    
    logger.info("=" * 80)
    logger.info("Starting evaluation")
    logger.info("=" * 80)
    
    results = []
    total_reward = 0.0
    total_success = 0
    task_stats = []  # 记录每个任务的统计信息
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        extra_info = row.get('extra_info', {})
        if isinstance(extra_info, dict):
            interaction_kwargs = extra_info.get('interaction_kwargs', {})
            session_id = interaction_kwargs.get('session_id', idx)
        else:
            session_id = idx
        
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Task {idx + 1}/{len(df)} - Session {session_id}")
        if args.num_samples_per_task > 1:
            logger.info(f"Will sample {args.num_samples_per_task} times for this task")
        logger.info(f"{'=' * 80}")
        
        # 对同一个任务采样多次
        task_results = []
        for sample_idx in range(args.num_samples_per_task):
            if args.num_samples_per_task > 1:
                logger.info(f"\n--- Sample {sample_idx + 1}/{args.num_samples_per_task} ---")
            
            try:
                result = await evaluate_one_episode(
                    agent=agent,
                    interaction=interaction,
                    session_id=session_id,
                    max_rounds=args.max_rounds
                )
                
                task_results.append(result)
                results.append(result)
                total_reward += result['reward']
                total_success += 1 if result['success'] else 0
                
                # 保存每次采样的结果
                with open(output_file, 'a') as f:
                    f.write(json.dumps({
                        "item_id": f"textcraft_{session_id}",
                        "session_id": session_id,
                        "sample_idx": sample_idx,
                        "reward": result['reward'],
                        "success": result['success'],
                        "num_turns": result['num_turns'],
                        "conversations": result['conversations']
                    }) + '\n')
                
                if args.num_samples_per_task > 1:
                    logger.info(f"Sample {sample_idx + 1} result: Success={result['success']}, Reward={result['reward']}, Turns={result['num_turns']}")
            
            except Exception as e:
                logger.error(f"Error evaluating session {session_id} sample {sample_idx}: {e}")
                logger.exception(e)
        
        # 计算该任务的统计信息
        if task_results:
            task_success_rate = sum(1 for r in task_results if r['success']) / len(task_results)
            task_avg_reward = sum(r['reward'] for r in task_results) / len(task_results)
            task_avg_turns = sum(r['num_turns'] for r in task_results) / len(task_results)
            
            task_stats.append({
                'task_id': session_id,
                'num_samples': len(task_results),
                'success_rate': task_success_rate,
                'avg_reward': task_avg_reward,
                'avg_turns': task_avg_turns
            })
            
            if args.num_samples_per_task > 1:
                logger.info(f"\nTask {idx + 1} Summary:")
                logger.info(f"  Success Rate: {task_success_rate:.2%} ({sum(1 for r in task_results if r['success'])}/{len(task_results)})")
                logger.info(f"  Avg Reward: {task_avg_reward:.4f}")
                logger.info(f"  Avg Turns: {task_avg_turns:.1f}")
    
    avg_reward = total_reward / len(results) if results else 0.0
    success_rate = total_success / len(results) if results else 0.0
    
    # 多次采样的统计信息
    if args.num_samples_per_task > 1 and task_stats:
        task_level_success_rate = sum(t['success_rate'] for t in task_stats) / len(task_stats)
        task_level_avg_reward = sum(t['avg_reward'] for t in task_stats) / len(task_stats)
        
        multi_sample_info = f"""
Multi-Sampling Statistics (Task-Level Average):
------------------------------------------------
Samples per Task: {args.num_samples_per_task}
Number of Tasks: {len(task_stats)}
Total Evaluations: {len(results)}

Task-Level Metrics (averaged across tasks):
  Success Rate: {task_level_success_rate:.4f} ({task_level_success_rate:.2%})
  Average Reward: {task_level_avg_reward:.4f}

Sample-Level Metrics (all evaluations):
  Success Rate: {success_rate:.4f} ({total_success}/{len(results)})
  Average Reward: {avg_reward:.4f}
"""
    else:
        multi_sample_info = ""
    
    summary = f"""
{'=' * 80}
TextCraft Evaluation Summary (vLLM)
{'=' * 80}
Model: {args.model_path}
Data: {args.data_path}
Total samples: {len(results)}
Max rounds: {args.max_rounds}
Random seed: {args.seed}
{multi_sample_info}
Results:
--------
Average Reward: {avg_reward:.4f}
Success Rate: {success_rate:.4f} ({total_success}/{len(results)})

Output files:
-------------
Results: {output_file}
Summary: {summary_file}
{'=' * 80}
"""
    
    print(summary)
    logger.info(summary)
    
    with open(summary_file, 'w') as f:
        f.write(summary)
    
    # 如果有多次采样，保存 task-level 统计信息
    if args.num_samples_per_task > 1 and task_stats:
        task_stats_file = os.path.join(args.output_dir, f"eval_task_stats_{timestamp}.json")
        with open(task_stats_file, 'w') as f:
            json.dump(task_stats, f, indent=2)
        logger.info(f"✓ Task-level statistics saved to {task_stats_file}")
    
    logger.info(f"✓ Evaluation complete!")
    logger.info(f"✓ Results saved to {output_file}")
    logger.info(f"✓ Summary saved to {summary_file}")


if __name__ == "__main__":
    asyncio.run(main())

