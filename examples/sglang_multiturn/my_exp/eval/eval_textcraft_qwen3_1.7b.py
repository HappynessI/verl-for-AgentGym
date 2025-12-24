#!/usr/bin/env python3
"""
TextCraft评估脚本 - 基于verl框架
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
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add verl to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from verl.interactions.textcraft_interaction import TextCraftInteraction

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleTextCraftAgent:
    """TextCraft Agent - 使用ADaPT风格的prompt和参数"""
    
    def __init__(self, model, tokenizer, max_length=8192, max_new_tokens=150, 
                 temperature=0.0, top_p=1.0, do_sample=False):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens  # ADaPT: 150
        self.temperature = temperature  # ADaPT: 0.0
        self.top_p = top_p  # ADaPT: 1.0
        self.do_sample = do_sample  # ADaPT: False (temperature=0)
        
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
        """构建新版提示词（双括号格式，严格action提取）"""
        return '''You are a Minecraft Assistant.

Core Protocol:

Think: First, output Think: ... to analyze the current situation.

Action: After deciding the next step, you MUST wrap the command inside [[ ]].

Format: Action: [[ command ]]

Example: Action: [[ inventory ]] or Action: [[ get 3 logs ]]

Core Command Set (API):

craft [target] using [ingredients]

get [item]

inventory

Interaction Example:

[Environment]
Goal: craft stick

[You]
Think: I need to check my inventory to see if I have planks.
Action: [[ inventory ]]

[Environment]
Inventory: [oak planks] (2)

[You]
Think: I have planks, so I can craft sticks directly.
Action: [[ craft 4 stick using 2 oak planks ]]

[Environment]
Crafted 4 sticks.

[You]
Think: Task completed.
Action: [[ Task Completed! ]]'''
    
    def generate(self, messages: List[Dict[str, str]]) -> str:
        """生成模型响应（ADaPT风格）
        
        注意：原本想用stop=['\n']来限制单次生成，但Qwen tokenizer会将换行符合并进复合token
        （如token 382 = '.\n\n'），导致从未生成单独的token 198（'\n'），stop机制失效。
        实际依赖max_new_tokens和模型从SFT学到的生成长度模式。
        """
        prompt = self._build_prompt(messages)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # ❌ Stop token机制已失效（Qwen tokenizer将\n合并进复合token，从未生成token 198）
        # newline_token_id = self.tokenizer.encode('\n', add_special_tokens=False)[0]
        
        with torch.no_grad():
            # ADaPT风格：temperature=0等价于do_sample=False（贪心解码）
            gen_kwargs = {
                'max_new_tokens': self.max_new_tokens,  # ADaPT: 150（实际控制生成长度）
                'do_sample': self.do_sample,  # ADaPT: False
                'pad_token_id': self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                # ❌ 注释掉无效的stop token配置
                # 'eos_token_id': [self.tokenizer.eos_token_id, newline_token_id],
            }
            
            # 只有do_sample=True时才传temperature和top_p
            if self.do_sample:
                gen_kwargs['temperature'] = self.temperature
                gen_kwargs['top_p'] = self.top_p
            
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
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
    max_rounds: int = 50,
    initial_prompt: str = None
) -> Dict[str, Any]:
    """评估一个episode
    
    Args:
        initial_prompt: 如果提供，则使用此固定prompt而不是从AgentGym获取新任务
                       这确保多次采样使用相同的任务
    """
    instance_id = f"eval_{session_id}"
    messages = []
    conversations = []
    done = False
    total_reward = 0.0  # 累积reward
    
    if initial_prompt is None:
        # 第一次采样：从AgentGym获取新任务
        try:
            await interaction.start_interaction(instance_id, session_id=session_id)
        except Exception as e:
            logger.error(f"Failed to start interaction for session {session_id}: {e}")
            return {
                "session_id": session_id,
                "reward": 0.0,
                "success": False,
                "num_turns": 0,
                "conversations": [],
                "initial_prompt": None
            }
        
        try:
            done, initial_obs, reward, extra = await interaction.generate_response(instance_id, messages)
            total_reward += reward  # 累积初始reward
            messages.append({"role": "user", "content": initial_obs})
            conversations.append({"role": "user", "content": initial_obs})
            initial_prompt = initial_obs  # 保存初始prompt供后续采样使用
        except Exception as e:
            logger.error(f"Failed to get initial observation: {e}")
            await interaction.finalize_interaction(instance_id)
            return {
                "session_id": session_id,
                "reward": 0.0,
                "success": False,
                "num_turns": 0,
                "conversations": [],
                "initial_prompt": None
            }
    else:
        # 后续采样：使用缓存的初始prompt
        try:
            await interaction.start_interaction(instance_id, session_id=session_id)
        except Exception as e:
            logger.error(f"Failed to start interaction for session {session_id}: {e}")
            return {
                "session_id": session_id,
                "reward": 0.0,
                "success": False,
                "num_turns": 0,
                "conversations": [],
                "initial_prompt": initial_prompt
            }
        
        # 直接使用缓存的prompt，不调用generate_response
        messages.append({"role": "user", "content": initial_prompt})
        conversations.append({"role": "user", "content": initial_prompt})
    
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
            
            # 检查终止条件（新格式：Action: [[ Task Completed! ]] 或 Think: task completed）
            response_lower = response.lower()
            if 'task completed' in response_lower:
                done = True
                logger.info(f"Session {session_id} Turn {turn}: Task Completed")
            elif 'task failed' in response_lower:
                done = True
                logger.info(f"Session {session_id} Turn {turn}: Task Failed")
            
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
        "conversations": conversations,
        "initial_prompt": initial_prompt
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
    logger.info("TextCraft Evaluation - Qwen3-1.7B")
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
    
    logger.info("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    device = "cuda:0"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map=device,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).eval()
    logger.info(f"✓ Model loaded on {device}")
    
    agent = SimpleTextCraftAgent(
        model, 
        tokenizer, 
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample
    )
    
    logger.info("=" * 80)
    logger.info("Generation Parameters (ADaPT风格):")
    logger.info(f"  max_length: {args.max_length}")
    logger.info(f"  max_new_tokens: {args.max_new_tokens} (ADaPT: 150)")
    logger.info(f"  temperature: {args.temperature} (ADaPT: 0.0)")
    logger.info(f"  top_p: {args.top_p} (ADaPT: 1.0)")
    logger.info(f"  do_sample: {args.do_sample} (ADaPT: False)")
    logger.info(f"  max_rounds: {args.max_rounds} (ADaPT: 40)")
    logger.info(f"  注意: stop token机制已失效（tokenizer将\\n合并进复合token），依赖max_new_tokens控制长度")
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
        # 使用 idx 作为 session_id，确保 shuffle 后每次采样的任务一致
        # 注意：不再使用 extra_info 中的 session_id，因为 shuffle 会打乱原始顺序
        session_id = idx
        
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Task {idx + 1}/{len(df)} - Session {session_id}")
        if args.num_samples_per_task > 1:
            logger.info(f"Will sample {args.num_samples_per_task} times for this task")
        logger.info(f"{'=' * 80}")
        
        # 对同一个任务采样多次
        task_results = []
        cached_initial_prompt = None  # 缓存第一次采样的初始prompt
        
        for sample_idx in range(args.num_samples_per_task):
            if args.num_samples_per_task > 1:
                logger.info(f"\n--- Sample {sample_idx + 1}/{args.num_samples_per_task} ---")
            
            try:
                result = await evaluate_one_episode(
                    agent=agent,
                    interaction=interaction,
                    session_id=session_id,
                    max_rounds=args.max_rounds,
                    initial_prompt=cached_initial_prompt  # 第一次为None，后续使用缓存
                )
                
                # 第一次采样后，缓存初始prompt供后续使用
                if sample_idx == 0 and result.get('initial_prompt'):
                    cached_initial_prompt = result['initial_prompt']
                    if args.num_samples_per_task > 1:
                        logger.info(f"Cached initial prompt for subsequent samples (Goal: {cached_initial_prompt.split('Goal: ')[1].split('.')[0] if 'Goal: ' in cached_initial_prompt else 'N/A'})")
                
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
TextCraft Evaluation Summary (Transformers)
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

