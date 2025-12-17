#!/usr/bin/env python3
"""
Lightweight standalone evaluation script for Webshop.
最简版本：最小化配置，让模型和环境自然交互。
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

import pyarrow.parquet as pq
from tqdm import tqdm
import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add verl to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from verl.interactions.webshop_interaction import WebshopInteraction

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleWebshopAgent:
    """极简Agent：只负责调用模型生成"""
    
    def __init__(self, model, tokenizer, max_length=4096):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 添加强调action输出的prompt
        self.system_prompt = (
            "You are web shopping.\n"
            "I will give you instructions about what to do.\n"
            "You have to follow the instructions.\n"
            "Every round I will give you an observation and a list of available actions, "
            "you have to respond an action based on the state and instruction.\n"
            "You can use search action if search is available.\n"
            "You can click one of the buttons in clickables.\n"
            "An action should be of the following structure:\n"
            "search[keywords]\n"
            "click[value]\n"
            "If the action is not valid, perform nothing.\n"
            "Keywords in search are up to you, but the value in click must be a value in the list of available actions.\n"
            "Remember that your keywords in search should be carefully designed.\n\n"
            "IMPORTANT: You must ALWAYS output a valid action (search[...] or click[...]) in your response. "
            "Keep your thinking very brief (1-2 sentences max), then immediately provide an action."
        )
    
    def generate(self, messages: List[Dict[str, str]]) -> str:
        """生成模型响应"""
        # 构建prompt
        prompt = self._build_prompt(messages)
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate - 关键：极低的temperature强制简洁输出
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=800,   # 足够的空间输出思考+action
                temperature=0.1,      # 极低温度：让模型选择最可能（最简洁）的输出
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        """构建prompt - 使用标准的chat template"""
        formatted_messages = [{"role": "system", "content": self.system_prompt}]
        formatted_messages.extend(messages)
        
        prompt = self.tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return prompt


async def evaluate_one_episode(
    agent: SimpleWebshopAgent,
    interaction: WebshopInteraction,
    session_id: int,
    max_rounds: int = 25
) -> Dict[str, Any]:
    """评估一个episode"""
    instance_id = f"eval_{session_id}"
    messages = []
    conversations = []
    done = False
    
    # 1. 启动interaction
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
    
    # 2. 获取初始observation
    try:
        done, initial_obs, reward, extra = await interaction.generate_response(instance_id, messages)
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
    
    # 3. 多轮交互循环
    for turn in range(max_rounds):
        if done:
            break
        
        # Agent生成response
        try:
            response = agent.generate(messages)
            messages.append({"role": "assistant", "content": response})
            conversations.append({"role": "assistant", "content": response})
        except Exception as e:
            logger.error(f"Generation error at turn {turn}: {e}")
            break
        
        # 环境执行并返回observation
        try:
            done, observation, step_reward, extra = await interaction.generate_response(instance_id, messages)
            messages.append({"role": "user", "content": observation})
            conversations.append({"role": "user", "content": observation})
        except Exception as e:
            logger.error(f"Environment error at turn {turn}: {e}")
            break
    
    # 4. 获取最终reward
    try:
        total_reward = await interaction.calculate_score(instance_id)
    except:
        total_reward = 0.0
    
    # 5. 清理
    try:
        await interaction.finalize_interaction(instance_id)
    except Exception as e:
        logger.warning(f"Failed to finalize: {e}")
    
    # 计算结果
    success = total_reward == 1.0
    num_turns = len([m for m in messages if m["role"] == "assistant"])
    
    # 输出任务完成日志
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
    parser = argparse.ArgumentParser(description='Lightweight Webshop Evaluation')
    parser.add_argument('--model_path', type=str, default='/Data/public/Qwen3-8B')
    parser.add_argument('--data_path', type=str, 
                       default='/Data/wyh/datasets/Verl-Data/webshop/train.parquet')
    parser.add_argument('--output_dir', type=str,
                       default='/Data/wyh/datasets/Verl-Data/outputs/webshop_eval_lightweight')
    parser.add_argument('--webshop_server', type=str,
                       default='http://127.0.0.1:36001')
    parser.add_argument('--max_rounds', type=int, default=25)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--max_length', type=int, default=4096)
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"eval_results_{timestamp}.jsonl")
    summary_file = os.path.join(args.output_dir, f"eval_summary_{timestamp}.txt")
    
    logger.info("=" * 80)
    logger.info("Webshop Lightweight Evaluation")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Data: {args.data_path}")
    logger.info(f"Output: {output_file}")
    logger.info(f"Webshop Server: {args.webshop_server}")
    logger.info("=" * 80)
    
    # Check server
    try:
        response = requests.get(args.webshop_server, timeout=5)
        logger.info("✓ Webshop server is accessible")
    except Exception as e:
        logger.error(f"✗ Cannot connect to Webshop server: {e}")
        return
    
    # Load model
    logger.info("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map=device,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).eval()
    logger.info(f"✓ Model loaded on {device}")
    
    # Create agent
    agent = SimpleWebshopAgent(model, tokenizer, max_length=args.max_length)
    
    # Initialize interaction
    logger.info("Initializing Webshop interaction...")
    interaction = WebshopInteraction({
        'env_server_base': args.webshop_server,
        'timeout': 600,
        'max_retries': 3
    })
    logger.info("✓ Interaction initialized")
    
    # Load data
    logger.info("Loading evaluation data...")
    table = pq.read_table(args.data_path)
    df = table.to_pandas()
    
    if args.max_samples:
        df = df.head(args.max_samples)
    
    logger.info(f"Total samples: {len(df)}")
    
    # Evaluation loop
    logger.info("=" * 80)
    logger.info("Starting evaluation")
    logger.info("=" * 80)
    
    results = []
    total_reward = 0.0
    total_success = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        extra_info = row.get('extra_info', {})
        if isinstance(extra_info, dict):
            interaction_kwargs = extra_info.get('interaction_kwargs', {})
            session_id = interaction_kwargs.get('session_id', idx)
        else:
            session_id = idx
        
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Starting Task {idx + 1}/{len(df)} - Session {session_id}")
        logger.info(f"{'=' * 80}")
        
        try:
            result = await evaluate_one_episode(
                agent=agent,
                interaction=interaction,
                session_id=session_id,
                max_rounds=args.max_rounds
            )
            
            results.append(result)
            total_reward += result['reward']
            total_success += 1 if result['success'] else 0
            
            # Save to jsonl
            with open(output_file, 'a') as f:
                f.write(json.dumps({
                    "item_id": f"webshop_{session_id}",
                    "session_id": session_id,
                    "reward": result['reward'],
                    "success": result['success'],
                    "num_turns": result['num_turns'],
                    "conversations": result['conversations']
                }) + '\n')
        
        except Exception as e:
            logger.error(f"Error evaluating session {session_id}: {e}")
            logger.exception(e)
    
    # Calculate metrics
    avg_reward = total_reward / len(results) if results else 0.0
    success_rate = total_success / len(results) if results else 0.0
    
    # Save summary
    summary = f"""
{'=' * 80}
Webshop Evaluation Summary
{'=' * 80}
Model: {args.model_path}
Data: {args.data_path}
Total samples: {len(results)}
Max rounds: {args.max_rounds}

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
    
    logger.info(f"✓ Evaluation complete!")
    logger.info(f"✓ Results saved to {output_file}")
    logger.info(f"✓ Summary saved to {summary_file}")


if __name__ == "__main__":
    asyncio.run(main())
