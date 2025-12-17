#!/usr/bin/env python3
"""
统一的AgentGym环境评估脚本
支持切换不同环境：webshop, babyai, alfworld, sciworld, sqlgym, textcraft, searchqa
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

# 导入所有环境的interaction类
from verl.interactions.webshop_interaction import WebshopInteraction
from verl.interactions.babyai_interaction import BabyAIInteraction
from verl.interactions.alfworld_interaction import ALFWorldInteraction
from verl.interactions.sciworld_interaction import SciWorldInteraction
from verl.interactions.sqlgym_interaction import SQLGymInteraction
from verl.interactions.textcraft_interaction import TextCraftInteraction
from verl.interactions.searchqa_interaction import SearchQAInteraction

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 环境配置
ENV_CONFIGS = {
    'webshop': {
        'interaction_class': WebshopInteraction,
        'default_port': 36003,
        'system_prompt': (
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
            "Remember that your keywords in search should be carefully designed."
        )
    },
    'babyai': {
        'interaction_class': BabyAIInteraction,
        'default_port': 36001,
        'system_prompt': (
            "You are controlling a robot in a grid world.\n"
            "I will give you instructions about what to do.\n"
            "You have to follow the instructions.\n"
            "Every round I will give you an observation, you have to respond with an action.\n"
            "Valid actions: turn left, turn right, go forward, pick up, drop, toggle.\n"
            "Respond with only the action."
        )
    },
    'alfworld': {
        'interaction_class': ALFWorldInteraction,
        'default_port': 36002,
        'system_prompt': (
            "You are an AI agent in a household environment.\n"
            "I will give you a task to complete.\n"
            "You have to interact with objects to complete the task.\n"
            "Every round I will give you an observation, you have to respond with an action.\n"
            "Example actions: go to <object>, take <object> from <receptacle>, "
            "put <object> in/on <receptacle>, examine <object>.\n"
            "Respond with your action."
        )
    },
    'sciworld': {
        'interaction_class': SciWorldInteraction,
        'default_port': 36004,
        'system_prompt': (
            "You are conducting a scientific experiment.\n"
            "I will give you a task to complete.\n"
            "You have to interact with scientific equipment and perform experiments.\n"
            "Every round I will give you an observation, you have to respond with an action.\n"
            "Example actions: move to <location>, take <object> from <location>, "
            "use <object> on <target>, read <object>, pour <liquid> from/to <container>.\n"
            "Respond with your action."
        )
    },
    'sqlgym': {
        'interaction_class': SQLGymInteraction,
        'default_port': 36005,
        'system_prompt': (
            "You are an SQL database assistant.\n"
            "I will give you a natural language question about a database.\n"
            "You have to write SQL queries to answer the question.\n"
            "Every round I will give you feedback, you can refine your query.\n"
            "Respond with a valid SQL query."
        )
    },
    'textcraft': {
        'interaction_class': TextCraftInteraction,
        'default_port': 36006,
        'system_prompt': (
            "You are playing a text-based crafting game (like Minecraft).\n"
            "I will give you a crafting task to complete.\n"
            "You have to craft items by gathering resources and combining them.\n"
            "Every round I will give you an observation, you have to respond with an action.\n"
            "Actions are in format: craft(item), mine(resource), get(item), goto(location).\n"
            "Respond with your action."
        )
    },
    'searchqa': {
        'interaction_class': SearchQAInteraction,
        'default_port': 36007,
        'system_prompt': (
            "You are a search-based question answering agent.\n"
            "I will give you a question to answer.\n"
            "You can search the web and click on results to find information.\n"
            "Every round I will give you an observation, you have to respond with an action.\n"
            "Actions: search[keywords], click[number], answer[your answer].\n"
            "Respond with your action."
        )
    }
}


class AgentGymAgent:
    """统一的Agent类，支持所有AgentGym环境"""
    
    def __init__(self, model, tokenizer, system_prompt, max_length=4096):
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.max_length = max_length
    
    def generate(self, messages: List[Dict[str, str]]) -> str:
        """生成模型响应"""
        # 构建prompt
        prompt = self._build_prompt(messages)
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=800,
                temperature=0.1,  # 低温度确保简洁输出
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        """构建prompt"""
        formatted_messages = [{"role": "system", "content": self.system_prompt}]
        formatted_messages.extend(messages)
        
        prompt = self.tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return prompt


async def evaluate_one_episode(
    agent: AgentGymAgent,
    interaction,
    session_id: int,
    max_rounds: int = 30
) -> Dict[str, Any]:
    """评估一个episode"""
    instance_id = f"eval_{session_id}"
    messages = []
    conversations = []
    done = False
    
    # 启动interaction
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
    
    # 获取初始observation
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
    
    # 多轮交互循环
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
    
    # 获取最终reward
    try:
        total_reward = await interaction.calculate_score(instance_id)
    except:
        total_reward = 0.0
    
    # 清理
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
    parser = argparse.ArgumentParser(description='Unified AgentGym Environments Evaluation')
    parser.add_argument('--env', type=str, required=True,
                       choices=list(ENV_CONFIGS.keys()),
                       help='Environment to evaluate')
    parser.add_argument('--model_path', type=str, default='/Data/public/Qwen3-8B')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to preprocessed data (parquet)')
    parser.add_argument('--output_dir', type=str,
                       default='/Data/wyh/datasets/Verl-Data/outputs',
                       help='Directory to save evaluation results')
    parser.add_argument('--env_server', type=str, default=None,
                       help='Environment server URL (default: http://127.0.0.1:<port>)')
    parser.add_argument('--max_rounds', type=int, default=30)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--max_length', type=int, default=4096)
    
    args = parser.parse_args()
    
    # 获取环境配置
    env_config = ENV_CONFIGS[args.env]
    
    # 设置服务器URL
    if args.env_server is None:
        args.env_server = f"http://127.0.0.1:{env_config['default_port']}"
    
    # 创建输出目录
    env_output_dir = os.path.join(args.output_dir, f"{args.env}_eval")
    os.makedirs(env_output_dir, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(env_output_dir, f"eval_results_{timestamp}.jsonl")
    summary_file = os.path.join(env_output_dir, f"eval_summary_{timestamp}.txt")
    
    logger.info("=" * 80)
    logger.info(f"AgentGym {args.env.upper()} Evaluation")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Data: {args.data_path}")
    logger.info(f"Output: {output_file}")
    logger.info(f"Server: {args.env_server}")
    logger.info("=" * 80)
    
    # 检查服务器
    try:
        response = requests.get(args.env_server, timeout=5)
        logger.info(f"✓ {args.env} server is accessible")
    except Exception as e:
        logger.error(f"✗ Cannot connect to {args.env} server: {e}")
        return
    
    # 加载模型
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
    
    # 创建agent
    agent = AgentGymAgent(
        model, 
        tokenizer, 
        env_config['system_prompt'],
        max_length=args.max_length
    )
    
    # 初始化interaction
    logger.info(f"Initializing {args.env} interaction...")
    interaction_class = env_config['interaction_class']
    interaction = interaction_class({
        'env_server_base': args.env_server,
        'timeout': 600,
        'max_retries': 3
    })
    logger.info("✓ Interaction initialized")
    
    # 加载数据
    logger.info("Loading evaluation data...")
    table = pq.read_table(args.data_path)
    df = table.to_pandas()
    
    if args.max_samples:
        df = df.head(args.max_samples)
    
    logger.info(f"Total samples: {len(df)}")
    
    # 评估循环
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
            
            # 保存到jsonl
            with open(output_file, 'a') as f:
                f.write(json.dumps({
                    "env": args.env,
                    "item_id": f"{args.env}_{session_id}",
                    "session_id": session_id,
                    "reward": result['reward'],
                    "success": result['success'],
                    "num_turns": result['num_turns'],
                    "conversations": result['conversations']
                }) + '\n')
        
        except Exception as e:
            logger.error(f"Error evaluating session {session_id}: {e}")
            logger.exception(e)
    
    # 计算指标
    avg_reward = total_reward / len(results) if results else 0.0
    success_rate = total_success / len(results) if results else 0.0
    
    # 保存总结
    summary = f"""
{'=' * 80}
AgentGym {args.env.upper()} Evaluation Summary
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

