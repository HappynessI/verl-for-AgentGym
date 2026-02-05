"""
TextCraft评估脚本 - Qwen2.5-7B (复现AgentGym实验)
采用AgentGym的ReAct格式和配置
- 模型：Qwen2.5-7B-Instruct
- 格式：Thought + Action（ReAct）
- 模板：chatml
- 推理：vLLM
"""

import os
import json
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Optional
import re
import random
import numpy as np

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# 尝试导入vLLM
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning("vLLM未安装，将使用transformers推理（速度较慢）")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleTextCraftAgent:
    """简单的TextCraft Agent - ReAct格式"""
    
    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 512,  # 允许完整的thought+action输出
        temperature: float = 0.7,   # 适度随机性
        top_p: float = 0.9,
        do_sample: bool = True,     # 启用采样
        device: str = "cuda",
        use_vllm: bool = True       # 是否使用vLLM
    ):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.use_vllm = use_vllm and VLLM_AVAILABLE
        
        logger.info(f"加载模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side='left'
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 选择推理引擎
        if self.use_vllm:
            logger.info("使用vLLM推理引擎（更快）")
            self.model = LLM(
                model=model_path,
                trust_remote_code=True,
                dtype="float16",
                gpu_memory_utilization=0.9,
                max_model_len=32768  # Qwen2.5-7B支持32K上下文
            )
        else:
            logger.info("使用transformers推理引擎")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            self.model.eval()
        
        # AgentGym ReAct风格的system prompt (带ADaPT few-shot examples)
        self.system_prompt = """You are given few useful crafting recipes to craft items in Minecraft. Crafting commands are of the format "craft [target object] using [input ingredients]".
Every round I will give you an observation, you have to respond an action based on the state and instruction. You can "get" an object (ingredients) from the inventory or the environment, look-up the game inventory by "inventory", or "craft" (target) using any of the crafting commands.

Your output must strictly follow this format:
"Thought:
your thoughts (keep it concise, 1-2 sentences).

Action:
your next action (ONLY ONE ACTION PER ROUND)"

Here are examples of how to solve tasks:

Example 1 - Crafting with ingredients in inventory:
Goal: craft dark oak sign

Thought:
I should check if I can get dark oak sign directly from the environment or inventory.

Action:
inventory

Observation: Inventory: [stick] (1) [dark oak planks] (8)

Thought:
I cannot get dark oak sign directly, I need to craft it. I have 6 dark oak planks and 1 stick needed for the recipe.

Action:
craft 1 dark oak sign using 6 dark oak planks, 1 stick

Observation: Crafted 1 minecraft:dark_oak_sign

Example 2 - Fetching from environment:
Goal: fetch 2 dark oak logs

Thought:
I should check my inventory first to see if I already have dark oak logs.

Action:
inventory

Observation: Inventory: [stick] (1)

Thought:
I don't have dark oak logs, I will try to get them from the environment.

Action:
get 2 dark oak logs

Observation: Got 2 dark oak logs

Example 3 - Crafting with missing ingredients:
Goal: craft 2 oak planks

Thought:
I should check my inventory first.

Action:
inventory

Observation: Inventory: [stick] (1)

Thought:
I don't have oak planks. I need to craft them using 1 oak log. Let me get the oak log first.

Action:
get 1 oak log

Observation: Got 1 oak log

Thought:
Now I have the oak log, I can craft 4 oak planks.

Action:
craft 4 oak planks using 1 oak log

Observation: Crafted 4 minecraft:oak_planks

CRITICAL REMINDERS: 
1. **Output EXACTLY ONE action per round**. Do NOT list multiple actions.
2. Always specify the quantity when using "get" and "craft" commands.
3. When using "get" command, do not specify whether the item comes from the inventory or the environment.
4. You can use ONLY crafting commands provided, do not use your own crafting commands. However, if the crafting command uses a generic ingredient like "planks", you can use special types of the same ingredient e.g. "dark oak planks" in the command instead.
5. If you cannot find an item after trying to get it, consider whether you need to craft it first or get its ingredients."""
        
        logger.info(f"模型加载完成")
        logger.info(f"参数: max_new_tokens={max_new_tokens}, temp={temperature}, top_p={top_p}, do_sample={do_sample}")

    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        """使用chat template构建prompt"""
        # 添加system prompt
        formatted_messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        formatted_messages.extend(messages)
        
        # 使用tokenizer的chat template (chatml格式)
        # Qwen2.5没有深度思考模式，使用标准chat template
        prompt = self.tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return prompt
    
    def generate(self, messages: List[Dict[str, str]]) -> str:
        """生成回复"""
        prompt = self._build_prompt(messages)
        
        # 检查prompt长度
        prompt_tokens = len(self.tokenizer.encode(prompt))
        if prompt_tokens > 15000:  # 留1000+ tokens给回复
            logger.warning(f"Prompt过长 ({prompt_tokens} tokens)，可能接近上限")
        
        if self.use_vllm:
            # vLLM推理
            sampling_params = SamplingParams(
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_new_tokens,
                # 不使用stop tokens，让模型自由生成
            )
            outputs = self.model.generate([prompt], sampling_params)
            response = outputs[0].outputs[0].text
        else:
            # transformers推理
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=self.do_sample,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
        
        return response.strip()


def extract_action_from_react(text: str) -> Optional[str]:
    """
    从ReAct格式输出中提取action
    
    期待格式:
    Thought: 
    some thinking...
    
    Action:
    the actual action
    """
    # 查找 Action: 标记后的内容
    action_match = re.search(r'Action:\s*\n?(.+?)(?:\n|$)', text, re.DOTALL)
    if action_match:
        action = action_match.group(1).strip()
        # 只取第一行作为action
        action = action.split('\n')[0].strip()
        return action
    
    # Fallback: 如果没有Action:标记，尝试直接查找action关键词
    for keyword in ['craft ', 'get ', 'inventory']:
        if keyword in text.lower():
            lines = text.split('\n')
            for line in lines:
                if keyword in line.lower():
                    return line.strip()
    
    return None


def evaluate_one_episode(
    agent: SimpleTextCraftAgent,
    textcraft_server: str,
    session_id: int,
    max_rounds: int = 50
) -> Dict:
    """评估一个episode"""
    import requests
    
    # 创建环境 (AgentGym使用/create端点，不需要预先指定goal)
    create_data = {
        "minecraft_dir": f"session_{session_id}/"
    }
    
    try:
        response = requests.post(f"{textcraft_server}/create", json=create_data, timeout=10)
        if response.status_code != 200:
            logger.error(f"Session {session_id}: 创建环境失败 - {response.text}")
            return {
                "session_id": session_id,
                "success": False,
                "reward": 0.0,
                "num_turns": 0,
                "error": "create_failed",
                "conversations": []
            }
        
        result = response.json()
        env_id = result["id"]
        initial_observation = result["observation"]
        
        logger.info(f"Session {session_id}: 环境已创建, env_id={env_id}")
        logger.info(f"Initial observation: {initial_observation[:200]}")
        
    except Exception as e:
        logger.error(f"Session {session_id}: 创建环境异常 - {e}")
        return {
            "session_id": session_id,
            "success": False,
            "reward": 0.0,
            "num_turns": 0,
            "error": str(e),
            "conversations": []
        }
    
    # 构建初始消息 (使用服务器返回的observation，包含goal和crafting_commands)
    messages = [{"role": "user", "content": initial_observation}]
    
    # 打印第一轮prompt用于调试
    if session_id == 0:
        first_prompt = agent._build_prompt(messages)
        logger.info(f"\n{'='*80}\n首轮完整Prompt:\n{'='*80}\n{first_prompt}\n{'='*80}")
    
    conversations = []
    total_reward = 0.0
    done = False
    consecutive_failures = 0  # 连续action提取失败次数
    MAX_CONSECUTIVE_FAILURES = 3  # 最多允许连续失败3次
    
    for turn in range(max_rounds):
        # 检查prompt长度，避免超出限制
        current_prompt = agent._build_prompt(messages)
        prompt_tokens = len(agent.tokenizer.encode(current_prompt))
        if prompt_tokens > 14000:  # 接近16K限制
            logger.warning(f"Session {session_id} Turn {turn}: Prompt过长 ({prompt_tokens} tokens)，提前终止")
            break
        
        # 生成回复
        response = agent.generate(messages)
        logger.debug(f"Session {session_id} Turn {turn}: Generated: {response[:200]}")
        
        # 提取action
        action = extract_action_from_react(response)
        
        if action is None:
            consecutive_failures += 1
            logger.warning(f"Session {session_id} Turn {turn}: 无法提取action ({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}) from: {response[:100]}")
            
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logger.error(f"Session {session_id}: 连续{MAX_CONSECUTIVE_FAILURES}次action提取失败，终止")
                break
            
            # 给模型明确的格式提示
            observation = "Please provide a valid action in the format:\nThought:\nyour thoughts.\n\nAction:\nyour next action"
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": observation})
            conversations.append({
                "role": "assistant",
                "content": response
            })
            conversations.append({
                "role": "user",
                "content": observation
            })
            continue
        
        # 成功提取action，重置失败计数
        consecutive_failures = 0
        
        logger.info(f"Session {session_id} Turn {turn}: Action: {action}")
        
        # 发送action到环境 (AgentGym需要env_id)
        try:
            step_data = {
                "id": env_id,
                "action": action
            }
            step_response = requests.post(f"{textcraft_server}/step", json=step_data, timeout=10)
            
            if step_response.status_code != 200:
                logger.error(f"Session {session_id} Turn {turn}: Step失败")
                break
                
            result = step_response.json()
            observation = result.get("observation", "")
            reward = result.get("reward", 0.0)
            done = result.get("done", False)
            
            logger.info(f"Session {session_id} Turn {turn}: Obs: {observation[:100]}, Reward: {reward}, Done: {done}")
            
            total_reward += reward
            
            # 更新对话历史
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": observation})
            
            conversations.append({
                "role": "user",
                "content": messages[-3]["content"] if len(messages) >= 3 else initial_message
            })
            conversations.append({
                "role": "assistant",
                "content": response
            })
            conversations.append({
                "role": "user",
                "content": observation
            })
            
            if done:
                logger.info(f"Session {session_id}: 完成！总reward={total_reward}")
                break
                
        except Exception as e:
            logger.error(f"Session {session_id} Turn {turn}: Step异常 - {e}")
            break
    
    # 重置环境 (AgentGym使用reset)
    try:
        requests.post(f"{textcraft_server}/reset", json={"id": env_id}, timeout=5)
    except:
        pass
    
    return {
        "session_id": session_id,
        "success": done and total_reward > 0,
        "reward": total_reward,
        "num_turns": len(conversations) // 3,  # 每3条消息一轮
        "conversations": conversations
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="/Data/wyh/datasets/Verl-Data/eval/textcraft/test.parquet")
    parser.add_argument("--output_dir", type=str, default="/Data/wyh/datasets/Verl-Data/outputs/textcraft_eval")
    parser.add_argument("--textcraft_server", type=str, default="http://127.0.0.1:36002")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--max_rounds", type=int, default=50)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--do_sample", action="store_true", default=True)
    parser.add_argument("--use_vllm", action="store_true", default=True, help="使用vLLM加速推理")
    parser.add_argument("--no_vllm", dest="use_vllm", action="store_false", help="不使用vLLM")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    logger.info(f"设置随机种子: {args.seed}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"eval_results_{timestamp}.jsonl")
    
    logger.info("=" * 80)
    logger.info("TextCraft评估 - ReAct格式")
    logger.info("=" * 80)
    logger.info(f"模型: {args.model_path}")
    logger.info(f"数据: {args.data_path}")
    logger.info(f"输出: {output_file}")
    logger.info(f"TextCraft服务器: {args.textcraft_server}")
    logger.info(f"Max rounds: {args.max_rounds}")
    logger.info(f"Max new tokens: {args.max_new_tokens}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Top-p: {args.top_p}")
    logger.info(f"Do sample: {args.do_sample}")
    logger.info(f"Use vLLM: {args.use_vllm}")
    
    # 加载数据
    df = pd.read_parquet(args.data_path)
    if args.max_samples > 0:
        df = df.head(args.max_samples)
    
    logger.info(f"加载了 {len(df)} 个测试样本")
    
    # 初始化agent
    agent = SimpleTextCraftAgent(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        use_vllm=args.use_vllm
    )
    
    # 评估
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="评估进度"):
        logger.info(f"\n{'='*80}")
        logger.info(f"样本 {idx}/{len(df)}: {row.get('item_id', f'sample_{idx}')}")
        
        result = evaluate_one_episode(
            agent=agent,
            textcraft_server=args.textcraft_server,
            session_id=idx,
            max_rounds=args.max_rounds
        )
        
        result["item_id"] = row.get("item_id", f"textcraft_{idx}")
        results.append(result)
        
        # 保存结果
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    # 统计
    total = len(results)
    success = sum(1 for r in results if r['success'])
    avg_reward = sum(r['reward'] for r in results) / total if total > 0 else 0
    
    logger.info("\n" + "=" * 80)
    logger.info("评估完成")
    logger.info("=" * 80)
    logger.info(f"总样本数: {total}")
    logger.info(f"成功数: {success}")
    logger.info(f"成功率: {success/total*100:.2f}%")
    logger.info(f"平均Reward: {avg_reward:.4f}")
    logger.info(f"结果已保存到: {output_file}")
    
    # 保存summary
    summary_file = output_file.replace('.jsonl', '_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"TextCraft评估摘要 - ReAct格式\n")
        f.write(f"{'='*80}\n")
        f.write(f"模型: {args.model_path}\n")
        f.write(f"总样本数: {total}\n")
        f.write(f"成功数: {success}\n")
        f.write(f"成功率: {success/total*100:.2f}%\n")
        f.write(f"平均Reward: {avg_reward:.4f}\n")
        f.write(f"参数: temp={args.temperature}, top_p={args.top_p}, max_new_tokens={args.max_new_tokens}\n")


if __name__ == "__main__":
    main()

