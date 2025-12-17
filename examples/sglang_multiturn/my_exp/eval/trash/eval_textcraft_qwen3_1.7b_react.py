"""
TextCraft评估脚本 - ADaPT风格配置
使用ADaPT的prompt和推理参数
- 贪婪解码 (temperature=0, max_tokens=150)
- 单行输出 (stop=['\n'])
- Few-shot examples
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
                max_model_len=16384  # 增加到16K以支持长对话
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
        
        # ADaPT完整的few-shot prompt（与24%成功率测试完全一致）
        self.system_prompt = '''You are given few useful crafting recipes to craft items in Minecraft. Crafting commands are of the format "craft [target object] using [input ingredients]". You can either "fetch" an object (ingredients) from the inventory or the environment or "craft" (target) using any of the crafting commands. 

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
        
        logger.info(f"模型加载完成")
        logger.info(f"参数: max_new_tokens={max_new_tokens}, temp={temperature}, top_p={top_p}, do_sample={do_sample}")

    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        """ADaPT风格：直接字符串拼接，不使用chat template"""
        # 构建prompt：system prompt + 对话历史
        prompt = self.system_prompt + "\n\n"
        
        for msg in messages:
            if msg['role'] == 'user':
                prompt += msg['content'] + "\n\n"
            elif msg['role'] == 'assistant':
                prompt += msg['content'] + "\n\n"
        
        # ADaPT使用">"作为生成提示符
        prompt += ">"
        return prompt
    
    def generate(self, messages: List[Dict[str, str]]) -> str:
        """生成回复"""
        prompt = self._build_prompt(messages)
        
        # 检查prompt长度
        prompt_tokens = len(self.tokenizer.encode(prompt))
        if prompt_tokens > 15000:  # 留1000+ tokens给回复
            logger.warning(f"Prompt过长 ({prompt_tokens} tokens)，可能接近上限")
        
        if self.use_vllm:
            # vLLM推理（ADaPT风格：单行输出）
            sampling_params = SamplingParams(
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_new_tokens,
                stop=['\n'],  # ADaPT: 只生成单行
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
    从ADaPT风格输出中提取action
    
    ADaPT期待格式:
    > think: some思考过程
    OK.
    
    > inventory
    或:
    > craft 1 item using ingredients
    """
    text = text.strip()
    
    # ADaPT格式：按行分割，查找以">"开头的action行
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        
        # 跳过空行和"OK."
        if not line or line == 'OK.':
            continue
        
        # 如果以">"开头
        if line.startswith('>'):
            action_text = line[1:].strip()
            
            # 跳过think:行
            if action_text.startswith('think:'):
                continue
            
            # 跳过inventory:行（这是observation的回显，不是action）
            if action_text.startswith('inventory:'):
                continue
            
            # 检查是否是有效的action
            if any(keyword in action_text.lower() for keyword in ['craft ', 'get ', 'inventory', 'look']):
                return action_text
    
    # Fallback：如果没有">"，直接查找有效的action关键词
    for line in lines:
        line = line.strip()
        if any(keyword in line.lower() for keyword in ['craft ', 'get ', 'inventory', 'look']):
            # 但跳过think:行
            if not line.lower().startswith('think:'):
                return line
    
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
        
        # ADaPT风格：检查think:中的任务完成/失败标记
        response_lower = response.lower()
        if 'task completed!' in response_lower:
            logger.info(f"Session {session_id} Turn {turn}: 检测到'Task Completed!'")
            done = True
            total_reward = 1.0  # ADaPT: 任务完成给予reward
            conversations.append({"role": "assistant", "content": response})
            break
        elif 'task failed!' in response_lower:
            logger.info(f"Session {session_id} Turn {turn}: 检测到'task failed!'")
            done = True
            conversations.append({"role": "assistant", "content": response})
            break
        
        # 提取action
        action = extract_action_from_react(response)
        
        if action is None:
            consecutive_failures += 1
            logger.warning(f"Session {session_id} Turn {turn}: 无法提取action ({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}) from: {response[:100]}")
            
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logger.error(f"Session {session_id}: 连续{MAX_CONSECUTIVE_FAILURES}次action提取失败，终止")
                break
            
            # ADaPT风格：简单提示
            observation = "Invalid action. Use one of: get [quantity] [item], craft [quantity] [item] using [ingredients], inventory"
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
                "content": messages[-3]["content"] if len(messages) >= 3 else initial_observation
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

