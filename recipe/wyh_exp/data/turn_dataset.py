# Copyright 2024 wyh
# Turn-based Prefix Dataset for Multi-turn Agent Interactions
"""
Turn-based 数据集实现

支持从多轮交互数据中：
1. 解析 turn 边界
2. 生成不同长度的 prefix
3. 支持两种训练模式的数据格式
"""

import copy
import logging
import os
import uuid
from collections import defaultdict
from typing import List, Optional, Union, Dict, Any

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

try:
    import verl.utils.torch_functional as verl_F
    from verl.utils.model import compute_position_id_with_mask
except ImportError:
    verl_F = None
    compute_position_id_with_mask = None

from .turn_parser import TurnParser, Turn, Trajectory, TurnRole

logger = logging.getLogger(__name__)


def collate_fn(data_list: list[dict]) -> dict:
    """Collate a batch of data."""
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


class TurnPrefixDataset(Dataset):
    """
    Turn-based Prefix 数据集
    
    支持两种训练模式：
    - mode="full_trajectory": 整个轨迹作为小模型的数据
    - mode="prefix_guided": 使用 prefix 引导的 RL
    
    数据格式要求：
    {
        "prompt_id": str,
        "query": str,               # 初始查询/任务
        "messages": [               # 多轮对话
            {"role": "observation", "content": "..."},
            {"role": "think", "content": "..."},
            {"role": "action", "content": "..."},
            ...
        ],
        "reward": float,            # 最终奖励
        "turn_rewards": [float],    # 可选：每个 turn 的奖励
    }
    """

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, (List, ListConfig)):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        # 基础配置
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.max_response_length = config.get("max_response_length", 2048)
        self.truncation = config.get("truncation", "error")
        
        # Turn-Prefix 特定配置
        self.mode = config.get("mode", "full_trajectory")  # full_trajectory 或 prefix_guided
        self.prefix_strategy = config.get("prefix_strategy", "random")  # random, fixed, progressive
        self.fixed_prefix_turns = config.get("fixed_prefix_turns", 1)
        self.min_prefix_turns = config.get("min_prefix_turns", 1)
        self.max_prefix_turns = config.get("max_prefix_turns", None)  # None 表示自动
        self.num_rollouts_per_prefix = config.get("num_rollouts_per_prefix", 1)
        
        # 数据字段配置
        self.prompt_key = config.get("prompt_key", "query")
        self.messages_key = config.get("messages_key", "messages")
        self.reward_key = config.get("reward_key", "reward")
        self.turn_rewards_key = config.get("turn_rewards_key", "turn_rewards")
        
        # Turn 解析器
        self.turn_parser = TurnParser(tokenizer)
        
        # 加载数据
        self._load_data()

    def _load_data(self):
        """加载并预处理数据"""
        all_data = []
        
        for data_file in self.data_files:
            logger.info(f"Loading data from {data_file}")
            
            # 支持多种格式
            if data_file.endswith('.parquet'):
                ds = datasets.load_dataset('parquet', data_files=data_file, split='train')
            elif data_file.endswith('.json') or data_file.endswith('.jsonl'):
                ds = datasets.load_dataset('json', data_files=data_file, split='train')
            else:
                ds = datasets.load_dataset(data_file, split='train')
            
            all_data.extend(ds)
        
        self.raw_data = all_data
        logger.info(f"Loaded {len(self.raw_data)} samples")
        
        # 预处理：解析 turns 并生成 prefix 变体
        self._preprocess_data()

    def _preprocess_data(self):
        """预处理数据：解析 turns 并生成训练样本"""
        self.processed_data = []
        
        for idx, item in enumerate(self.raw_data):
            try:
                # 获取基础信息
                prompt_id = item.get("prompt_id", f"sample_{idx}")
                query = item.get(self.prompt_key, "")
                messages = item.get(self.messages_key, [])
                reward = item.get(self.reward_key, 0.0)
                turn_rewards = item.get(self.turn_rewards_key, None)
                
                # 解析 turns
                trajectory = self.turn_parser.parse_messages(messages)
                trajectory.prompt_id = prompt_id
                trajectory.query = query
                trajectory.reward = reward
                
                # 根据模式生成训练样本
                if self.mode == "full_trajectory":
                    # 方式A：整个轨迹作为一个样本
                    sample = self._create_full_trajectory_sample(
                        trajectory, messages, reward, turn_rewards
                    )
                    if sample:
                        self.processed_data.append(sample)
                
                elif self.mode == "prefix_guided":
                    # 方式B：生成不同 prefix 长度的变体
                    samples = self._create_prefix_guided_samples(
                        trajectory, messages, reward, turn_rewards
                    )
                    self.processed_data.extend(samples)
                
            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {e}")
                continue
        
        logger.info(f"Preprocessed {len(self.processed_data)} training samples")

    def _create_full_trajectory_sample(
        self,
        trajectory: Trajectory,
        messages: List[Dict],
        reward: float,
        turn_rewards: Optional[List[float]],
    ) -> Optional[Dict[str, Any]]:
        """
        方式A：创建全轨迹训练样本
        
        整个轨迹都参与训练，没有 prefix/rollout 区分
        """
        # 构建完整的 prompt（通常是 system + user 的第一条消息）
        prompt_messages = []
        response_messages = []
        
        # 分离 prompt 和 response
        for msg in messages:
            role = msg.get("role", "assistant")
            if role in ["system", "user"] and not response_messages:
                prompt_messages.append(msg)
            else:
                response_messages.append(msg)
        
        if not response_messages:
            return None
        
        # Tokenize
        prompt_text = self._messages_to_text(prompt_messages)
        response_text = self._messages_to_text(response_messages)
        
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        response_ids = self.tokenizer.encode(response_text, add_special_tokens=False)
        
        # 检查长度
        if len(prompt_ids) > self.max_prompt_length:
            if self.truncation == "error":
                return None
            prompt_ids = prompt_ids[-self.max_prompt_length:]
        
        if len(response_ids) > self.max_response_length:
            if self.truncation == "error":
                return None
            response_ids = response_ids[:self.max_response_length]
        
        # 计算 turn boundaries（相对于 response 部分）
        turn_boundaries = self._compute_turn_boundaries(response_messages)
        
        return {
            "prompt_id": trajectory.prompt_id,
            "prefix_id": "full",  # 全轨迹模式没有 prefix
            "prompt_ids": prompt_ids,
            "response_ids": response_ids,
            "reward": reward,
            "turn_rewards": turn_rewards,
            "turn_boundaries": turn_boundaries,
            "num_turns": trajectory.num_turns,
            "prefix_turns": 0,  # 全轨迹模式没有 prefix
            "mode": "full_trajectory",
        }

    def _create_prefix_guided_samples(
        self,
        trajectory: Trajectory,
        messages: List[Dict],
        reward: float,
        turn_rewards: Optional[List[float]],
    ) -> List[Dict[str, Any]]:
        """
        方式B：创建前缀引导训练样本
        
        根据配置的 prefix 策略，生成不同 prefix 长度的变体
        """
        samples = []
        
        # 分离 system/user 和 assistant 消息
        prompt_messages = []
        assistant_messages = []
        
        for msg in messages:
            role = msg.get("role", "assistant")
            if role in ["system", "user"] and not assistant_messages:
                prompt_messages.append(msg)
            else:
                assistant_messages.append(msg)
        
        if len(assistant_messages) < 2:
            # 至少需要 2 个 turn 才能做 prefix-rollout
            return samples
        
        # 确定 prefix 长度范围
        max_prefix = self.max_prefix_turns or (len(assistant_messages) - 1)
        max_prefix = min(max_prefix, len(assistant_messages) - 1)
        
        # 根据策略选择 prefix 长度
        if self.prefix_strategy == "fixed":
            prefix_lengths = [min(self.fixed_prefix_turns, max_prefix)]
        elif self.prefix_strategy == "random":
            prefix_lengths = [np.random.randint(self.min_prefix_turns, max_prefix + 1)]
        elif self.prefix_strategy == "all":
            prefix_lengths = list(range(self.min_prefix_turns, max_prefix + 1))
        else:
            prefix_lengths = [self.min_prefix_turns]
        
        # 为每个 prefix 长度创建样本
        for prefix_len in prefix_lengths:
            prefix_messages = assistant_messages[:prefix_len]
            rollout_messages = assistant_messages[prefix_len:]
            
            # Tokenize
            prompt_text = self._messages_to_text(prompt_messages)
            prefix_text = self._messages_to_text(prefix_messages)
            rollout_text = self._messages_to_text(rollout_messages)
            
            prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
            rollout_ids = self.tokenizer.encode(rollout_text, add_special_tokens=False)
            
            # 完整的 response = prefix + rollout
            response_ids = prefix_ids + rollout_ids
            
            # 检查长度
            if len(prompt_ids) > self.max_prompt_length:
                if self.truncation == "error":
                    continue
                prompt_ids = prompt_ids[-self.max_prompt_length:]
            
            if len(response_ids) > self.max_response_length:
                if self.truncation == "error":
                    continue
                response_ids = response_ids[:self.max_response_length]
                # 重新计算 prefix 和 rollout 的分界
                prefix_ids = response_ids[:len(prefix_ids)]
                rollout_ids = response_ids[len(prefix_ids):]
            
            # 生成 prefix mask
            prefix_mask = [1] * len(prefix_ids) + [0] * len(rollout_ids)
            
            # 计算 turn boundaries
            turn_boundaries = self._compute_turn_boundaries(assistant_messages)
            
            # 生成唯一的 prefix_id
            prefix_id = f"{trajectory.prompt_id}_p{prefix_len}_{uuid.uuid4().hex[:8]}"
            
            sample = {
                "prompt_id": trajectory.prompt_id,
                "prefix_id": prefix_id,
                "prompt_ids": prompt_ids,
                "response_ids": response_ids,
                "prefix_mask": prefix_mask,
                "prefix_length": len(prefix_ids),
                "rollout_length": len(rollout_ids),
                "reward": reward,
                "turn_rewards": turn_rewards,
                "turn_boundaries": turn_boundaries,
                "num_turns": len(assistant_messages),
                "prefix_turns": prefix_len,
                "mode": "prefix_guided",
            }
            
            samples.append(sample)
        
        return samples

    def _messages_to_text(self, messages: List[Dict]) -> str:
        """将消息列表转换为文本"""
        # 使用 tokenizer 的 chat template（如果有）
        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
            except:
                pass
        
        # 简单拼接
        text_parts = []
        for msg in messages:
            role = msg.get("role", "assistant")
            content = msg.get("content", "")
            text_parts.append(f"{role}: {content}")
        
        return "\n".join(text_parts)

    def _compute_turn_boundaries(
        self, 
        messages: List[Dict]
    ) -> List[List[int]]:
        """计算每个 turn 的 token 边界"""
        boundaries = []
        current_pos = 0
        
        for msg in messages:
            content = msg.get("content", "")
            tokens = self.tokenizer.encode(content, add_special_tokens=False)
            token_length = len(tokens)
            
            boundaries.append([current_pos, current_pos + token_length])
            current_pos += token_length
        
        return boundaries

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx) -> Dict[str, Any]:
        """获取单个训练样本"""
        item = self.processed_data[idx]
        
        # 转换为 tensor
        prompt_ids = torch.tensor(item["prompt_ids"], dtype=torch.long)
        response_ids = torch.tensor(item["response_ids"], dtype=torch.long)
        
        # Padding
        if verl_F is not None:
            prompt_ids = verl_F.pad_sequence_to_length(
                prompt_ids.unsqueeze(0),
                max_seq_len=self.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id or 0,
                left_pad=True
            ).squeeze(0)
            
            response_ids = verl_F.pad_sequence_to_length(
                response_ids.unsqueeze(0),
                max_seq_len=self.max_response_length,
                pad_token_id=self.tokenizer.pad_token_id or 0,
                left_pad=False
            ).squeeze(0)
        
        # 构建返回数据
        result = {
            "input_ids": prompt_ids,
            "response_ids": response_ids,
            "attention_mask": (prompt_ids != (self.tokenizer.pad_token_id or 0)).long(),
            "prompt_id": item["prompt_id"],
            "prefix_id": item["prefix_id"],
            "reward": item["reward"],
            "mode": item["mode"],
        }
        
        # 添加 prefix 相关字段（方式B）
        if item["mode"] == "prefix_guided" and "prefix_mask" in item:
            prefix_mask = item["prefix_mask"]
            # Padding prefix_mask
            if len(prefix_mask) < self.max_response_length:
                prefix_mask = prefix_mask + [0] * (self.max_response_length - len(prefix_mask))
            result["prefix_mask"] = torch.tensor(prefix_mask[:self.max_response_length], dtype=torch.long)
            result["prefix_turns"] = item["prefix_turns"]
        
        # 添加 turn boundaries
        if "turn_boundaries" in item:
            # 转换为 tensor，padding 到固定大小
            max_turns = 50  # 最大 turn 数
            boundaries = item["turn_boundaries"]
            padded_boundaries = boundaries + [[0, 0]] * (max_turns - len(boundaries))
            result["turn_boundaries"] = torch.tensor(
                padded_boundaries[:max_turns], dtype=torch.long
            )
        
        return result


# 兼容 verl 的 collate_fn
TurnPrefixDataset.collate_fn = staticmethod(collate_fn)

