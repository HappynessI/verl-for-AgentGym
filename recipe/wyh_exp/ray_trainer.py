# Copyright 2024 wyh
# Ray Trainer for Turn-based Prefix RL
"""
Ray Trainer 实现

复用 verl 的 RayPPOTrainer，只扩展必要的功能：
1. 支持 Turn-Prefix 数据加载
2. 支持两种训练模式的优势计算
3. 支持 prefix_mask 和 prefix_index 的传递
"""

import logging
import os
from typing import Dict, Optional, Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

# 复用 verl 的训练器
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl import DataProto

# 导入自定义的优势估计器（注册到 verl）
from recipe.wyh_exp.algos import core_algos  # noqa: F401 - 触发注册

logger = logging.getLogger(__name__)


class TurnPrefixRayTrainer(RayPPOTrainer):
    """
    Turn-based Prefix RL 的 Ray Trainer
    
    继承自 verl 的 RayPPOTrainer，扩展以下功能：
    1. 处理 prefix_mask 和 prefix_index
    2. 支持两种训练模式
    3. 记录 turn-level 的指标
    """
    
    def __init__(self, config: DictConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        
        # Turn-Prefix 特定配置
        self.training_mode = config.get("algorithm", {}).get("adv_estimator", "turn_full_trajectory")
        self.prefix_advantage_mode = config.get("algorithm", {}).get("prefix_advantage_mode", "zero")
        
        logger.info(f"TurnPrefixRayTrainer initialized with mode: {self.training_mode}")
    
    def _compute_advantages(self, data: DataProto) -> DataProto:
        """
        计算优势值
        
        扩展以支持 prefix_mask 和 prefix_index
        """
        from verl.trainer.ppo.core_algos import get_adv_estimator_fn
        
        # 获取基础数据
        token_level_rewards = data.batch["token_level_rewards"]
        response_mask = data.batch["response_mask"]
        
        # 获取 index（用于组内归一化）
        if "index" in data.non_tensor_batch:
            index = data.non_tensor_batch["index"]
        else:
            # 使用 prompt_id 作为 index
            index = data.non_tensor_batch.get("prompt_id", np.arange(token_level_rewards.shape[0]))
        
        # 准备额外参数
        kwargs = {}
        
        # 获取 prefix 相关数据（方式B 需要）
        if "prefix_mask" in data.batch:
            kwargs["prefix_mask"] = data.batch["prefix_mask"]
        
        if "prefix_index" in data.non_tensor_batch:
            kwargs["prefix_index"] = data.non_tensor_batch["prefix_index"]
        elif "prefix_id" in data.non_tensor_batch:
            kwargs["prefix_index"] = data.non_tensor_batch["prefix_id"]
        
        # 获取 turn 相关数据
        if "turn_boundaries" in data.batch:
            kwargs["turn_boundaries"] = data.batch["turn_boundaries"]
        
        if "turn_rewards" in data.batch:
            kwargs["turn_rewards"] = data.batch["turn_rewards"]
        
        # 获取优势估计函数
        adv_fn = get_adv_estimator_fn(self.training_mode)
        
        # 计算优势
        advantages, returns = adv_fn(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            config=self.config.get("algorithm", None),
            **kwargs,
        )
        
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        
        return data
    
    def _compute_metrics(self, data: DataProto, metrics: Dict) -> Dict:
        """
        计算训练指标
        
        扩展以记录 turn-level 的指标
        """
        # 调用父类方法
        metrics = super()._compute_metrics(data, metrics) if hasattr(super(), '_compute_metrics') else metrics
        
        # 添加 turn-prefix 相关指标
        if "prefix_mask" in data.batch:
            prefix_mask = data.batch["prefix_mask"]
            response_mask = data.batch["response_mask"]
            
            # 计算 prefix 和 rollout 的比例
            prefix_tokens = (prefix_mask * response_mask).sum().item()
            total_tokens = response_mask.sum().item()
            
            if total_tokens > 0:
                metrics["turn_prefix/prefix_ratio"] = prefix_tokens / total_tokens
                metrics["turn_prefix/rollout_ratio"] = 1 - prefix_tokens / total_tokens
        
        # 记录优势的统计信息
        if "advantages" in data.batch:
            advantages = data.batch["advantages"]
            response_mask = data.batch["response_mask"]
            
            masked_adv = advantages * response_mask
            valid_adv = masked_adv[response_mask.bool()]
            
            if len(valid_adv) > 0:
                metrics["turn_prefix/adv_mean"] = valid_adv.mean().item()
                metrics["turn_prefix/adv_std"] = valid_adv.std().item()
                metrics["turn_prefix/adv_max"] = valid_adv.max().item()
                metrics["turn_prefix/adv_min"] = valid_adv.min().item()
        
        return metrics


def create_trainer(config: DictConfig) -> TurnPrefixRayTrainer:
    """
    创建 TurnPrefixRayTrainer 实例
    
    这是一个工厂函数，用于从配置创建训练器
    """
    return TurnPrefixRayTrainer(config)


# 为了兼容 verl 的启动方式，提供一个简单的入口
if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig
    
    @hydra.main(config_path="config", config_name="base_config", version_base=None)
    def main(config: DictConfig):
        trainer = create_trainer(config)
        trainer.fit()
    
    main()

