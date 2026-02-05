# Copyright 2024 wyh
# Core algorithms for Turn-based Prefix RL
"""
核心算法实现：两种训练方式

通过 verl 的注册机制，注册自定义的优势估计器：
- turn_full_trajectory: 方式A，整个轨迹作为小模型自己的轨迹进行 RL
- turn_prefix_guided: 方式B，Prefix 作为参考，只对 rollout 部分进行 RL
- turn_prefix_guided_dr: 方式B 的 Dr.GRPO 变体

使用方法：
    在配置文件中设置 algorithm.adv_estimator = "turn_full_trajectory" 或其他
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from omegaconf import DictConfig

# 导入 verl 的核心模块
from verl.trainer.ppo.core_algos import (
    register_adv_est,
    ADV_ESTIMATOR_REGISTRY,
)

try:
    import verl.utils.torch_functional as verl_F
    from verl.utils import group_mean_std
except ImportError:
    verl_F = None
    group_mean_std = None


# ============================================================================
# 方式A: Full-Trajectory RL (全轨迹强化学习)
# 注册为 verl 的优势估计器
# ============================================================================

@register_adv_est("turn_full_trajectory")
def compute_turn_full_trajectory_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    config: Optional[DictConfig] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    方式A: Full-Trajectory RL 优势计算
    
    整个轨迹（包括 prefix 和 rollout）都参与训练，使用统一的 GRPO 优势估计。
    这种方式类似于 ARPO，将整个轨迹视为模型自己生成的。
    
    Args:
        token_level_rewards: (batch_size, response_length) 
            token 级别的奖励，通常最后一个 token 有 outcome reward
        response_mask: (batch_size, response_length)
            响应部分的 mask，1 表示有效 token
        index: (batch_size,)
            每个样本对应的 prompt id，用于组内归一化
        config: 配置对象
        **kwargs: 可能包含 turn_boundaries, turn_weights 等
    
    Returns:
        advantages: (batch_size, response_length) 优势值
        returns: (batch_size, response_length) 回报值
    """
    # 从 config 或 kwargs 获取参数
    epsilon = 1e-6
    norm_adv_by_std = True
    if config is not None:
        norm_adv_by_std = getattr(config, "norm_adv_by_std_in_grpo", True)
    
    # 获取可选参数
    turn_boundaries = kwargs.get("turn_boundaries", None)
    turn_weighting = kwargs.get("turn_weighting", False)
    turn_weights = kwargs.get("turn_weights", None)
    
    # 计算 outcome reward (整个序列的总奖励)
    scores = token_level_rewards.sum(dim=-1)  # (batch_size,)
    
    # 按 prompt 分组计算统计量
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    
    with torch.no_grad():
        bsz = scores.shape[0]
        device = scores.device
        
        # 按 prompt id 分组
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        
        # 计算组内均值和标准差
        for idx in id2score:
            group_scores = id2score[idx]
            if len(group_scores) == 1:
                id2mean[idx] = torch.tensor(0.0, device=device)
                id2std[idx] = torch.tensor(1.0, device=device)
            else:
                stacked = torch.stack(group_scores)
                id2mean[idx] = torch.mean(stacked)
                id2std[idx] = torch.std(stacked)
        
        # 计算归一化优势
        for i in range(bsz):
            if norm_adv_by_std:
                # GRPO 风格：减均值除以标准差
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                # Dr.GRPO 风格：只减均值
                scores[i] = scores[i] - id2mean[index[i]]
        
        # 应用 turn 权重（可选）
        if turn_weighting and turn_weights is not None:
            scores = scores * turn_weights.mean(dim=-1)
        
        # 广播到所有 token
        advantages = scores.unsqueeze(-1) * response_mask
    
    return advantages, advantages


@register_adv_est("turn_full_trajectory_with_turn_bonus")
def compute_turn_full_trajectory_with_turn_bonus(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    config: Optional[DictConfig] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    方式A 扩展: 结合 turn-level 奖励的全轨迹优势
    
    除了 outcome reward，还考虑每个 turn 的中间奖励。
    需要在 kwargs 中提供 turn_rewards。
    """
    epsilon = 1e-6
    outcome_weight = 0.7
    turn_weight = 0.3
    
    if config is not None:
        outcome_weight = getattr(config, "outcome_weight", 0.7)
        turn_weight = getattr(config, "turn_weight", 0.3)
    
    turn_rewards = kwargs.get("turn_rewards", None)
    
    # 计算 outcome reward
    outcome_scores = token_level_rewards.sum(dim=-1)
    
    # 计算 turn-level reward 的总和
    if turn_rewards is not None:
        turn_scores = turn_rewards.sum(dim=-1)
        combined_scores = outcome_weight * outcome_scores + turn_weight * turn_scores
    else:
        combined_scores = outcome_scores
    
    # 按 prompt 分组归一化
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    
    with torch.no_grad():
        bsz = combined_scores.shape[0]
        device = combined_scores.device
        
        for i in range(bsz):
            id2score[index[i]].append(combined_scores[i])
        
        for idx in id2score:
            group_scores = id2score[idx]
            if len(group_scores) == 1:
                id2mean[idx] = torch.tensor(0.0, device=device)
                id2std[idx] = torch.tensor(1.0, device=device)
            else:
                stacked = torch.stack(group_scores)
                id2mean[idx] = torch.mean(stacked)
                id2std[idx] = torch.std(stacked)
        
        for i in range(bsz):
            combined_scores[i] = (combined_scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        
        advantages = combined_scores.unsqueeze(-1) * response_mask
    
    return advantages, advantages


# ============================================================================
# 方式B: Prefix-Guided RL (前缀引导强化学习)
# ============================================================================

@register_adv_est("turn_prefix_guided")
def compute_turn_prefix_guided_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    config: Optional[DictConfig] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    方式B: Prefix-Guided RL 优势计算
    
    Prefix 部分作为参考/引导，只对 rollout 部分计算真正的 RL 优势。
    Prefix 部分可以使用不同的处理方式。
    
    需要在 kwargs 中提供:
    - prefix_mask: (batch_size, response_length) prefix 部分为 1
    - prefix_index: (batch_size,) prefix 的唯一标识
    
    Args:
        token_level_rewards: (batch_size, response_length)
        response_mask: (batch_size, response_length)
        index: (batch_size,) prompt id
        config: 配置对象
        **kwargs: 必须包含 prefix_mask 和 prefix_index
    
    Returns:
        advantages, returns
    """
    epsilon = 1e-6
    norm_adv_by_std = True
    prefix_advantage_mode = "zero"
    
    if config is not None:
        norm_adv_by_std = getattr(config, "norm_adv_by_std_in_grpo", True)
        prefix_advantage_mode = getattr(config, "prefix_advantage_mode", "zero")
    
    # 获取必需参数
    prefix_mask = kwargs.get("prefix_mask")
    prefix_index = kwargs.get("prefix_index")
    
    if prefix_mask is None or prefix_index is None:
        # 如果没有 prefix 信息，退化为 full_trajectory 模式
        return compute_turn_full_trajectory_advantage(
            token_level_rewards, response_mask, index, config, **kwargs
        )
    
    # 计算 outcome reward
    scores = token_level_rewards.sum(dim=-1)
    
    # Rollout 部分的 mask
    rollout_mask = response_mask * (1 - prefix_mask.float())
    
    # 按 prompt 和 prefix 分组
    id2prefix2scores = defaultdict(lambda: defaultdict(list))
    id2prefix2mean = defaultdict(dict)
    id2prefix2std = defaultdict(dict)
    id2all_mean = {}
    
    with torch.no_grad():
        bsz = scores.shape[0]
        device = scores.device
        
        # 分组收集 scores
        for i in range(bsz):
            id2prefix2scores[index[i]][prefix_index[i]].append(scores[i])
        
        # 计算每个 prefix 组的统计量
        for idx in id2prefix2scores:
            all_scores_for_prompt = []
            
            for p_idx in id2prefix2scores[idx]:
                group_scores = id2prefix2scores[idx][p_idx]
                all_scores_for_prompt.extend(group_scores)
                
                if len(group_scores) == 1:
                    id2prefix2mean[idx][p_idx] = torch.tensor(0.0, device=device)
                    id2prefix2std[idx][p_idx] = torch.tensor(1.0, device=device)
                else:
                    stacked = torch.stack(group_scores)
                    id2prefix2mean[idx][p_idx] = torch.mean(stacked)
                    id2prefix2std[idx][p_idx] = torch.std(stacked)
            
            # 计算所有 rollout 的均值（用于 prefix 相对优势）
            if all_scores_for_prompt:
                id2all_mean[idx] = torch.mean(torch.stack(all_scores_for_prompt))
        
        # 计算优势
        rollout_advantages = torch.zeros_like(scores)
        prefix_advantages = torch.zeros_like(scores)
        
        for i in range(bsz):
            # Rollout 优势：相对于同 prefix 组的均值
            if norm_adv_by_std:
                rollout_advantages[i] = (
                    scores[i] - id2prefix2mean[index[i]][prefix_index[i]]
                ) / (id2prefix2std[index[i]][prefix_index[i]] + epsilon)
            else:
                rollout_advantages[i] = scores[i] - id2prefix2mean[index[i]][prefix_index[i]]
            
            # Prefix 优势：根据模式处理
            if prefix_advantage_mode == "zero":
                prefix_advantages[i] = 0.0
            elif prefix_advantage_mode == "sft":
                prefix_advantages[i] = 1.0  # 相当于 SFT loss
            elif prefix_advantage_mode == "relative":
                # 相对于所有 rollout 的均值
                if index[i] in id2all_mean:
                    prefix_advantages[i] = (
                        scores[i] - id2all_mean[index[i]]
                    ) / (id2prefix2std[index[i]][prefix_index[i]] + epsilon)
        
        # 组合优势
        rollout_adv_expanded = rollout_advantages.unsqueeze(-1) * rollout_mask
        prefix_adv_expanded = prefix_advantages.unsqueeze(-1) * prefix_mask
        
        if prefix_advantage_mode == "zero":
            # prefix 部分优势为 0，只训练 rollout 部分
            advantages = rollout_adv_expanded
        else:
            advantages = rollout_adv_expanded + prefix_adv_expanded
    
    return advantages, advantages


@register_adv_est("turn_prefix_guided_dr")
def compute_turn_prefix_guided_dr_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    config: Optional[DictConfig] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    方式B 的 Dr.GRPO 变体: 只减均值，不除以标准差
    
    这是一个简化版本，更接近 prefix_rft 原始实现。
    """
    epsilon = 1e-6
    
    prefix_mask = kwargs.get("prefix_mask")
    prefix_index = kwargs.get("prefix_index")
    
    if prefix_mask is None or prefix_index is None:
        # 退化为 full_trajectory 模式
        return compute_turn_full_trajectory_advantage(
            token_level_rewards, response_mask, index, config, 
            norm_adv_by_std=False, **kwargs
        )
    
    scores = token_level_rewards.sum(dim=-1)
    rollout_mask = response_mask * (1 - prefix_mask.float())
    
    id2prefix2scores = defaultdict(lambda: defaultdict(list))
    id2prefix2mean = defaultdict(dict)
    
    with torch.no_grad():
        bsz = scores.shape[0]
        device = scores.device
        
        for i in range(bsz):
            id2prefix2scores[index[i]][prefix_index[i]].append(scores[i])
        
        for idx in id2prefix2scores:
            for p_idx in id2prefix2scores[idx]:
                group_scores = id2prefix2scores[idx][p_idx]
                if len(group_scores) == 1:
                    id2prefix2mean[idx][p_idx] = torch.tensor(0.0, device=device)
                else:
                    id2prefix2mean[idx][p_idx] = torch.mean(torch.stack(group_scores))
        
        for i in range(bsz):
            scores[i] = scores[i] - id2prefix2mean[index[i]][prefix_index[i]]
        
        advantages = scores.unsqueeze(-1) * rollout_mask
    
    return advantages, advantages


# ============================================================================
# 辅助函数
# ============================================================================

def create_prefix_mask_from_turns(
    turn_boundaries: torch.Tensor,
    prefix_turns: int,
    total_length: int,
    device: torch.device = None,
) -> torch.Tensor:
    """
    根据 turn 边界创建 prefix mask
    
    Args:
        turn_boundaries: (batch_size, max_turns, 2) 每个 turn 的 [start, end]
        prefix_turns: int 前 N 个 turn 作为 prefix
        total_length: int 总序列长度
        device: torch.device
    
    Returns:
        prefix_mask: (batch_size, total_length)
    """
    batch_size = turn_boundaries.shape[0]
    device = device or turn_boundaries.device
    
    prefix_mask = torch.zeros(batch_size, total_length, device=device)
    
    for i in range(batch_size):
        if prefix_turns > 0 and prefix_turns <= turn_boundaries.shape[1]:
            # 获取前 prefix_turns 个 turn 的结束位置
            prefix_end = turn_boundaries[i, prefix_turns - 1, 1].item()
            prefix_end = int(min(prefix_end, total_length))
            prefix_mask[i, :prefix_end] = 1.0
    
    return prefix_mask


def get_available_adv_estimators() -> List[str]:
    """获取所有可用的优势估计器名称"""
    return list(ADV_ESTIMATOR_REGISTRY.keys())


# ============================================================================
# 统一接口（向后兼容）
# ============================================================================

def compute_turn_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    prefix_mask: Optional[torch.Tensor],
    turn_boundaries: Optional[torch.Tensor],
    index: np.ndarray,
    prefix_index: Optional[np.ndarray] = None,
    mode: str = "turn_full_trajectory",
    config: Optional[DictConfig] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    统一的优势计算接口
    
    Args:
        mode: str
            - "turn_full_trajectory": 方式A，全轨迹 RL
            - "turn_prefix_guided": 方式B，前缀引导 RL
            - "turn_prefix_guided_dr": 方式B 的 Dr.GRPO 变体
    
    Returns:
        advantages, returns
    """
    from verl.trainer.ppo.core_algos import get_adv_estimator_fn
    
    # 准备 kwargs
    full_kwargs = {
        "prefix_mask": prefix_mask,
        "turn_boundaries": turn_boundaries,
        "prefix_index": prefix_index,
        **kwargs,
    }
    
    # 获取对应的优势估计函数
    adv_fn = get_adv_estimator_fn(mode)
    
    return adv_fn(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        config=config,
        **full_kwargs,
    )
