# Copyright 2024 wyh
# FSDP Workers for Turn-based Prefix RL
"""
FSDP Workers 实现

支持两种训练模式:
- 方式A (Full-Trajectory RL): 整个轨迹参与训练
- 方式B (Prefix-Guided RL): 只有 rollout 部分参与训练
"""

import logging
import os
from typing import Dict, Optional, Union

import torch
import torch.distributed
from codetiming import Timer
from omegaconf import DictConfig, open_dict

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_tokenizer
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.fsdp_utils import (
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    load_fsdp_model_to_gpu,
    offload_fsdp_model_to_cpu,
)
from verl.utils.model import compute_position_id_with_mask

try:
    from verl.workers.fsdp_workers import ActorRolloutRefWorker as BaseActorRolloutRefWorker
except ImportError:
    BaseActorRolloutRefWorker = Worker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class TurnPrefixActorRolloutRefWorker(BaseActorRolloutRefWorker):
    """
    支持 Turn-based Prefix RL 的 Actor/Rollout/Ref Worker
    
    扩展了基础 Worker 以支持:
    1. Prefix mask 的处理
    2. Turn boundaries 的处理
    3. 两种训练模式的切换
    """
    
    def __init__(self, config: DictConfig, role: str):
        super().__init__(config, role)
        
        # Turn-Prefix 特定配置
        self.training_mode = config.get("training_mode", "full_trajectory")
        self.prefix_advantage_mode = config.get("prefix_advantage_mode", "zero")
        self.turn_weighting = config.get("turn_weighting", False)
        
        logger.info(f"TurnPrefixActorRolloutRefWorker initialized with mode: {self.training_mode}")
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob(self, data: DataProto) -> DataProto:
        """
        计算 log probability
        
        扩展以支持 prefix mask
        """
        data = data.to("cuda")
        
        # 获取基础数据
        input_ids = data.batch["input_ids"]
        attention_mask = data.batch["attention_mask"]
        response_ids = data.batch["responses"]
        response_mask = data.batch["response_mask"]
        
        # 获取 prefix 相关数据（如果存在）
        prefix_mask = data.batch.get("prefix_mask", None)
        turn_boundaries = data.batch.get("turn_boundaries", None)
        
        # 合并 input 和 response
        input_ids = torch.cat([input_ids, response_ids], dim=-1)
        
        # 构建完整的 attention mask
        batch_size = input_ids.size(0)
        response_length = response_ids.size(1)
        attention_mask = torch.cat([
            attention_mask,
            torch.ones(batch_size, response_length, device=attention_mask.device, dtype=attention_mask.dtype)
        ], dim=-1)
        
        # 计算 position ids
        position_ids = compute_position_id_with_mask(attention_mask)
        
        # 前向传播
        with torch.no_grad():
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
            logits = output.logits
        
        # 计算 log prob
        log_probs = self._compute_log_probs_from_logits(logits, input_ids, response_mask)
        
        # 根据训练模式处理 log_probs
        if self.training_mode == "prefix_guided" and prefix_mask is not None:
            # 方式B: 只保留 rollout 部分的 log_probs
            rollout_mask = response_mask * (1 - prefix_mask.float())
            log_probs = log_probs * rollout_mask
        
        data.batch["old_log_probs"] = log_probs
        
        return data
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data: DataProto) -> DataProto:
        """
        计算参考模型的 log probability
        """
        data = data.to("cuda")
        
        input_ids = data.batch["input_ids"]
        attention_mask = data.batch["attention_mask"]
        response_ids = data.batch["responses"]
        response_mask = data.batch["response_mask"]
        
        prefix_mask = data.batch.get("prefix_mask", None)
        
        input_ids = torch.cat([input_ids, response_ids], dim=-1)
        
        batch_size = input_ids.size(0)
        response_length = response_ids.size(1)
        attention_mask = torch.cat([
            attention_mask,
            torch.ones(batch_size, response_length, device=attention_mask.device, dtype=attention_mask.dtype)
        ], dim=-1)
        
        position_ids = compute_position_id_with_mask(attention_mask)
        
        with torch.no_grad():
            output = self.ref_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
            logits = output.logits
        
        log_probs = self._compute_log_probs_from_logits(logits, input_ids, response_mask)
        
        if self.training_mode == "prefix_guided" and prefix_mask is not None:
            rollout_mask = response_mask * (1 - prefix_mask.float())
            log_probs = log_probs * rollout_mask
        
        data.batch["ref_log_probs"] = log_probs
        
        return data
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto) -> Dict:
        """
        更新 Actor 模型
        
        支持两种训练模式:
        - full_trajectory: 整个轨迹参与训练
        - prefix_guided: 只有 rollout 部分参与训练
        """
        data = data.to("cuda")
        
        # 获取数据
        input_ids = data.batch["input_ids"]
        attention_mask = data.batch["attention_mask"]
        response_ids = data.batch["responses"]
        response_mask = data.batch["response_mask"]
        old_log_probs = data.batch["old_log_probs"]
        advantages = data.batch["advantages"]
        
        prefix_mask = data.batch.get("prefix_mask", None)
        
        # 根据训练模式调整 loss mask
        if self.training_mode == "prefix_guided" and prefix_mask is not None:
            if self.prefix_advantage_mode == "zero":
                # Prefix 部分不参与训练
                loss_mask = response_mask * (1 - prefix_mask.float())
            else:
                # Prefix 部分也参与训练（使用不同的 advantage）
                loss_mask = response_mask
        else:
            # 全轨迹训练
            loss_mask = response_mask
        
        # 合并 input 和 response
        input_ids = torch.cat([input_ids, response_ids], dim=-1)
        
        batch_size = input_ids.size(0)
        response_length = response_ids.size(1)
        attention_mask = torch.cat([
            attention_mask,
            torch.ones(batch_size, response_length, device=attention_mask.device, dtype=attention_mask.dtype)
        ], dim=-1)
        
        position_ids = compute_position_id_with_mask(attention_mask)
        
        # 前向传播
        output = self.actor_module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        logits = output.logits
        
        # 计算新的 log prob
        new_log_probs = self._compute_log_probs_from_logits(logits, input_ids, loss_mask)
        
        # 计算 policy loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        clip_ratio = self.config.actor.get("clip_ratio", 0.2)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
        pg_loss = torch.max(pg_loss1, pg_loss2)
        
        # 应用 loss mask
        pg_loss = (pg_loss * loss_mask).sum() / loss_mask.sum()
        
        # 反向传播
        self.actor_optimizer.zero_grad()
        pg_loss.backward()
        
        # 梯度裁剪
        grad_clip = self.config.actor.get("grad_clip", 1.0)
        torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), grad_clip)
        
        self.actor_optimizer.step()
        
        # 返回指标
        metrics = {
            "actor/loss": pg_loss.item(),
            "actor/ratio_mean": ratio.mean().item(),
            "actor/ratio_std": ratio.std().item(),
        }
        
        if prefix_mask is not None:
            metrics["actor/prefix_ratio"] = prefix_mask.float().mean().item()
            metrics["actor/rollout_ratio"] = (1 - prefix_mask.float()).mean().item()
        
        return metrics
    
    def _compute_log_probs_from_logits(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """从 logits 计算 log probabilities"""
        # Shift logits and labels
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        # 计算 log softmax
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        
        # 收集目标 token 的 log prob
        log_probs = torch.gather(
            log_probs, 
            dim=-1, 
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # 应用 mask（需要对齐长度）
        if mask.size(-1) == log_probs.size(-1) + 1:
            mask = mask[..., 1:]
        
        return log_probs * mask


class TurnPrefixCriticWorker(Worker):
    """
    Critic Worker（可选，用于 Actor-Critic 方法）
    
    对于 GRPO 等无 Critic 的方法，不需要这个 Worker
    """
    
    def __init__(self, config: DictConfig, role: str):
        super().__init__()
        self.config = config
        logger.info("TurnPrefixCriticWorker initialized")
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, data: DataProto) -> DataProto:
        """计算价值估计（用于 GAE）"""
        # 对于 GRPO，我们不需要 Critic
        # 这里提供一个简单的实现，返回零值
        response_mask = data.batch["response_mask"]
        data.batch["values"] = torch.zeros_like(response_mask, dtype=torch.float)
        return data

