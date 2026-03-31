#!/usr/bin/env python3
"""
Debug logging for TextCraft GRPO training - MINIMAL VIABLE VERSION

修复：
1. 使用 instance 属性而不是 contextvars 来传递 debug 信息（解决 Ray worker 进程隔离问题）
2. reward_score 类型安全提取（支持多元素 tensor，降级时取 mean）
3. global_step 自增计数器
"""

import os
import sys
import json
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEBUG_CONFIG = {
    "dump_enabled": True,
    "dump_dir": "/Data/wyh/verl/examples/sglang_multiturn/my_exp/debug/logs/rollout_dumps",
    "dump_samples_per_step": 2,
    "initialized": False,
    "_tokenizer": None,
    "_global_step": 0,
}


def extract_scalar_reward(rs: Any) -> Tuple[float, bool]:
    """
    安全提取标量 reward，支持多种类型。
    
    返回: (value, is_fallback) - value 是提取的 float，is_fallback 表示是否发生了降级解析
    
    处理策略：
    - Python int/float: 直接返回
    - 0-d tensor/numpy: 取 item()
    - 单元素 (size=1): 取唯一元素
    - 多元素: 记录 warning，返回 mean（不返回 0.0 以免污染统计）
    - None: 返回 0.0, True
    - 无法解析: 记录 warning，返回 0.0, True
    """
    is_fallback = False
    
    if rs is None:
        return 0.0, True
    
    # 如果已经是 Python 数值
    if isinstance(rs, (int, float)):
        return float(rs), False
    
    # 如果是 torch.Tensor
    if hasattr(rs, 'detach'):
        rs = rs.detach()
    if hasattr(rs, 'cpu'):
        rs = rs.cpu()
    if hasattr(rs, 'numpy'):
        rs = rs.numpy()
    
    # 现在应该是 numpy 数组或标量
    if hasattr(rs, 'flat'):
        rs = rs.flat
    
    if hasattr(rs, '__iter__'):
        arr = np.asarray(rs)
        
        # 0-d 数组
        if arr.ndim == 0:
            return float(arr.item()), False
        
        # 单元素
        if arr.size == 1:
            return float(arr.item()), False
        
        # 多元素：降级策略 - 取 mean 并记录 warning
        is_fallback = True
        mean_val = float(arr.mean())
        logger.warning(
            f"[DEBUG] reward_score has multiple elements (shape={arr.shape}), "
            f"using mean={mean_val:.4f} instead of 0.0"
        )
        return mean_val, is_fallback
    
    # 尝试直接转 float
    try:
        return float(rs), False
    except Exception as e:
        is_fallback = True
        logger.warning(f"[DEBUG] Failed to parse reward_score: {e}, returning 0.0")
        return 0.0, True


def init_debug_logging(dump_dir: str = None, samples_per_step: int = 2, tokenizer=None):
    """Initialize debug logging configuration."""
    if dump_dir:
        DEBUG_CONFIG["dump_dir"] = dump_dir
    DEBUG_CONFIG["dump_samples_per_step"] = samples_per_step
    if tokenizer:
        DEBUG_CONFIG["_tokenizer"] = tokenizer
    
    os.makedirs(DEBUG_CONFIG["dump_dir"], exist_ok=True)
    
    # Clear old dump files
    for f in os.listdir(DEBUG_CONFIG["dump_dir"]):
        if f.endswith(".jsonl"):
            os.remove(os.path.join(DEBUG_CONFIG["dump_dir"], f))
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    DEBUG_CONFIG["initialized"] = True
    DEBUG_CONFIG["_global_step"] = 0
    
    logger.info(f"[DEBUG] Debug logging initialized. Dump dir: {DEBUG_CONFIG['dump_dir']}")


def patch_tool_agent_loop():
    """Patch ToolAgentLoop methods to write TRUE debug fields to output.extra_fields.
    
    使用 self 属性来存储 debug 信息（可以在 Ray worker 进程间传递）。
    """
    try:
        from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop, AgentData, AgentState
        
        # Store original methods
        orig_generate = ToolAgentLoop._handle_generating_state
        orig_tools = ToolAgentLoop._handle_processing_tools_state
        orig_run = ToolAgentLoop.run
        
        # Patch _handle_generating_state to capture end_reason
        async def patched_generate(self, agent_data, sp):
            result = await orig_generate(self, agent_data, sp)
            if result == AgentState.TERMINATED:
                resp_len = len(agent_data.response_mask)
                if resp_len >= self.response_length:
                    self._debug_end_reason = 'max_response_length'
                elif self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
                    self._debug_end_reason = 'max_assistant_turns'
                elif self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
                    self._debug_end_reason = 'max_user_turns'
                else:
                    self._debug_end_reason = 'eos'
            return result
        
        # Patch _handle_processing_tools_state to count tool calls
        async def patched_tools(self, agent_data):
            if agent_data.tool_calls:
                if not hasattr(self, '_debug_tool_call_count'):
                    self._debug_tool_call_count = 0
                self._debug_tool_call_count += len(agent_data.tool_calls[:self.max_parallel_calls])
            return await orig_tools(self, agent_data)
        
        # Patch run() to initialize and finalize debug fields
        async def patched_run(self, sampling_params, **kwargs):
            # Reset debug state for this run
            self._debug_tool_call_count = 0
            self._debug_end_reason = 'unknown'
            
            # Call original run
            output = await orig_run(self, sampling_params, **kwargs)
            
            # Write debug fields to extra_fields
            output.extra_fields['debug_tool_call_count'] = self._debug_tool_call_count
            output.extra_fields['debug_end_reason'] = self._debug_end_reason
            
            return output
        
        # Apply patches
        ToolAgentLoop._handle_generating_state = patched_generate
        ToolAgentLoop._handle_processing_tools_state = patched_tools
        ToolAgentLoop.run = patched_run
        
        logger.info("[DEBUG] Patched ToolAgentLoop (self-attribute-based)")
        return True
        
    except Exception as e:
        logger.error(f"[DEBUG] Failed to patch ToolAgentLoop: {e}")
        import traceback
        traceback.print_exc()
        return False


def patch_agent_loop_worker():
    """Patch AgentLoopWorker to:
    1. Capture tokenizer
    2. Aggregate debug metrics in _postprocess() from inputs[i].extra_fields
    3. Dump samples
    """
    try:
        from verl.experimental.agent_loop.agent_loop import AgentLoopWorker
        
        # ===== KEY FIX: Also patch _agent_loop_postprocess to forward extra_fields =====
        # The issue is: ToolAgentLoop.run writes debug fields to output.extra_fields,
        # but _agent_loop_postprocess creates _InternalAgentLoopOutput and passes it.
        # We need to ensure these fields are preserved in the output.
        
        # First, check if there's already a patched version running
        # If not, we need to patch at the source
        
        # Patch __init__ to capture tokenizer
        orig_init = AgentLoopWorker.__init__
        
        def patched_init(self, config, server_handles, reward_router_address=None):
            orig_init(self, config, server_handles, reward_router_address)
            DEBUG_CONFIG["_tokenizer"] = self.tokenizer
            logger.info("[DEBUG] Captured tokenizer from AgentLoopWorker")
        
        AgentLoopWorker.__init__ = patched_init
        
        # Patch _postprocess to aggregate from inputs[i].extra_fields
        orig_postprocess = AgentLoopWorker._postprocess
        
        def patched_postprocess(self, inputs):
            """Patched _postprocess - read debug fields from inputs[i].extra_fields."""
            output = orig_postprocess(self, inputs)
            
            # Track if any reward extraction had fallback
            any_reward_fallback = False
            
            try:
                # ===== STEP 1: Read debug fields from inputs (NOT from tool_extra_fields!) =====
                # inputs is list of _InternalAgentLoopOutput, each has .extra_fields dict
                tool_call_counts = []
                end_reasons = []
                
                for input_item in inputs:
                    ef = input_item.extra_fields
                    tool_call_counts.append(ef.get('debug_tool_call_count', 0))
                    end_reasons.append(ef.get('debug_end_reason', 'unknown'))
                
                # Get other fields
                num_turns = output.non_tensor_batch.get('__num_turns__', np.ones(len(inputs)))
                if hasattr(num_turns, 'tolist'):
                    num_turns = num_turns.tolist()
                
                # Response lengths
                attn_mask = output.batch['attention_mask']
                prompt_len = output.batch['prompts'].shape[1]
                response_lengths = attn_mask[:, prompt_len:].sum(dim=1).tolist()
                
                # Rewards - use type-safe extraction with fallback tracking
                rewards = []
                for input_item in inputs:
                    rs = input_item.reward_score
                    val, is_fb = extract_scalar_reward(rs)
                    rewards.append(val)
                    if is_fb:
                        any_reward_fallback = True
                
                # Get config
                max_resp_len = self.config.actor_rollout_ref.rollout.response_length
                
                # Convert to numpy
                tool_call_counts = np.array(tool_call_counts)
                end_reasons = np.array(end_reasons, dtype=object)
                response_lengths = np.array(response_lengths)
                rewards = np.array(rewards)
                num_turns = np.array(num_turns)
                
                # ===== STEP 2: Compute batch metrics =====
                tc_mean = float(tool_call_counts.mean())
                tc_max = int(tool_call_counts.max())
                tc_min = int(tool_call_counts.min())
                tc_ratio = float((tool_call_counts > 0).sum() / len(tool_call_counts))
                
                # End reason breakdown
                unique_reasons, reason_counts = np.unique(end_reasons, return_counts=True)
                reason_stats = {r: {'count': int(c), 'ratio': float(c/len(end_reasons))} 
                              for r, c in zip(unique_reasons, reason_counts)}
                
                hit_max_ratio = float((response_lengths >= max_resp_len * 0.95).sum() / len(response_lengths))
                resp_mean = float(response_lengths.mean())
                nt_mean = float(num_turns.mean())
                reward_mean = float(rewards.mean())
                reward_std = float(rewards.std())
                
                # Count reward statistics
                n_zero_reward = int((rewards == 0).sum())
                n_positive_reward = int((rewards > 0).sum())
                n_negative_reward = int((rewards < 0).sum())
                
                # ===== STEP 3: Log to console =====
                logger.info("=" * 60)
                logger.info("[DEBUG BATCH METRICS]")
                logger.info(f"  tool_call_count: mean={tc_mean:.2f}, max={tc_max}, min={tc_min}")
                logger.info(f"  tool_call_ratio: {tc_ratio:.3f}")
                logger.info(f"  end_reasons: {reason_stats}")
                logger.info(f"  hit_max_length_ratio: {hit_max_ratio:.3f}")
                logger.info(f"  response_length/mean: {resp_mean:.1f}")
                logger.info(f"  num_turns/mean: {nt_mean:.1f}")
                logger.info(f"  reward/mean: {reward_mean:.3f}, std: {reward_std:.3f}")
                logger.info(f"  reward/zero: {n_zero_reward}, positive: {n_positive_reward}, negative: {n_negative_reward}")
                if any_reward_fallback:
                    logger.warning("[DEBUG] Some rewards had fallback parsing (multi-element tensors -> mean)")
                logger.info("=" * 60)
                
                # ===== STEP 4: Dump samples with global_step =====
                if DEBUG_CONFIG.get('dump_enabled', False):
                    tokenizer = DEBUG_CONFIG.get('_tokenizer')
                    
                    if tokenizer is not None:
                        prompts = output.batch['prompts']
                        responses = output.batch['responses']
                        
                        # Increment global_step for each postprocess call
                        global_step = DEBUG_CONFIG["_global_step"]
                        DEBUG_CONFIG["_global_step"] += 1
                        
                        # Find samples to dump
                        dump_idx = []
                        for i in range(len(inputs)):
                            if response_lengths[i] >= max_resp_len * 0.95:
                                dump_idx.append((i, 10))
                            elif rewards[i] == 0:
                                dump_idx.append((i, 5))
                            elif tool_call_counts[i] == 0:
                                dump_idx.append((i, 3))
                        
                        dump_idx.sort(key=lambda x: x[1], reverse=True)
                        dump_idx = [x[0] for x in dump_idx[:DEBUG_CONFIG.get('dump_samples_per_step', 2)]]
                        
                        for idx in dump_idx:
                            try:
                                pt = tokenizer.decode(prompts[idx].tolist(), skip_special_tokens=True)
                                rt = tokenizer.decode(responses[idx].tolist(), skip_special_tokens=True)
                                
                                if len(pt) > 500:
                                    pt = pt[:500] + "...[trunc]"
                                if len(rt) > 1500:
                                    rt = rt[:1500] + "...[trunc]"
                                
                                sample = {
                                    'sample_idx': int(idx),
                                    'tool_call_count': int(tool_call_counts[idx]),
                                    'num_turns': int(num_turns[idx]),
                                    'reward': float(rewards[idx]),
                                    'response_length': int(response_lengths[idx]),
                                    'max_response_length': int(max_resp_len),
                                    'hit_max_length_ratio': float(response_lengths[idx] / max_resp_len),
                                    'end_reason': str(end_reasons[idx]),
                                    'prompt_text': pt,
                                    'response_text': rt,
                                    '_global_step': global_step,
                                }
                                
                                dump_path = os.path.join(DEBUG_CONFIG['dump_dir'], f"rollout_step{global_step:05d}.jsonl")
                                with open(dump_path, 'a') as f:
                                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                                
                                logger.info(f"[DEBUG] Dumped sample {idx} at step {global_step}")
                            except Exception as e:
                                logger.warning(f"[DEBUG] Dump failed: {e}")
                
                # ===== STEP 5: Add debug metrics to meta_info for WandB =====
                debug_metrics = {
                    'debug/tool_call_count/mean': tc_mean,
                    'debug/tool_call_count/max': tc_max,
                    'debug/tool_call_count/min': tc_min,
                    'debug/tool_call_ratio': tc_ratio,
                    'debug/hit_max_length_ratio': hit_max_ratio,
                    'debug/response_length/mean': resp_mean,
                    'debug/num_turns/mean': nt_mean,
                    'debug/reward/mean': reward_mean,
                    'debug/reward/std': reward_std,
                    'debug/reward/zero_count': n_zero_reward,
                    'debug/reward/positive_count': n_positive_reward,
                    'debug/reward/negative_count': n_negative_reward,
                }
                
                for reason, stats in reason_stats.items():
                    key = reason.replace('/', '_').replace(' ', '_')
                    debug_metrics[f'debug/end_reason/{key}_ratio'] = stats['ratio']
                
                # Merge into meta_info
                if 'metrics' in output.meta_info and output.meta_info['metrics']:
                    output.meta_info['metrics'][0].update(debug_metrics)
                
            except Exception as e:
                logger.warning(f"[DEBUG] Metrics computation failed: {e}")
                import traceback
                traceback.print_exc()
            
            return output
        
        AgentLoopWorker._postprocess = patched_postprocess
        logger.info("[DEBUG] Patched AgentLoopWorker._postprocess")
        return True
        
    except Exception as e:
        logger.error(f"[DEBUG] Failed to patch AgentLoopWorker: {e}")
        import traceback
        traceback.print_exc()
        return False


def apply_debug_patches(dump_dir=None, samples_per_step=2, tokenizer=None):
    """Apply all debug patches. MUST be called in training process."""
    logger.info("[DEBUG] Applying debug patches...")
    
    init_debug_logging(dump_dir, samples_per_step, tokenizer)
    
    success1 = patch_tool_agent_loop()
    success2 = patch_agent_loop_worker()
    
    if success1 and success2:
        logger.info("[DEBUG] All patches applied successfully!")
    else:
        logger.error("[DEBUG] Some patches failed!")


if __name__ == "__main__":
    print("This module must be imported and apply_debug_patches() called in training process.")
    print("Use debug_entry.py to run with debug logging.")
