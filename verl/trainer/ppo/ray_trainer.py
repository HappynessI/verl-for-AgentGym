# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import logging
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional


import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, using max_colocate_count=3: actor_critic_ref, rollout, reward model (optional)
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=3, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def _find_subsequence_positions(full_ids: list[int], subseq: list[int], max_search_start: int = None) -> list[int]:
    """Find all positions where subseq occurs in full_ids.

    Returns the list of starting positions of each occurrence.
    """
    if not subseq or not full_ids:
        return []
    n, m = len(full_ids), len(subseq)
    if m > n:
        return []
    if max_search_start is None:
        max_search_start = n - m + 1
    positions = []
    for i in range(min(max_search_start, n - m + 1)):
        if full_ids[i:i + m] == subseq:
            positions.append(i)
    return positions


def _normalize_single_prompt(raw_prompt):
    """Normalize a single prompt entry from parquet to a list of message dicts.

    Handles the common storage patterns for a single prompt:
    - np.array([[{'role': ..., 'content': ...}]])  -> shape (1, 1)
    - np.array([{'role': ..., 'content': ...}])   -> shape (1,)
    - [{'role': ..., 'content': ...}]             -> already a list
    - string literal JSON                           -> str

    Returns:
        list[dict]: list of message dicts
    """
    import ast
    import json

    if raw_prompt is None:
        return []

    if hasattr(raw_prompt, "tolist"):
        # numpy array case
        inner = raw_prompt.tolist()
        if isinstance(inner, list) and len(inner) > 0:
            first = inner[0]
            if isinstance(first, dict):
                return inner
            elif isinstance(first, list):
                return first
    elif isinstance(raw_prompt, str):
        for parser in [ast.literal_eval, json.loads]:
            try:
                parsed = parser(raw_prompt)
                if isinstance(parsed, list) and len(parsed) > 0:
                    first = parsed[0]
                    if isinstance(first, dict):
                        return parsed
                    elif isinstance(first, list):
                        return first
            except Exception:
                pass
    elif isinstance(raw_prompt, list):
        if len(raw_prompt) > 0 and isinstance(raw_prompt[0], dict):
            return raw_prompt

    return []


def compute_prefix_mask(data: DataProto, tokenizer=None):
    """Compute the mask for assistant prefix tokens in the prompt.

    This function identifies which tokens in the prompt correspond to assistant messages
    (teacher history from the prompt). Used when optimize_prefix_tokens=True.

    IMPORTANT: Only assistant role tokens are included in the prefix mask.
    System and user tokens are NOT included in prefix optimization.

    This function prefers the pre-computed `prefix_mask` from the parquet (authoritative)
    over runtime computation. If the pre-computed mask is not available, it falls back
    to computing from raw_prompt.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        tokenizer: Tokenizer for processing chat format prompts. REQUIRED when
                   optimize_prefix_tokens=True and pre-computed mask is not available.

    Returns:
        torch.Tensor: The mask for assistant prefix tokens (shape: batch_size, seq_len).
                      1 for assistant prefix tokens, 0 for others (including system/user).

    Raises:
        ValueError: If tokenizer is None or cannot parse assistant token spans.
    """
    batch_size, response_len = data.batch["responses"].shape
    attention_mask = data.batch["attention_mask"]
    total_length = attention_mask.size(1)
    prompt_length = total_length - response_len

    # === DEBUG: 精确报告 runtime prompt_length 来源 ===
    attn_sum = int(attention_mask.sum().item())
    print(f"[DEBUG_PROMPT_LEN] responses.shape={data.batch['responses'].shape}, attn_mask.shape={attention_mask.shape}, attn_mask.sum()={attn_sum}, total_length={total_length}, response_len={response_len}, prompt_length={prompt_length}, attn_mask.sum()==prompt_length: {attn_sum==prompt_length}", flush=True)
    if "prefix_mask" in data.non_tensor_batch:
        raw_mask_arr = data.non_tensor_batch["prefix_mask"]
        print(f"[DEBUG_PROMPT_LEN] raw_mask_arr.shape={raw_mask_arr.shape}, dtype={raw_mask_arr.dtype}", flush=True)
        for i in range(min(2, batch_size)):
            raw_mask_i = raw_mask_arr[i]
            mask_len = int(np.array(raw_mask_i).reshape(-1).shape[0]) if hasattr(raw_mask_i,'reshape') else len(raw_mask_i)
            mask_sum = int(np.array(raw_mask_i).sum())
            print(f"[DEBUG_PROMPT_LEN]   sample {i}: raw_mask_len={mask_len}, mask_sum={mask_sum}", flush=True)
    # === end DEBUG ===

    # Check if we have pre-computed prefix_masks from the parquet.
    # The parquet stores per-sample 1D arrays: np.array([list_of_0_or_1]).
    # After collate_fn, non_tensor_batch["prefix_mask"] is dtype=object array of shape (batch_size,)
    # where each element is the per-sample 1D mask.
    if "prefix_mask" in data.non_tensor_batch:
        raw_mask_arr = data.non_tensor_batch["prefix_mask"]

        # Validate structure: must be object-array of shape (batch_size,) containing per-sample 1D arrays
        if not isinstance(raw_mask_arr, np.ndarray):
            raise ValueError(f"[PREFIX_MASK] prefix_mask in non_tensor_batch must be np.ndarray, got {type(raw_mask_arr)}")
        if raw_mask_arr.dtype != object:
            raise ValueError(
                f"[PREFIX_MASK] prefix_mask dtype must be object (from collate_fn), got {raw_mask_arr.dtype}. "
                f"This function reads from non_tensor_batch which should have the collated object-array, "
                f"not raw parquet data. Use the trainer-side batch-processing path instead."
            )
        if raw_mask_arr.shape[0] != batch_size:
            raise ValueError(
                f"[PREFIX_MASK] prefix_mask array length {raw_mask_arr.shape[0]} != batch_size {batch_size}"
            )

        # Process each sample individually: per-sample mask must match its prompt_length
        mask_rows = []
        for i in range(batch_size):
            raw_mask_i = raw_mask_arr[i]

            # Convert per-sample mask to 1D tensor
            if isinstance(raw_mask_i, list):
                mask_i = torch.tensor(raw_mask_i, dtype=torch.float32)
            else:
                mask_i = torch.as_tensor(raw_mask_i, dtype=torch.float32)

            # Flatten to 1D
            mask_i = mask_i.reshape(-1)
            if mask_i.dim() != 1:
                raise ValueError(
                    f"[PREFIX_MASK] per-sample mask for sample {i} must be 1D after flatten, got shape={mask_i.shape}"
                )

            # STRICT: per-sample mask length must equal runtime prompt_length
            if mask_i.shape[0] != prompt_length:
                raise ValueError(
                    f"[PREFIX_MASK] Per-sample mismatch at index {i}: "
                    f"mask length {mask_i.shape[0]} != runtime prompt_length {prompt_length}. "
                    f"This means parquet generation and runtime tokenization are inconsistent for this sample. "
                    f"REGENERATE the parquet with matching tokenizer/settings."
                )

            # Validate: sum must equal assistant_prefix_old_log_probs length for this sample
            if "assistant_prefix_old_log_probs" in data.non_tensor_batch:
                lp_arr = data.non_tensor_batch["assistant_prefix_old_log_probs"]
                lp_i = lp_arr[i]
                lp_len = lp_i.shape[-1] if hasattr(lp_i, 'shape') else len(lp_i)
                mask_sum = int(mask_i.sum().item())
                if mask_sum != lp_len:
                    raise ValueError(
                        f"[PREFIX_MASK] Per-sample mismatch at index {i}: "
                        f"mask.sum()={mask_sum} != assistant_prefix_old_log_probs length={lp_len}. "
                        f"Pre-computed old_logprobs were generated on a different tokenization. "
                        f"REGENERATE the parquet."
                    )

            mask_rows.append(mask_i)

        # Stack into (batch_size, prompt_length)
        return torch.stack(mask_rows, dim=0)

    # Fallback: compute from raw_prompt
    # WARNING: this may diverge from the parquet generation logic!
    if tokenizer is None:
        raise ValueError(
            "tokenizer is required for computing assistant prefix mask when "
            "pre-computed 'prefix_mask' is not in the parquet. "
            "Please ensure compute_prefix_mask() is called with a valid tokenizer, "
            "or regenerate the parquet with build_prefix_old_logprob_dataset.py."
        )

    raw_prompts = data.non_tensor_batch.get(
        "prompt",
        data.non_tensor_batch.get("raw_prompt", None)
    )

    available_keys = list(data.non_tensor_batch.keys()) if hasattr(data.non_tensor_batch, 'keys') else []
    logging.getLogger(__name__).info(f"[PREFIX_MASK] No pre-computed prefix_mask found. Computing from raw_prompt. keys={available_keys}")

    if raw_prompts is not None and not isinstance(raw_prompts, list):
        if isinstance(raw_prompts, np.ndarray):
            # Handle the parquet storage format: np.array([[{...}]]) or np.array([{...}])
            # Convert each to the list of message dicts
            raw_prompts = [
                _normalize_single_prompt(p) for p in raw_prompts
            ]
        else:
            raw_prompts = list(raw_prompts)

    if raw_prompts is None or len(raw_prompts) == 0:
        has_raw_prompt = "raw_prompt" in data.non_tensor_batch if hasattr(data.non_tensor_batch, '__contains__') else False
        raise ValueError(
            f"prompt field is missing or empty in batch.non_tensor_batch. "
            f"Cannot compute assistant prefix mask without prompt data.\n"
            f"  - Available keys: {available_keys}\n"
            f"  - Has 'raw_prompt': {has_raw_prompt}\n"
            f"  - Batch size: {attention_mask.shape[0] if attention_mask is not None else 'unknown'}\n"
            f"  - Attempted to find: 'prompt' or 'raw_prompt'"
        )

    device = attention_mask.device
    assistant_mask = compute_assistant_token_mask_from_prompt(
        raw_prompts=raw_prompts,
        tokenizer=tokenizer,
        prompt_length=prompt_length,
        batch_size=batch_size,
        device=device
    )

    # Validate: warn if mask sum is suspiciously different from parquet expectations
    if "assistant_prefix_old_log_probs" in data.non_tensor_batch or "assistant_prefix_old_logprobs" in data.non_tensor_batch:
        key = data.non_tensor_batch.get("assistant_prefix_old_log_probs") or data.non_tensor_batch.get("assistant_prefix_old_logprobs")
        if isinstance(key, list):
            expected_count = len(key)
        else:
            expected_count = key.shape[-1] if hasattr(key, 'shape') else 0
        mask_sum = int(assistant_mask.sum().item())
        if mask_sum != expected_count:
            logging.getLogger(__name__).warning(
                f"[PREFIX_MASK] Runtime mask.sum()={mask_sum} != "
                f"cached old_logprobs length={expected_count}. "
                f"Consider regenerating the parquet with build_prefix_old_logprob_dataset.py."
            )

    return assistant_mask


def compute_assistant_token_mask_from_prompt(raw_prompts, tokenizer, prompt_length, batch_size, device):
    """Compute mask for assistant role tokens - EXACTLY matching build_prefix_old_logprob_dataset.py.

    This function replicates the tokenization logic from build_prefix_old_logprob_dataset.py:
    1. For each assistant message: role_text + content_text (without add_generation_prompt)
    2. role_text = f"<|im_start|>{role}\n" tokenized WITHOUT special tokens
    3. content_text = f"{content}<|im_end|>\n" tokenized WITHOUT special tokens
    4. Search for [role_ids + content_ids] sequence in the full tokenization
    5. Mark ALL matched positions (including the trailing <|im_end|> token) as assistant

    Args:
        raw_prompts: List of prompts (each is a list of chat messages with 'role' and 'content')
        tokenizer: Tokenizer for processing
        prompt_length: Expected prompt length
        batch_size: Batch size
        device: Device to place the mask tensor

    Returns:
        torch.Tensor: Mask with 1 for assistant tokens, 0 for others
    """
    mask = torch.zeros((batch_size, prompt_length), dtype=torch.float32, device=device)

    for i, raw_prompt in enumerate(raw_prompts):
        if i >= batch_size:
            break

        try:
            # Normalize raw_prompt to a list of message dicts
            if hasattr(raw_prompt, 'tolist'):
                chat_messages = raw_prompt.tolist()
            else:
                chat_messages = list(raw_prompt)

            if not isinstance(chat_messages, (list, tuple)) or len(chat_messages) == 0:
                continue

            # Tokenize the FULL prompt (no generation prompt, matching build script)
            full_text = tokenizer.apply_chat_template(
                chat_messages,
                add_generation_prompt=False,
                tokenize=False
            )
            tokens = tokenizer(full_text, add_special_tokens=True, return_tensors="pt")
            full_ids = tokens.input_ids[0].tolist()

            # For each assistant message, find its position in full_ids and mark as assistant
            for msg in chat_messages:
                role = msg.get('role', '')
                content = msg.get('content', '')

                if role != 'assistant':
                    continue

                # EXACTLY matching build_prefix_old_logprob_dataset.py:
                # role_text = "<|im_start|>{role}\n"
                role_text = f"<|im_start|>{role}\n"
                role_ids = tokenizer(role_text, add_special_tokens=False).input_ids

                # content_text = "{content}<|im_end|>\n" (note: content is NOT prefixed with \n)
                content_text = f"{content}<|im_end|>\n"
                content_ids = tokenizer(content_text, add_special_tokens=False).input_ids

                msg_ids = list(role_ids) + list(content_ids)

                # Find the subsequence position
                positions = _find_subsequence_positions(full_ids, msg_ids)
                if not positions:
                    # Fallback search with limited range
                    positions = _find_subsequence_positions(
                        full_ids, msg_ids, max_search_start=max(1, len(full_ids) - len(msg_ids) - 500)
                    )

                if not positions:
                    continue

                # Mark ALL positions in the matched sequence (including trailing <|im_end|>)
                start_pos = positions[0]
                end_pos = start_pos + len(msg_ids)

                # Clip to prompt_length
                end_pos = min(end_pos, prompt_length)
                start_pos = min(start_pos, prompt_length)

                if start_pos < end_pos:
                    mask[i, start_pos:end_pos] = 1.0

        except Exception as e:
            # Skip this sample on error
            continue

    return mask


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_module_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = (
            config.actor_rollout_ref.model.get("lora_rank", 0) > 0
            or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        )

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured module_logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured module_logger
        self.validation_generations_module_logger.log(self.config.trainer.module_logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid", "turn_scores"}) & batch.non_tensor_batch.keys()
        print(f"[DEBUG_GEN_BATCH] batch.non_tensor_batch keys = {list(batch.non_tensor_batch.keys())}", flush=True)
        print(f"[DEBUG_GEN_BATCH] reward_model_keys = {reward_model_keys}", flush=True)
        if "turn_scores" in batch.non_tensor_batch:
            ts = batch.non_tensor_batch["turn_scores"]
            print(f"[DEBUG_GEN_BATCH] turn_scores IN reward_model_keys: {'turn_scores' in reward_model_keys}", flush=True)
            print(f"[DEBUG_GEN_BATCH] turn_scores value = {ts}", flush=True)
        if "prefix_mask" in batch.non_tensor_batch:
            v = batch.non_tensor_batch["prefix_mask"]
            print(f"[DEBUG_GEN_BATCH] prefix_mask in batch.non_tensor_batch: type={type(v).__name__}, dtype={getattr(v,'dtype','N/A')}, shape={getattr(v,'shape','N/A')}", flush=True)
            if hasattr(v, 'dtype') and v.dtype == object and len(v) > 0:
                print(f"[DEBUG_GEN_BATCH]   prefix_mask[0] type={type(v[0]).__name__}", flush=True)

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        if "prefix_mask" in gen_batch.non_tensor_batch:
            v = gen_batch.non_tensor_batch["prefix_mask"]
            print(f"[DEBUG_GEN_BATCH] prefix_mask in gen_batch.non_tensor_batch: type={type(v).__name__}, dtype={getattr(v,'dtype','N/A')}, shape={getattr(v,'shape','N/A')}", flush=True)
            if hasattr(v, 'dtype') and v.dtype == object and len(v) > 0:
                print(f"[DEBUG_GEN_BATCH]   prefix_mask[0] type={type(v[0]).__name__}, len={len(v[0]) if hasattr(v[0],'__len__') else 'N/A'}", flush=True)
        return gen_batch

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
                rm_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            else:
                rm_resource_pool = None

            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
                rm_resource_pool=rm_resource_pool,
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        world_size = self.actor_rollout_wg.world_size
        if keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(world_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=world_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(
                workload_lst, k_partitions=world_size, equal_size=True
            )
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        for idx, partition in enumerate(global_partition_lst):
            partition.sort(key=lambda x: (workload_lst[x], x))
            ordered_partition = partition[::2] + partition[1::2][::-1]
            global_partition_lst[idx] = ordered_partition
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        self.tracking_logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            self.tracking_logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                batch.meta_info["logprob_temperature"] = 1.0

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # NOTE:
                # DataProto.repeat() already repeats non_tensor_batch, including ragged object arrays.
                # Expanding prefix sidecars here causes a second repeat inside gen_batch.repeat(),
                # which breaks batch-size consistency when rollout.n > 1.
                n_rollouts = self.config.actor_rollout_ref.rollout.n

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=n_rollouts, interleave=True
                )
                # Preserve the repeated pre-rollout non-tensor sidecars because the rollout output
                # replaces non_tensor_batch with reward/interaction fields only.
                restore_non_tensor_batch = gen_batch_output.non_tensor_batch.copy()

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                rm_scores = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(rm_scores)
                            reward_baseline_tensor, _ = compute_reward(batch, self.reward_fn)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat batch (which has basic keys) and union with rollout output
                    # NOTE: _get_gen_batch() pops prefix sidecars out of the original batch and places them
                    # into gen_batch.non_tensor_batch. After rollout, gen_batch_output.non_tensor_batch only
                    # contains reward/interaction fields, so we must restore prefix keys from the preserved
                    # pre-rollout repeated non-tensor batch, not from batch.non_tensor_batch.
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                    # === DEBUG: Check repeated prefix keys in batch.non_tensor_batch ===
                    for _pk in ["assistant_prefix_old_log_probs", "prefix_mask"]:
                        if _pk in batch.non_tensor_batch:
                            _v = batch.non_tensor_batch[_pk]
                            if isinstance(_v, np.ndarray) and _v.dtype == object:
                                def _safe_shape(x):
                                    return getattr(x, 'shape', f"no_shape({type(x).__name__})")
                                print(f"[DEBUG_EXPAND] batch.non_tensor_batch[{_pk}] shape={_v.shape}, first_elem_type={type(_v[0]).__name__}, first_elem={_safe_shape(_v[0])}", flush=True)
                            else:
                                print(f"[DEBUG_EXPAND] batch.non_tensor_batch[{_pk}] type={type(_v)}, len={len(_v) if hasattr(_v,'__len__') else 'N/A'}", flush=True)
                    # === end DEBUG ===

                    # CRITICAL: Restore prefix keys to batch.batch BEFORE union with gen_batch_output.
                    # gen_batch_output only contains rollout tensor keys (responses, etc.) and has its OWN
                    # non_tensor_batch from the reward computation. It does NOT contain assistant_prefix_old_log_probs.
                    # The union only merges keys FROM gen_batch_output INTO batch, not the other way around.
                    # So we must copy already-repeated prefix keys from the preserved pre-rollout non_tensor_batch NOW.
                    print(f"[DEBUG_BATCH] gen_batch_output.batch.keys() = {list(gen_batch_output.batch.keys())}", flush=True)
                    print(f"[DEBUG_BATCH] gen_batch_output.non_tensor_batch.keys() = {list(gen_batch_output.non_tensor_batch.keys())}", flush=True)

                    prefix_keys_to_restore = [
                        "assistant_prefix_old_log_probs",
                        "assistant_prefix_old_logprobs",
                        "prefix_token_count",
                        "prefix_mask",
                        "assistant_prefix_span",
                        "prompt",
                        "raw_prompt",
                    ]
                    restoredKeys = []
                    for key in prefix_keys_to_restore:
                        if key in restore_non_tensor_batch:
                            val = restore_non_tensor_batch[key]
                            print(f"[DEBUG_RESTORE] Copying key={key} from restore_non_tensor_batch to batch.batch", flush=True)
                            if isinstance(val, np.ndarray) and val.dtype == object:
                                per_sample_lens = []
                                for _i in range(min(5, val.shape[0])):
                                    elem = val[_i]
                                    l = len(elem) if hasattr(elem, '__len__') else 'N/A'
                                    per_sample_lens.append(l)
                                print(f"[DEBUG_RESTORE]   value is ragged object array: shape={val.shape}, first_5_lens={per_sample_lens}{'...' if val.shape[0] > 5 else ''}", flush=True)
                            batch.batch[key] = val
                            # Immediately verify shape after assignment
                            if key in batch.batch:
                                bv = batch.batch[key]
                                print(f"[DEBUG_RESTORE]   AFTER assignment: batch.batch['{key}'] type={type(bv).__name__}, is_Tensor={isinstance(bv,torch.Tensor)}, is_ndarray={isinstance(bv,np.ndarray)}, dtype={getattr(bv,'dtype','N/A')}, shape={bv.shape if hasattr(bv,'shape') else getattr(bv,'shape','N/A')}", flush=True)
                            restoredKeys.append(key)

                    # === DEBUG_PREFIX_TOKEN_COUNT at restore ===
                    for key in ["prefix_token_count"]:
                        if key in batch.non_tensor_batch:
                            val = batch.non_tensor_batch[key]
                            print(f"[DEBUG_PREFIX_TOKEN_COUNT] batch.non_tensor_batch['{key}']: type={type(val).__name__}, dtype={getattr(val,'dtype','N/A')}, shape={getattr(val,'shape','N/A') if hasattr(val,'shape') else len(val)}", flush=True)
                            if hasattr(val, '__len__') and not isinstance(val, np.ndarray):
                                print(f"[DEBUG_PREFIX_TOKEN_COUNT]   first 5 values: {list(val[:5])}", flush=True)
                            elif isinstance(val, np.ndarray) and val.dtype == object:
                                print(f"[DEBUG_PREFIX_TOKEN_COUNT]   first 5 values: {[val[i] for i in range(min(5,len(val)))]}", flush=True)
                            elif isinstance(val, np.ndarray):
                                print(f"[DEBUG_PREFIX_TOKEN_COUNT]   first 5 values: {val[:5].tolist()}", flush=True)
                        if key in batch.batch:
                            bv = batch.batch[key]
                            print(f"[DEBUG_PREFIX_TOKEN_COUNT] batch.batch['{key}']: type={type(bv).__name__}, dtype={getattr(bv,'dtype','N/A')}, shape={getattr(bv,'shape','N/A') if hasattr(bv,'shape') else len(bv)}", flush=True)
                            if isinstance(bv, np.ndarray):
                                print(f"[DEBUG_PREFIX_TOKEN_COUNT]   first 5 values: {bv[:5].tolist() if bv.dtype!=object else [bv[i] for i in range(min(5,len(bv)))]}", flush=True)
                            elif isinstance(bv, torch.Tensor):
                                print(f"[DEBUG_PREFIX_TOKEN_COUNT]   first 5 values: {bv[:5].tolist()}", flush=True)
                    print(f"[DEBUG_BATCH] Restored prefix keys to batch.batch: {restoredKeys}", flush=True)

                    batch = batch.union(gen_batch_output)

                    print(f"[DEBUG_BATCH] batch.batch.keys() after union = {list(batch.batch.keys())}", flush=True)
                    # === Check if union overwrote prefix keys ===
                    for rk in restoredKeys:
                        if rk in batch.batch:
                            bv = batch.batch[rk]
                            print(f"[DEBUG_AFTER_UNION] batch.batch['{rk}'] type={type(bv)}, is_Tensor={isinstance(bv,torch.Tensor)}, shape={bv.shape if hasattr(bv,'shape') else getattr(bv,'shape','N/A')}", flush=True)
                    # === DEBUG: Check restored value type and content ===
                    for _pk in restoredKeys:
                        if _pk in batch.batch:
                            _v = batch.batch[_pk]
                            if isinstance(_v, np.ndarray):
                                print(f"[DEBUG_AFTER] batch.batch[{_pk}] is np.ndarray, shape={_v.shape}, dtype={_v.dtype}", flush=True)
                            elif isinstance(_v, torch.Tensor):
                                print(f"[DEBUG_AFTER] batch.batch[{_pk}] is Tensor, shape={_v.shape}", flush=True)
                            else:
                                print(f"[DEBUG_AFTER] batch.batch[{_pk}] type={type(_v)}", flush=True)
                    # === end DEBUG ===

                    if restoredKeys and self.config.algorithm.get("optimize_prefix_tokens", False):
                        logging.getLogger(__name__).info(f"[PREFIX_OPT] Restored prefix keys from gen_batch.non_tensor_batch (n={n_rollouts}): {restoredKeys}")

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    
                    # Compute prefix_mask when optimize_prefix_tokens is enabled
                    # This is used to include prefix tokens in the GRPO loss
                    if self.config.algorithm.get("optimize_prefix_tokens", False):
                        # CRITICAL: Fail fast if cached prefix old_logprobs are missing
                        # prefix optimization REQUIRES pre-cached SFT old_logprobs
                        # NOTE: After restore, prefix keys are in batch.batch (restored from gen_batch.non_tensor_batch)
                        prefix_logprobs_key = None
                        for key in ["assistant_prefix_old_log_probs", "assistant_prefix_old_logprobs"]:
                            if key in batch.batch:
                                prefix_logprobs_key = key
                                break

                        if prefix_logprobs_key is None:
                            batch_keys = list(batch.batch.keys())
                            raise ValueError(
                                f"[PREFIX_OPT] Missing 'assistant_prefix_old_log_probs' in batch.batch. "
                                f"batch.batch keys: {batch_keys}."
                            )

                        device = batch.batch["attention_mask"].device
                        batch_size, response_len = batch.batch["responses"].shape
                        total_length = batch.batch["attention_mask"].shape[1]
                        prompt_len = total_length - response_len

                        # === Load cached prefix old_log_probs from batch.batch ===
                        # Reconstruct into a dense [B, max_prefix_len] tensor.
                        # Per-sample lengths come from prefix_token_count (may be Tensor or ndarray).
                        cached_olp = batch.batch[prefix_logprobs_key]

                        # Get prefix_lens in numpy format regardless of source type
                        ptc = batch.batch["prefix_token_count"]
                        if isinstance(ptc, torch.Tensor):
                            prefix_lens = ptc.cpu().numpy()
                        else:
                            # Already a regular numeric array (np.repeat already made it int64)
                            prefix_lens = np.array(ptc, dtype=np.int64)
                        B = len(prefix_lens)

                        if isinstance(cached_olp, torch.Tensor):
                            # Already a tensor (edge case)
                            cached_prefix_logprobs_full_seq = cached_olp.float()
                        else:
                            # cached_olp is a ragged np.ndarray dtype=object of lists
                            assert isinstance(cached_olp, np.ndarray) and cached_olp.dtype == object, \
                                (f"Expected ragged object array, got {type(cached_olp)} "
                                 f"dtype={getattr(cached_olp, 'dtype', 'N/A')}")
                            max_len = int(prefix_lens.max())
                            dense = np.full((B, max_len), 0.0, dtype=np.float32)
                            for b in range(B):
                                sample_lp = cached_olp[b]  # list of floats
                                actual_len = int(prefix_lens[b])
                                dense[b, :actual_len] = np.array(sample_lp, dtype=np.float32)
                            cached_prefix_logprobs_full_seq = torch.from_numpy(dense).float()

                        # === CHECK 1: per-sample cached old_logprobs length vs prefix_token_count ===
                        for b in range(B):
                            cached_len = len(cached_olp[b])
                            expected_len = int(prefix_lens[b])
                            assert cached_len == expected_len, (
                                f"[CHECK_1 FAIL] sample {b}: cached old_logprobs len={cached_len} "
                                f"!= prefix_token_count[{b}]={expected_len}. "
                                f"Will be silently truncated/padded — aborting."
                            )
                        print(f"[CHECK_1 PASS] All {B} samples: len(cached_olp[b]) == prefix_token_count[b]", flush=True)

                        print(f"[DEBUG_1592] cached_prefix_logprobs_full_seq: type={type(cached_prefix_logprobs_full_seq).__name__}, dtype={cached_prefix_logprobs_full_seq.dtype}, shape={cached_prefix_logprobs_full_seq.shape}", flush=True)

                        # === Materialize prefix_mask from ragged object array ===
                        # prefix_mask is a ragged list of 0/1 ints per sample.
                        # Convert to dense [B, max_prompt_len] float tensor.
                        raw_prefix_mask = batch.batch["prefix_mask"]
                        if isinstance(raw_prefix_mask, torch.Tensor):
                            mk_tensor = raw_prefix_mask.float()
                        else:
                            # raw_prefix_mask is ragged np.ndarray dtype=object of lists of ints
                            assert isinstance(raw_prefix_mask, np.ndarray) and raw_prefix_mask.dtype == object, \
                                f"Expected ragged object array for prefix_mask, got {type(raw_prefix_mask)}"
                            pm_lens = np.array([len(raw_prefix_mask[b]) for b in range(B)])
                            max_pm_len = int(pm_lens.max())
                            dense_pm = np.zeros((B, max_pm_len), dtype=np.float32)
                            for b in range(B):
                                dense_pm[b, :pm_lens[b]] = np.array(raw_prefix_mask[b], dtype=np.float32)
                            mk_tensor = torch.from_numpy(dense_pm).float()
                        mk_tensor = mk_tensor.to(device)

                        # === CHECK 2: per-sample prefix_mask ones count vs prefix_token_count ===
                        # Recompute pm_lens from the dense tensor (all rows have same padded length)
                        for b in range(B):
                            pm_ones = int(mk_tensor[b].sum().item())
                            expected_len = int(prefix_lens[b])
                            assert pm_ones == expected_len, (
                                f"[CHECK_2 FAIL] sample {b}: prefix_mask ones={pm_ones} "
                                f"!= prefix_token_count[{b}]={expected_len}. "
                                f"Discrepancy between prefix_mask and prefix_token_count."
                            )
                        print(f"[CHECK_2 PASS] All {B} samples: prefix_mask.sum() == prefix_token_count", flush=True)

                        # === Write dense tensors back to batch.batch for Ray serialization ===
                        # Replace ragged object arrays with dense tensors so they can be
                        # serialized and sent to actor workers.
                        batch.batch["assistant_prefix_old_log_probs"] = cached_prefix_logprobs_full_seq
                        batch.batch["prefix_mask"] = mk_tensor
                        # Clean up remaining ragged object arrays that actor doesn't need
                        for _rag_key in ["raw_prompt"]:
                            if _rag_key in batch.batch:
                                batch.batch.pop(_rag_key)

                        # === SUMMARY: trainer → actor prefix tensors ===
                        olp_final = batch.batch["assistant_prefix_old_log_probs"]
                        pm_final  = batch.batch["prefix_mask"]
                        ptc_final = batch.batch["prefix_token_count"]
                        print(f"[MATERIALIZE_SUMMARY] === TRAINER → ACTOR PREFIX TENSORS ===", flush=True)
                        print(f"  batch.batch['assistant_prefix_old_log_probs']:", flush=True)
                        print(f"    type={type(olp_final).__name__}, dtype={olp_final.dtype}, shape={olp_final.shape}", flush=True)
                        print(f"  batch.batch['prefix_mask']:", flush=True)
                        print(f"    type={type(pm_final).__name__}, dtype={pm_final.dtype}, shape={pm_final.shape}", flush=True)
                        print(f"  batch.batch['prefix_token_count']:", flush=True)
                        print(f"    type={type(ptc_final).__name__}, dtype={getattr(ptc_final, 'dtype', 'N/A')}, shape={ptc_final.shape if hasattr(ptc_final, 'shape') else len(ptc_final)}", flush=True)
                        if hasattr(ptc_final, 'tolist'):
                            print(f"    values (first {min(5, B)}): {ptc_final[:min(5,B)].tolist()}", flush=True)
                        elif isinstance(ptc_final, np.ndarray):
                            print(f"    values (first {min(5, B)}): {ptc_final[:min(5,B)].tolist()}", flush=True)
                        print(f"  B={B}, max_prefix_len={olp_final.shape[1]}, max_prefix_mask_len={pm_final.shape[1]}", flush=True)
                        print(f"[MATERIALIZE_SUMMARY] === END ===", flush=True)

                        # Log alignment info
                        metrics["actor/use_cached_prefix_old_logprob"] = True
                        metrics["actor/prefix_token_count"] = int(cached_prefix_logprobs_full_seq.shape[1])
                        metrics["actor/prefix_mask_sum"] = int(mk_tensor.sum().item())
                    
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_rollout_correction

                        apply_rollout_correction(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        print(f"[REF_POLICY] use_reference_policy=True → will call compute_ref_log_prob", flush=True)
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)
                    else:
                        print(f"[REF_POLICY] use_reference_policy=False → ref policy skipped, NO compute_ref_log_prob call", flush=True)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute seq_level_rewards for DRPO
                        # DRPO requires uid and seq_level_rewards to compute the loss
                        # This is safe for GRPO/PPO as they don't use these fields
                        batch.batch['seq_level_rewards'] = batch.batch['token_level_scores'].sum(dim=-1)
                        if 'uid' in batch.non_tensor_batch:
                            # Keep uid as numpy array of strings - GRPO can use it directly for grouping
                            batch.batch['uid'] = np.array(
                                [str(u) for u in batch.non_tensor_batch['uid']], dtype=object
                            )

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            rollout_config = self.config.actor_rollout_ref.rollout
                            batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
                            # TODO: Make "temperature" single source of truth from generation.
                            batch.meta_info["temperature"] = rollout_config.temperature
                            batch.meta_info["logprob_temperature"] = 1.0
                            
                            # Pass prefix optimization config to actor update
                            if self.config.algorithm.get("optimize_prefix_tokens", False):
                                batch.meta_info["optimize_prefix_tokens"] = True
                                batch.meta_info["prefix_loss_weight"] = self.config.algorithm.get("prefix_loss_weight", 1.0)
                            
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                self.tracking_logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
