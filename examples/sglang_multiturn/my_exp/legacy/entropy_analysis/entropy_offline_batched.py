#!/usr/bin/env python3
"""
Batched Offline Entropy Analysis on MiniMax-M2.1 Trajectories
==============================================================
核心优化：将同一位置（相同前缀长度）的所有 turn 打包成 batch，
一次 forward 完成整个 batch 的熵计算。

吞吐量提升：9597条 × 20turns × 10s = ~55小时  →  几十个 batch × 1s ≈ 几分钟

用法：
  CUDA_VISIBLE_DEVICES=5 python entropy_offline_batched.py \
      --traj_dir /Data/wyh/datasets/Sampling-Data/alfworld_MiniMax-M2.1_20260313_212024 \
      --output_dir /Data/wyh/datasets/Verl-Data/outputs/entropy_offline_minimax \
      --batch_size 32
"""

import os
import sys
import json
import math
import re
import logging
import argparse
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from collections import defaultdict

from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------- 全局模型/Tokenizer ----------
TOKENIZER = None
MODEL = None
MODEL_DEVICE = None


def get_tokenizer(tokenizer_name: str = None):
    global TOKENIZER
    if TOKENIZER is None:
        TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name or "/Data/public/Qwen3-1.7B", trust_remote_code=True)
    return TOKENIZER


def get_model(model_path: str, cuda_device: int = 0):
    global MODEL, MODEL_DEVICE
    if MODEL is not None:
        return MODEL
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    MODEL_DEVICE = torch.device(f"cuda:{cuda_device}")
    MODEL = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    MODEL.to(MODEL_DEVICE)
    MODEL.eval()
    return MODEL


# ---------- 辅助函数 ----------
def apply_qwen_chat_template(messages: List[Dict], tokenizer) -> str:
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return formatted


def compute_entropy(top_logprobs: Dict[str, float]) -> float:
    """从 top-k logprobs 计算 entropy（单位：nat）。"""
    if not top_logprobs:
        return 0.0
    log_ps = list(top_logprobs.values())
    max_lp = max(log_ps)
    reweighted = [lp - max_lp for lp in log_ps]
    log_Z = max_lp + math.log(sum(math.exp(r) for r in reweighted))
    entropy = -sum(math.exp(lp - log_Z) * lp for lp in log_ps)
    return max(0.0, entropy)


def reversible_normalize(s: str):
    """可逆规范化，返回 (normalized_s, norm_spans)"""
    normalized_chars = []
    norm_spans = []
    buffer = []
    buffer_spans = []

    i = 0
    while i < len(s):
        c = s[i]
        if c == '\r' and i + 1 < len(s) and s[i + 1] == '\n':
            while buffer and (buffer[-1] == ' ' or buffer[-1] == '\t'):
                buffer.pop(); buffer_spans.pop()
            normalized_chars.extend(buffer); norm_spans.extend(buffer_spans)
            normalized_chars.append('\n'); norm_spans.append((i, i + 2))
            buffer.clear(); buffer_spans.clear()
            i += 2; continue
        if c == '\r':
            while buffer and (buffer[-1] == ' ' or buffer[-1] == '\t'):
                buffer.pop(); buffer_spans.pop()
            normalized_chars.extend(buffer); norm_spans.extend(buffer_spans)
            normalized_chars.append('\n'); norm_spans.append((i, i + 1))
            buffer.clear(); buffer_spans.clear()
            i += 1; continue
        if c == '\n':
            while buffer and (buffer[-1] == ' ' or buffer[-1] == '\t'):
                buffer.pop(); buffer_spans.pop()
            normalized_chars.extend(buffer); norm_spans.extend(buffer_spans)
            normalized_chars.append('\n'); norm_spans.append((i, i + 1))
            buffer.clear(); buffer_spans.clear()
            i += 1; continue
        if c == ' ' or c == '\t':
            buffer.append(c); buffer_spans.append((i, i + 1))
        else:
            if buffer:
                normalized_chars.append(' '); norm_spans.append(buffer_spans[-1])
                buffer.clear(); buffer_spans.clear()
            normalized_chars.append(c); norm_spans.append((i, i + 1))
        i += 1

    if buffer:
        normalized_chars.append(' '); norm_spans.append(buffer_spans[-1])
    return ''.join(normalized_chars), norm_spans


def find_char_range_in_normalized(target: str, norm_s: str, norm_spans: list) -> Tuple[int, int]:
    """在规范化文本中定位 target 的字符区间（返回规范化后的区间）"""
    # 精确匹配
    pos = norm_s.find(target)
    if pos != -1:
        return pos, pos + len(target)

    # 规范化匹配
    norm_target, _ = reversible_normalize(target)
    pos = norm_s.find(norm_target)
    if pos != -1:
        return pos, pos + len(norm_target)

    # 双锚点匹配
    t_stripped = target.strip()
    idx1 = norm_s.find(t_stripped[: max(1, len(t_stripped) // 4)])
    idx2 = norm_s.rfind(t_stripped[-max(1, len(t_stripped) // 4):])
    if idx1 != -1 and idx2 != -1 and idx1 <= idx2:
        return idx1, idx2 + max(1, len(t_stripped) // 4)
    return -1, -1


# ---------- 数据加载 ----------
def load_trajectories(traj_dir: str, max_samples: int = -1) -> List[Dict]:
    jsonl_name = os.path.basename(traj_dir).replace("_MiniMax-M2.1_", "_trajectories.jsonl").split("_")[0]
    for candidate in [f"{jsonl_name}_trajectories.jsonl", "trajectories.jsonl",
                       "alfworld_trajectories.jsonl", "webshop_trajectories.jsonl",
                       "babyai_trajectories.jsonl", "sciworld_trajectories.jsonl"]:
        path = os.path.join(traj_dir, candidate)
        if os.path.exists(path):
            jsonl_path = path
            break
    else:
        files = [f for f in os.listdir(traj_dir) if f.endswith(".jsonl")]
        jsonl_path = os.path.join(traj_dir, files[0])

    trajectories = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                trajectories.append(json.loads(line))
    logger.info(f"Loaded {len(trajectories)} trajectories from {jsonl_path}")
    if max_samples > 0:
        trajectories = trajectories[:max_samples]
        logger.info(f"Using first {max_samples} trajectories")
    return trajectories


def parse_assistant_turns(conversations: List[Dict]) -> List[str]:
    return [c["content"] for c in conversations if c.get("role") == "assistant"]


def parse_user_turns(conversations: List[Dict]) -> List[str]:
    return [c["content"] for c in conversations if c.get("role") == "user"]


# ---------- 提取所有 Turn 信息（用于批处理） ----------
class TurnInfo(NamedTuple):
    traj_idx: int
    item_id: str
    role: str          # "assistant" or "user"
    turn_idx: int
    prefix_messages: List[Dict]
    target_content: str
    success: float
    reward: float


def build_prefix_for_turn(conversations: List[Dict], turn_idx: int, role: str) -> List[Dict]:
    """构建指定 role 第 turn_idx 个 turn 之前的上下文（不含 target turn 本身）。"""
    prefix_messages = []
    role_count = 0
    for msg in conversations:
        msg_role = msg.get("role")
        if msg_role == role:
            if role_count == turn_idx:
                break
            role_count += 1
        prefix_messages.append(msg)
    return prefix_messages


def extract_all_turns(trajectories: List[Dict], max_turns: int = -1) -> List[TurnInfo]:
    """遍历所有轨迹，提取每个 turn 的完整信息"""
    all_turns = []
    for traj_idx, traj in enumerate(tqdm(trajectories, desc="Extracting turns")):
        conversations = traj.get("conversations", [])
        item_id = traj.get("item_id", f"traj_{traj_idx}")
        success = float(traj.get("success", 0))
        reward = float(traj.get("reward", 0))
        assistant_turns = parse_assistant_turns(conversations)
        user_turns = parse_user_turns(conversations)

        n_a = len(assistant_turns) if max_turns == -1 else min(len(assistant_turns), max_turns)
        n_u = len(user_turns) if max_turns == -1 else min(len(user_turns), max_turns)

        for turn_idx in range(n_a):
            prefix_msgs = build_prefix_for_turn(conversations, turn_idx, role="assistant")
            all_turns.append(TurnInfo(
                traj_idx=traj_idx,
                item_id=item_id,
                role="assistant",
                turn_idx=turn_idx,
                prefix_messages=prefix_msgs,
                target_content=assistant_turns[turn_idx],
                success=success,
                reward=reward,
            ))

        for turn_idx in range(n_u):
            prefix_msgs = build_prefix_for_turn(conversations, turn_idx, role="user")
            all_turns.append(TurnInfo(
                traj_idx=traj_idx,
                item_id=item_id,
                role="user",
                turn_idx=turn_idx,
                prefix_messages=prefix_msgs,
                target_content=user_turns[turn_idx],
                success=success,
                reward=reward,
            ))
    return all_turns


# ---------- 核心：Tokenize 所有 prompt ----------
class TokenizedTurn(NamedTuple):
    turn_info: TurnInfo
    full_ids: List[int]
    target_start: int   # target token 在 full_ids 中的起始 index
    target_end: int     # 结束 index（不含）
    prefix_len: int


def tokenize_all_turns(all_turns: List[TurnInfo], tokenizer) -> List[TokenizedTurn]:
    """对所有 turn 做 tokenize，记录 target token 范围"""
    result = []
    for turn in tqdm(all_turns, desc="Tokenizing prompts"):
        try:
            prefix_msgs = turn.prefix_messages
            target_msg = {"role": turn.role, "content": turn.target_content}
            full_prompt = apply_qwen_chat_template(prefix_msgs + [target_msg], tokenizer)
            enc = tokenizer(
                full_prompt,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            full_ids = enc["input_ids"]
            offsets = enc["offset_mapping"]
            if hasattr(offsets, "tolist"):
                offsets = offsets.tolist()

            # 找 prefix 长度（不含 target）
            prefix_prompt = apply_qwen_chat_template(prefix_msgs, tokenizer) if prefix_msgs else ""
            prefix_enc = tokenizer(prefix_prompt, add_special_tokens=False, return_offsets_mapping=True)
            prefix_ids = prefix_enc["input_ids"]
            prefix_len = len(prefix_ids)

            # target_ids = full_ids[prefix_len:]
            # 但由于 chat template 可能导致 prefix 内容有微小差异，用实际匹配
            target_ids = full_ids[prefix_len:]

            result.append(TokenizedTurn(
                turn_info=turn,
                full_ids=full_ids,
                target_start=prefix_len,
                target_end=len(full_ids),
                prefix_len=prefix_len,
            ))
        except Exception as e:
            logger.warning(f"Tokenize error for {turn.item_id}/{turn.role}/turn{turn.turn_idx}: {e}")
    return result


# ---------- 批次化 forward ----------
def forward_batched(
    tokenized_turns: List[TokenizedTurn],
    model,
    batch_size: int = 32,
) -> Dict[Tuple, Tuple[List[Dict], Optional[str]]]:
    """
    分 batch 做 forward，按序列长度排序 + 动态 batch size 避免 OOM。
    长序列用小 batch，短序列用大 batch。
    """
    # 按长度排序
    sorted_turns = sorted(tokenized_turns, key=lambda t: len(t.full_ids))

    # 建立 (traj_idx, role, turn_idx) -> None 槽位
    key_to_turn = {
        (t.turn_info.traj_idx, t.turn_info.role, t.turn_info.turn_idx): t
        for t in sorted_turns
    }
    results = {k: ([], None) for k in key_to_turn}

    # GPU 显存估算：batch_size * seq_len * vocab_size * 2bytes(bf16)
    # 约 44GB 总显存，扣除模型 3-4GB，预留 35GB 给 activations
    # vocab_size ≈ 152K, bf16=2B → 每 token ≈ 0.3MB
    # logits = batch * seq_len * vocab → 35GB / (seq_len * 0.3MB)
    def compute_dynamic_bs(seq_len: int, base_bs: int = 32) -> int:
        mem_per_token_mb = 0.3
        available_gb = 35.0
        max_bs = int(available_gb * 1024 / (seq_len * mem_per_token_mb))
        return max(1, min(base_bs, max_bs))

    i = 0
    pbar = tqdm(total=len(sorted_turns), desc="Batched forward")

    while i < len(sorted_turns):
        seq_len = len(sorted_turns[i].full_ids)
        dyn_bs = compute_dynamic_bs(seq_len, batch_size)

        # 收集当前 batch：seq_len 在 [seq_len, seq_len * 1.5) 范围内
        batch_end = i + 1
        while batch_end < len(sorted_turns):
            next_len = len(sorted_turns[batch_end].full_ids)
            if next_len <= seq_len * 1.5:
                batch_end += 1
            else:
                break
        batch_turns = sorted_turns[i:min(i + dyn_bs, batch_end)]

        max_len = max(len(t.full_ids) for t in batch_turns)

        input_ids_list = []
        attention_mask_list = []
        for t in batch_turns:
            pad_len = max_len - len(t.full_ids)
            ids = t.full_ids + [0] * pad_len
            mask = [1] * len(t.full_ids) + [0] * pad_len
            input_ids_list.append(ids)
            attention_mask_list.append(mask)

        input_ids = torch.tensor(input_ids_list, device=MODEL_DEVICE)
        attention_mask = torch.tensor(attention_mask_list, device=MODEL_DEVICE)

        try:
            with torch.inference_mode():
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                log_probs = torch.log_softmax(logits.float(), dim=-1)
        except torch.OutOfMemoryError:
            logger.warning(f"OOM at seq_len={max_len}, retrying batch_size=1")
            del input_ids, attention_mask
            torch.cuda.empty_cache()
            for t in batch_turns:
                try:
                    single_ids = torch.tensor([t.full_ids], device=MODEL_DEVICE)
                    with torch.inference_mode():
                        single_logits = model(input_ids=single_ids).logits
                        single_lp = torch.log_softmax(single_logits.float(), dim=-1)
                    ts, te = t.target_start, t.target_end
                    token_list = []
                    for j in range(ts, te):
                        t_logit_idx = j - 1
                        if t_logit_idx < 0:
                            continue
                        gold_id = t.full_ids[j]
                        gold_lp = float(single_lp[0, t_logit_idx, gold_id].item())
                        k = 100
                        topv, topi = torch.topk(single_lp[0, t_logit_idx, :], k=k)
                        top_logprobs = {}
                        for v, idx in zip(topv.tolist(), topi.tolist()):
                            decoded = tokenizer.decode([idx], clean_up_tokenization_spaces=False)
                            top_logprobs[decoded] = v
                        token_list.append({
                            "decoded": tokenizer.decode([gold_id], clean_up_tokenization_spaces=False),
                            "gold_logprob": gold_lp,
                            "top_logprobs": top_logprobs,
                        })
                    del single_ids, single_logits, single_lp
                except Exception as e2:
                    token_list = []
                key = (t.turn_info.traj_idx, t.turn_info.role, t.turn_info.turn_idx)
                results[key] = (token_list, None)
            torch.cuda.empty_cache()
            pbar.update(len(batch_turns))
            i += len(batch_turns)
            continue

        for local_idx, t in enumerate(batch_turns):
            try:
                full_ids = t.full_ids
                ts, te = t.target_start, t.target_end
                token_list = []

                for j in range(ts, te):
                    t_logit_idx = j - 1
                    if t_logit_idx < 0:
                        continue
                    gold_id = full_ids[j]
                    gold_lp = float(log_probs[local_idx, t_logit_idx, gold_id].item())

                    k = 100
                    topv, topi = torch.topk(log_probs[local_idx, t_logit_idx, :], k=k)
                    top_logprobs = {}
                    for v, idx in zip(topv.tolist(), topi.tolist()):
                        decoded = tokenizer.decode([idx], clean_up_tokenization_spaces=False)
                        top_logprobs[decoded] = v

                    token_list.append({
                        "decoded": tokenizer.decode([gold_id], clean_up_tokenization_spaces=False),
                        "gold_logprob": gold_lp,
                        "top_logprobs": top_logprobs,
                    })

                key = (t.turn_info.traj_idx, t.turn_info.role, t.turn_info.turn_idx)
                results[key] = (token_list, None)
            except Exception as e:
                key = (t.turn_info.traj_idx, t.turn_info.role, t.turn_info.turn_idx)
                results[key] = ([], str(e))

        del input_ids, attention_mask, logits, log_probs
        torch.cuda.empty_cache()
        pbar.update(len(batch_turns))
        i += len(batch_turns)

    pbar.close()
    return results


def find_action_token_idx(token_list: List[Dict]) -> Optional[int]:
    text = "".join(t["decoded"] for t in token_list)
    m = re.search(r"Action:\s*\n?", text)
    if m:
        target = m.end()
        char_count = 0
        for i, t in enumerate(token_list):
            char_count += len(t["decoded"])
            if char_count >= target:
                return min(i + 1, len(token_list) - 1)
        return None
    m = re.search(r"\[\[", text)
    if not m:
        return None
    target = m.end()
    char_count = 0
    for i, t in enumerate(token_list):
        char_count += len(t["decoded"])
        if char_count >= target:
            return min(i + 1, len(token_list) - 1)
    return None


# ---------- 聚合结果 ----------
def aggregate_results(
    trajectories: List[Dict],
    all_turns: List[TurnInfo],
    turn_results: Dict[Tuple, Tuple[List[Dict], Optional[str]]],
) -> List[Dict]:
    """按轨迹聚合每个 turn 的熵，生成与原脚本一致的输出格式"""
    # turn_results 已经是 Dict，直接用
    turn_map = {k: v[0] for k, v in turn_results.items()}

    output = []
    for traj_idx, traj in enumerate(tqdm(trajectories, desc="Aggregating results")):
        conversations = traj.get("conversations", [])
        item_id = traj.get("item_id", f"traj_{traj_idx}")
        success = float(traj.get("success", 0))
        reward = float(traj.get("reward", 0))

        assistant_turns_list = parse_assistant_turns(conversations)
        user_turns_list = parse_user_turns(conversations)

        entropy_A_assistant, entropy_B_assistant, entropy_C_assistant = [], [], []
        entropy_A_user, entropy_C_user = [], []
        turn_lengths_assistant, turn_lengths_user = [], []
        cumsum_lengths_assistant, cumsum_lengths_user = [], []
        entropy_per_token_assistant, entropy_per_token_user = [], []

        # Assistant
        cumsum_a = 0
        for turn_idx in range(len(assistant_turns_list)):
            key = (traj_idx, "assistant", turn_idx)
            token_list = turn_map.get(key, [])
            n_tok = len(token_list)
            turn_lengths_assistant.append(n_tok)
            cumsum_a += n_tok
            cumsum_lengths_assistant.append(cumsum_a)

            if n_tok == 0:
                entropy_A_assistant.append(0.0)
                entropy_B_assistant.append(None)
                entropy_C_assistant.append(0.0)
                entropy_per_token_assistant.append([])
                continue

            ent_A = compute_entropy(token_list[0]["top_logprobs"]) if token_list else 0.0
            all_ents = [compute_entropy(t["top_logprobs"]) for t in token_list]
            entropy_C = sum(all_ents) / n_tok if n_tok > 0 else 0.0
            entropy_A_assistant.append(ent_A)
            entropy_C_assistant.append(entropy_C)
            entropy_per_token_assistant.append(all_ents)

            act_idx = find_action_token_idx(token_list)
            entropy_B_assistant.append(
                compute_entropy(token_list[act_idx]["top_logprobs"]) if act_idx is not None and act_idx < n_tok else None
            )

        # User
        cumsum_u = 0
        for turn_idx in range(len(user_turns_list)):
            key = (traj_idx, "user", turn_idx)
            token_list = turn_map.get(key, [])
            n_tok = len(token_list)
            turn_lengths_user.append(n_tok)
            cumsum_u += n_tok
            cumsum_lengths_user.append(cumsum_u)

            if n_tok == 0:
                entropy_A_user.append(0.0)
                entropy_C_user.append(0.0)
                entropy_per_token_user.append([])
                continue

            ent_A = compute_entropy(token_list[0]["top_logprobs"]) if token_list else 0.0
            all_ents = [compute_entropy(t["top_logprobs"]) for t in token_list]
            entropy_C = sum(all_ents) / n_tok if n_tok > 0 else 0.0
            entropy_A_user.append(ent_A)
            entropy_C_user.append(entropy_C)
            entropy_per_token_user.append(all_ents)

        n_turns = max(len(assistant_turns_list), len(user_turns_list))
        relative_positions = (
            [turn_idx / (n_turns - 1) for turn_idx in range(n_turns)] if n_turns > 1 else [0.0]
        )

        output.append({
            "item_id": item_id,
            "success": success,
            "reward": reward,
            "num_assistant_turns": len(assistant_turns_list),
            "num_user_turns": len(user_turns_list),
            "total_tokens": cumsum_lengths_assistant[-1] if cumsum_lengths_assistant else 0,
            "turn_lengths_assistant": turn_lengths_assistant,
            "turn_lengths_user": turn_lengths_user,
            "cumsum_lengths_assistant": cumsum_lengths_assistant,
            "cumsum_lengths_user": cumsum_lengths_user,
            "relative_positions": relative_positions,
            "entropy_A_assistant": entropy_A_assistant,
            "entropy_B_assistant": entropy_B_assistant,
            "entropy_C_assistant": entropy_C_assistant,
            "entropy_A_user": entropy_A_user,
            "entropy_C_user": entropy_C_user,
            "entropy_per_token_assistant": entropy_per_token_assistant,
            "entropy_per_token_user": entropy_per_token_user,
        })

    return output


# ---------- 主流程 ----------
def run_batched_analysis(args):
    global tokenizer

    trajectories = load_trajectories(args.traj_dir, args.max_samples)
    tokenizer = get_tokenizer(args.tokenizer_name)
    model = get_model(args.model_path, args.cuda_device)

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_name = os.path.basename(args.traj_dir).split("_")[0]
    output_file = os.path.join(args.output_dir, f"offline_results_{ts}.jsonl")

    logger.info(f"Extracting all turns from {len(trajectories)} trajectories...")
    all_turns = extract_all_turns(trajectories, max_turns=args.max_turns)
    logger.info(f"Total turns: {len(all_turns)}")

    logger.info("Tokenizing all prompts...")
    tokenized_turns = tokenize_all_turns(all_turns, tokenizer)
    valid_turns = [t for t in tokenized_turns if t.full_ids]
    logger.info(f"Tokenized: {len(valid_turns)} valid turns, "
                f"max length: {max(len(t.full_ids) for t in valid_turns)} tokens, "
                f"median: {int(np.median([len(t.full_ids) for t in valid_turns]))} tokens")

    # 显存检查
    max_len = max(len(t.full_ids) for t in valid_turns)
    approx_mem_gb = (max_len * args.batch_size * 16 * 4) / (1024**3)  # bfloat16
    logger.info(f"Estimated memory per batch: ~{approx_mem_gb:.1f} GB")

    logger.info("Running batched forward...")
    turn_results = forward_batched(valid_turns, model, batch_size=args.batch_size)

    errors = sum(1 for _, e in turn_results.values() if e is not None)
    if errors > 0:
        logger.warning(f"Errors in {errors}/{len(turn_results)} turns")

    logger.info("Aggregating results...")
    results = aggregate_results(trajectories, all_turns, turn_results)

    logger.info(f"Writing {len(results)} results to {output_file}")
    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    logger.info(f"Done! Results: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Batched offline entropy analysis")
    parser.add_argument("--traj_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="/Data/public/Qwen3-1.7B")
    parser.add_argument("--tokenizer_name", type=str, default="/Data/public/Qwen3-1.7B")
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--max_turns", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    run_batched_analysis(args)


if __name__ == "__main__":
    main()
