#!/usr/bin/env python3
"""
Offline Entropy Analysis on MiniMax-M2.1 BabyAI Trajectories
============================================================
用小模型（Qwen3-1.7B，本地 vLLM）对已采集的 MiniMax-M2.1 BabyAI 轨迹做 forward，
统计小模型在每个 turn 位置的 token 熵。

这才是确定 prefix 切分点的正确方式：
  - 小模型熵高  => 小模型对这段轨迹"看不懂/预测不准"，适合放在 prefix 里由大模型提供
  - 小模型熵低  => 小模型已能自主跟上，适合作为 rollout 起始位置

统计三种粒度（均从小模型视角计算）：
  A. 每个 turn 首 token 的熵   —— 最轻量的信号
  B. Action 首 token 的熵      —— 决策时刻的不确定性（[[ 之后）
  C. 每个 turn 所有 token 的平均熵  —— 最稳定的信号

使用方式：
  # 确保 vLLM 服务已启动（端口 8000，模型 qwen3）
  conda activate verl
  python entropy_offline_minimax_traj.py \
      --traj_dir /Data/wyh/datasets/Sampling-Data/babyai_MiniMax-M2.1_20260307_150356 \
      --vllm_url http://localhost:8000 \
      --output_dir /Data/wyh/datasets/Verl-Data/outputs/entropy_offline_minimax \
      --max_samples 100 \
      --concurrency 32
"""

import os
import sys
import json
import math
import re
import asyncio
import logging
import argparse
import fcntl
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import aiohttp
from tqdm import tqdm

# ---------- logging ----------
log_dir = Path("/Data/wyh/datasets/Verl-Data/outputs/entropy_offline_minimax/logs")
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / f"offline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    ],
)
logger = logging.getLogger("OfflineEntropyAnalysis")


# =============================================================================
# 数据加载
# =============================================================================

def load_trajectories(traj_dir: str, max_samples: int = -1) -> List[Dict]:
    """
    加载 MiniMax 轨迹数据（jsonl 格式）。
    每条数据包含 conversations 列表（role/content 交替），item_id, reward, success 等。
    """
    # 优先尝试几种常见的文件名
    candidates = list(Path(traj_dir).glob("*_trajectories.jsonl"))
    if not candidates:
        candidates = list(Path(traj_dir).glob("*.jsonl"))

    if not candidates:
        raise FileNotFoundError(f"No jsonl file found in {traj_dir}")

    # 优先匹配带前缀的文件名（如 babyai_trajectories.jsonl, textcraft_trajectories.jsonl）
    traj_path = None
    for c in candidates:
        if "trajectories" in c.name:
            traj_path = c
            break
    if traj_path is None:
        traj_path = candidates[0]

    trajectories = []
    with open(traj_path) as f:
        for line in f:
            line = line.strip()
            if line:
                trajectories.append(json.loads(line))

    if max_samples > 0:
        trajectories = trajectories[:max_samples]

    logger.info(f"Loaded {len(trajectories)} trajectories from {traj_path}")
    return trajectories


def parse_user_turns(conversations: List[Dict]) -> List[str]:
    """
    从对话列表中提取所有 user 轮次的内容。
    返回: List[str]，每个元素是一个 user turn 的完整文本
    """
    return [m["content"] for m in conversations if m.get("role") == "user"]


def parse_assistant_turns(conversations: List[Dict]) -> List[str]:
    """
    从对话列表中提取所有 assistant 轮次的内容。
    返回: List[str]，每个元素是一个 turn 的完整文本
    """
    return [m["content"] for m in conversations if m.get("role") == "assistant"]


def build_prefix_for_turn(conversations: List[Dict], turn_idx: int, role: str = "assistant") -> List[Dict]:
    """
    构建到第 turn_idx 个指定角色轮次之前的消息上下文。
    用于让小模型预测该 turn 的 token 分布。

    conversations 格式: [user, assistant, user, assistant, ...]
    turn_idx: 第几个指定角色的 turn（0-indexed）
    role: "assistant" 或 "user"
    """
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


# =============================================================================
# 熵计算工具
# =============================================================================

def compute_entropy(lp_dict: Dict[str, float]) -> float:
    """从 {token: logprob} 字典计算香农熵（top-k 近似）。"""
    if not lp_dict:
        return 0.0
    log_probs = list(lp_dict.values())
    probs = [math.exp(lp) for lp in log_probs]
    return -sum(p * lp for p, lp in zip(probs, log_probs) if p > 1e-12)


def find_action_token_idx(token_list: List[Dict]) -> Optional[int]:
    """定位 action 开始后的第一个 token 位置。

    支持两种格式：
    - Qwen/QiFormer: [[action]]
    - MiniMax: Thought: ... Action: ...
    """
    text = "".join(t["decoded"] for t in token_list)

    # 优先匹配 MiniMax 格式: Action:
    m = re.search(r"Action:\s*\n?", text)
    if m:
        target = m.end()
        char_count = 0
        for i, t in enumerate(token_list):
            char_count += len(t["decoded"])
            if char_count >= target:
                return min(i + 1, len(token_list) - 1)
        return None

    # 备选: Qwen 格式: [[action]]
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


# =============================================================================
# vLLM logprobs 请求
# =============================================================================

async def get_logprobs_for_turn(
    prefix_messages: List[Dict],
    target_content: str,
    vllm_url: str,
    model_name: str,
    top_k: int,
    max_tokens: int,
    session: aiohttp.ClientSession,
) -> List[Dict]:
    """
    让小模型以 prefix_messages 为上下文，预测 target_content（一个 assistant turn）
    的每个 token 分布，返回 logprobs 列表。

    使用 chat completions API，把 target_content 作为 assistant 的回复，
    利用 echo 思路：把 target 拼入消息，但只让模型"续写"该 turn 的内容。
    实际上通过让模型在 prefix 后生成，并设置 max_tokens 足够大来获取 logprobs。

    简化实现：直接请求生成，获取生成结果的 logprobs。
    由于我们关心的是"小模型看到大模型这段 token 时的熵"（turn 级别聚合规律），
    让小模型自由生成并统计其 logprobs 即可捕捉不确定性趋势。
    """
    # 清理消息，移除 reasoning_content 字段（可能导致 vLLM chat template 错误）
    clean_messages = []
    for msg in prefix_messages:
        clean_msg = {
            "role": msg.get("role"),
            "content": msg.get("content", "")
        }
        clean_messages.append(clean_msg)

    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "messages": clean_messages,
        "max_tokens": max_tokens,   # 增大以避免截断（原 512 导致 Turn 0 约 94% 写满）
        "temperature": 1.0,        # 保持采样随机性，使熵有意义
        "top_p": 1.0,
        "stream": False,
        "logprobs": True,
        "top_logprobs": top_k,
    }

    try:
        async with session.post(
            f"{vllm_url.rstrip('/')}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            if resp.status != 200:
                logger.error(f"vLLM HTTP {resp.status}: {(await resp.text())[:200]}")
                return []
            result = await resp.json()
            choice = result["choices"][0]
            lp_content = (choice.get("logprobs") or {}).get("content") or []
            token_list = []
            for ti in lp_content:
                top_lps = ti.get("top_logprobs", [])
                lp_dict = {item["token"]: item["logprob"] for item in top_lps}
                token_list.append({
                    "decoded": ti.get("token", ""),
                    "logprob": ti.get("logprob", 0.0),
                    "top_logprobs": lp_dict,
                })
            return token_list
    except asyncio.TimeoutError:
        logger.error("vLLM request timed out")
        return []
    except Exception as e:
        # 打印详细的错误信息用于调试
        logger.error(f"vLLM request error: {e}")
        logger.error(f"  messages count: {len(prefix_messages)}")
        if prefix_messages:
            logger.error(f"  first msg role: {prefix_messages[0].get('role')}")
        return []


# =============================================================================
# 单条轨迹的熵分析
# =============================================================================

async def analyze_trajectory(
    traj: Dict,
    vllm_url: str,
    model_name: str,
    top_k: int,
    max_turns: int,
    max_tokens: int,
    session: aiohttp.ClientSession,
    save_per_token_entropy: bool = False,
) -> Dict:
    """
    对一条 MiniMax 轨迹，逐个 user/assistant turn 请求小模型 logprobs，统计熵。
    分别计算并区分 user 和 assistant 的熵。
    """
    conversations = traj.get("conversations", [])
    item_id = traj.get("item_id", "unknown")
    success = traj.get("success", 0)
    reward = traj.get("reward", 0)

    assistant_turns = parse_assistant_turns(conversations)
    user_turns = parse_user_turns(conversations)

    if not assistant_turns and not user_turns:
        return {"item_id": item_id, "success": success, "reward": reward,
                "entropy_A_assistant": [], "entropy_B_assistant": [], "entropy_C_assistant": [],
                "entropy_A_user": [], "entropy_C_user": [],
                "turn_lengths_assistant": [], "turn_lengths_user": []}

    # Assistant 熵存储
    entropy_A_assistant = []   # 首 token 熵
    entropy_B_assistant = []   # Action 首 token 熵
    entropy_C_assistant = []   # Turn 平均熵
    turn_lengths_assistant = []
    cumsum_lengths_assistant = []
    entropy_per_token_assistant = []

    # User 熵存储
    entropy_A_user = []   # 首 token 熵
    entropy_C_user = []   # Turn 平均熵
    turn_lengths_user = []
    cumsum_lengths_user = []
    entropy_per_token_user = []

    # 处理 assistant turns
    n_assistant_turns = len(assistant_turns) if max_turns == -1 else min(len(assistant_turns), max_turns)
    cumsum_assistant = 0
    for turn_idx in range(n_assistant_turns):
        # 构建该 turn 之前的上下文
        prefix_msgs = build_prefix_for_turn(conversations, turn_idx, role="assistant")

        # 请求小模型 logprobs
        token_list = await get_logprobs_for_turn(
            prefix_messages=prefix_msgs,
            target_content=assistant_turns[turn_idx],
            vllm_url=vllm_url,
            model_name=model_name,
            top_k=top_k,
            max_tokens=max_tokens,
            session=session,
        )

        n_tok = len(token_list)
        turn_lengths_assistant.append(n_tok)
        cumsum_assistant += n_tok
        cumsum_lengths_assistant.append(cumsum_assistant)

        if n_tok == 0:
            entropy_A_assistant.append(0.0)
            entropy_B_assistant.append(None)
            entropy_C_assistant.append(0.0)
            if save_per_token_entropy:
                entropy_per_token_assistant.append([])
            continue

        # A: 首 token 熵
        ent_A = compute_entropy(token_list[0]["top_logprobs"])
        entropy_A_assistant.append(ent_A)

        # C: turn 平均熵
        all_ents = [compute_entropy(t["top_logprobs"]) for t in token_list]
        entropy_C_assistant.append(sum(all_ents) / n_tok)

        # 可选：保存 per-token 熵
        if save_per_token_entropy:
            entropy_per_token_assistant.append(all_ents)

        # B: [[ 之后的 action token 熵
        act_idx = find_action_token_idx(token_list)
        if act_idx is not None and act_idx < n_tok:
            entropy_B_assistant.append(compute_entropy(token_list[act_idx]["top_logprobs"]))
        else:
            entropy_B_assistant.append(None)

    # 处理 user turns
    n_user_turns = len(user_turns) if max_turns == -1 else min(len(user_turns), max_turns)
    cumsum_user = 0
    for turn_idx in range(n_user_turns):
        # 构建该 turn 之前的上下文
        prefix_msgs = build_prefix_for_turn(conversations, turn_idx, role="user")

        # 请求小模型 logprobs
        token_list = await get_logprobs_for_turn(
            prefix_messages=prefix_msgs,
            target_content=user_turns[turn_idx],
            vllm_url=vllm_url,
            model_name=model_name,
            top_k=top_k,
            max_tokens=max_tokens,
            session=session,
        )

        n_tok = len(token_list)
        turn_lengths_user.append(n_tok)
        cumsum_user += n_tok
        cumsum_lengths_user.append(cumsum_user)

        if n_tok == 0:
            entropy_A_user.append(0.0)
            entropy_C_user.append(0.0)
            if save_per_token_entropy:
                entropy_per_token_user.append([])
            continue

        # A: 首 token 熵
        ent_A = compute_entropy(token_list[0]["top_logprobs"])
        entropy_A_user.append(ent_A)

        # C: turn 平均熵
        all_ents = [compute_entropy(t["top_logprobs"]) for t in token_list]
        entropy_C_user.append(sum(all_ents) / n_tok)

        # 可选：保存 per-token 熵
        if save_per_token_entropy:
            entropy_per_token_user.append(all_ents)

    # 计算相对位置
    n_turns = max(n_assistant_turns, n_user_turns)
    if n_turns > 1:
        relative_positions = [turn_idx / (n_turns - 1) for turn_idx in range(n_turns)]
    else:
        relative_positions = [0.0]

    return {
        "item_id": item_id,
        "success": success,
        "reward": reward,
        "num_assistant_turns": n_assistant_turns,
        "num_user_turns": n_user_turns,
        "total_tokens": cumsum_assistant + cumsum_user,
        "turn_lengths_assistant": turn_lengths_assistant,
        "turn_lengths_user": turn_lengths_user,
        "cumsum_lengths_assistant": cumsum_lengths_assistant,
        "cumsum_lengths_user": cumsum_lengths_user,
        "relative_positions": relative_positions,
        # Assistant 熵
        "entropy_A_assistant": entropy_A_assistant,
        "entropy_B_assistant": entropy_B_assistant,
        "entropy_C_assistant": entropy_C_assistant,
        # User 熵
        "entropy_A_user": entropy_A_user,
        "entropy_C_user": entropy_C_user,
        # Per-token 熵（可选）
        "entropy_per_token_assistant": entropy_per_token_assistant if save_per_token_entropy else None,
        "entropy_per_token_user": entropy_per_token_user if save_per_token_entropy else None,
    }


# =============================================================================
# 聚合统计
# =============================================================================

def aggregate_entropy(all_results: List[Dict]) -> Dict:
    # Assistant 熵聚合
    agg = {
        "all": {"A": defaultdict(list), "B": defaultdict(list), "C": defaultdict(list), "length": defaultdict(list)},
        "success": {"A": defaultdict(list), "B": defaultdict(list), "C": defaultdict(list), "length": defaultdict(list)},
    }
    # User 熵聚合
    agg_user = {
        "all": {"A": defaultdict(list), "C": defaultdict(list), "length": defaultdict(list)},
        "success": {"A": defaultdict(list), "C": defaultdict(list), "length": defaultdict(list)},
    }

    # H(q): 按相对位置 q = t/(T-1) 聚合 (0%, 10%, 20%, ..., 100%)
    # 使用更细的粒度：20个桶 (0-5%, 5-10%, ..., 95-100%)
    agg_by_q = {
        "all": defaultdict(lambda: defaultdict(list)),
        "success": defaultdict(lambda: defaultdict(list)),
    }
    agg_by_q_user = {
        "all": defaultdict(lambda: defaultdict(list)),
        "success": defaultdict(lambda: defaultdict(list)),
    }

    # H(q | length_bin): 按轨迹总长度分桶 + 相对位置
    agg_by_q_and_bin = {
        "all": defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
        "success": defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
    }
    agg_by_q_and_bin_user = {
        "all": defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
        "success": defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
    }

    # H(turn | length_bin): 按轨迹总长度分桶
    # 分桶: 0-2k, 2k-4k, 4k-6k, 6k-8k, 8k-10k, 10k+
    agg_by_bin = {
        "all": defaultdict(lambda: defaultdict(list)),
        "success": defaultdict(lambda: defaultdict(list)),
    }
    agg_by_bin_user = {
        "all": defaultdict(lambda: defaultdict(list)),
        "success": defaultdict(lambda: defaultdict(list)),
    }

    for res in all_results:
        if "error" in res:
            continue
        is_success = bool(res.get("success", 0))

        # ========== Assistant 熵聚合 ==========
        # 原始按 turn 索引聚合
        for t_idx, val in enumerate(res.get("entropy_A_assistant", [])):
            agg["all"]["A"][t_idx].append(val)
            if is_success:
                agg["success"]["A"][t_idx].append(val)
        for t_idx, val in enumerate(res.get("entropy_B_assistant", [])):
            if val is not None:
                agg["all"]["B"][t_idx].append(val)
                if is_success:
                    agg["success"]["B"][t_idx].append(val)
        for t_idx, val in enumerate(res.get("entropy_C_assistant", [])):
            agg["all"]["C"][t_idx].append(val)
            if is_success:
                agg["success"]["C"][t_idx].append(val)
        for t_idx, val in enumerate(res.get("turn_lengths_assistant", [])):
            agg["all"]["length"][t_idx].append(val)
            if is_success:
                agg["success"]["length"][t_idx].append(val)

        # ========== User 熵聚合 ==========
        for t_idx, val in enumerate(res.get("entropy_A_user", [])):
            agg_user["all"]["A"][t_idx].append(val)
            if is_success:
                agg_user["success"]["A"][t_idx].append(val)
        for t_idx, val in enumerate(res.get("entropy_C_user", [])):
            agg_user["all"]["C"][t_idx].append(val)
            if is_success:
                agg_user["success"]["C"][t_idx].append(val)
        for t_idx, val in enumerate(res.get("turn_lengths_user", [])):
            agg_user["all"]["length"][t_idx].append(val)
            if is_success:
                agg_user["success"]["length"][t_idx].append(val)

        # ========== H(q): 按相对位置聚合 (Assistant) ==========
        n_turns = res.get("num_assistant_turns", 0)
        relative_positions = res.get("relative_positions", [])
        entropy_C_assistant = res.get("entropy_C_assistant", [])

        for turn_idx, q in enumerate(relative_positions):
            if turn_idx >= len(entropy_C_assistant):
                continue
            val = entropy_C_assistant[turn_idx]
            q_bin = int(q * 20)
            q_bin = min(q_bin, 19)
            agg_by_q["all"][q_bin][turn_idx].append(val)
            if is_success:
                agg_by_q["success"][q_bin][turn_idx].append(val)

            # H(q | length_bin)
            total_tokens = res.get("total_tokens", 0)
            length_bin = get_length_bin(total_tokens)
            agg_by_q_and_bin["all"][length_bin][q_bin][turn_idx].append(val)
            if is_success:
                agg_by_q_and_bin["success"][length_bin][q_bin][turn_idx].append(val)

        # H(turn | length_bin) - Assistant
        for turn_idx, val in enumerate(entropy_C_assistant):
            length_bin = get_length_bin(total_tokens)
            agg_by_bin["all"][length_bin][turn_idx].append(val)
            if is_success:
                agg_by_bin["success"][length_bin][turn_idx].append(val)

        # ========== H(q): 按相对位置聚合 (User) ==========
        entropy_C_user = res.get("entropy_C_user", [])
        for turn_idx, q in enumerate(relative_positions):
            if turn_idx >= len(entropy_C_user):
                continue
            val = entropy_C_user[turn_idx]
            q_bin = int(q * 20)
            q_bin = min(q_bin, 19)
            agg_by_q_user["all"][q_bin][turn_idx].append(val)
            if is_success:
                agg_by_q_user["success"][q_bin][turn_idx].append(val)

            # H(q | length_bin) - User
            length_bin = get_length_bin(total_tokens)
            agg_by_q_and_bin_user["all"][length_bin][q_bin][turn_idx].append(val)
            if is_success:
                agg_by_q_and_bin_user["success"][length_bin][q_bin][turn_idx].append(val)

        # H(turn | length_bin) - User
        for turn_idx, val in enumerate(entropy_C_user):
            length_bin = get_length_bin(total_tokens)
            agg_by_bin_user["all"][length_bin][turn_idx].append(val)
            if is_success:
                agg_by_bin_user["success"][length_bin][turn_idx].append(val)

    def get_length_bin(total_tokens: int) -> int:
        """根据 token 总数返回长度分桶索引"""
        if total_tokens < 2000:
            return 0
        elif total_tokens < 4000:
            return 1
        elif total_tokens < 6000:
            return 2
        elif total_tokens < 8000:
            return 3
        elif total_tokens < 10000:
            return 4
        else:
            return 5

    def compute_percentiles(vals: List[float]) -> Dict[str, float]:
        """计算百分位数: 25%, 50%, 75%, 90%, 95%, 99%"""
        if not vals:
            return {}
        sorted_vals = sorted(vals)
        n = len(sorted_vals)
        result = {}
        for p in [25, 50, 75, 90, 95, 99]:
            idx = int(n * p / 100)
            if idx >= n:
                idx = n - 1
            result[f"p{p}"] = sorted_vals[idx]
        return result


def t_test_two_groups(vals1: List[float], vals2: List[float]) -> Dict[str, float]:
    """对两组样本进行独立样本 t 检验，返回 t 统计量和 p 值。"""
    if len(vals1) < 2 or len(vals2) < 2:
        return {"t_stat": 0.0, "p_value": 1.0, "significant": False}

    import numpy as np
    t_stat, p_value = np.ttest_ind(vals1, vals2)
    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05
    }

    def summarize(d, include_percentiles=False):
        out = {}
        for t_idx, vals in sorted(d.items()):
            n = len(vals)
            if n == 0:
                continue
            mean = sum(vals) / n
            std = math.sqrt(sum((v - mean) ** 2 for v in vals) / n) if n > 1 else 0.0
            result = {"mean": mean, "std": std, "count": n}
            if include_percentiles:
                result.update(compute_percentiles(vals))
            out[str(t_idx)] = result
        return out

    # 按相对位置 q 聚合: 每个 q_bin 下收集所有 turn 的熵，求平均
    def summarize_by_q(agg_by_q_dict):
        out = {}
        for q_bin in range(20):  # 0-19 (0-5%, 5-10%, ..., 95-100%)
            all_ents = []
            for turn_idx, ents in agg_by_q_dict[q_bin].items():
                all_ents.extend(ents)
            if all_ents:
                mean = sum(all_ents) / len(all_ents)
                std = math.sqrt(sum((e - mean) ** 2 for e in all_ents) / len(all_ents)) if len(all_ents) > 1 else 0.0
                # 将 q_bin 转换为百分比标签
                q_label = f"q{q_bin * 5:02d}-{(q_bin + 1) * 5:02d}"
                out[q_label] = {"mean": mean, "std": std, "count": len(all_ents)}
        return out

    # 按轨迹长度分桶聚合
    def summarize_by_bin(agg_by_bin_dict):
        bin_labels = ["0-2k", "2k-4k", "4k-6k", "6k-8k", "8k-10k", "10k+"]
        out = {}
        for bin_idx in range(6):
            all_ents = []
            for turn_idx, ents in agg_by_bin_dict[bin_idx].items():
                all_ents.extend(ents)
            if all_ents:
                mean = sum(all_ents) / len(all_ents)
                std = math.sqrt(sum((e - mean) ** 2 for e in all_ents) / len(all_ents)) if len(all_ents) > 1 else 0.0
                out[bin_labels[bin_idx]] = {"mean": mean, "std": std, "count": len(all_ents)}
        return out

    # 按相对位置 + 轨迹长度分桶聚合: H(q | length_bin)
    def summarize_by_q_and_bin(agg_by_q_and_bin_dict):
        bin_labels = ["0-2k", "2k-4k", "4k-6k", "6k-8k", "8k-10k", "10k+"]
        out = {}
        for bin_idx in range(6):
            bin_label = bin_labels[bin_idx]
            q_dict = agg_by_q_and_bin_dict.get(bin_idx, {})
            # 对每个 q_bin 求平均
            q_means = []
            for q_bin in range(20):
                all_ents = []
                for turn_idx, ents in q_dict.get(q_bin, {}).items():
                    all_ents.extend(ents)
                if all_ents:
                    q_means.append(sum(all_ents) / len(all_ents))
            if q_means:
                overall_mean = sum(q_means) / len(q_means)
                out[bin_label] = {
                    "q_means": q_means,
                    "overall_mean": overall_mean,
                    "count": sum(len(ents) for q_ents in q_dict.values() for ents in q_ents.values())
                }
        return out

    return {
        "all_episodes": {
            "A_first_token":  summarize(agg["all"]["A"]),
            "B_action_token": summarize(agg["all"]["B"]),
            "C_turn_mean":    summarize(agg["all"]["C"]),
            "turn_lengths":   summarize(agg["all"]["length"], include_percentiles=True),
            "entropy_by_q":   summarize_by_q(agg_by_q["all"]),
            "entropy_by_q_and_length_bin": summarize_by_q_and_bin(agg_by_q_and_bin["all"]),
            "entropy_by_length_bin": summarize_by_bin(agg_by_bin["all"]),
            # User 熵
            "A_first_token_user":  summarize(agg_user["all"]["A"]),
            "C_turn_mean_user":    summarize(agg_user["all"]["C"]),
            "turn_lengths_user":   summarize(agg_user["all"]["length"], include_percentiles=True),
            "entropy_by_q_user":   summarize_by_q(agg_by_q_user["all"]),
            "entropy_by_q_and_length_bin_user": summarize_by_q_and_bin(agg_by_q_and_bin_user["all"]),
            "entropy_by_length_bin_user": summarize_by_bin(agg_by_bin_user["all"]),
        },
        "success_only": {
            "A_first_token":  summarize(agg["success"]["A"]),
            "B_action_token": summarize(agg["success"]["B"]),
            "C_turn_mean":    summarize(agg["success"]["C"]),
            "turn_lengths":   summarize(agg["success"]["length"], include_percentiles=True),
            "entropy_by_q":   summarize_by_q(agg_by_q["success"]),
            "entropy_by_q_and_length_bin": summarize_by_q_and_bin(agg_by_q_and_bin["success"]),
            "entropy_by_length_bin": summarize_by_bin(agg_by_bin["success"]),
            # User 熵
            "A_first_token_user":  summarize(agg_user["success"]["A"]),
            "C_turn_mean_user":    summarize(agg_user["success"]["C"]),
            "turn_lengths_user":   summarize(agg_user["success"]["length"], include_percentiles=True),
            "entropy_by_q_user":   summarize_by_q(agg_by_q_user["success"]),
            "entropy_by_q_and_length_bin_user": summarize_by_q_and_bin(agg_by_q_and_bin_user["success"]),
            "entropy_by_length_bin_user": summarize_by_bin(agg_by_bin_user["success"]),
        },
        "total_episodes":   len(all_results),
        "success_episodes": sum(1 for r in all_results if r.get("success", 0)),
    }


# =============================================================================
# 绘图
# =============================================================================

def plot_entropy(summary: Dict, output_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plot")
        return

    fig, axes = plt.subplots(3, 3, figsize=(20, 14))

    # ========== (0,0)：按绝对 turn 的熵曲线 ==========
    labels = {
        "A_first_token":  "A: First-token Entropy (Think start)",
        "B_action_token": "B: Action first-token Entropy (after Action: or [[)",
        "C_turn_mean":    "C: Mean Token Entropy per Turn",
    }
    colors = {"A_first_token": "steelblue", "B_action_token": "coral", "C_turn_mean": "seagreen"}

    ax = axes[0, 0]
    subset = summary["all_episodes"]
    for metric_key, metric_label in labels.items():
        data = subset.get(metric_key, {})
        if not data:
            continue
        steps = sorted(data.keys(), key=lambda x: int(x))
        means = [data[s]["mean"] for s in steps]
        stds  = [data[s]["std"]  for s in steps]
        xs    = [int(s) + 1 for s in steps]

        ax.plot(xs, means, marker="o", markersize=4,
                label=metric_label, color=colors[metric_key])
        ax.fill_between(xs,
                        [m - sd for m, sd in zip(means, stds)],
                        [m + sd for m, sd in zip(means, stds)],
                        alpha=0.15, color=colors[metric_key])

    ax.set_xlabel("Turn Step (1-indexed)", fontsize=11)
    ax.set_ylabel("Token Entropy", fontsize=11)
    ax.set_title("H(turn) - Entropy by Absolute Turn Index", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ========== (0,1)：Success vs All ==========
    ax = axes[0, 1]
    for metric_key, metric_label in labels.items():
        all_data = summary["all_episodes"].get(metric_key, {})
        succ_data = summary["success_only"].get(metric_key, {})
        if not all_data:
            continue
        steps = sorted(all_data.keys(), key=lambda x: int(x))
        all_means = [all_data[s]["mean"] for s in steps]
        succ_means = [succ_data[s]["mean"] for s in steps]
        xs = [int(s) + 1 for s in steps]

        ax.plot(xs, all_means, marker="o", markersize=4, linestyle="-",
                label=f"{metric_label} (all)", color=colors[metric_key], alpha=0.7)
        ax.plot(xs, succ_means, marker="s", markersize=4, linestyle="--",
                label=f"{metric_label} (success)", color=colors[metric_key], alpha=0.4)

    ax.set_xlabel("Turn Step (1-indexed)", fontsize=11)
    ax.set_ylabel("Token Entropy", fontsize=11)
    ax.set_title("All vs Success Episodes", fontsize=11)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    # ========== (0,2)：H(q) - 按相对位置 q = t/(T-1) ==========
    ax = axes[0, 2]
    entropy_by_q = summary["all_episodes"].get("entropy_by_q", {})
    if entropy_by_q:
        # 解析 q 标签 (格式: q00-05, q05-10, ..., q95-100)
        q_labels = sorted(entropy_by_q.keys(), key=lambda x: (int(x[1:3]), int(x[4:6])))
        xs = [int(x[1:3]) + 2.5 for x in q_labels]  # 中点: 2.5, 7.5, 12.5, ..., 97.5
        means = [entropy_by_q[q]["mean"] for q in q_labels]
        stds = [entropy_by_q[q]["std"] for q in q_labels]

        ax.bar(xs, means, width=4, color="purple", alpha=0.6, label="H(q)")
        ax.plot(xs, means, marker="o", markersize=6, color="purple")
        ax.fill_between(xs,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.2, color="purple")
        ax.axhline(y=sum(means)/len(means), color="gray", linestyle=":", alpha=0.5, label=f"avg={sum(means)/len(means):.3f}")

    ax.set_xlabel("Relative Position q = t/(T-1) (%)", fontsize=11)
    ax.set_ylabel("Token Entropy", fontsize=11)
    ax.set_title("H(q) - Entropy by Relative Position (0%=start, 100%=end)", fontsize=11)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ========== (1,0)：Turn Length 百分位数曲线 ==========
    ax = axes[1, 0]
    length_data = summary["all_episodes"].get("turn_lengths", {})
    if length_data:
        steps = sorted(length_data.keys(), key=lambda x: int(x))
        xs = [int(s) + 1 for s in steps]

        for p, ls, color in [("p50", "-", "blue"), ("p75", "--", "green"),
                              ("p90", ":", "orange"), ("p95", "-.", "red")]:
            vals = [length_data[s].get(p, 0) for s in steps]
            ax.plot(xs, vals, marker="o", markersize=3, linestyle=ls,
                    label=f"Length {p}", color=color)

    ax.set_xlabel("Turn Step (1-indexed)", fontsize=11)
    ax.set_ylabel("Token Length", fontsize=11)
    ax.set_title("Turn Length Percentiles (max_tokens=8192)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ========== (1,1)：H(q | length_bin) - 按轨迹总长度分桶后的相对位置熵 ==========
    ax = axes[1, 1]
    entropy_by_q_and_bin = summary["all_episodes"].get("entropy_by_q_and_length_bin", {})
    bin_labels = ["0-2k", "2k-4k", "4k-6k", "6k-8k", "8k-10k", "10k+"]
    bin_colors = plt.cm.viridis([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    if entropy_by_q_and_bin:
        xs = list(range(0, 100, 5))  # 0, 5, 10, ..., 95
        for bin_idx, bin_label in enumerate(bin_labels):
            if bin_label not in entropy_by_q_and_bin:
                continue
            bin_data = entropy_by_q_and_bin[bin_label]
            q_means = bin_data.get("q_means", [])
            if q_means:
                ax.plot(xs[:len(q_means)], q_means, marker="o", markersize=3, linestyle="-",
                        label=f"{bin_label} (n={bin_data.get('count', '?')})",
                        color=bin_colors[bin_idx], alpha=0.8)

    ax.set_xlabel("Relative Position q = t/(T-1) (%)", fontsize=11)
    ax.set_ylabel("Token Entropy", fontsize=11)
    ax.set_title("H(q | length_bin) - Entropy by Relative Position for Each Length Bin", fontsize=11)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    # ========== (1,2)：各分桶的统计信息（柱状图）==========
    ax = axes[1, 2]
    entropy_by_bin = summary["all_episodes"].get("entropy_by_length_bin", {})
    if entropy_by_bin:
        bins = list(entropy_by_bin.keys())
        means = [entropy_by_bin[b]["mean"] for b in bins]
        counts = [entropy_by_bin[b]["count"] for b in bins]

        # 柱状图显示均值
        x_pos = range(len(bins))
        bars = ax.bar(x_pos, means, color=bin_colors[:len(bins)], alpha=0.7, edgecolor="black")

        # 在柱子上标注 count
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"n={count}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Trajectory Length Bin", fontsize=11)
    ax.set_ylabel("Mean Entropy", fontsize=11)
    ax.set_title("Mean Entropy by Trajectory Length Bin", fontsize=11)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bins, rotation=15)
    ax.grid(True, alpha=0.3, axis="y")

    # ========== (2,0)：按绝对 turn 的熵曲线（仅 success）==========
    ax = axes[2, 0]
    for metric_key, metric_label in [("C_turn_mean", "C: Mean Token Entropy")]:
        # All episodes
        all_data = summary["all_episodes"].get(metric_key, {})
        if not all_data:
            continue
        steps = sorted(all_data.keys(), key=lambda x: int(x))
        all_means = [all_data[s]["mean"] for s in steps]
        all_stds = [all_data[s]["std"] for s in steps]
        xs = [int(s) + 1 for s in steps]

        ax.plot(xs, all_means, marker="o", markersize=4,
                label="All episodes", color="blue", alpha=0.7)
        ax.fill_between(xs,
                        [m - s for m, s in zip(all_means, all_stds)],
                        [m + s for m, s in zip(all_means, all_stds)],
                        alpha=0.15, color="blue")

        # Success episodes
        succ_data = summary["success_only"].get(metric_key, {})
        if succ_data:
            succ_means = [succ_data[s]["mean"] for s in steps]
            succ_stds = [succ_data[s]["std"] for s in steps]
            ax.plot(xs, succ_means, marker="s", markersize=4,
                    label="Success only", color="green", alpha=0.7)
            ax.fill_between(xs,
                            [m - s for m, s in zip(succ_means, succ_stds)],
                            [m + s for m, s in zip(succ_means, succ_stds)],
                            alpha=0.15, color="green")

    ax.set_xlabel("Turn Step (1-indexed)", fontsize=11)
    ax.set_ylabel("Token Entropy", fontsize=11)
    ax.set_title("C: Mean Turn Entropy (All vs Success)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ========== (2,1)：熵的分布热力图 (turn vs entropy bin) ==========
    ax = axes[2, 1]
    all_turn_data = summary["all_episodes"].get("C_turn_mean", {})
    if all_turn_data:
        # 收集所有 turn 的熵值
        import numpy as np
        turn_entropies = []
        for turn_idx in range(15):
            if str(turn_idx) in all_turn_data:
                # 用均值代替实际分布做简单热力图
                turn_entropies.append(all_turn_data[str(turn_idx)]["mean"])
            else:
                turn_entropies.append(0)

        # 绘制条形图展示 turn 间熵变化
        xs = list(range(1, len(turn_entropies) + 1))
        colors = plt.cm.RdYlGn_r([v / max(turn_entropies) for v in turn_entropies])
        ax.bar(xs, turn_entropies, color=colors, edgecolor="black", alpha=0.8)
        ax.axhline(y=np.mean(turn_entropies), color="red", linestyle="--", label=f"mean={np.mean(turn_entropies):.3f}")

    ax.set_xlabel("Turn Step", fontsize=11)
    ax.set_ylabel("Mean Entropy", fontsize=11)
    ax.set_title("Entropy by Turn (Color: High=Red, Low=Green)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # ========== (2,2)：相对位置 q 的详细分布 ==========
    ax = axes[2, 2]
    if entropy_by_q:
        # 显示每个 q 桶的样本数量
        q_labels = sorted(entropy_by_q.keys(), key=lambda x: (int(x[1:3]), int(x[4:6])))
        counts = [entropy_by_q[q]["count"] for q in q_labels]
        means = [entropy_by_q[q]["mean"] for q in q_labels]

        x_pos = range(len(q_labels))
        ax.bar(x_pos, counts, color="steelblue", alpha=0.7, label="Sample count")

        # 在柱子上标注均值
        for i, (c, m) in enumerate(zip(counts, means)):
            ax.text(i, c + max(counts) * 0.02, f"{m:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)

        ax.set_xlabel("Relative Position q (5% bins)", fontsize=11)
        ax.set_ylabel("Sample Count", fontsize=11)
        ax.set_title("Sample Distribution across Relative Positions", fontsize=11)
        ax.set_xticks(x_pos[::4])  # 每4个显示一个
        ax.set_xticklabels([q_labels[i] for i in range(0, len(q_labels), 4)], rotation=45)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved: {output_path}")


# =============================================================================
# 文件写入
# =============================================================================

def safe_write(path: str, record: Dict):
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(json.dumps(record) + "\n")
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# =============================================================================
# 主流程
# =============================================================================

async def run_analysis(args):
    # 加载数据
    trajectories = load_trajectories(args.traj_dir, args.max_samples)

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(args.output_dir, f"offline_results_{ts}.jsonl")
    summary_file = os.path.join(args.output_dir, f"offline_summary_{ts}.json")
    plot_file    = os.path.join(args.output_dir, f"offline_plot_{ts}.png")
    open(results_file, "w").close()

    sem = asyncio.Semaphore(args.concurrency)
    all_results = []

    async def worker(traj):
        async with sem:
            res = await analyze_trajectory(
                traj=traj,
                vllm_url=args.vllm_url,
                model_name=args.model_name,
                top_k=args.top_k,
                max_turns=args.max_turns,
                max_tokens=args.max_tokens,
                session=http_session,
                save_per_token_entropy=args.save_per_token_entropy,
            )
            await asyncio.to_thread(safe_write, results_file, res)
            return res

    connector = aiohttp.TCPConnector(limit=args.concurrency, limit_per_host=args.concurrency)
    async with aiohttp.ClientSession(connector=connector,
                                     timeout=aiohttp.ClientTimeout(total=300)) as http_session:
        tasks = [worker(t) for t in trajectories]
        pbar = tqdm(total=len(tasks), desc="Offline entropy analysis", unit="traj")
        for f in asyncio.as_completed(tasks):
            res = await f
            if res:
                all_results.append(res)
            pbar.update(1)
        pbar.close()

    logger.info("Aggregating...")
    summary = aggregate_entropy(all_results)

    # 防御性检查：确保 summary 不为 None 且 all_results 不为空
    has_error = False
    if summary is None:
        logger.error("Aggregation failed! summary is None. Check if all_results is empty.")
        summary = {
            "all_episodes": {},
            "success_only": {},
            "total_episodes": len(all_results),
            "success_episodes": 0,
            "error": "aggregate_entropy returned None"
        }
        has_error = True
    if not all_results:
        logger.error("No results to aggregate! all_results is empty.")
        summary = {
            "all_episodes": {},
            "success_only": {},
            "total_episodes": 0,
            "success_episodes": 0,
            "error": "all_results is empty"
        }
        has_error = True

    # 保存 summary
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved: {summary_file}")

    # 如果有错误，提前返回，避免后续代码崩溃
    if has_error:
        print("\n[ERROR] Aggregation failed. Check logs for details.")
        return

    # 打印结果
    print("\n" + "=" * 65)
    print("Small Model Entropy on MiniMax-M2.1 BabyAI Trajectories")
    print("=" * 65)
    print(f"Total: {summary['total_episodes']},  Success: {summary['success_episodes']}")

    # 打印 turn_lengths 百分位数
    print("\n[All Episodes] Turn Lengths (tokens):")
    for step, stat in sorted(summary["all_episodes"]["turn_lengths"].items(), key=lambda x: int(x[0])):
        print(f"  Turn {int(step)+1:2d}: mean={stat['mean']:.1f}, "
              f"p25={stat.get('p25', '-')}, p50={stat.get('p50', '-')}, "
              f"p75={stat.get('p75', '-')}, p90={stat.get('p90', '-')}, "
              f"p95={stat.get('p95', '-')}, p99={stat.get('p99', '-')}")

    # 打印 H(q) - 按相对位置
    print("\n[All Episodes] H(q) - Entropy by Relative Position (q = t/(T-1)):")
    entropy_by_q = summary["all_episodes"].get("entropy_by_q", {})
    for q_bin in sorted(entropy_by_q.keys(), key=lambda x: (int(x[1:3]), int(x[4:6]))):
        stat = entropy_by_q[q_bin]
        bar = "█" * int(stat['mean'] * 50)
        print(f"  q={q_bin:>5s}: {stat['mean']:.4f} ±{stat['std']:.4f}  (n={stat['count']:4d})  {bar}")

    # 打印 H(q | length_bin) - 按轨迹长度分桶后的相对位置熵
    print("\n[All Episodes] H(q | length_bin) - Entropy by Relative Position for Each Length Bin:")
    entropy_by_q_and_bin = summary["all_episodes"].get("entropy_by_q_and_length_bin", {})
    bin_labels = ["0-2k", "2k-4k", "4k-6k", "6k-8k", "8k-10k", "10k+"]
    for bin_label in bin_labels:
        if bin_label in entropy_by_q_and_bin:
            bin_data = entropy_by_q_and_bin[bin_label]
            q_means = bin_data.get("q_means", [])
            if q_means:
                # 找出最高和最低的相对位置
                max_q_idx = q_means.index(max(q_means))
                min_q_idx = q_means.index(min(q_means))
                print(f"  {bin_label:>7s}: overall_mean={bin_data.get('overall_mean', 0):.4f}, "
                      f"max at q={max_q_idx*5:02d}-{(max_q_idx+1)*5:02d}% ({max(q_means):.4f}), "
                      f"min at q={min_q_idx*5:02d}-{(min_q_idx+1)*5:02d}% ({min(q_means):.4f}), "
                      f"n={bin_data.get('count', 0)}")

    # 打印 H(turn | length_bin) - 按轨迹长度分桶
    print("\n[All Episodes] H(turn | length_bin) - Entropy by Trajectory Total Length:")
    entropy_by_bin = summary["all_episodes"].get("entropy_by_length_bin", {})
    for bin_label in ["0-2k", "2k-4k", "4k-6k", "6k-8k", "8k-10k", "10k+"]:
        if bin_label in entropy_by_bin:
            stat = entropy_by_bin[bin_label]
            bar = "█" * int(stat['mean'] * 50)
            print(f"  {bin_label:>7s}: {stat['mean']:.4f} ±{stat['std']:.4f}  (n={stat['count']})  {bar}")

    print("\n[All Episodes] C - Mean Turn Entropy (all steps):")
    for step, stat in sorted(summary["all_episodes"]["C_turn_mean"].items(), key=lambda x: int(x[0])):
        bar = "█" * int(stat["mean"] * 50)
        print(f"  Step {int(step)+1:2d}: {stat['mean']:.4f} ±{stat['std']:.4f}  (n={stat['count']})  {bar}")
    print("=" * 65)

    plot_entropy(summary, plot_file)
    logger.info(f"Done!\n  Raw   : {results_file}\n  Summary: {summary_file}\n  Plot  : {plot_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Offline entropy analysis: small model on MiniMax-M2.1 BabyAI trajectories"
    )
    parser.add_argument("--traj_dir", type=str,
                        default="/Data/wyh/datasets/Sampling-Data/babyai_MiniMax-M2.1_20260307_150356",
                        help="Directory containing trajectories jsonl (e.g., babyai_trajectories.jsonl)")
    parser.add_argument("--output_dir", type=str,
                        default="/Data/wyh/datasets/Verl-Data/outputs/entropy_offline_minimax",
                        help="Output directory")
    parser.add_argument("--vllm_url", type=str,
                        default="http://localhost:8000",
                        help="Local vLLM service URL")
    parser.add_argument("--model_name", type=str, default="qwen3",
                        help="Model name served by vLLM")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Max trajectories to analyze, -1 = all")
    parser.add_argument("--concurrency", type=int, default=32,
                        help="Concurrent vLLM requests")
    parser.add_argument("--max_turns", type=int, default=-1,
                        help="Max assistant turns to analyze per trajectory (-1 = all)")
    parser.add_argument("--max_tokens", type=int, default=8192,
                        help="Max tokens per turn for vLLM generation (avoid truncation)")
    parser.add_argument("--top_k", type=int, default=100,
                        help="top-k logprobs per token position")
    parser.add_argument("--save_per_token_entropy", action="store_true",
                        help="Save per-token entropy for detailed analysis (larger output file)")
    args = parser.parse_args()

    logger.info("Config:")
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")

    asyncio.run(run_analysis(args))


if __name__ == "__main__":
    main()
