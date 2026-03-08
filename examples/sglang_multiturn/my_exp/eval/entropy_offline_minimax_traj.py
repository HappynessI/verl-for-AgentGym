#!/usr/bin/env python3
"""
Offline Entropy Analysis on MiniMax Trajectories
==================================================
用小模型（Qwen3-1.7B，本地 vLLM）对已采集的 MiniMax 轨迹做 forward，
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
      --traj_dir /Data/wyh/datasets/Sampling-Data/textcraft_MiniMax-M2.1_20260307_150412 \
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
    traj_path = Path(traj_dir) / "textcraft_trajectories.jsonl"
    if not traj_path.exists():
        # 尝试找 jsonl 文件
        candidates = list(Path(traj_dir).glob("*.jsonl"))
        if not candidates:
            raise FileNotFoundError(f"No jsonl file found in {traj_dir}")
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


def parse_assistant_turns(conversations: List[Dict]) -> List[str]:
    """
    从对话列表中提取所有 assistant 轮次的内容。
    返回: List[str]，每个元素是一个 turn 的完整文本
    """
    return [m["content"] for m in conversations if m.get("role") == "assistant"]


def build_prefix_for_turn(conversations: List[Dict], turn_idx: int) -> List[Dict]:
    """
    构建到第 turn_idx 个 assistant turn 之前的消息上下文。
    用于让小模型预测该 turn 的 token 分布。

    conversations 格式: [user, assistant, user, assistant, ...]
    turn_idx: 第几个 assistant turn（0-indexed）
    """
    prefix_messages = []
    assistant_count = 0
    for msg in conversations:
        if msg.get("role") == "assistant":
            if assistant_count == turn_idx:
                break
            assistant_count += 1
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
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "messages": prefix_messages,
        "max_tokens": 512,       # 统计前 512 token 的熵即可
        "temperature": 1.0,      # 保持采样随机性，使熵有意义
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
            timeout=aiohttp.ClientTimeout(total=60),
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
        logger.error(f"vLLM request error: {e}")
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
    session: aiohttp.ClientSession,
) -> Dict:
    """
    对一条 MiniMax 轨迹，逐个 assistant turn 请求小模型 logprobs，统计熵。
    """
    conversations = traj.get("conversations", [])
    item_id = traj.get("item_id", "unknown")
    success = traj.get("success", 0)
    reward = traj.get("reward", 0)

    assistant_turns = parse_assistant_turns(conversations)
    if not assistant_turns:
        return {"item_id": item_id, "success": success, "reward": reward,
                "entropy_A": [], "entropy_B": [], "entropy_C": [], "turn_lengths": []}

    entropy_A = []   # 首 token 熵
    entropy_B = []   # Action 首 token 熵
    entropy_C = []   # Turn 平均熵
    turn_lengths = []

    n_turns = min(len(assistant_turns), max_turns)
    for turn_idx in range(n_turns):
        # 构建该 turn 之前的上下文
        prefix_msgs = build_prefix_for_turn(conversations, turn_idx)

        # 请求小模型 logprobs
        token_list = await get_logprobs_for_turn(
            prefix_messages=prefix_msgs,
            target_content=assistant_turns[turn_idx],
            vllm_url=vllm_url,
            model_name=model_name,
            top_k=top_k,
            session=session,
        )

        n_tok = len(token_list)
        turn_lengths.append(n_tok)

        if n_tok == 0:
            entropy_A.append(0.0)
            entropy_B.append(None)
            entropy_C.append(0.0)
            continue

        # A: 首 token 熵
        ent_A = compute_entropy(token_list[0]["top_logprobs"])
        entropy_A.append(ent_A)

        # C: turn 平均熵
        all_ents = [compute_entropy(t["top_logprobs"]) for t in token_list]
        entropy_C.append(sum(all_ents) / n_tok)

        # B: [[ 之后的 action token 熵
        act_idx = find_action_token_idx(token_list)
        if act_idx is not None and act_idx < n_tok:
            entropy_B.append(compute_entropy(token_list[act_idx]["top_logprobs"]))
        else:
            entropy_B.append(None)

    return {
        "item_id": item_id,
        "success": success,
        "reward": reward,
        "num_turns": n_turns,
        "turn_lengths": turn_lengths,
        "entropy_A": entropy_A,
        "entropy_B": entropy_B,
        "entropy_C": entropy_C,
    }


# =============================================================================
# 聚合统计
# =============================================================================

def aggregate_entropy(all_results: List[Dict]) -> Dict:
    agg = {
        "all": {"A": defaultdict(list), "B": defaultdict(list), "C": defaultdict(list)},
        "success": {"A": defaultdict(list), "B": defaultdict(list), "C": defaultdict(list)},
    }

    for res in all_results:
        if "error" in res:
            continue
        is_success = bool(res.get("success", 0))
        for t_idx, val in enumerate(res["entropy_A"]):
            agg["all"]["A"][t_idx].append(val)
            if is_success:
                agg["success"]["A"][t_idx].append(val)
        for t_idx, val in enumerate(res["entropy_B"]):
            if val is not None:
                agg["all"]["B"][t_idx].append(val)
                if is_success:
                    agg["success"]["B"][t_idx].append(val)
        for t_idx, val in enumerate(res["entropy_C"]):
            agg["all"]["C"][t_idx].append(val)
            if is_success:
                agg["success"]["C"][t_idx].append(val)

    def summarize(d):
        out = {}
        for t_idx, vals in sorted(d.items()):
            n = len(vals)
            if n == 0:
                continue
            mean = sum(vals) / n
            std = math.sqrt(sum((v - mean) ** 2 for v in vals) / n) if n > 1 else 0.0
            out[str(t_idx)] = {"mean": mean, "std": std, "count": n}
        return out

    return {
        "all_episodes": {
            "A_first_token":  summarize(agg["all"]["A"]),
            "B_action_token": summarize(agg["all"]["B"]),
            "C_turn_mean":    summarize(agg["all"]["C"]),
        },
        "success_only": {
            "A_first_token":  summarize(agg["success"]["A"]),
            "B_action_token": summarize(agg["success"]["B"]),
            "C_turn_mean":    summarize(agg["success"]["C"]),
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

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    labels = {
        "A_first_token":  "A: First-token Entropy (Think start)",
        "B_action_token": "B: Action first-token Entropy (after [[)",
        "C_turn_mean":    "C: Mean Token Entropy per Turn",
    }
    colors = {"A_first_token": "steelblue", "B_action_token": "coral", "C_turn_mean": "seagreen"}

    for ax_idx, (subset_key, subset_label) in enumerate([
        ("all_episodes", "All Episodes"),
        ("success_only",  "Success Episodes Only"),
    ]):
        ax = axes[ax_idx]
        subset = summary[subset_key]
        for metric_key, metric_label in labels.items():
            data = subset[metric_key]
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

        ax.set_xlabel("Turn Step (1-indexed)", fontsize=12)
        ax.set_ylabel("Token Entropy (Qwen3-1.7B on MiniMax trajectories)", fontsize=10)
        ax.set_title(
            f"Small Model Entropy on MiniMax-M2.1 Trajectories\n"
            f"({subset_label}, total={summary['total_episodes']}, "
            f"success={summary['success_episodes']})",
            fontsize=11,
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

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
                session=http_session,
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

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved: {summary_file}")

    # 打印结果
    print("\n" + "=" * 65)
    print("Small Model Entropy on MiniMax-M2.1 Trajectories")
    print("=" * 65)
    print(f"Total: {summary['total_episodes']},  Success: {summary['success_episodes']}")
    print("\n[All Episodes] C - Mean Turn Entropy (all steps):")
    for step, stat in sorted(summary["all_episodes"]["C_turn_mean"].items(), key=lambda x: int(x[0])):
        bar = "█" * int(stat["mean"] * 50)
        print(f"  Step {int(step)+1:2d}: {stat['mean']:.4f} ±{stat['std']:.4f}  (n={stat['count']})  {bar}")
    print("=" * 65)

    plot_entropy(summary, plot_file)
    logger.info(f"Done!\n  Raw   : {results_file}\n  Summary: {summary_file}\n  Plot  : {plot_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Offline entropy analysis: small model on MiniMax trajectories"
    )
    parser.add_argument("--traj_dir", type=str,
                        default="/Data/wyh/datasets/Sampling-Data/textcraft_MiniMax-M2.1_20260307_150412",
                        help="Directory containing textcraft_trajectories.jsonl")
    parser.add_argument("--output_dir", type=str,
                        default="/Data/wyh/datasets/Verl-Data/outputs/entropy_offline_minimax")
    parser.add_argument("--vllm_url", type=str,
                        default="http://localhost:8000",
                        help="Local vLLM service URL")
    parser.add_argument("--model_name", type=str, default="qwen3",
                        help="Model name served by vLLM")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Max trajectories to analyze, -1 = all")
    parser.add_argument("--concurrency", type=int, default=32,
                        help="Concurrent vLLM requests")
    parser.add_argument("--max_turns", type=int, default=20,
                        help="Max assistant turns to analyze per trajectory")
    parser.add_argument("--top_k", type=int, default=20,
                        help="top-k logprobs per token position")
    args = parser.parse_args()

    logger.info("Config:")
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")

    asyncio.run(run_analysis(args))


if __name__ == "__main__":
    main()
