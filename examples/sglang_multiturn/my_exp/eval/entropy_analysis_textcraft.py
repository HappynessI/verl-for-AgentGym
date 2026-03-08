#!/usr/bin/env python3
"""
TextCraft Token Entropy Analysis Script  —— MiniMax-M2.1 版
=============================================================
让大模型（MiniMax-M2.1）与 TextCraft 环境交互，统计每个 step 的 token 熵，
用于分析蒸馏轨迹数据的切分位点。

统计三种粒度：
  A. Turn 首 token 熵      —— 模型刚开始生成时的不确定性（Thought 第一个 token）
  B. Action 首 token 熵    —— [[ 之后第一个 token 的不确定性（决策时刻）
  C. Turn 内平均 token 熵  —— 整个 assistant 回复的平均熵

输出：
  - entropy_results_minimax_*.jsonl   每个 episode 的原始数据
  - entropy_summary_minimax_*.json    按 turn_index 聚合的统计结果（均值/std）
  - entropy_plot_minimax_*.png        三种粒度折线图

用法：
  # 启动 textcraft 环境服务器（端口 36001）后运行：
  python entropy_analysis_textcraft.py \
      --api_key sk-xxx \
      --textcraft_server http://127.0.0.1:36001 \
      --max_samples 10 \
      --concurrency 4 \
      --output_dir /Data/wyh/datasets/entropy_analysis_minimax

注意：MiniMax 外部 API 有限速，并发建议 4~8。
"""

import os
import sys
import json
import math
import logging
import argparse
import uuid
import asyncio
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

import pyarrow.parquet as pq
from tqdm import tqdm
import aiohttp
import fcntl

# ---------- path setup ----------
project_root = Path(__file__).parent.parent.parent.parent.parent
if not (project_root / "verl").exists():
    raise RuntimeError(f"verl not found in {project_root}")
sys.path.insert(0, str(project_root))

from verl.interactions.textcraft_interaction import TextCraftInteraction

# ---------- logging ----------
log_dir = Path("/Data/wyh/datasets/Verl-Data/outputs/entropy_analysis/logs")
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / f"entropy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger("EntropyAnalysis")


# =============================================================================
# 熵计算工具函数
# =============================================================================

def compute_entropy_from_logprobs(logprob_dict: Dict) -> float:
    """
    从 logprobs 字典计算香农熵。
    字典格式: {token_str: logprob_float}  （已在 generate_with_logprobs 中转换好）
    用 top-k logprobs 做近似估计（下界）。
    """
    if not logprob_dict:
        return 0.0
    # 值直接是 float（logprob），不是对象
    log_probs = list(logprob_dict.values())
    probs = [math.exp(lp) for lp in log_probs]
    entropy = -sum(p * lp for p, lp in zip(probs, log_probs) if p > 1e-12)
    return entropy


def find_action_token_index(token_logprobs: List[Dict], decoded_tokens: List[str]) -> Optional[int]:
    """
    找到 [[ 之后第一个有意义 token 的位置。
    decoded_tokens: 每个 token 的解码字符串列表。
    返回该 token 在列表中的 index，找不到则返回 None。
    """
    full_text = "".join(decoded_tokens)
    action_match = re.search(r'\[\[', full_text)
    if not action_match:
        return None

    # 找到 [[ 结束后的字符位置
    target_char_pos = action_match.end()
    # 定位到对应 token index
    char_count = 0
    for i, tok in enumerate(decoded_tokens):
        char_count += len(tok)
        if char_count >= target_char_pos:
            # 返回 [[ 之后的下一个 token（如果有）
            return min(i + 1, len(decoded_tokens) - 1)
    return None


# =============================================================================
# Agent —— MiniMax 外部 API（带 logprobs 的生成）
# =============================================================================

class EntropyTrackingAgent:
    """
    调用 MiniMax-M2.1 外部 API，生成时请求 logprobs 用于熵统计。

    MiniMax API 兼容 OpenAI 格式，端点：
        POST https://api.minimaxi.com/v1/chat/completions
    鉴权：Header  Authorization: Bearer {api_key}

    注意：MiniMax 是否支持 logprobs 视版本而定。
    若返回体中没有 logprobs 字段，token_logprobs_list 为空列表，
    熵值将全部记为 0，不会报错。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.minimaxi.com/v1",
        model: str = "MiniMax-M2.1",
        max_new_tokens: int = 16384,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_logprobs: int = 20,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_logprobs = top_logprobs
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        # 与 api_eval.py react 格式保持一致
        return (
            "You are an agent in the TextCraft environment. "
            "Your goal is to craft items by gathering resources and following recipes.\n\n"
            "At each step, you must output:\n"
            "  Thought: <your reasoning>\n"
            "  Action: [[ <command> ]]\n\n"
            "Available commands:\n"
            "  get <item>                         - gather a resource\n"
            "  craft <target> using <ingredients> - craft an item\n"
            "  inventory                          - check your inventory\n\n"
            "Rules:\n"
            "1. Output exactly ONE action per turn, wrapped in [[ ]].\n"
            "2. Do NOT simulate the environment response.\n"
            "3. Stop after outputting the Action line."
        )

    async def generate_with_logprobs(
        self,
        messages: List[Dict[str, str]],
        session: aiohttp.ClientSession,
    ) -> Tuple[str, List[Dict]]:
        """
        调用 MiniMax API 生成回复，同时解析 logprobs。
        返回: (response_text, token_logprobs_list)
          token_logprobs_list[i] = {
              "decoded": str,          # 该 token 的字符串
              "logprob": float,        # 该 token 自身的 log prob
              "top_logprobs": {token: logprob, ...}  # top-k 候选
          }
        若 API 不返回 logprobs，则 token_logprobs_list = []。
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                *messages,
            ],
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": False,
            # 请求 logprobs（OpenAI 兼容格式）
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
        }

        try:
            async with session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=180),  # 外部 API 超时更长
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"MiniMax API HTTP {response.status}: {error_text[:300]}")
                    return "", []

                result = await response.json()
                if "choices" not in result or not result["choices"]:
                    logger.error(f"Unexpected response format: {str(result)[:200]}")
                    return "", []

                choice = result["choices"][0]
                text = (choice.get("message") or {}).get("content", "").strip()

                # 解析 logprobs（若 MiniMax 支持则有值，否则为空）
                token_logprobs_list = []
                lp_content = (choice.get("logprobs") or {}).get("content") or []
                for token_info in lp_content:
                    top_lps = token_info.get("top_logprobs", [])
                    lp_dict = {item["token"]: item["logprob"] for item in top_lps}
                    token_logprobs_list.append({
                        "decoded": token_info.get("token", ""),
                        "logprob": token_info.get("logprob", 0.0),
                        "top_logprobs": lp_dict,
                    })

                return text, token_logprobs_list

        except asyncio.TimeoutError:
            logger.error("MiniMax API request timed out (180s)")
            return "", []
        except Exception as e:
            logger.error(f"MiniMax API request failed: {e}")
            return "", []


# =============================================================================
# 单 episode 的熵分析
# =============================================================================

async def analyze_one_episode(
    agent: EntropyTrackingAgent,
    interaction: TextCraftInteraction,
    session_id: int,
    http_session: aiohttp.ClientSession,
    max_rounds: int = 40,
) -> Dict[str, Any]:
    """
    运行一个完整 episode，收集每个 turn 的三种粒度熵值。
    """
    instance_id = f"entropy_{session_id}_{uuid.uuid4().hex[:8]}"
    messages = []
    done = False
    total_reward = 0.0

    # 三种粒度的熵序列（按 turn_index 记录）
    entropy_first_token = []   # A: 每个 turn 第一个生成 token 的熵
    entropy_action_token = []  # B: [[ 之后第一个 token 的熵（找不到则记 None）
    entropy_turn_mean = []     # C: 整个 turn 所有 token 的平均熵
    turn_lengths = []          # 每个 turn 的 token 数（用于分析）

    try:
        await interaction.start_interaction(instance_id, session_id=session_id)
        done, initial_obs, reward, _ = await interaction.generate_response(instance_id, messages)
        total_reward += reward
        messages.append({"role": "user", "content": initial_obs})
    except Exception as e:
        logger.error(f"Start failed for session {session_id}: {e}")
        return {"session_id": session_id, "error": str(e), "entropy_A": [], "entropy_B": [], "entropy_C": []}

    for turn_idx in range(max_rounds):
        if done:
            break

        # 生成（带 logprobs）
        try:
            response, token_lp_list = await agent.generate_with_logprobs(messages, http_session)
        except Exception as e:
            logger.error(f"Generation error session {session_id} turn {turn_idx}: {e}")
            break

        if not response:
            break

        # ---------- 熵计算 ----------
        num_tokens = len(token_lp_list)
        turn_lengths.append(num_tokens)

        # A: 第一个 token 的熵
        if num_tokens > 0:
            ent_A = compute_entropy_from_logprobs(token_lp_list[0]["top_logprobs"])
        else:
            ent_A = 0.0
        entropy_first_token.append(ent_A)

        # C: 所有 token 的平均熵
        if num_tokens > 0:
            all_entropies = [
                compute_entropy_from_logprobs(tlp["top_logprobs"])
                for tlp in token_lp_list
            ]
            ent_C = sum(all_entropies) / num_tokens
        else:
            ent_C = 0.0
        entropy_turn_mean.append(ent_C)

        # B: [[ 之后第一个 token 的熵
        decoded_tokens = [tlp["decoded"] for tlp in token_lp_list]
        action_idx = find_action_token_index(token_lp_list, decoded_tokens)
        if action_idx is not None and action_idx < num_tokens:
            ent_B = compute_entropy_from_logprobs(token_lp_list[action_idx]["top_logprobs"])
        else:
            ent_B = None  # 这个 turn 没有 action（可能是对话 turn）
        entropy_action_token.append(ent_B)
        # ----------------------------

        # 判断终止
        response_lower = response.lower()
        if 'task completed' in response_lower or 'task failed' in response_lower:
            done = True

        messages.append({"role": "assistant", "content": response})

        # 环境交互
        try:
            done, observation, step_reward, _ = await interaction.generate_response(instance_id, messages)
            total_reward += step_reward
            messages.append({"role": "user", "content": observation})
        except Exception as e:
            logger.error(f"Env error session {session_id} turn {turn_idx}: {e}")
            break

    try:
        await interaction.finalize_interaction(instance_id)
    except Exception:
        pass

    success = total_reward > 0.0
    num_turns = len(entropy_first_token)

    return {
        "session_id": session_id,
        "success": success,
        "reward": total_reward,
        "num_turns": num_turns,
        "turn_lengths": turn_lengths,
        "entropy_A": entropy_first_token,    # List[float]，长度 = num_turns
        "entropy_B": entropy_action_token,   # List[Optional[float]]
        "entropy_C": entropy_turn_mean,      # List[float]
    }


# =============================================================================
# 文件写入工具
# =============================================================================

def safe_write_record(output_file: str, record: Dict):
    try:
        with open(output_file, 'a') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(json.dumps(record) + '\n')
                f.flush()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except Exception as e:
        logger.error(f"Write failed: {e}")


# =============================================================================
# 聚合分析：按 turn_index 统计平均熵
# =============================================================================

def aggregate_entropy(all_results: List[Dict]) -> Dict:
    """
    按 turn_index（0-based）聚合三种熵，输出每个 step 的均值和标准差。
    只统计成功的 episode（可配置）。
    """
    from collections import defaultdict

    # turn_index -> List[value]
    agg_A = defaultdict(list)
    agg_B = defaultdict(list)
    agg_C = defaultdict(list)
    agg_success_A = defaultdict(list)
    agg_success_B = defaultdict(list)
    agg_success_C = defaultdict(list)

    for res in all_results:
        if "error" in res:
            continue
        success = res.get("success", False)
        for t_idx, val in enumerate(res["entropy_A"]):
            agg_A[t_idx].append(val)
            if success:
                agg_success_A[t_idx].append(val)
        for t_idx, val in enumerate(res["entropy_B"]):
            if val is not None:
                agg_B[t_idx].append(val)
                if success:
                    agg_success_B[t_idx].append(val)
        for t_idx, val in enumerate(res["entropy_C"]):
            agg_C[t_idx].append(val)
            if success:
                agg_success_C[t_idx].append(val)

    def summarize(agg_dict):
        result = {}
        for t_idx in sorted(agg_dict.keys()):
            vals = agg_dict[t_idx]
            n = len(vals)
            mean = sum(vals) / n
            std = math.sqrt(sum((v - mean) ** 2 for v in vals) / n) if n > 1 else 0.0
            result[t_idx] = {"mean": mean, "std": std, "count": n}
        return result

    return {
        "all_episodes": {
            "A_first_token": summarize(agg_A),
            "B_action_token": summarize(agg_B),
            "C_turn_mean": summarize(agg_C),
        },
        "success_only": {
            "A_first_token": summarize(agg_success_A),
            "B_action_token": summarize(agg_success_B),
            "C_turn_mean": summarize(agg_success_C),
        },
        "total_episodes": len(all_results),
        "success_episodes": sum(1 for r in all_results if r.get("success", False)),
    }


# =============================================================================
# 绘图（可选）
# =============================================================================

def plot_entropy(summary: Dict, output_path: str):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not available, skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    labels = {
        "A_first_token": "A: First-token Entropy (Think start)",
        "B_action_token": "B: Action first-token Entropy (after [[)",
        "C_turn_mean": "C: Mean Token Entropy per Turn",
    }
    colors = {"A_first_token": "steelblue", "B_action_token": "coral", "C_turn_mean": "seagreen"}

    for ax_idx, (subset_key, subset_label) in enumerate([
        ("all_episodes", "All Episodes"),
        ("success_only", "Success Episodes Only"),
    ]):
        ax = axes[ax_idx]
        subset = summary[subset_key]
        for metric_key, metric_label in labels.items():
            data = subset[metric_key]
            if not data:
                continue
            steps = sorted(int(k) for k in data.keys())
            means = [data[s]["mean"] for s in steps]
            stds = [data[s]["std"] for s in steps]
            xs = [s + 1 for s in steps]  # 1-based display

            ax.plot(xs, means, marker='o', markersize=4,
                    label=metric_label, color=colors[metric_key])
            ax.fill_between(
                xs,
                [m - s for m, s in zip(means, stds)],
                [m + s for m, s in zip(means, stds)],
                alpha=0.15, color=colors[metric_key]
            )

        ax.set_xlabel("Turn Step (1-indexed)", fontsize=12)
        ax.set_ylabel("Token Entropy", fontsize=12)
        ax.set_title(
            f"TextCraft Token Entropy vs. Step\n"
            f"({subset_label}, total={summary['total_episodes']} episodes, "
            f"success={summary['success_episodes']})",
            fontsize=11
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Plot saved to: {output_path}")
    plt.close()


# =============================================================================
# 主流程
# =============================================================================

async def run_analysis(args: argparse.Namespace):
    interaction = TextCraftInteraction({
        'env_server_base': args.textcraft_server,
        'timeout': 600,
        'max_retries': 3
    })

    agent = EntropyTrackingAgent(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_logprobs=args.logprobs,
    )

    logger.info(f"Using model: {args.model}  base_url: {args.base_url}")
    # 加载数据集
    logger.info(f"Loading dataset: {args.data_path}")
    table = pq.read_table(args.data_path)
    num_rows = table.num_rows
    logger.info(f"数据集共 {num_rows} 个 query")

    total = num_rows if args.max_samples <= 0 else min(args.max_samples, num_rows)

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = args.model.replace("/", "-").replace(" ", "_")
    results_file = os.path.join(args.output_dir, f"entropy_results_{model_tag}_{timestamp}.jsonl")
    summary_file = os.path.join(args.output_dir, f"entropy_summary_{model_tag}_{timestamp}.json")
    plot_file    = os.path.join(args.output_dir, f"entropy_plot_{model_tag}_{timestamp}.png")

    # 清空结果文件
    open(results_file, 'w').close()

    sem = asyncio.Semaphore(args.concurrency)
    all_results = []

    async def worker(session_id: int):
        async with sem:
            result = await analyze_one_episode(
                agent=agent,
                interaction=interaction,
                session_id=session_id,
                http_session=http_session,
                max_rounds=args.max_rounds,
            )
            await asyncio.to_thread(safe_write_record, results_file, result)
            return result

    connector = aiohttp.TCPConnector(limit=args.concurrency, limit_per_host=args.concurrency)
    timeout = aiohttp.ClientTimeout(total=300)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as http_session:
        tasks = [worker(i) for i in range(total)]
        pbar = tqdm(total=total, desc=f"Analyzing [{args.model}]", unit="episode")
        for f in asyncio.as_completed(tasks):
            res = await f
            if res:
                all_results.append(res)
            pbar.update(1)
        pbar.close()

    # 聚合分析
    logger.info("Aggregating entropy statistics...")
    summary = aggregate_entropy(all_results)

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"Summary saved: {summary_file}")

    # 打印关键统计
    print("\n" + "=" * 65)
    print(f"TextCraft Token Entropy Analysis  [{args.model}]")
    print("=" * 65)
    print(f"Total episodes: {summary['total_episodes']},  Success: {summary['success_episodes']}")
    lp_note = "(logprobs unavailable - all zeros)" \
        if all(v["mean"] == 0.0 for v in summary["all_episodes"]["A_first_token"].values()) else ""
    if lp_note:
        print(f"WARNING: {lp_note}  MiniMax may not support logprobs for this model.")
    print("\n[All Episodes] A - First-token Entropy (top 10 steps):")
    for step, stat in sorted(summary["all_episodes"]["A_first_token"].items(), key=lambda x: int(x[0]))[:10]:
        print(f"  Step {int(step)+1:2d}: mean={stat['mean']:.4f}  std={stat['std']:.4f}  (n={stat['count']})")
    print("\n[All Episodes] C - Mean Turn Entropy (top 10 steps):")
    for step, stat in sorted(summary["all_episodes"]["C_turn_mean"].items(), key=lambda x: int(x[0]))[:10]:
        print(f"  Step {int(step)+1:2d}: mean={stat['mean']:.4f}  std={stat['std']:.4f}  (n={stat['count']})")
    print("=" * 65)

    # 绘图
    plot_entropy(summary, plot_file)

    logger.info(f"Done!\n  Raw data : {results_file}\n  Summary  : {summary_file}\n  Plot     : {plot_file}")


def main():
    parser = argparse.ArgumentParser(description='TextCraft Token Entropy Analysis')

    parser.add_argument('--data_path', type=str,
                        default='/Data/wyh/datasets/Verl-Data/train/textcraft/train.parquet')
    parser.add_argument('--output_dir', type=str,
                        default='/Data/wyh/datasets/Verl-Data/outputs/entropy_analysis_minimax')
    parser.add_argument('--textcraft_server', type=str,
                        default='http://127.0.0.1:36001')

    # MiniMax API 配置
    parser.add_argument('--api_key', type=str,
                        default='sk-api-mwLPAgumrEeAGUp-DwHKCG_GDHGHrDip50YQ94ucOr1V73g1wnDnlZVurGT638HZpbHij2CX27qyY-Pvti6_CuPzP47LOpS1VudhHn6mTUoqoPzVj7x9HLo',
                        help='MiniMax API key')
    parser.add_argument('--base_url', type=str,
                        default='https://api.minimaxi.com/v1',
                        help='MiniMax API base URL')
    parser.add_argument('--model', type=str,
                        default='MiniMax-M2.1',
                        help='Model name')

    parser.add_argument('--max_rounds', type=int, default=20,
                        help='Max interaction rounds per episode (same as sampling script)')
    parser.add_argument('--max_samples', type=int, default=-1,
                        help='Max queries to analyze, -1 = all')
    parser.add_argument('--concurrency', type=int, default=4,
                        help='Concurrent requests (keep low for external API, recommend 4~8)')
    parser.add_argument('--max_new_tokens', type=int, default=16384,
                        help='Max tokens per generation')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--logprobs', type=int, default=20,
                        help='top-k logprobs per token position (used if API supports it)')

    args = parser.parse_args()

    logger.info("📋 分析配置:")
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")

    try:
        asyncio.run(run_analysis(args))
    except KeyboardInterrupt:
        logger.info("已中断")
    except Exception as e:
        logger.exception(f"分析失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
