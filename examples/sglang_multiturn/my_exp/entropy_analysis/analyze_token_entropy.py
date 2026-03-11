#!/usr/bin/env python3
"""
逐 Token 熵分析脚本
===================
分析 MiniMax 轨迹数据中的逐 token 熵信息。

数据格式要求（每条记录）：
  - item_id: str
  - success: int (0/1)
  - reward: float
  - num_turns: int
  - total_tokens: int
  - turn_lengths: List[int]  # 每个 turn 的 token 数
  - cumsum_lengths: List[int]  # 累计 token 数
  - relative_positions: List[float]  # 相对位置 0~1
  - entropy_per_token: List[List[float]]  # 二维：每个 turn 的逐 token 熵

统计维度：
  1. 按 token 位置聚合：所有轨迹在同一绝对位置的熵
  2. 按相对位置 q 聚合：按轨迹进度的百分比聚合
  3. 按 turn 索引聚合：每个 turn 的平均熵
  4. 成功 vs 失败对比

用法：
  python analyze_token_entropy.py \
      --input_dir /Data/wyh/datasets/Verl-Data/outputs/entropy_offline_20260309_152233 \
      --output_dir /Data/wyh/datasets/Verl-Data/outputs/token_entropy_analysis \
      --max_samples -1
"""

import os
import sys
import json
import math
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import numpy as np

# ---------- logging ----------
log_dir = Path("/Data/wyh/datasets/Verl-Data/outputs/token_entropy_analysis/logs")
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / f"token_entropy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    ],
)
logger = logging.getLogger("TokenEntropyAnalysis")


# =============================================================================
# 数据加载
# =============================================================================

def load_token_entropy_data(input_path: str, max_samples: int = -1) -> List[Dict]:
    """
    加载包含逐 token 熵的数据。
    """
    records = []
    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if max_samples > 0:
        records = records[:max_samples]

    logger.info(f"Loaded {len(records)} records from {input_path}")
    return records


def validate_record(record: Dict) -> bool:
    """验证记录是否包含必要的字段和有效的逐 token 熵数据"""
    required_fields = ["item_id", "entropy_per_token", "turn_lengths", "total_tokens"]
    for field in required_fields:
        if field not in record:
            logger.warning(f"Missing field: {field}")
            return False

    entropy_per_token = record.get("entropy_per_token", [])
    turn_lengths = record.get("turn_lengths", [])

    if not entropy_per_token:
        logger.warning(f"Empty entropy_per_token for {record.get('item_id')}")
        return False

    # 验证每个 turn 的熵长度是否与 turn_lengths 一致
    for i, (ents, length) in enumerate(zip(entropy_per_token, turn_lengths)):
        if length > 0 and len(ents) != length:
            logger.warning(
                f"Mismatch at turn {i}: entropy_per_token has {len(ents)} tokens, "
                f"but turn_lengths says {length}"
            )
            # 不作为失败条件，因为可能有 padding

    return True


# =============================================================================
# 聚合分析
# =============================================================================

def compute_percentiles(data: List[float], percentiles: List[int] = None) -> Dict:
    """计算百分位数"""
    if percentiles is None:
        percentiles = [25, 50, 75, 90, 95, 99]
    if not data:
        return {f"p{p}": 0.0 for p in percentiles}

    sorted_data = sorted(data)
    n = len(sorted_data)
    result = {}
    for p in percentiles:
        idx = int(n * p / 100)
        idx = min(idx, n - 1)
        result[f"p{p}"] = sorted_data[idx]
    return result


def aggregate_token_entropy(records: List[Dict]) -> Dict:
    """
    逐 token 熵聚合分析。

    聚合维度：
    1. all_tokens: 所有 token 位置的统计
    2. by_token_position: 按绝对 token 位置 (0, 100, 200, ...)
    3. by_relative_position: 按相对位置 q (0-5%, 5-10%, ...)
    4. by_turn_index: 按 turn 索引
    5. by_length_bin: 按轨迹总长度分桶
    """
    # 1. 全局统计
    all_entropies = []

    # 2. 按 token 绝对位置聚合 (每 100 个 token 为一个 bucket)
    # key: token_position // 100, value: list of entropies
    entropy_by_abs_pos = defaultdict(list)

    # 3. 按相对位置 q 聚合 (20 个 bucket: 0-5%, 5-10%, ..., 95-100%)
    entropy_by_q = defaultdict(list)

    # 4. 按 turn 索引聚合
    entropy_by_turn = defaultdict(list)

    # 5. 按轨迹长度分桶 (0-2k, 2k-4k, 4k-6k, 6k-8k, 8k-10k, 10k+)
    entropy_by_length_bin = defaultdict(list)

    # 6. 成功 vs 失败分开统计
    success_entropies = []
    failure_entropies = []

    # 7. 每个 token 位置的详细信息 (前 500 个位置)
    token_pos_details = defaultdict(list)

    valid_count = 0
    error_count = 0

    for record in records:
        if not validate_record(record):
            error_count += 1
            continue

        is_success = bool(record.get("success", 0))
        total_tokens = record.get("total_tokens", 0)
        turn_lengths = record.get("turn_lengths", [])
        cumsum_lengths = record.get("cumsum_lengths", [0])
        entropy_per_token = record.get("entropy_per_token", [])
        relative_positions = record.get("relative_positions", [])

        # 轨迹长度分桶
        if total_tokens < 2000:
            length_bin = 0
        elif total_tokens < 4000:
            length_bin = 1
        elif total_tokens < 6000:
            length_bin = 2
        elif total_tokens < 8000:
            length_bin = 3
        elif total_tokens < 10000:
            length_bin = 4
        else:
            length_bin = 5

        # 遍历每个 turn
        for turn_idx, token_entropies in enumerate(entropy_per_token):
            if not token_entropies:
                continue

            # 该 turn 的平均熵
            turn_mean_entropy = sum(token_entropies) / len(token_entropies)
            entropy_by_turn[turn_idx].append(turn_mean_entropy)

            # 该 turn 开始的绝对位置
            turn_start_pos = cumsum_lengths[turn_idx] if turn_idx < len(cumsum_lengths) else 0

            # 相对位置
            q = relative_positions[turn_idx] if turn_idx < len(relative_positions) else 0.0
            q_bin = int(q * 20)  # 0-19
            q_bin = min(q_bin, 19)

            # 遍历每个 token
            for token_idx, entropy_val in enumerate(token_entropies):
                if entropy_val is None or entropy_val < 0:
                    continue

                all_entropies.append(entropy_val)

                if is_success:
                    success_entropies.append(entropy_val)
                else:
                    failure_entropies.append(entropy_val)

                # 绝对位置
                abs_pos = turn_start_pos + token_idx
                abs_pos_bin = abs_pos // 100  # 每 100 token 一个 bucket
                entropy_by_abs_pos[abs_pos_bin].append(entropy_val)

                # 相对位置 q
                entropy_by_q[q_bin].append(entropy_val)

                # 长度分桶
                entropy_by_length_bin[length_bin].append(entropy_val)

                # 记录前 500 个位置的详细信息
                if abs_pos < 500:
                    token_pos_details[abs_pos].append(entropy_val)

        valid_count += 1

    logger.info(f"Valid records: {valid_count}, Error records: {error_count}")

    # ========== 汇总统计函数 ==========
    def summarize_entropies(entropy_list: List[float], include_percentiles: bool = True) -> Dict:
        if not entropy_list:
            return {}
        n = len(entropy_list)
        mean = sum(entropy_list) / n
        std = math.sqrt(sum((e - mean) ** 2 for e in entropy_list) / n) if n > 1 else 0.0
        result = {
            "mean": mean,
            "std": std,
            "count": n,
            "min": min(entropy_list),
            "max": max(entropy_list),
        }
        if include_percentiles:
            result.update(compute_percentiles(entropy_list))
        return result

    # 按 token 位置聚合 (每 100 token)
    def summarize_by_abs_pos(agg_dict: Dict) -> Dict:
        result = {}
        for pos_bin in sorted(agg_dict.keys()):
            ents = agg_dict[pos_bin]
            if ents:
                pos_label = pos_bin * 100
                result[str(pos_label)] = summarize_entropies(ents)
        return result

    # 按相对位置 q 聚合
    def summarize_by_q(agg_dict: Dict) -> Dict:
        result = {}
        for q_bin in range(20):
            ents = agg_dict.get(q_bin, [])
            if ents:
                q_label = f"q{q_bin * 5:02d}-{(q_bin + 1) * 5:02d}"
                result[q_label] = summarize_entropies(ents)
        return result

    # 按 turn 索引聚合
    def summarize_by_turn(agg_dict: Dict) -> Dict:
        result = {}
        for turn_idx in sorted(agg_dict.keys()):
            ents = agg_dict[turn_idx]
            if ents:
                result[str(turn_idx)] = summarize_entropies(ents)
        return result

    # 按长度分桶聚合
    def summarize_by_length_bin(agg_dict: Dict) -> Dict:
        bin_labels = ["0-2k", "2k-4k", "4k-6k", "6k-8k", "8k-10k", "10k+"]
        result = {}
        for bin_idx in range(6):
            ents = agg_dict.get(bin_idx, [])
            if ents:
                result[bin_labels[bin_idx]] = summarize_entropies(ents)
        return result

    # 前 500 个位置的详细统计
    def summarize_token_pos_details(agg_dict: Dict) -> Dict:
        result = {}
        for pos in sorted(agg_dict.keys())[:500]:
            ents = agg_dict[pos]
            if ents:
                result[str(pos)] = summarize_entropies(ents)
        return result

    return {
        "global": {
            "all": summarize_entropies(all_entropies),
            "success": summarize_entropies(success_entropies),
            "failure": summarize_entropies(failure_entropies),
        },
        "by_token_position": summarize_by_abs_pos(entropy_by_abs_pos),
        "by_relative_position": {
            "all": summarize_by_q(entropy_by_q),
        },
        "by_turn_index": {
            "all": summarize_by_turn(entropy_by_turn),
        },
        "by_length_bin": {
            "all": summarize_by_length_bin(entropy_by_length_bin),
        },
        "token_position_detail": summarize_token_pos_details(token_pos_details),
        "total_records": len(records),
        "valid_records": valid_count,
        "error_records": error_count,
        "success_count": sum(1 for r in records if r.get("success", 0)),
    }


# =============================================================================
# 绘图
# =============================================================================

def plot_token_entropy(summary: Dict, output_dir: Path):
    """绘制逐 token 熵分析图"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plot")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 全局分布：成功 vs 失败
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 左上：全局熵分布直方图
    ax = axes[0, 0]
    global_all = summary.get("global", {}).get("all", {})
    # 这里没有原始数据分布，只能用统计信息
    # 绘制 by_relative_position 的均值趋势
    by_q = summary.get("by_relative_position", {}).get("all", {})
    if by_q:
        q_labels = list(by_q.keys())
        means = [by_q[q].get("mean", 0) for q in q_labels]
        stds = [by_q[q].get("std", 0) for q in q_labels]
        x = range(len(q_labels))
        ax.plot(x, means, marker='o', label='Mean Entropy')
        ax.fill_between(x, [m - s for m, s in zip(means, stds)],
                       [m + s for m, s in zip(means, stds)], alpha=0.2)
        ax.set_xticks(x)
        ax.set_xticklabels(q_labels, rotation=45, ha='right', fontsize=8)
        ax.set_xlabel("Relative Position (q)")
        ax.set_ylabel("Token Entropy")
        ax.set_title("Entropy vs Relative Position (0-100%)")
        ax.grid(True, alpha=0.3)

    # 右上：按 turn 索引
    ax = axes[0, 1]
    by_turn = summary.get("by_turn_index", {}).get("all", {})
    if by_turn:
        turn_labels = list(by_turn.keys())
        means = [by_turn[t].get("mean", 0) for t in turn_labels]
        stds = [by_turn[t].get("std", 0) for t in turn_labels]
        x = [int(t) for t in turn_labels]
        ax.plot(x, means, marker='s', label='Mean Entropy')
        ax.fill_between(x, [m - s for m, s in zip(means, stds)],
                       [m + s for m, s in zip(means, stds)], alpha=0.2)
        ax.set_xlabel("Turn Index")
        ax.set_ylabel("Token Entropy")
        ax.set_title("Entropy vs Turn Index")
        ax.grid(True, alpha=0.3)

    # 左下：按 token 绝对位置
    ax = axes[1, 0]
    by_pos = summary.get("by_token_position", {})
    if by_pos:
        pos_labels = sorted(by_pos.keys(), key=lambda x: int(x))
        means = [by_pos[p].get("mean", 0) for p in pos_labels]
        stds = [by_pos[p].get("std", 0) for p in pos_labels]
        x = [int(p) for p in pos_labels]
        ax.plot(x, means, marker='.', label='Mean Entropy')
        ax.fill_between(x, [m - s for m, s in zip(means, stds)],
                       [m + s for m, s in zip(means, stds)], alpha=0.2)
        ax.set_xlabel("Token Position (absolute)")
        ax.set_ylabel("Token Entropy")
        ax.set_title("Entropy vs Token Position")
        ax.grid(True, alpha=0.3)

    # 右下：按轨迹长度分桶
    ax = axes[1, 1]
    by_bin = summary.get("by_length_bin", {}).get("all", {})
    if by_bin:
        bin_labels = list(by_bin.keys())
        means = [by_bin[b].get("mean", 0) for b in bin_labels]
        stds = [by_bin[b].get("std", 0) for b in bin_labels]
        x = range(len(bin_labels))
        ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, rotation=45, ha='right')
        ax.set_xlabel("Trajectory Length Bin")
        ax.set_ylabel("Token Entropy")
        ax.set_title("Entropy vs Trajectory Length")
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_path = output_dir / "token_entropy_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Plot saved to: {plot_path}")

    # 2. 前 500 个 token 位置的详细熵值
    fig, ax = plt.subplots(figsize=(14, 6))
    pos_detail = summary.get("token_position_detail", {})
    if pos_detail:
        positions = sorted([int(p) for p in pos_detail.keys()])
        means = [pos_detail[str(p)].get("mean", 0) for p in positions]
        stds = [pos_detail[str(p)].get("std", 0) for p in positions]
        ax.plot(positions, means, marker='.', markersize=3, label='Mean Entropy')
        ax.fill_between(positions, [m - s for m, s in zip(means, stds)],
                       [m + s for m, s in zip(means, stds)], alpha=0.2)
        ax.set_xlabel("Token Position (0-500)")
        ax.set_ylabel("Token Entropy")
        ax.set_title("Detailed Entropy for First 500 Tokens")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plot_path2 = output_dir / "token_entropy_first_500.png"
    plt.savefig(plot_path2, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Plot saved to: {plot_path2}")


# =============================================================================
# 主流程
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='逐 Token 熵分析')

    parser.add_argument('--input_dir', type=str,
                        default='/Data/wyh/datasets/Verl-Data/outputs/entropy_offline_20260309_152233',
                        help='输入目录或文件路径')
    parser.add_argument('--input_file', type=str,
                        default='offline_results_20260309_152233.jsonl',
                        help='输入的 jsonl 文件名')
    parser.add_argument('--output_dir', type=str,
                        default='/Data/wyh/datasets/Verl-Data/outputs/token_entropy_analysis',
                        help='输出目录')
    parser.add_argument('--max_samples', type=int, default=-1,
                        help='最大样本数，-1 表示全部')

    args = parser.parse_args()

    # 确定输入路径
    input_path = Path(args.input_dir)
    if input_path.is_file():
        input_file = input_path
    else:
        input_file = input_path / args.input_file

    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        sys.exit(1)

    logger.info(f"Loading data from: {input_file}")
    records = load_token_entropy_data(str(input_file), args.max_samples)

    if not records:
        logger.error("No records loaded")
        sys.exit(1)

    # 聚合分析
    logger.info("Aggregating token entropy...")
    summary = aggregate_token_entropy(records)

    # 保存 summary
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = output_dir / f"token_entropy_summary_{timestamp}.json"

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"Summary saved to: {summary_file}")

    # 绘图
    logger.info("Generating plots...")
    plot_token_entropy(summary, output_dir)

    # 打印关键统计
    print("\n" + "=" * 70)
    print("逐 Token 熵分析结果")
    print("=" * 70)
    print(f"总记录数: {summary['total_records']}")
    print(f"有效记录数: {summary['valid_records']}")
    print(f"成功记录数: {summary['success_count']}")

    global_stats = summary.get("global", {})
    print("\n[全局统计]")
    all_stats = global_stats.get("all", {})
    print(f"  所有 token: mean={all_stats.get('mean', 0):.4f}, std={all_stats.get('std', 0):.4f}, "
          f"min={all_stats.get('min', 0):.4f}, max={all_stats.get('max', 0):.4f}")

    success_stats = global_stats.get("success", {})
    if success_stats:
        print(f"  成功轨迹: mean={success_stats.get('mean', 0):.4f}, std={success_stats.get('std', 0):.4f}")

    failure_stats = global_stats.get("failure", {})
    if failure_stats:
        print(f"  失败轨迹: mean={failure_stats.get('mean', 0):.4f}, std={failure_stats.get('std', 0):.4f}")

    print("\n[按相对位置 q]")
    by_q = summary.get("by_relative_position", {}).get("all", {})
    for q_label in list(by_q.keys())[:5]:
        stats = by_q[q_label]
        print(f"  {q_label}: mean={stats.get('mean', 0):.4f}, count={stats.get('count', 0)}")
    if len(by_q) > 5:
        print("  ...")

    print("\n[按 turn 索引]")
    by_turn = summary.get("by_turn_index", {}).get("all", {})
    for turn_label in list(by_turn.keys())[:5]:
        stats = by_turn[turn_label]
        print(f"  Turn {turn_label}: mean={stats.get('mean', 0):.4f}, count={stats.get('count', 0)}")
    if len(by_turn) > 5:
        print("  ...")

    print("=" * 70)
    print(f"\n输出目录: {output_dir}")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()
