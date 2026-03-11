#!/usr/bin/env python3
"""
Per-Token Entropy Analysis on Pre-computed Trajectories
=======================================================
分析已收集的逐 token 熵数据，生成详细的统计和可视化。

这个脚本用于处理已经包含 entropy_per_token 的数据，
不需要再调用 vLLM 进行 forward 计算。

用法：
  python entropy_analyze_per_token.py \
      --input_file /Data/wyh/datasets/Verl-Data/outputs/entropy_offline_20260309_152233/offline_results_20260309_152233.jsonl \
      --output_dir /Data/wyh/datasets/Verl-Data/outputs/entropy_per_token_analysis
"""

import os
import sys
import json
import math
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

import numpy as np
from tqdm import tqdm


# ---------- logging ----------
log_dir = Path("/Data/wyh/datasets/Verl-Data/outputs/entropy_per_token_analysis/logs")
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / f"per_token_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    ],
)
logger = logging.getLogger("PerTokenEntropyAnalysis")


# =============================================================================
# 数据加载
# =============================================================================

def load_entropy_results(input_file: str, max_samples: int = -1) -> List[Dict]:
    """加载已经计算好的逐 token 熵结果。"""
    results = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    if max_samples > 0:
        results = results[:max_samples]

    logger.info(f"Loaded {len(results)} trajectory results from {input_file}")
    return results


# =============================================================================
# 逐 token 熵分析核心函数
# =============================================================================

def compute_statistics(values: List[float]) -> Dict[str, float]:
    """计算基础统计量。"""
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
    
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "count": len(values),
    }


def compute_percentiles(values: List[float], percentiles: List[int] = [25, 50, 75, 90, 95, 99]) -> Dict[str, float]:
    """计算百分位数。"""
    if not values:
        return {f"p{p}": 0.0 for p in percentiles}
    
    arr = np.array(values)
    result = {}
    for p in percentiles:
        result[f"p{p}"] = float(np.percentile(arr, p))
    return result


def analyze_per_token_entropy(
    results: List[Dict],
    max_turns: int = 20,
) -> Dict[str, Any]:
    """
    对逐 token 熵数据进行全面分析。
    
    分析维度：
    1. 按 turn 索引聚合（每个 turn 的平均熵）
    2. 按 token 位置聚合（每个 token 位置的平均熵）
    3. 按相对位置 q = token_idx / total_tokens 聚合
    4. 熵的分布分析（高熵/低熵 token 比例）
    """
    
    # ---------- 1. 按 turn 索引聚合 ----------
    turn_entropy_stats = defaultdict(list)  # turn_idx -> List[mean_entropy]
    turn_token_counts = defaultdict(list)   # turn_idx -> List[token_count]
    
    # ---------- 2. 按 token 位置聚合 ----------
    # 记录每个 turn 内相同位置的 token 熵
    position_entropy = defaultdict(list)  # position_in_turn -> List[entropy]
    
    # ---------- 3. 按相对位置 q 聚合 ----------
    # q = token_idx / total_tokens_in_turn，范围 [0, 1]
    # 分成 20 个桶: 0-5%, 5-10%, ..., 95-100%
    q_entropy = defaultdict(list)  # q_bin -> List[entropy]
    
    # ---------- 4. 整体统计 ----------
    all_token_entropies = []  # 所有 token 的熵值
    total_valid_turns = 0
    total_empty_turns = 0
    
    success_count = 0
    total_count = len(results)
    
    for res in results:
        is_success = bool(res.get("success", 0))
        if is_success:
            success_count += 1
            
        entropy_per_token = res.get("entropy_per_token", [])
        turn_lengths = res.get("turn_lengths", [])
        
        num_turns = min(len(entropy_per_token), max_turns) if max_turns > 0 else len(entropy_per_token)
        
        for turn_idx in range(num_turns):
            if turn_idx >= len(entropy_per_token):
                break
                
            token_entropies = entropy_per_token[turn_idx]
            
            if not token_entropies:
                total_empty_turns += 1
                continue
                
            total_valid_turns += 1
            
            # Turn 级别的统计
            turn_mean_entropy = sum(token_entropies) / len(token_entropies)
            turn_entropy_stats[turn_idx].append(turn_mean_entropy)
            
            # Token 数量
            if turn_idx < len(turn_lengths):
                turn_token_counts[turn_idx].append(turn_lengths[turn_idx])
            
            # 记录每个 token 的熵
            for token_idx, entropy in enumerate(token_entropies):
                all_token_entropies.append(entropy)
                
                # 按 token 位置聚合（最多取前 200 个位置）
                if token_idx < 200:
                    position_entropy[token_idx].append(entropy)
                
                # 按相对位置 q 聚合
                q = token_idx / len(token_entropies)
                q_bin = int(q * 20)  # 0-19
                q_bin = min(q_bin, 19)
                q_entropy[q_bin].append(entropy)
    
    # ---------- 汇总统计 ----------
    
    # 1. Turn 级别统计
    turn_summary = {}
    for turn_idx in sorted(turn_entropy_stats.keys()):
        values = turn_entropy_stats[turn_idx]
        stats = compute_statistics(values)
        stats.update(compute_percentiles(values))
        turn_summary[f"turn_{turn_idx}"] = stats
    
    # 2. Token 位置级别统计
    position_summary = {}
    for pos in sorted(position_entropy.keys()):
        values = position_entropy[pos]
        stats = compute_statistics(values)
        position_summary[f"pos_{pos}"] = stats
    
    # 3. 相对位置 q 统计
    q_summary = {}
    for q_bin in range(20):
        values = q_entropy.get(q_bin, [])
        if values:
            stats = compute_statistics(values)
            q_label = f"q{q_bin * 5:02d}-{(q_bin + 1) * 5:02d}"
            q_summary[q_label] = stats
    
    # 4. 整体统计
    overall_stats = compute_statistics(all_token_entropies)
    overall_stats.update(compute_percentiles(all_token_entropies))
    
    # 5. 熵的分布：高熵 vs 低熵 token
    high_entropy_ratio = sum(1 for e in all_token_entropies if e > 1.0) / len(all_token_entropies) if all_token_entropies else 0
    low_entropy_ratio = sum(1 for e in all_token_entropies if e < 0.1) / len(all_token_entropies) if all_token_entropies else 0
    
    entropy_distribution = {
        "high_entropy_ratio (e > 1.0)": high_entropy_ratio,
        "medium_entropy_ratio (0.1 <= e <= 1.0)": 1.0 - high_entropy_ratio - low_entropy_ratio,
        "low_entropy_ratio (e < 0.1)": low_entropy_ratio,
    }
    
    return {
        "total_episodes": total_count,
        "success_episodes": success_count,
        "success_rate": success_count / total_count if total_count > 0 else 0,
        "total_valid_turns": total_valid_turns,
        "total_empty_turns": total_empty_turns,
        "total_tokens_analyzed": len(all_token_entropies),
        "overall_entropy": overall_stats,
        "entropy_distribution": entropy_distribution,
        "entropy_by_turn": turn_summary,
        "entropy_by_position": position_summary,
        "entropy_by_relative_position": q_summary,
    }


# =============================================================================
# 绘图
# =============================================================================

def plot_per_token_entropy(summary: Dict, output_path: str):
    """绘制逐 token 熵分析图。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plot")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # ---------- (0,0): 按 turn 索引的熵曲线 ----------
    ax = axes[0, 0]
    entropy_by_turn = summary.get("entropy_by_turn", {})
    if entropy_by_turn:
        turns = sorted(entropy_by_turn.keys(), key=lambda x: int(x.split('_')[1]))
        means = [entropy_by_turn[t]["mean"] for t in turns]
        stds = [entropy_by_turn[t]["std"] for t in turns]
        xs = [int(t.split('_')[1]) + 1 for t in turns]  # 1-indexed
        
        ax.plot(xs, means, marker='o', markersize=4, color='steelblue', label='Mean entropy')
        ax.fill_between(xs, 
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.2, color='steelblue')
        
    ax.set_xlabel("Turn Index (1-indexed)", fontsize=11)
    ax.set_ylabel("Mean Token Entropy", fontsize=11)
    ax.set_title("Entropy by Turn Index", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # ---------- (0,1): 按相对位置 q 的熵 ----------
    ax = axes[0, 1]
    q_summary = summary.get("entropy_by_relative_position", {})
    if q_summary:
        q_labels = sorted(q_summary.keys(), key=lambda x: (int(x[1:3]), int(x[4:6])))
        means = [q_summary[q]["mean"] for q in q_labels]
        stds = [q_summary[q]["std"] for q in q_labels]
        xs = [int(q[1:3]) + 2.5 for q in q_labels]  # 中点
        
        ax.bar(xs, means, width=4, color='purple', alpha=0.6, label='Mean entropy')
        ax.plot(xs, means, marker='o', markersize=6, color='purple')
        ax.fill_between(xs,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.2, color='purple')
        
    ax.set_xlabel("Relative Position q = token_idx / total_tokens (%)", fontsize=11)
    ax.set_ylabel("Mean Token Entropy", fontsize=11)
    ax.set_title("Entropy by Relative Position in Turn", fontsize=12)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(True, alpha=0.3)
    
    # ---------- (0,2): 熵的分布直方图 ----------
    ax = axes[0, 2]
    overall = summary.get("overall_entropy", {})
    # 绘制简化的分布 - 这里需要原始数据才能绘制直方图
    # 用百分位数代替
    percentiles = [overall.get(f"p{p}", 0) for p in [25, 50, 75, 90, 95]]
    labels = ['p25', 'p50', 'p75', 'p90', 'p95']
    ax.bar(labels, percentiles, color='seagreen', alpha=0.7)
    ax.set_xlabel("Percentile", fontsize=11)
    ax.set_ylabel("Entropy Value", fontsize=11)
    ax.set_title("Entropy Percentiles (Overall)", fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # ---------- (1,0): 按 token 位置的热力图 ----------
    ax = axes[1, 0]
    position_summary = summary.get("entropy_by_position", {})
    if position_summary:
        # 取前 50 个位置
        positions = sorted(position_summary.keys(), key=lambda x: int(x.split('_')[1]))
        positions = [p for p in positions if int(p.split('_')[1]) < 50]
        
        if positions:
            means = [position_summary[p]["mean"] for p in positions]
            xs = [int(p.split('_')[1]) for p in positions]
            
            ax.plot(xs, means, marker='.', markersize=3, color='coral')
            ax.fill_between(xs, means, alpha=0.3, color='coral')
            
    ax.set_xlabel("Token Position in Turn", fontsize=11)
    ax.set_ylabel("Mean Entropy", fontsize=11)
    ax.set_title("Entropy by Token Position (First 50)", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # ---------- (1,1): 熵的分布饼图 ----------
    ax = axes[1, 1]
    dist = summary.get("entropy_distribution", {})
    if dist:
        labels = ['Low (e<0.1)', 'Medium', 'High (e>1.0)']
        sizes = [dist.get("low_entropy_ratio (e < 0.1)", 0),
                 dist.get("medium_entropy_ratio (0.1 <= e <= 1.0)", 0),
                 dist.get("high_entropy_ratio (e > 1.0)", 0)]
        colors = ['#2ecc71', '#f1c40f', '#e74c3c']
        explode = (0.05, 0, 0.05)
        
        ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
               shadow=True, startangle=90)
        ax.set_title("Entropy Distribution", fontsize=12)
    
    # ---------- (1,2): Turn 内熵变化趋势 (前 10 个 turn) ----------
    ax = axes[1, 2]
    # 用摘要数据绘制
    if entropy_by_turn:
        turns = sorted(entropy_by_turn.keys(), key=lambda x: int(x.split('_')[1]))[:10]
        means = [entropy_by_turn[t]["mean"] for t in turns]
        counts = [entropy_by_turn[t]["count"] for t in turns]
        
        x_pos = range(len(turns))
        ax.bar(x_pos, means, color='steelblue', alpha=0.7)
        
        for i, (m, c) in enumerate(zip(means, counts)):
            ax.text(i, m + 0.02, f'n={c}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel("Turn Index", fontsize=11)
        ax.set_ylabel("Mean Entropy", fontsize=11)
        ax.set_title("Entropy by Turn (First 10)", fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"Turn {int(t.split('_')[1])+1}" for t in turns], rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Plot saved: {output_path}")


# =============================================================================
# 主流程
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze per-token entropy from pre-computed trajectory results"
    )
    parser.add_argument("--input_file", type=str,
                        default="/Data/wyh/datasets/Verl-Data/outputs/entropy_offline_20260309_152233/offline_results_20260309_152233.jsonl",
                        help="Input jsonl file with entropy_per_token data")
    parser.add_argument("--output_dir", type=str,
                        default="/Data/wyh/datasets/Verl-Data/outputs/entropy_per_token_analysis",
                        help="Output directory")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Max samples to analyze, -1 = all")
    parser.add_argument("--max_turns", type=int, default=20,
                        help="Max turns to analyze per trajectory")
    args = parser.parse_args()

    # 加载数据
    logger.info(f"Loading data from: {args.input_file}")
    results = load_entropy_results(args.input_file, args.max_samples)
    
    if not results:
        logger.error("No data loaded!")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 分析
    logger.info("Analyzing per-token entropy...")
    summary = analyze_per_token_entropy(results, max_turns=args.max_turns)
    
    # 保存 summary
    summary_file = os.path.join(args.output_dir, f"per_token_summary_{ts}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Summary saved: {summary_file}")
    
    # 绘图
    plot_file = os.path.join(args.output_dir, f"per_token_plot_{ts}.png")
    plot_per_token_entropy(summary, plot_file)
    
    # 打印摘要
    print("\n" + "=" * 70)
    print("Per-Token Entropy Analysis Summary")
    print("=" * 70)
    print(f"Total episodes: {summary['total_episodes']}")
    print(f"Success episodes: {summary['success_episodes']} ({summary['success_rate']:.1%})")
    print(f"Total valid turns: {summary['total_valid_turns']}")
    print(f"Total tokens analyzed: {summary['total_tokens_analyzed']}")
    
    overall = summary['overall_entropy']
    print(f"\n[Overall Entropy Statistics]")
    print(f"  Mean: {overall['mean']:.4f} ± {overall['std']:.4f}")
    print(f"  Min: {overall['min']:.4f}, Max: {overall['max']:.4f}")
    print(f"  Median: {overall['median']:.4f}")
    
    dist = summary['entropy_distribution']
    print(f"\n[Entropy Distribution]")
    print(f"  Low (e < 0.1):    {dist['low_entropy_ratio (e < 0.1)']:.1%}")
    print(f"  Medium:           {dist['medium_entropy_ratio (0.1 <= e <= 1.0)']:.1%}")
    print(f"  High (e > 1.0):   {dist['high_entropy_ratio (e > 1.0)']:.1%}")
    
    # 按 turn 显示
    turn_summary = summary['entropy_by_turn']
    print(f"\n[Entropy by Turn] (showing first 10)")
    for turn_key in sorted(turn_summary.keys(), key=lambda x: int(x.split('_')[1]))[:10]:
        stats = turn_summary[turn_key]
        turn_idx = int(turn_key.split('_')[1])
        print(f"  Turn {turn_idx + 1:2d}: mean={stats['mean']:.4f} ± {stats['std']:.4f}  (n={stats['count']})")
    
    # 按相对位置显示
    q_summary = summary['entropy_by_relative_position']
    print(f"\n[Entropy by Relative Position q]")
    for q_key in sorted(q_summary.keys(), key=lambda x: (int(x[1:3]), int(x[4:6]))):
        stats = q_summary[q_key]
        print(f"  {q_key}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  Summary: {summary_file}")
    print(f"  Plot: {plot_file}")
    

if __name__ == "__main__":
    main()
