#!/usr/bin/env python3
"""
Entropy Prefix Cut Point Analysis
==================================
基于 Qwen3-1.7B 在 MiniMax-M2.1 TextCraft 轨迹上的 token entropy，
实现三种 prefix cut point 策略的分析与可视化。

三种策略：
1. top-k entropy peaks - 基于熵峰值的 cut point 选择
2. fixed position / ratio - 固定比例位置的 cut point
3. cumulative information curve - 基于累积信息量的 cut point

输出：
- 结构化的分析结果（jsonl）
- 可视化图表
"""

import os
import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import numpy as np


# =============================================================================
# 数据加载
# =============================================================================

def load_entropy_results(jsonl_path: str) -> List[Dict]:
    """加载 entropy 分析结果"""
    results = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    print(f"Loaded {len(results)} trajectories from {jsonl_path}")
    return results


def validate_trajectory(traj: Dict) -> bool:
    """验证轨迹数据是否有效"""
    required_fields = [
        'item_id', 'success', 'reward', 
        'num_assistant_turns', 'num_user_turns',
        'turn_lengths_assistant', 'turn_lengths_user',
        'cumsum_lengths_assistant', 'cumsum_lengths_user',
        'relative_positions',
        'entropy_C_assistant', 'entropy_C_user'
    ]
    for field in required_fields:
        if field not in traj:
            return False
    return True


# =============================================================================
# 策略 1: Top-K Entropy Peaks
# =============================================================================

def find_entropy_peaks(
    entropy_values: List[float],
    k: int = 3,
    min_distance: int = 1,
    threshold_ratio: float = 0.5
) -> List[int]:
    """
    找到 entropy 曲线中的 top-k 峰值位置
    
    Args:
        entropy_values: 熵值列表
        k: 选择 top-k 个峰值
        min_distance: 峰值之间的最小距离
        threshold_ratio: 峰值阈值（相对于最大值）
    
    Returns:
        峰值位置的索引列表
    """
    if not entropy_values or len(entropy_values) < 2:
        return []
    
    n = len(entropy_values)
    max_entropy = max(entropy_values)
    threshold = max_entropy * threshold_ratio
    
    # 找局部最大值
    peaks = []
    for i in range(n):
        val = entropy_values[i]
        if val < threshold:
            continue
        
        # 检查是否是局部最大值
        is_peak = True
        for j in range(max(0, i - min_distance), min(n, i + min_distance + 1)):
            if j != i and entropy_values[j] >= val:
                is_peak = False
                break
        
        if is_peak:
            peaks.append((i, val))
    
    # 按熵值降序排序，取 top-k
    peaks.sort(key=lambda x: x[1], reverse=True)
    peak_indices = [p[0] for p in peaks[:k]]
    
    return sorted(peak_indices)


def apply_topk_peaks_strategy(
    traj: Dict,
    k: int = 3,
    entropy_type: str = 'entropy_C_assistant'
) -> Dict[str, Any]:
    """
    应用 top-k entropy peaks 策略
    
    Args:
        traj: 轨迹数据
        k: 选择 top-k 个峰值
        entropy_type: 使用哪种熵曲线
    
    Returns:
        策略结果
    """
    entropy_values = traj.get(entropy_type, [])
    
    if not entropy_values:
        return {
            'cut_points': [],
            'cut_turn_indices': [],
            'cut_relative_positions': [],
            'cut_cumsum_tokens': [],
            'strategy': 'top-k peaks',
            'k': k
        }
    
    # 找峰值
    peak_indices = find_entropy_peaks(entropy_values, k=k)
    
    if not peak_indices:
        # 如果没有找到峰值，使用最高熵位置
        peak_indices = [int(np.argmax(entropy_values))]
    
    # 转换为各种坐标
    n_turns = len(entropy_values)
    relative_positions = traj.get('relative_positions', [])
    cumsum_lengths = traj.get('cumsum_lengths_assistant', [])
    
    cut_turn_indices = peak_indices
    cut_relative_positions = []
    cut_cumsum_tokens = []
    
    for idx in peak_indices:
        if idx < len(relative_positions):
            cut_relative_positions.append(relative_positions[idx])
        else:
            cut_relative_positions.append(None)
        
        if idx < len(cumsum_lengths):
            cut_cumsum_tokens.append(cumsum_lengths[idx])
        else:
            cut_cumsum_tokens.append(None)
    
    return {
        'cut_points': peak_indices,
        'cut_turn_indices': cut_turn_indices,
        'cut_relative_positions': cut_relative_positions,
        'cut_cumsum_tokens': cut_cumsum_tokens,
        'strategy': 'top-k peaks',
        'k': k,
        'entropy_values': entropy_values,
        'peak_values': [entropy_values[i] for i in peak_indices if i < len(entropy_values)]
    }


# =============================================================================
# 策略 2: Fixed Position / Ratio
# =============================================================================

def apply_fixed_ratio_strategy(
    traj: Dict,
    ratios: List[float] = [0.25, 0.5, 0.75],
    entropy_type: str = 'entropy_C_assistant'
) -> Dict[str, Any]:
    """
    应用固定比例位置策略
    
    使用实际 entropy 序列长度来计算 cut point，确保 cut point 有对应的有效信号
    
    Args:
        traj: 轨迹数据
        ratios: 比例列表
        entropy_type: 使用哪种熵曲线
    
    Returns:
        策略结果
    """
    # 使用实际 entropy 长度，而不是 num_assistant_turns
    entropy_values = traj.get(entropy_type, [])
    n_turns = len(entropy_values)
    
    if n_turns == 0:
        return {
            'cut_points': [],
            'cut_turn_indices': [],
            'cut_relative_positions': [],
            'cut_cumsum_tokens': [],
            'strategy': 'fixed ratio',
            'ratios': ratios,
            'entropy_type': entropy_type,
            'n_turns_used': 0
        }
    
    # 根据 entropy_type 选择对应的坐标字段
    # assistant 类型使用 cumsum_lengths_assistant
    # user 类型使用 cumsum_lengths_user
    if 'assistant' in entropy_type.lower():
        cumsum_lengths = traj.get('cumsum_lengths_assistant', [])
    else:  # user 类型
        cumsum_lengths = traj.get('cumsum_lengths_user', [])
    
    # relative_positions 是共用的（基于总 turn 数）
    relative_positions = traj.get('relative_positions', [])
    
    cut_turn_indices = []
    cut_relative_positions = []
    cut_cumsum_tokens = []
    
    for ratio in ratios:
        # 计算对应的 turn 索引 - 基于实际 entropy 长度
        turn_idx = int(n_turns * ratio)
        turn_idx = min(turn_idx, n_turns - 1)  # 不超过最后一个 turn
        
        cut_turn_indices.append(turn_idx)
        
        if turn_idx < len(relative_positions):
            cut_relative_positions.append(relative_positions[turn_idx])
        else:
            cut_relative_positions.append(None)
        
        if turn_idx < len(cumsum_lengths):
            cut_cumsum_tokens.append(cumsum_lengths[turn_idx])
        else:
            cut_cumsum_tokens.append(None)
    
    return {
        'cut_points': cut_turn_indices,
        'cut_turn_indices': cut_turn_indices,
        'cut_relative_positions': cut_relative_positions,
        'cut_cumsum_tokens': cut_cumsum_tokens,
        'strategy': 'fixed ratio',
        'ratios': ratios,
        'entropy_type': entropy_type,
        'n_turns_used': n_turns
    }


# =============================================================================
# 策略 3: Cumulative Information Curve
# =============================================================================

def compute_cumulative_info(
    entropy_values: List[float],
    normalize: bool = True
) -> Tuple[List[float], List[float]]:
    """
    计算累积信息量曲线
    
    Args:
        entropy_values: 熵值列表
        normalize: 是否归一化
    
    Returns:
        (累积信息量列表, 归一化累积比例列表)
    """
    if not entropy_values:
        return [], []
    
    # 累积和
    cumsum = []
    total = 0
    for val in entropy_values:
        total += val
        cumsum.append(total)
    
    if normalize and total > 0:
        # 归一化到 [0, 1]
        cumsum_ratio = [c / total for c in cumsum]
    else:
        cumsum_ratio = cumsum
    
    return cumsum, cumsum_ratio


def apply_cumulative_info_strategy(
    traj: Dict,
    thresholds: List[float] = [0.3, 0.5, 0.7],
    entropy_type: str = 'entropy_C_assistant'
) -> Dict[str, Any]:
    """
    应用累积信息量策略
    
    基于累积熵占比选择 cut point：
    - 30% 阈值：累积信息量达到 30% 时的位置
    - 50% 阈值：中位数信息量位置
    - 70% 阈值：累积信息量达到 70% 时的位置
    
    Args:
        traj: 轨迹数据
        thresholds: 累积信息量阈值列表
        entropy_type: 使用哪种熵曲线
    
    Returns:
        策略结果
    """
    entropy_values = traj.get(entropy_type, [])
    
    if not entropy_values:
        return {
            'cut_points': [],
            'cut_turn_indices': [],
            'cut_relative_positions': [],
            'cut_cumsum_tokens': [],
            'strategy': 'cumulative info',
            'thresholds': thresholds,
            'cumsum_ratio': []
        }
    
    # 计算累积信息量
    cumsum, cumsum_ratio = compute_cumulative_info(entropy_values, normalize=True)
    
    # 找每个阈值对应的位置
    cut_turn_indices = []
    for threshold in thresholds:
        # 找到第一个累积比例 >= threshold 的位置
        cut_idx = None
        for i, ratio in enumerate(cumsum_ratio):
            if ratio >= threshold:
                cut_idx = i
                break
        
        if cut_idx is None:
            # 如果没有达到阈值，使用最后一个位置
            cut_idx = len(cumsum_ratio) - 1
        
        cut_turn_indices.append(cut_idx)
    
    # 转换为各种坐标
    # 根据 entropy_type 选择对应的坐标字段
    if 'assistant' in entropy_type.lower():
        cumsum_lengths = traj.get('cumsum_lengths_assistant', [])
    else:  # user 类型
        cumsum_lengths = traj.get('cumsum_lengths_user', [])
    
    relative_positions = traj.get('relative_positions', [])
    
    cut_relative_positions = []
    cut_cumsum_tokens = []
    
    for idx in cut_turn_indices:
        if idx < len(relative_positions):
            cut_relative_positions.append(relative_positions[idx])
        else:
            cut_relative_positions.append(None)
        
        if idx < len(cumsum_lengths):
            cut_cumsum_tokens.append(cumsum_lengths[idx])
        else:
            cut_cumsum_tokens.append(None)
    
    return {
        'cut_points': cut_turn_indices,
        'cut_turn_indices': cut_turn_indices,
        'cut_relative_positions': cut_relative_positions,
        'cut_cumsum_tokens': cut_cumsum_tokens,
        'strategy': 'cumulative info',
        'thresholds': thresholds,
        'cumsum_ratio': cumsum_ratio,
        'entropy_type': entropy_type
    }


# =============================================================================
# 统一策略接口
# =============================================================================

def apply_all_strategies(
    traj: Dict,
    topk_k: int = 3,
    fixed_ratios: List[float] = [0.25, 0.5, 0.75],
    cumulative_thresholds: List[float] = [0.3, 0.5, 0.7]
) -> Dict[str, Any]:
    """
    应用所有三种策略
    
    Returns:
        包含三种策略结果的字典
    """
    # 策略 1: Top-K Entropy Peaks (使用 assistant entropy_C)
    topk_result = apply_topk_peaks_strategy(
        traj, k=topk_k, entropy_type='entropy_C_assistant'
    )
    
    # 策略 2: Fixed Ratio
    fixed_result = apply_fixed_ratio_strategy(
        traj, ratios=fixed_ratios, entropy_type='entropy_C_assistant'
    )
    
    # 策略 3: Cumulative Information
    cumulative_result = apply_cumulative_info_strategy(
        traj, thresholds=cumulative_thresholds, entropy_type='entropy_C_assistant'
    )
    
    return {
        'item_id': traj.get('item_id', 'unknown'),
        'success': traj.get('success', 0),
        'reward': traj.get('reward', 0),
        'num_assistant_turns': traj.get('num_assistant_turns', 0),
        'num_user_turns': traj.get('num_user_turns', 0),
        'total_tokens': traj.get('total_tokens', 0),
        
        # 原始坐标信息
        'turn_lengths_assistant': traj.get('turn_lengths_assistant', []),
        'turn_lengths_user': traj.get('turn_lengths_user', []),
        'cumsum_lengths_assistant': traj.get('cumsum_lengths_assistant', []),
        'cumsum_lengths_user': traj.get('cumsum_lengths_user', []),
        'relative_positions': traj.get('relative_positions', []),
        
        # 熵数据
        'entropy_A_assistant': traj.get('entropy_A_assistant', []),
        'entropy_B_assistant': traj.get('entropy_B_assistant', []),
        'entropy_C_assistant': traj.get('entropy_C_assistant', []),
        'entropy_A_user': traj.get('entropy_A_user', []),
        'entropy_C_user': traj.get('entropy_C_user', []),
        
        # 策略结果
        'cut_topk_peaks': topk_result['cut_turn_indices'],
        'cut_topk_peaks_relative': topk_result['cut_relative_positions'],
        'cut_topk_peaks_tokens': topk_result['cut_cumsum_tokens'],
        
        'cut_fixed_ratio': fixed_result['cut_turn_indices'],
        'cut_fixed_ratio_relative': fixed_result['cut_relative_positions'],
        'cut_fixed_ratio_tokens': fixed_result['cut_cumsum_tokens'],
        
        'cut_cumulative_info': cumulative_result['cut_turn_indices'],
        'cut_cumulative_info_relative': cumulative_result['cut_relative_positions'],
        'cut_cumulative_info_tokens': cumulative_result['cut_cumsum_tokens'],
        
        # 额外信息
        'cumsum_ratio_assistant': cumulative_result.get('cumsum_ratio', []),
    }


def process_all_trajectories(
    results: List[Dict],
    topk_k: int = 3,
    fixed_ratios: List[float] = [0.25, 0.5, 0.75],
    cumulative_thresholds: List[float] = [0.3, 0.5, 0.7]
) -> List[Dict]:
    """
    处理所有轨迹，应用所有策略
    """
    processed = []
    
    for i, traj in enumerate(results):
        # 验证数据
        if not validate_trajectory(traj):
            print(f"Warning: Invalid trajectory at index {i}: {traj.get('item_id', 'unknown')}")
            continue
        
        try:
            result = apply_all_strategies(
                traj,
                topk_k=topk_k,
                fixed_ratios=fixed_ratios,
                cumulative_thresholds=cumulative_thresholds
            )
            processed.append(result)
        except Exception as e:
            print(f"Error processing trajectory {traj.get('item_id', i)}: {e}")
            continue
    
    return processed


# =============================================================================
# 统计分析
# =============================================================================

def compute_strategy_statistics(processed_results: List[Dict]) -> Dict:
    """
    计算策略的全局统计信息
    """
    stats = {
        'total_trajectories': len(processed_results),
        'success_count': sum(1 for r in processed_results if r.get('success', 0) == 1),
        'failure_count': sum(1 for r in processed_results if r.get('success', 0) == 0),
    }
    
    # 按策略收集 cut point 分布
    for strategy_name, key in [
        ('topk', 'cut_topk_peaks'),
        ('fixed_ratio', 'cut_fixed_ratio'),
        ('cumulative', 'cut_cumulative_info')
    ]:
        # 收集所有 cut point
        all_cut_points = []
        success_cut_points = []
        failure_cut_points = []
        
        for r in processed_results:
            cuts = r.get(key, [])
            if not cuts:
                continue
            
            for cut in cuts:
                if cut is not None:
                    all_cut_points.append(cut)
                    if r.get('success', 0) == 1:
                        success_cut_points.append(cut)
                    else:
                        failure_cut_points.append(cut)
        
        # 统计
        if all_cut_points:
            stats[f'{strategy_name}_all_mean'] = np.mean(all_cut_points)
            stats[f'{strategy_name}_all_std'] = np.std(all_cut_points)
            stats[f'{strategy_name}_all_median'] = np.median(all_cut_points)
            stats[f'{strategy_name}_all_count'] = len(all_cut_points)
        
        if success_cut_points:
            stats[f'{strategy_name}_success_mean'] = np.mean(success_cut_points)
            stats[f'{strategy_name}_success_median'] = np.median(success_cut_points)
        
        if failure_cut_points:
            stats[f'{strategy_name}_failure_mean'] = np.mean(failure_cut_points)
            stats[f'{strategy_name}_failure_median'] = np.median(failure_cut_points)
    
    return stats


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Entropy Prefix Cut Point Analysis"
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='/Data/wyh/datasets/Verl-Data/outputs/entropy_offline_textcraft_20260312_202044/offline_results_20260312_202051.jsonl',
        help='Input entropy results jsonl file'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='/Data/wyh/datasets/Verl-Data/outputs/entropy_analysis',
        help='Output directory'
    )
    parser.add_argument(
        '--topk_k', '-k',
        type=int,
        default=3,
        help='Top-K peaks parameter'
    )
    parser.add_argument(
        '--fixed_ratios',
        type=str,
        default='0.25,0.5,0.75',
        help='Fixed ratio values (comma-separated)'
    )
    parser.add_argument(
        '--cumulative_thresholds',
        type=str,
        default='0.3,0.5,0.7',
        help='Cumulative info thresholds (comma-separated)'
    )
    parser.add_argument(
        '--max_samples', '-n',
        type=int,
        default=-1,
        help='Max samples to process (-1 = all)'
    )
    
    args = parser.parse_args()
    
    # 解析参数
    fixed_ratios = [float(x) for x in args.fixed_ratios.split(',')]
    cumulative_thresholds = [float(x) for x in args.cumulative_thresholds.split(',')]
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    print(f"Loading data from {args.input}...")
    results = load_entropy_results(args.input)
    
    if args.max_samples > 0:
        results = results[:args.max_samples]
    
    # 处理轨迹
    print(f"Processing {len(results)} trajectories...")
    processed = process_all_trajectories(
        results,
        topk_k=args.topk_k,
        fixed_ratios=fixed_ratios,
        cumulative_thresholds=cumulative_thresholds
    )
    
    # 保存处理结果
    output_jsonl = os.path.join(args.output_dir, 'processed_trajectories.jsonl')
    with open(output_jsonl, 'w') as f:
        for r in processed:
            f.write(json.dumps(r) + '\n')
    print(f"Saved processed results to {output_jsonl}")
    
    # 计算统计信息
    print("Computing statistics...")
    stats = compute_strategy_statistics(processed)
    
    # 保存统计信息
    stats_file = os.path.join(args.output_dir, 'strategy_statistics.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to {stats_file}")
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"Total trajectories: {stats['total_trajectories']}")
    print(f"Success: {stats['success_count']}")
    print(f"Failure: {stats['failure_count']}")
    print("\nStrategy Statistics:")
    for key, val in stats.items():
        if 'mean' in key or 'median' in key or 'count' in key:
            print(f"  {key}: {val:.3f}" if isinstance(val, float) else f"  {key}: {val}")


if __name__ == '__main__':
    main()
