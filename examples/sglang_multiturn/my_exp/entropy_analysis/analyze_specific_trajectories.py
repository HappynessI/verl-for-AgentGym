#!/usr/bin/env python3
"""
Trajectory Entropy Analysis - 查看具体轨迹的逐token熵变化
==========================================================
分析成功和失败轨迹的熵分布差异

用法:
  python analyze_specific_trajectories.py \
      --input_file /Data/wyh/datasets/Verl-Data/outputs/entropy_offline_20260309_152233/offline_results_20260309_152233.jsonl \
      --output_dir /Data/wyh/datasets/Verl-Data/outputs/trajectory_entropy_analysis
"""

import os
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from tqdm import tqdm

# ========== Logging ==========
log_dir = Path("/Data/wyh/datasets/Verl-Data/outputs/trajectory_entropy_analysis/logs")
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / f"traj_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    ],
)
logger = logging.getLogger("TrajectoryEntropyAnalysis")


# ========== Data Loading ==========

def load_trajectories(input_file: str) -> List[Dict]:
    """加载所有轨迹数据"""
    results = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    logger.info(f"Loaded {len(results)} trajectories")
    return results


def get_trajectory_by_id(results: List[Dict], item_id: str) -> Tuple[Dict, int]:
    """根据item_id查找轨迹，返回(轨迹, 索引)"""
    for idx, res in enumerate(results):
        if res.get('item_id') == item_id:
            return res, idx
    return None, -1


# ========== Trajectory Analysis ==========

def analyze_trajectory_summary(traj: Dict) -> Dict:
    """分析单条轨迹的摘要信息"""
    entropy_per_token = traj.get('entropy_per_token', [])
    turn_lengths = traj.get('turn_lengths', [])
    
    # 有效turn数量（非空）
    valid_turns = sum(1 for tokens in entropy_per_token if tokens)
    total_tokens = traj.get('total_tokens', 0)
    
    # 整体熵统计
    all_entropies = []
    for turn_tokens in entropy_per_token:
        all_entropies.extend(turn_tokens)
    
    if all_entropies:
        mean_entropy = np.mean(all_entropies)
        max_entropy = np.max(all_entropies)
        min_entropy = np.min(all_entropies)
    else:
        mean_entropy = max_entropy = min_entropy = 0.0
    
    # 每个turn的熵统计
    turn_stats = []
    for i, tokens in enumerate(entropy_per_token):
        if tokens:
            turn_entropies = tokens
            turn_stats.append({
                'turn_idx': i,
                'token_count': turn_lengths[i] if i < len(turn_lengths) else len(tokens),
                'mean_entropy': float(np.mean(turn_entropies)),
                'max_entropy': float(np.max(turn_entropies)),
                'min_entropy': float(np.min(turn_entropies)),
                'std_entropy': float(np.std(turn_entropies)),
            })
    
    return {
        'item_id': traj.get('item_id'),
        'success': traj.get('success'),
        'reward': traj.get('reward'),
        'num_turns': traj.get('num_turns'),
        'valid_turns': valid_turns,
        'total_tokens': total_tokens,
        'mean_entropy': float(mean_entropy),
        'max_entropy': float(max_entropy),
        'min_entropy': float(min_entropy),
        'turn_stats': turn_stats,
    }


def find_diverse_trajectories(results: List[Dict], n_success: int = 5, n_fail: int = 3, 
                               min_valid_turns: int = 3) -> Dict:
    """找到多样化的成功和失败轨迹
    
    Args:
        min_valid_turns: 最少需要的有效turn数（有熵数据的turn）
    """
    
    # 分类
    success_trajs = [r for r in results if r.get('success', 0) == 1]
    fail_trajs = [r for r in results if r.get('success', 0) == 0]
    
    logger.info(f"Found {len(success_trajs)} success, {len(fail_trajs)} fail trajectories")
    
    # 按turn数量和token数量选择多样化的样本
    def get_trajectory_features(traj):
        valid_turns = sum(1 for tokens in traj.get('entropy_per_token', []) if tokens)
        num_turns = traj.get('num_turns', valid_turns)
        total_tokens = traj.get('total_tokens', 0)
        return valid_turns, num_turns, total_tokens
    
    # 过滤出数据完整的轨迹（valid_turns >= min_valid_turns）
    def is_complete(traj):
        valid_turns, num_turns, _ = get_trajectory_features(traj)
        return valid_turns >= min_valid_turns
    
    complete_success = [t for t in success_trajs if is_complete(t)]
    complete_fail = [t for t in fail_trajs if is_complete(t)]
    
    logger.info(f"Complete trajectories (>={min_valid_turns} valid turns): {len(complete_success)} success, {len(complete_fail)} fail")
    
    # 如果完整轨迹不够，降低阈值
    if len(complete_success) < n_success:
        logger.warning(f"Not enough complete success trajectories, using all available")
        complete_success = success_trajs
    if len(complete_fail) < n_fail:
        logger.warning(f"Not enough complete fail trajectories, using all available")
        complete_fail = fail_trajs
    
    # 成功轨迹：选择不同valid_turn数量的（避免重复）
    success_by_turns = {}
    for traj in complete_success:
        valid_turns, num_turns, tokens = get_trajectory_features(traj)
        if valid_turns not in success_by_turns:
            success_by_turns[valid_turns] = []
        success_by_turns[valid_turns].append((traj, tokens))
    
    selected_success = []
    selected_ids = set()
    # 优先选择 turn 数多的
    for turns in sorted(success_by_turns.keys(), reverse=True):
        trajs = success_by_turns[turns]
        # 按 token 数排序，选择不同的样本
        trajs_sorted = sorted(trajs, key=lambda x: x[1])
        for traj, tokens in trajs_sorted:
            item_id = traj.get('item_id')
            if item_id not in selected_ids:
                selected_success.append(traj)
                selected_ids.add(item_id)
                break
        if len(selected_success) >= n_success:
            break
    
    # 失败轨迹：选择不同特征的（避免重复）
    fail_by_turns = {}
    for traj in complete_fail:
        valid_turns, num_turns, tokens = get_trajectory_features(traj)
        key = (valid_turns, tokens // 3000)
        if key not in fail_by_turns:
            fail_by_turns[key] = []
        fail_by_turns[key].append(traj)
    
    selected_fail = []
    selected_fail_ids = set()
    for key in sorted(fail_by_turns.keys(), reverse=True):
        for traj in fail_by_turns[key]:
            item_id = traj.get('item_id')
            if item_id not in selected_fail_ids:
                selected_fail.append(traj)
                selected_fail_ids.add(item_id)
                break
        if len(selected_fail) >= n_fail:
            break
    
    return {
        'success': selected_success,
        'fail': selected_fail,
    }


# ========== Visualization ==========

def segment_and_average(entropies: List[float], n_segments: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """将熵值分成 n_segments 个段，每段取平均"""
    if len(entropies) == 0:
        return np.array([]), np.array([])
    
    entropies = np.array(entropies)
    n = len(entropies)
    
    if n <= n_segments:
        x = np.linspace(0, 1, n)
        return x, entropies
    
    # 分段平均
    segment_size = n / n_segments
    x = np.linspace(0, 1, n_segments)
    y = np.zeros(n_segments)
    
    for i in range(n_segments):
        start = int(i * segment_size)
        end = int((i + 1) * segment_size)
        if end > n:
            end = n
        if start < end:
            y[i] = np.mean(entropies[start:end])
    
    return x, y


def find_peaks_and_valleys(y: np.ndarray, n_top: int = 3) -> Tuple[List[int], List[int]]:
    """找到 top-n peaks 和 top-n valleys（按 prominence 排序）"""
    try:
        from scipy.signal import find_peaks
    except ImportError:
        return [], []
    
    if len(y) < 3:
        return [], []
    
    # 找 peaks（prominence 越大越重要）
    peaks, peak_props = find_peaks(y, prominence=0.01)
    if len(peaks) > 0 and 'prominences' in peak_props:
        # 按 prominence 降序排序
        sorted_peak_idx = np.argsort(peak_props['prominences'])[::-1]
        peaks = peaks[sorted_peak_idx[:n_top]]
    else:
        peaks = peaks[:n_top]
    
    # 找 valleys（对 -y 找 peaks）
    valleys, valley_props = find_peaks(-y, prominence=0.01)
    if len(valleys) > 0 and 'prominences' in valley_props:
        sorted_valley_idx = np.argsort(valley_props['prominences'])[::-1]
        valleys = valleys[sorted_valley_idx[:n_top]]
    else:
        valleys = valleys[:n_top]
    
    return list(peaks), list(valleys)


def plot_trajectory_entropies(trajectories: Dict, output_dir: str, n_segments: int = 200):
    """绘制轨迹熵可视化图（分段平均 + Peak/Valley 检测）"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available")
        return
    
    # 合并成功和失败轨迹
    all_trajs = []
    for traj in trajectories.get('success', []):
        traj['_type'] = 'success'
        all_trajs.append(traj)
    for traj in trajectories.get('fail', []):
        traj['_type'] = 'fail'
        all_trajs.append(traj)
    
    n_trajs = len(all_trajs)
    n_cols = 3
    n_rows = (n_trajs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 3.5*n_rows))
    if n_trajs == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # 设置总标题
    fig.suptitle(f"Segmented Entropy Curves ({n_segments} segments) — Red ▼=top-3 peaks, Blue ▲=top-3 valleys (by prominence)", 
                 fontsize=11, y=1.02)
    
    for idx, traj in enumerate(all_trajs):
        ax = axes[idx]
        
        entropy_per_token = traj.get('entropy_per_token', [])
        
        # 展平所有 turn 的熵值
        all_entropies = []
        for tokens in entropy_per_token:
            if tokens:
                all_entropies.extend(tokens)
        
        if not all_entropies:
            ax.set_visible(False)
            continue
        
        # 分段平均
        x, y = segment_and_average(all_entropies, n_segments)
        
        # 绘制折线
        ax.plot(x, y, color='black', linewidth=0.8, alpha=0.9)
        
        # 找 peaks 和 valleys
        peaks, valleys = find_peaks_and_valleys(y, n_top=3)
        
        # 标记 peaks (红色倒三角)
        for p in peaks:
            ax.plot(x[p], y[p], 'rv', markersize=8, markeredgecolor='darkred', markeredgewidth=0.5)
        
        # 标记 valleys (蓝色正三角)
        for v in valleys:
            ax.plot(x[v], y[v], 'b^', markersize=8, markeredgecolor='darkblue', markeredgewidth=0.5)
        
        # 添加 turn 边界虚线（可选，较淡）
        turn_lengths = traj.get('turn_lengths', [])
        total_tokens = sum(turn_lengths)
        if total_tokens > 0:
            cumsum = 0
            for tl in turn_lengths[:-1]:
                if tl > 0:
                    cumsum += tl
                    norm_pos = cumsum / total_tokens
                    ax.axvline(x=norm_pos, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
        
        # 统计 turn 数
        valid_turns = sum(1 for tokens in entropy_per_token if tokens)
        num_turns = traj.get('num_turns', valid_turns)
        
        # 简洁标题（格式与师兄一致）
        status = 'OK' if traj.get('success') == 1 else 'FAIL'
        # 显示 num_turns，如果有数据缺失则加括号标注有效数
        if valid_turns < num_turns:
            turn_str = f"{num_turns}t({valid_turns}v)"
        else:
            turn_str = f"{num_turns}t"
        title = f"{traj.get('item_id')} | {status} | {turn_str} | P{len(peaks)}V{len(valleys)}"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Position (norm)", fontsize=9)
        ax.set_ylabel("Avg Entropy", fontsize=9)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.tick_params(labelsize=8)
    
    # 隐藏多余的subplot
    for idx in range(len(all_trajs), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "trajectory_entropy_profiles.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Plot saved: {output_path}")
    return output_path


def plot_trajectory_entropies_detailed(trajectories: Dict, output_dir: str):
    """绘制原始详细的轨迹熵可视化图（保留原来的功能）"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available")
        return
    
    # 合并成功和失败轨迹
    all_trajs = []
    for traj in trajectories.get('success', []):
        traj['_type'] = 'success'
        all_trajs.append(traj)
    for traj in trajectories.get('fail', []):
        traj['_type'] = 'fail'
        all_trajs.append(traj)
    
    n_trajs = len(all_trajs)
    n_cols = 3
    n_rows = (n_trajs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    if n_trajs == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, traj in enumerate(all_trajs):
        ax = axes[idx]
        
        entropy_per_token = traj.get('entropy_per_token', [])
        
        # 绘制每个turn的熵
        colors = plt.cm.viridis(np.linspace(0, 1, len([e for e in entropy_per_token if e])))
        color_idx = 0
        
        cumulative_tokens = 0
        for turn_idx, tokens in enumerate(entropy_per_token):
            if not tokens:
                continue
            
            x_positions = np.arange(cumulative_tokens, cumulative_tokens + len(tokens))
            ax.plot(x_positions, tokens, color=colors[color_idx], alpha=0.7, linewidth=0.5)
            color_idx += 1
            cumulative_tokens += len(tokens)
        
        # 标记turn边界
        turn_lengths = traj.get('turn_lengths', [])
        cumsum = [0]
        for tl in turn_lengths:
            cumsum.append(cumsum[-1] + tl)
        
        for cs in cumsum[1:-1]:
            ax.axvline(x=cs, color='red', linestyle='--', alpha=0.3, linewidth=0.5)
        
        # 标题
        traj_type = '✓ Success' if traj.get('success') == 1 else '✗ Fail'
        ax.set_title(f"{traj.get('item_id')}\n{traj_type} | {traj.get('num_turns')} turns | {traj.get('total_tokens')} tokens")
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Entropy")
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的subplot
    for idx in range(len(all_trajs), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "trajectory_entropy_profiles_detailed.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Plot saved: {output_path}")
    return output_path


def plot_turn_level_comparison(trajectories: Dict, output_dir: str):
    """绘制turn级别的成功vs失败对比"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available")
        return
    
    # 收集所有turn的统计 - 从原始轨迹中获取turn_stats
    success_turn_means = []
    fail_turn_means = []
    success_turn_maxs = []
    fail_turn_maxs = []
    
    for traj in trajectories.get('success', []):
        summary = analyze_trajectory_summary(traj)
        for turn_stat in summary['turn_stats']:
            success_turn_means.append(turn_stat['mean_entropy'])
            success_turn_maxs.append(turn_stat['max_entropy'])
    
    for traj in trajectories.get('fail', []):
        summary = analyze_trajectory_summary(traj)
        for turn_stat in summary['turn_stats']:
            fail_turn_means.append(turn_stat['mean_entropy'])
            fail_turn_maxs.append(turn_stat['max_entropy'])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 均值对比
    ax = axes[0]
    data_means = [success_turn_means, fail_turn_means]
    bp = ax.boxplot(data_means, labels=['Success', 'Fail'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax.set_ylabel("Mean Entropy per Turn")
    ax.set_title("Turn Mean Entropy: Success vs Fail")
    ax.grid(True, alpha=0.3)
    
    # 最大值对比
    ax = axes[1]
    data_maxs = [success_turn_maxs, fail_turn_maxs]
    bp = ax.boxplot(data_maxs, labels=['Success', 'Fail'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax.set_ylabel("Max Entropy per Turn")
    ax.set_title("Turn Max Entropy: Success vs Fail")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "success_vs_fail_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Plot saved: {output_path}")
    
    # 打印统计
    print("\n" + "=" * 60)
    print("Success vs Fail Entropy Comparison")
    print("=" * 60)
    print(f"\n[Turn Mean Entropy]")
    print(f"  Success: mean={np.mean(success_turn_means):.4f}, std={np.std(success_turn_means):.4f}, n={len(success_turn_means)}")
    print(f"  Fail:    mean={np.mean(fail_turn_means):.4f}, std={np.std(fail_turn_means):.4f}, n={len(fail_turn_means)}")
    print(f"\n[Turn Max Entropy]")
    print(f"  Success: mean={np.mean(success_turn_maxs):.4f}, std={np.std(success_turn_maxs):.4f}")
    print(f"  Fail:    mean={np.mean(fail_turn_maxs):.4f}, std={np.std(fail_turn_maxs):.4f}")
    print("=" * 60)


def generate_trajectory_report(trajectories: Dict, output_dir: str) -> str:
    """生成详细的轨迹分析报告"""
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("Trajectory Entropy Analysis Report")
    report_lines.append("=" * 80)
    
    # 成功轨迹
    report_lines.append("\n## Success Trajectories\n")
    for i, traj in enumerate(trajectories.get('success', [])):
        summary = analyze_trajectory_summary(traj)
        report_lines.append(f"### {i+1}. {summary['item_id']}")
        report_lines.append(f"   - Success: {summary['success']}, Reward: {summary['reward']}")
        report_lines.append(f"   - Total Turns: {summary['num_turns']}, Valid Turns: {summary['valid_turns']}")
        report_lines.append(f"   - Total Tokens: {summary['total_tokens']}")
        report_lines.append(f"   - Overall Entropy: mean={summary['mean_entropy']:.4f}, max={summary['max_entropy']:.4f}")
        report_lines.append(f"   - Turn-level stats:")
        for ts in summary['turn_stats']:
            report_lines.append(f"     Turn {ts['turn_idx']+1}: {ts['token_count']} tokens, mean={ts['mean_entropy']:.4f}, max={ts['max_entropy']:.4f}")
        report_lines.append("")
    
    # 失败轨迹
    report_lines.append("\n## Fail Trajectories\n")
    for i, traj in enumerate(trajectories.get('fail', [])):
        summary = analyze_trajectory_summary(traj)
        report_lines.append(f"### {i+1}. {summary['item_id']}")
        report_lines.append(f"   - Success: {summary['success']}, Reward: {summary['reward']}")
        report_lines.append(f"   - Total Turns: {summary['num_turns']}, Valid Turns: {summary['valid_turns']}")
        report_lines.append(f"   - Total Tokens: {summary['total_tokens']}")
        report_lines.append(f"   - Overall Entropy: mean={summary['mean_entropy']:.4f}, max={summary['max_entropy']:.4f}")
        report_lines.append(f"   - Turn-level stats:")
        for ts in summary['turn_stats']:
            report_lines.append(f"     Turn {ts['turn_idx']+1}: {ts['token_count']} tokens, mean={ts['mean_entropy']:.4f}, max={ts['max_entropy']:.4f}")
        report_lines.append("")
    
    report_content = "\n".join(report_lines)
    
    # 保存报告
    report_path = os.path.join(output_dir, "trajectory_analysis_report.md")
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    logger.info(f"Report saved: {report_path}")
    return report_path


def export_trajectory_tokens(trajectories: Dict, output_dir: str):
    """导出每个轨迹的详细逐token熵数据为JSON"""
    
    for traj_type in ['success', 'fail']:
        trajs = trajectories.get(traj_type, [])
        
        for traj in trajs:
            item_id = traj.get('item_id')
            entropy_per_token = traj.get('entropy_per_token', [])
            turn_lengths = traj.get('turn_lengths', [])
            
            # 展平所有token的熵
            all_entropies = []
            turn_info = []
            cumulative = 0
            for turn_idx, tokens in enumerate(entropy_per_token):
                if tokens:
                    for token_idx, entropy in enumerate(tokens):
                        all_entropies.append({
                            'global_pos': cumulative + token_idx,
                            'turn': turn_idx,
                            'pos_in_turn': token_idx,
                            'entropy': entropy,
                        })
                    turn_info.append({
                        'turn': turn_idx,
                        'token_count': len(tokens),
                        'start_pos': cumulative,
                        'mean_entropy': float(np.mean(tokens)),
                        'max_entropy': float(np.max(tokens)),
                    })
                    cumulative += len(tokens)
            
            output_data = {
                'item_id': item_id,
                'type': traj_type,
                'success': traj.get('success'),
                'reward': traj.get('reward'),
                'total_tokens': cumulative,
                'turn_info': turn_info,
                'token_entropies': all_entropies,
            }
            
            output_path = os.path.join(output_dir, f"{item_id}_entropy.json")
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported: {output_path}")


# ========== Main ==========

def main():
    parser = argparse.ArgumentParser(
        description="Analyze specific trajectories from entropy dataset"
    )
    parser.add_argument("--input_file", type=str,
                        default="/Data/wyh/datasets/Verl-Data/outputs/entropy_offline_20260309_152233/offline_results_20260309_152233.jsonl",
                        help="Input jsonl file")
    parser.add_argument("--output_dir", type=str,
                        default="/Data/wyh/datasets/Verl-Data/outputs/trajectory_entropy_analysis",
                        help="Output directory")
    parser.add_argument("--n_success", type=int, default=6,
                        help="Number of success trajectories to analyze")
    parser.add_argument("--n_fail", type=int, default=5,
                        help="Number of fail trajectories to analyze")
    parser.add_argument("--specific_id", type=str, default=None,
                        help="Analyze specific trajectory by item_id")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    logger.info(f"Loading data from: {args.input_file}")
    results = load_trajectories(args.input_file)
    
    if args.specific_id:
        # 分析特定轨迹
        traj, idx = get_trajectory_by_id(results, args.specific_id)
        if traj:
            trajectories = {'success': [traj] if traj.get('success') == 1 else [], 
                           'fail': [traj] if traj.get('success') == 0 else []}
            logger.info(f"Analyzing specific trajectory: {args.specific_id}")
        else:
            logger.error(f"Trajectory {args.specific_id} not found")
            return
    else:
        # 选择多样化的轨迹
        trajectories = find_diverse_trajectories(results, args.n_success, args.n_fail)
    
    # 生成分析
    print("\n" + "=" * 60)
    print("Selected Trajectories")
    print("=" * 60)
    print(f"Success trajectories: {[t.get('item_id') for t in trajectories.get('success', [])]}")
    print(f"Fail trajectories: {[t.get('item_id') for t in trajectories.get('fail', [])]}")
    
    # 1. 生成报告
    report_path = generate_trajectory_report(trajectories, args.output_dir)
    
    # 2. 绘制轨迹熵图
    plot_path = plot_trajectory_entropies(trajectories, args.output_dir)
    
    # 3. 成功vs失败对比
    plot_turn_level_comparison(trajectories, args.output_dir)
    
    # 4. 导出详细数据
    export_trajectory_tokens(trajectories, args.output_dir)
    
    print(f"\nAnalysis complete!")
    print(f"  Report: {report_path}")
    print(f"  Plots: {args.output_dir}")
    print(f"  Data: {args.output_dir}/*.json")


if __name__ == "__main__":
    main()
