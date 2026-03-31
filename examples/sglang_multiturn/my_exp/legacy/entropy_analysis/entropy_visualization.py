#!/usr/bin/env python3
"""
Entropy Prefix Cut Point Visualization
=====================================
可视化三种 prefix cut point 策略的分析结果

图表：
1. 单轨迹 overlay 图 - 展示 entropy 曲线和 cut points
2. 全局 cut point 分布图 - 展示三种策略的分布
3. Success / Failure 对比图 - 比较成功和失败轨迹的 cut points
4. 长度分桶后的策略对比图
"""

import os
import json
import argparse
from typing import Dict, List, Any, Tuple
import numpy as np

# 尝试导入 matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available, skipping visualization")


# =============================================================================
# 数据加载
# =============================================================================

def load_processed_results(jsonl_path: str) -> List[Dict]:
    """加载处理后的结果"""
    results = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    print(f"Loaded {len(results)} processed trajectories from {jsonl_path}")
    return results


# =============================================================================
# 可视化 1: 单轨迹 Overlay 图
# =============================================================================

def plot_single_trajectory_overlay(
    traj: Dict,
    output_path: str,
    traj_idx: int = 0
):
    """
    绘制单条轨迹的 entropy 曲线和 cut points
    
    颜色约定：
    - Assistant: 蓝色系 (blue, cyan)
    - User: 橙色系 (orange, red)
    - 策略: 不同 marker 和线型区分
    
    X 轴使用统一的相对位置 q，确保 assistant 和 user 在同一坐标系下对齐
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 获取数据
    entropy_A = traj.get('entropy_C_assistant', [])
    entropy_user = traj.get('entropy_C_user', [])
    
    n_turns = len(entropy_A)
    n_user_turns = len(entropy_user)
    
    if n_turns == 0:
        print(f"Warning: No entropy data for trajectory {traj.get('item_id', traj_idx)}")
        return
    
    # X 轴: 使用相对位置 q (0 到 1)
    # Assistant: 0 到 1 均匀分布
    if n_turns > 1:
        rel_positions_A = np.linspace(0, 1, n_turns)
    else:
        rel_positions_A = [0.5]
    
    # User: 0 到 1 均匀分布
    if n_user_turns > 1:
        rel_positions_U = np.linspace(0, 1, n_user_turns)
    else:
        rel_positions_U = [0.5]
    
    # 绘制 Assistant Entropy (蓝色) - 使用相对位置
    ax.plot(rel_positions_A, entropy_A, '-o', linewidth=2, markersize=8, 
            label='Assistant Entropy (C)', color='steelblue', alpha=0.8)
    
    # 绘制 User Entropy (橙色) - 使用相对位置
    if entropy_user:
        ax.plot(rel_positions_U, entropy_user, '-s', linewidth=2, markersize=6,
                label='User Entropy (C)', color='coral', alpha=0.6)
    
    # 获取 cut points - 使用相对位置
    topk_cuts = traj.get('cut_topk_peaks', [])
    topk_rel = traj.get('cut_topk_peaks_relative', [])
    fixed_cuts = traj.get('cut_fixed_ratio', [])
    fixed_rel = traj.get('cut_fixed_ratio_relative', [])
    cumulative_cuts = traj.get('cut_cumulative_info', [])
    cumulative_rel = traj.get('cut_cumulative_info_relative', [])
    
    # 绘制 cut points - 使用相对位置 q
    # Top-K Peaks: 三角形
    for i, cut_idx in enumerate(topk_cuts):
        if cut_idx is not None and cut_idx < n_turns:
            q_pos = topk_rel[i] if i < len(topk_rel) else cut_idx / max(n_turns - 1, 1)
            ax.axvline(x=q_pos, color='green', linestyle='--', linewidth=2, alpha=0.7)
            ax.plot(q_pos, entropy_A[cut_idx], '^', markersize=15, 
                   color='green', label=f'Top-K Peak @ q={q_pos:.2f}')
    
    # Fixed Ratio: 方形
    for i, cut_idx in enumerate(fixed_cuts):
        if cut_idx is not None and cut_idx < n_turns:
            q_pos = fixed_rel[i] if i < len(fixed_rel) else cut_idx / max(n_turns - 1, 1)
            ax.axvline(x=q_pos, color='purple', linestyle='-.', linewidth=2, alpha=0.7)
            ax.plot(q_pos, entropy_A[cut_idx], 's', markersize=12,
                   color='purple', label=f'Fixed Ratio @ q={q_pos:.2f}')
    
    # Cumulative Info: 菱形
    for i, cut_idx in enumerate(cumulative_cuts):
        if cut_idx is not None and cut_idx < n_turns:
            q_pos = cumulative_rel[i] if i < len(cumulative_rel) else cut_idx / max(n_turns - 1, 1)
            ax.axvline(x=q_pos, color='red', linestyle=':', linewidth=2, alpha=0.7)
            ax.plot(q_pos, entropy_A[cut_idx], 'D', markersize=12,
                   color='red', label=f'Cumulative Info @ q={q_pos:.2f}')
    
    # 设置图表
    ax.set_xlabel('Relative Position q', fontsize=12)
    ax.set_ylabel('Token Entropy', fontsize=12)
    ax.set_title(f"Trajectory {traj.get('item_id', traj_idx)} - Entropy & Cut Points\n"
                f"(Success: {traj.get('success', 0)}, "
                f"Assistant Turns: {n_turns}, User Turns: {n_user_turns}, "
                f"Total Tokens: {traj.get('total_tokens', 0)}) [x-axis: relative position q]",
                fontsize=13)
    
    # 设置图例 - 避免重复
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=9)
    
    ax.grid(True, alpha=0.3)
    
    # 设置 x 轴范围
    ax.set_xlim(-0.02, 1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved single trajectory plot to {output_path}")


# =============================================================================
# 可视化 2: 全局 Cut Point 分布图
# =============================================================================

def plot_global_cut_distribution(
    results: List[Dict],
    output_path: str
):
    """
    绘制全局 cut point 分布
    展示三种策略在全数据上的 turn index、relative position、cumulative token 分布
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    
    # 收集数据
    strategies = [
        ('Top-K Peaks', 'cut_topk_peaks', 'cut_topk_peaks_relative', 'cut_topk_peaks_tokens'),
        ('Fixed Ratio', 'cut_fixed_ratio', 'cut_fixed_ratio_relative', 'cut_fixed_ratio_tokens'),
        ('Cumulative Info', 'cut_cumulative_info', 'cut_cumulative_info_relative', 'cut_cumulative_info_tokens'),
    ]
    
    colors = ['steelblue', 'coral', 'seagreen']
    
    # 第一行: Turn Index 分布
    for idx, (name, key, _, _) in enumerate(strategies):
        ax = axes[0, idx]
        
        all_cuts = []
        for r in results:
            cuts = r.get(key, [])
            for c in cuts:
                if c is not None:
                    all_cuts.append(c)
        
        if all_cuts:
            ax.hist(all_cuts, bins=20, color=colors[idx], alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(all_cuts), color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {np.mean(all_cuts):.2f}')
            ax.axvline(np.median(all_cuts), color='orange', linestyle=':',
                      linewidth=2, label=f'Median: {np.median(all_cuts):.2f}')
        
        ax.set_xlabel('Turn Index', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'{name}\nTurn Index Distribution', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # 第二行: Relative Position 分布
    for idx, (name, _, key_rel, _) in enumerate(strategies):
        ax = axes[1, idx]
        
        all_rel = []
        for r in results:
            rels = r.get(key_rel, [])
            for rel in rels:
                if rel is not None:
                    all_rel.append(rel)
        
        if all_rel:
            ax.hist(all_rel, bins=20, color=colors[idx], alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(all_rel), color='red', linestyle='--',
                      linewidth=2, label=f'Mean: {np.mean(all_rel):.2f}')
        
        ax.set_xlabel('Relative Position q', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'{name}\nRelative Position Distribution', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # 第三行: Cumulative Token Position 分布
    for idx, (name, _, _, key_tokens) in enumerate(strategies):
        ax = axes[2, idx]
        
        all_tokens = []
        for r in results:
            tokens = r.get(key_tokens, [])
            for t in tokens:
                if t is not None:
                    all_tokens.append(t)
        
        if all_tokens:
            ax.hist(all_tokens, bins=30, color=colors[idx], alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(all_tokens), color='red', linestyle='--',
                      linewidth=2, label=f'Mean: {np.mean(all_tokens):.0f}')
            ax.axvline(np.median(all_tokens), color='orange', linestyle=':',
                      linewidth=2, label=f'Median: {np.median(all_tokens):.0f}')
        
        ax.set_xlabel('Cumulative Token Position', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'{name}\nCumulative Token Distribution', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Global Cut Point Distribution by Strategy', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved global distribution plot to {output_path}")


# =============================================================================
# 可视化 3: Success / Failure 对比图
# =============================================================================

def plot_success_failure_comparison(
    results: List[Dict],
    output_path: str
):
    """
    比较三种策略在 success 和 failure 两组中的 cut point 分布差异
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    strategies = [
        ('Top-K Peaks', 'cut_topk_peaks'),
        ('Fixed Ratio', 'cut_fixed_ratio'),
        ('Cumulative Info', 'cut_cumulative_info'),
    ]
    
    colors = {'success': 'green', 'failure': 'red'}
    
    for idx, (name, key) in enumerate(strategies):
        ax = axes[0, idx]
        
        # 收集 success 和 failure 的 cut points
        success_cuts = []
        failure_cuts = []
        
        for r in results:
            cuts = r.get(key, [])
            is_success = r.get('success', 0) == 1
            
            for c in cuts:
                if c is not None:
                    if is_success:
                        success_cuts.append(c)
                    else:
                        failure_cuts.append(c)
        
        # 绘制箱线图
        data_to_plot = []
        labels = []
        if success_cuts:
            data_to_plot.append(success_cuts)
            labels.append(f'Success (n={len(success_cuts)})')
        if failure_cuts:
            data_to_plot.append(failure_cuts)
            labels.append(f'Failure (n={len(failure_cuts)})')
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            colors_box = [colors['success'], colors['failure']][:len(data_to_plot)]
            for patch, color in zip(bp['boxes'], colors_box):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
        
        ax.set_xlabel('Group', fontsize=11)
        ax.set_ylabel('Cut Point (Turn Index)', fontsize=11)
        ax.set_title(f'{name}\nSuccess vs Failure (Turn Index)', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 第二行: 相对位置对比
        ax2 = axes[1, idx]
        
        # 使用相对位置
        key_rel = key + '_relative'
        
        success_rel = []
        failure_rel = []
        
        for r in results:
            rels = r.get(key_rel, [])
            is_success = r.get('success', 0) == 1
            
            for rel in rels:
                if rel is not None:
                    if is_success:
                        success_rel.append(rel)
                    else:
                        failure_rel.append(rel)
        
        data_to_plot2 = []
        labels2 = []
        if success_rel:
            data_to_plot2.append(success_rel)
            labels2.append(f'Success (n={len(success_rel)})')
        if failure_rel:
            data_to_plot2.append(failure_rel)
            labels2.append(f'Failure (n={len(failure_rel)})')
        
        if data_to_plot2:
            bp2 = ax2.boxplot(data_to_plot2, labels=labels2, patch_artist=True)
            colors_box2 = [colors['success'], colors['failure']][:len(data_to_plot2)]
            for patch, color in zip(bp2['boxes'], colors_box2):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
        
        ax2.set_xlabel('Group', fontsize=11)
        ax2.set_ylabel('Cut Point (Relative Position)', fontsize=11)
        ax2.set_title(f'{name}\nSuccess vs Failure (Relative Position)', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Success vs Failure Cut Point Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved success/failure comparison plot to {output_path}")


# =============================================================================
# 可视化 4: 长度分桶后的策略对比图
# =============================================================================

def plot_length_binned_comparison(
    results: List[Dict],
    output_path: str
):
    """
    按 trajectory total length 分桶后，比较三种策略的 cut point 分布
    控制长度偏置
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 定义长度分桶
    def get_length_bin(total_tokens: int) -> str:
        if total_tokens < 500:
            return '0-500'
        elif total_tokens < 1000:
            return '500-1k'
        elif total_tokens < 2000:
            return '1k-2k'
        elif total_tokens < 3000:
            return '2k-3k'
        else:
            return '3k+'
    
    # 按长度分桶
    bins = ['0-500', '500-1k', '1k-2k', '2k-3k', '3k+']
    binned_data = {bin_name: {'topk': [], 'fixed': [], 'cumulative': []} for bin_name in bins}
    
    for r in results:
        total_tokens = r.get('total_tokens', 0)
        bin_name = get_length_bin(total_tokens)
        
        # 收集各策略的 cut points
        for cut in r.get('cut_topk_peaks', []):
            if cut is not None:
                binned_data[bin_name]['topk'].append(cut)
        for cut in r.get('cut_fixed_ratio', []):
            if cut is not None:
                binned_data[bin_name]['fixed'].append(cut)
        for cut in r.get('cut_cumulative_info', []):
            if cut is not None:
                binned_data[bin_name]['cumulative'].append(cut)
    
    # 绘制每种策略在不同长度分桶下的分布
    strategies = [('topk', 'Top-K Peaks'), ('fixed', 'Fixed Ratio'), ('cumulative', 'Cumulative Info')]
    colors = ['steelblue', 'coral', 'seagreen']
    
    # 第一行: 各策略在不同分桶下的均值
    ax = axes[0, 0]
    x = np.arange(len(bins))
    width = 0.25
    
    for idx, (key, name) in enumerate(strategies):
        means = []
        stds = []
        counts = []
        for bin_name in bins:
            data = binned_data[bin_name][key]
            if data:
                means.append(np.mean(data))
                stds.append(np.std(data))
                counts.append(len(data))
            else:
                means.append(0)
                stds.append(0)
                counts.append(0)
        
        ax.bar(x + idx * width, means, width, label=name, color=colors[idx], alpha=0.7)
    
    ax.set_xlabel('Trajectory Length Bin', fontsize=11)
    ax.set_ylabel('Mean Cut Point (Turn Index)', fontsize=11)
    ax.set_title('Mean Cut Point by Length Bin', fontsize=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels(bins)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 样本数量分布
    ax2 = axes[0, 1]
    for idx, (key, name) in enumerate(strategies):
        counts = [len(binned_data[bin_name][key]) for bin_name in bins]
        ax2.bar(x + idx * width, counts, width, label=name, color=colors[idx], alpha=0.7)
    
    ax2.set_xlabel('Trajectory Length Bin', fontsize=11)
    ax2.set_ylabel('Sample Count', fontsize=11)
    ax2.set_title('Sample Count by Length Bin', fontsize=12)
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(bins)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 热力图: 相对位置 vs 长度分桶
    ax3 = axes[1, 0]
    
    # 构建热力图数据
    heatmap_data = []
    for bin_name in bins:
        row = []
        for key, _ in strategies:
            rels = []
            for r in results:
                if get_length_bin(r.get('total_tokens', 0)) == bin_name:
                    key_rel = key + '_' if key == 'topk' else key + '_'
                    if key == 'topk':
                        key_rel = 'cut_topk_peaks_relative'
                    elif key == 'fixed':
                        key_rel = 'cut_fixed_ratio_relative'
                    else:
                        key_rel = 'cut_cumulative_info_relative'
                    
                    for rel in r.get(key_rel, []):
                        if rel is not None:
                            rels.append(rel)
            if rels:
                row.append(np.mean(rels))
            else:
                row.append(0)
        heatmap_data.append(row)
    
    heatmap_array = np.array(heatmap_data)
    im = ax3.imshow(heatmap_array.T, cmap='YlOrRd', aspect='auto')
    
    ax3.set_xticks(np.arange(len(bins)))
    ax3.set_yticks(np.arange(3))
    ax3.set_xticklabels(bins)
    ax3.set_yticklabels(['Top-K', 'Fixed Ratio', 'Cumulative'])
    ax3.set_xlabel('Trajectory Length Bin', fontsize=11)
    ax3.set_ylabel('Strategy', fontsize=11)
    ax3.set_title('Mean Relative Position by Length Bin', fontsize=12)
    
    # 添加数值标注
    for i in range(3):
        for j in range(len(bins)):
            text = ax3.text(j, i, f'{heatmap_array[j, i]:.2f}',
                           ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im, ax=ax3)
    
    # 相对位置分布对比
    ax4 = axes[1, 1]
    
    all_data_by_strategy = {key: [] for key, _ in strategies}
    
    for r in results:
        for key, _ in strategies:
            if key == 'topk':
                key_rel = 'cut_topk_peaks_relative'
            elif key == 'fixed':
                key_rel = 'cut_fixed_ratio_relative'
            else:
                key_rel = 'cut_cumulative_info_relative'
            
            for rel in r.get(key_rel, []):
                if rel is not None:
                    all_data_by_strategy[key].append(rel)
    
    # 绘制小提琴图
    data_violin = [all_data_by_strategy[key] for key, _ in strategies]
    parts = ax4.violinplot(data_violin, positions=range(3), showmeans=True, showmedians=True)
    
    for idx, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[idx])
        pc.set_alpha(0.6)
    
    ax4.set_xticks(range(3))
    ax4.set_xticklabels(['Top-K', 'Fixed Ratio', 'Cumulative'])
    ax4.set_xlabel('Strategy', fontsize=11)
    ax4.set_ylabel('Relative Position q', fontsize=11)
    ax4.set_title('Overall Relative Position Distribution', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Cut Point Analysis by Trajectory Length', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved length-binned comparison plot to {output_path}")


# =============================================================================
# 可视化 5: 多轨迹 Overlay 图（批量）
# =============================================================================

def plot_multiple_trajectories(
    results: List[Dict],
    output_dir: str,
    n_samples: int = 20
):
    """
    批量绘制多条轨迹的 overlay 图
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 采样
    n_total = len(results)
    if n_samples > 0 and n_samples < n_total:
        indices = np.random.choice(n_total, n_samples, replace=False)
    else:
        indices = list(range(n_total))
    
    for idx in indices:
        traj = results[idx]
        output_path = os.path.join(output_dir, f"traj_{traj.get('item_id', idx)}.png")
        try:
            plot_single_trajectory_overlay(traj, output_path, idx)
        except Exception as e:
            print(f"Error plotting trajectory {idx}: {e}")


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Entropy Prefix Cut Point Visualization"
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='/Data/wyh/datasets/Verl-Data/outputs/entropy_analysis/processed_trajectories.jsonl',
        help='Input processed results jsonl file'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='/Data/wyh/datasets/Verl-Data/outputs/entropy_analysis/visualizations',
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--plot_samples', '-n',
        type=int,
        default=20,
        help='Number of sample trajectories to plot individually'
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Random seed for sampling'
    )
    
    args = parser.parse_args()
    
    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib is required for visualization")
        return
    
    # 设置随机种子
    np.random.seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    print(f"Loading data from {args.input}...")
    results = load_processed_results(args.input)
    
    # 1. 全局 cut point 分布图
    plot_global_cut_distribution(
        results,
        os.path.join(args.output_dir, 'global_cut_distribution.png')
    )
    
    # 2. Success / Failure 对比图
    plot_success_failure_comparison(
        results,
        os.path.join(args.output_dir, 'success_failure_comparison.png')
    )
    
    # 3. 长度分桶后的策略对比图
    plot_length_binned_comparison(
        results,
        os.path.join(args.output_dir, 'length_binned_comparison.png')
    )
    
    # 4. 批量单轨迹图
    if args.plot_samples > 0:
        plot_multiple_trajectories(
            results,
            os.path.join(args.output_dir, 'sample_trajectories'),
            n_samples=args.plot_samples
        )
    
    print("\n" + "=" * 60)
    print("Visualization Complete!")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print("Generated plots:")
    print("  - global_cut_distribution.png")
    print("  - success_failure_comparison.png")
    print("  - length_binned_comparison.png")
    if args.plot_samples > 0:
        print(f"  - sample_trajectories/ ({args.plot_samples} trajectories)")


if __name__ == '__main__':
    main()
