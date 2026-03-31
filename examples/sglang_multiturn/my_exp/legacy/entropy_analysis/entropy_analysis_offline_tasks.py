#!/usr/bin/env python3
"""
Entropy Analysis for SciWorld and BabyAI
=========================================
分析 sciworld 和 babyai 任务上的 token entropy 数据

生成的图表：
1. 全局平均 entropy 曲线 (按 relative position 对齐)
2. 按长度分桶的平均 entropy 曲线
3. Success / Failure 对比图
4. 统计摘要
"""

import json
import os
import argparse
from typing import List, Dict, Tuple
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available, skipping visualization")


def load_trajectories(jsonl_path: str) -> List[Dict]:
    """加载轨迹数据"""
    trajectories = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                trajectories.append(json.loads(line))
    print(f"Loaded {len(trajectories)} trajectories from {jsonl_path}")
    
    # 检查任务类型并设置success字段
    # BabyAI: 使用reward字段，成功标准是环境反馈"任务完成"
    # 检查是否有success字段，如果没有则根据reward判断
    has_success_field = any('success' in t for t in trajectories[:10])
    has_reward_field = any('reward' in t for t in trajectories[:10])
    
    task_type = None
    if has_success_field:
        # 检查success字段是否有效（有0和1）
        success_values = set()
        for t in trajectories[:100]:
            if 'success' in t:
                success_values.add(t['success'])
        if len(success_values) > 1:
            task_type = 'sciworld'  # 有有效的success字段
        else:
            task_type = 'babyai'  # success字段全为0或1，无效，使用reward
    elif has_reward_field:
        # 检查reward分布，判断是否是BabyAI类型
        rewards = [t.get('reward', 0) for t in trajectories[:100]]
        if any(r > 1 for r in rewards):
            task_type = 'babyai'  # BabyAI类型，reward可能大于1
        else:
            task_type = 'unknown'
    
    if task_type == 'babyai':
        print(f"  Detected BabyAI task type - using reward > 0.9 as success criterion")
        for t in trajectories:
            # BabyAI: reward > 0.9 通常表示任务完成
            t['_success'] = 1 if t.get('reward', 0) > 0.9 else 0
    else:
        print(f"  Detected {task_type or 'unknown'} task type - using success field")
        for t in trajectories:
            t['_success'] = t.get('success', 0)
    
    return trajectories


def interpolate_to_relative_positions(
    entropy_values: List[float],
    n_points: int = 20
) -> np.ndarray:
    """将 entropy 曲线插值到统一的 relative position 网格上"""
    if not entropy_values or len(entropy_values) < 2:
        return np.zeros(n_points)
    
    n = len(entropy_values)
    original_positions = np.linspace(0, 1, n)
    target_positions = np.linspace(0, 1, n_points)
    interpolated = np.interp(target_positions, original_positions, entropy_values)
    return interpolated


def plot_global_entropy_curves(
    trajectories: List[Dict],
    output_path: str,
    task_name: str
):
    """绘制全局平均 entropy 曲线"""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    success_traj = [t for t in trajectories if t.get('_success', 0) == 1]
    failure_traj = [t for t in trajectories if t.get('_success', 0) == 0]
    
    print(f"  Success: {len(success_traj)}, Failure: {len(failure_traj)}")
    
    n_points = 20
    rel_positions = np.linspace(0, 1, n_points)
    
    # Assistant entropy
    success_assistant_curves = []
    failure_assistant_curves = []
    success_user_curves = []
    failure_user_curves = []
    
    for t in success_traj:
        ent_a = t.get('entropy_C_assistant', [])
        ent_u = t.get('entropy_C_user', [])
        if ent_a:
            success_assistant_curves.append(interpolate_to_relative_positions(ent_a, n_points))
        if ent_u:
            success_user_curves.append(interpolate_to_relative_positions(ent_u, n_points))
    
    for t in failure_traj:
        ent_a = t.get('entropy_C_assistant', [])
        ent_u = t.get('entropy_C_user', [])
        if ent_a:
            failure_assistant_curves.append(interpolate_to_relative_positions(ent_a, n_points))
        if ent_u:
            failure_user_curves.append(interpolate_to_relative_positions(ent_u, n_points))
    
    success_a_mean = np.mean(success_assistant_curves, axis=0) if success_assistant_curves else np.zeros(n_points)
    failure_a_mean = np.mean(failure_assistant_curves, axis=0) if failure_assistant_curves else np.zeros(n_points)
    success_u_mean = np.mean(success_user_curves, axis=0) if success_user_curves else np.zeros(n_points)
    failure_u_mean = np.mean(failure_user_curves, axis=0) if failure_user_curves else np.zeros(n_points)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    ax = axes[0]
    ax.plot(rel_positions, success_a_mean, 'b-', linewidth=2.5, label='Success', alpha=0.9)
    ax.plot(rel_positions, failure_a_mean, 'r-', linewidth=2.5, label='Failure', alpha=0.9)
    ax.fill_between(rel_positions, success_a_mean * 0.8, success_a_mean * 1.2, alpha=0.2, color='blue')
    ax.fill_between(rel_positions, failure_a_mean * 0.8, failure_a_mean * 1.2, alpha=0.2, color='red')
    
    ax.set_xlabel('Relative Position q', fontsize=12)
    ax.set_ylabel('Entropy (C) - Assistant', fontsize=12)
    ax.set_title(f'{task_name}\nAssistant Entropy Curve (Success vs Failure)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    
    ax = axes[1]
    ax.plot(rel_positions, success_u_mean, 'b-', linewidth=2.5, label='Success', alpha=0.9)
    ax.plot(rel_positions, failure_u_mean, 'r-', linewidth=2.5, label='Failure', alpha=0.9)
    ax.fill_between(rel_positions, success_u_mean * 0.8, success_u_mean * 1.2, alpha=0.2, color='blue')
    ax.fill_between(rel_positions, failure_u_mean * 0.8, failure_u_mean * 1.2, alpha=0.2, color='red')
    
    ax.set_xlabel('Relative Position q', fontsize=12)
    ax.set_ylabel('Entropy (C) - User', fontsize=12)
    ax.set_title(f'{task_name}\nUser Entropy Curve (Success vs Failure)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_length_binned_entropy_curves(
    trajectories: List[Dict],
    output_path: str,
    task_name: str
):
    """按轨迹长度分桶后分别绘制平均 entropy 曲线"""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    buckets = [
        ('Short (<800)', lambda t: t.get('total_tokens', 0) < 800),
        ('Medium (800-1500)', lambda t: 800 <= t.get('total_tokens', 0) < 1500),
        ('Long (1500-2500)', lambda t: 1500 <= t.get('total_tokens', 0) < 2500),
        ('Very Long (>2500)', lambda t: t.get('total_tokens', 0) >= 2500),
    ]
    
    n_points = 20
    rel_positions = np.linspace(0, 1, n_points)
    
    n_buckets = len(buckets)
    fig, axes = plt.subplots(2, n_buckets, figsize=(20, 10))
    
    for col, (bucket_name, filter_fn) in enumerate(buckets):
        bucket_traj = [t for t in trajectories if filter_fn(t)]
        
        if not bucket_traj:
            continue
        
        success_traj = [t for t in bucket_traj if t.get('_success', 0) == 1]
        failure_traj = [t for t in bucket_traj if t.get('_success', 0) == 0]
        
        success_curves = []
        failure_curves = []
        
        for t in success_traj:
            ent = t.get('entropy_C_assistant', [])
            if ent:
                success_curves.append(interpolate_to_relative_positions(ent, n_points))
        
        for t in failure_traj:
            ent = t.get('entropy_C_assistant', [])
            if ent:
                failure_curves.append(interpolate_to_relative_positions(ent, n_points))
        
        ax = axes[0, col]
        
        if success_curves:
            succ_mean = np.mean(success_curves, axis=0)
            ax.plot(rel_positions, succ_mean, 'b-', linewidth=2, label=f'Success (n={len(success_curves)})')
        
        if failure_curves:
            fail_mean = np.mean(failure_curves, axis=0)
            ax.plot(rel_positions, fail_mean, 'r-', linewidth=2, label=f'Failure (n={len(failure_curves)})')
        
        ax.set_title(f'{bucket_name}\nAssistant Entropy', fontsize=11)
        ax.set_xlabel('Relative Position q', fontsize=10)
        ax.set_ylabel('Entropy (C)', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        
        ax = axes[1, col]
        
        success_user_curves = []
        failure_user_curves = []
        
        for t in success_traj:
            ent = t.get('entropy_C_user', [])
            if ent:
                success_user_curves.append(interpolate_to_relative_positions(ent, n_points))
        
        for t in failure_traj:
            ent = t.get('entropy_C_user', [])
            if ent:
                failure_user_curves.append(interpolate_to_relative_positions(ent, n_points))
        
        if success_user_curves:
            succ_u_mean = np.mean(success_user_curves, axis=0)
            ax.plot(rel_positions, succ_u_mean, 'b-', linewidth=2, label=f'Success')
        
        if failure_user_curves:
            fail_u_mean = np.mean(failure_user_curves, axis=0)
            ax.plot(rel_positions, fail_u_mean, 'r-', linewidth=2, label=f'Failure')
        
        ax.set_title(f'{bucket_name}\nUser Entropy', fontsize=11)
        ax.set_xlabel('Relative Position q', fontsize=10)
        ax.set_ylabel('Entropy (C)', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
    
    plt.suptitle(f'{task_name}\nEntropy Curves by Trajectory Length Bucket', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_detailed_trajectories(
    trajectories: List[Dict],
    output_dir: str,
    n_samples: int = 10
):
    """绘制代表性轨迹的详细分析图"""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    success_traj = [t for t in trajectories if t.get('_success', 0) == 1]
    failure_traj = [t for t in trajectories if t.get('_success', 0) == 0]
    
    # 选取样本
    def select_samples(traj_list, n):
        if len(traj_list) <= n:
            return traj_list
        traj_list_sorted = sorted(traj_list, key=lambda t: t.get('total_tokens', 0))
        indices = np.linspace(0, len(traj_list_sorted)-1, min(n, len(traj_list_sorted)))
        return [traj_list_sorted[int(i)] for i in indices]
    
    success_samples = select_samples(success_traj, n_samples // 2)
    failure_samples = select_samples(failure_traj, n_samples // 2)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i, traj in enumerate(success_samples):
        traj_id = traj.get('item_id', f'success_{i}')
        output_path = os.path.join(output_dir, f"traj_{traj_id}.png")
        try:
            plot_single_trajectory(traj, output_path, f"Success: {traj_id}")
        except Exception as e:
            print(f"  Warning: Error plotting {traj_id}: {e}")
    
    for i, traj in enumerate(failure_samples):
        traj_id = traj.get('item_id', f'failure_{i}')
        output_path = os.path.join(output_dir, f"traj_{traj_id}.png")
        try:
            plot_single_trajectory(traj, output_path, f"Failure: {traj_id}")
        except Exception as e:
            print(f"  Warning: Error plotting {traj_id}: {e}")
    
    print(f"  Saved {len(success_samples) + len(failure_samples)} trajectory plots to {output_dir}")


def plot_single_trajectory(
    traj: Dict,
    output_path: str,
    title_suffix: str
):
    """绘制单条轨迹的详细分析图"""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    entropy_A = traj.get('entropy_C_assistant', [])
    entropy_U = traj.get('entropy_C_user', [])
    
    n_turns = len(entropy_A)
    if n_turns == 0:
        return
    
    rel_positions = np.linspace(0, 1, n_turns) if n_turns > 1 else np.array([0.5])
    
    n_user = len(entropy_U)
    user_rel = np.linspace(0, 1, n_user) if n_user > 1 else np.array([0.5])
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    ax = axes[0]
    ax.plot(rel_positions, entropy_A, '-o', linewidth=2, markersize=8, 
            label='Entropy C (Assistant)', color='steelblue', alpha=0.9)
    
    ax.set_ylabel('Entropy', fontsize=12)
    ax.set_title(f"Trajectory {traj.get('item_id', 'unknown')} - Assistant\n"
                f"Success: {traj.get('_success', 0)}, Turns: {n_turns}, "
                f"Tokens: {traj.get('total_tokens', 0)} [{title_suffix}]",
                fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    
    ax = axes[1]
    if n_user > 0:
        ax.plot(user_rel, entropy_U, '-s', linewidth=2, markersize=7,
                label='Entropy C (User)', color='coral', alpha=0.9)
    
    ax.set_xlabel('Relative Position q', fontsize=12)
    ax.set_ylabel('Entropy', fontsize=12)
    ax.set_title("User Entropy", fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def compute_statistics(trajectories: List[Dict], task_name: str) -> Dict:
    """计算统计摘要"""
    success_traj = [t for t in trajectories if t.get('_success', 0) == 1]
    failure_traj = [t for t in trajectories if t.get('_success', 0) == 0]
    
    stats = {
        'task_name': task_name,
        'total_trajectories': len(trajectories),
        'success_count': len(success_traj),
        'failure_count': len(failure_traj),
        'success_rate': float(len(success_traj) / len(trajectories)) if trajectories else 0,
    }
    
    # Token 统计
    all_tokens = [int(t.get('total_tokens', 0)) for t in trajectories]
    stats['total_tokens'] = {
        'mean': float(np.mean(all_tokens)),
        'std': float(np.std(all_tokens)),
        'min': int(np.min(all_tokens)),
        'max': int(np.max(all_tokens)),
    }
    
    # Turn 统计
    all_turns = [int(t.get('num_assistant_turns', 0)) for t in trajectories]
    stats['assistant_turns'] = {
        'mean': float(np.mean(all_turns)),
        'std': float(np.std(all_turns)),
    }
    
    # Entropy 统计
    success_entropies = []
    failure_entropies = []
    for t in success_traj:
        ent = t.get('entropy_C_assistant', [])
        if ent:
            success_entropies.extend([float(e) for e in ent])
    for t in failure_traj:
        ent = t.get('entropy_C_assistant', [])
        if ent:
            failure_entropies.extend([float(e) for e in ent])
    
    stats['entropy_C_assistant'] = {
        'success_mean': float(np.mean(success_entropies)) if success_entropies else 0,
        'success_std': float(np.std(success_entropies)) if success_entropies else 0,
        'failure_mean': float(np.mean(failure_entropies)) if failure_entropies else 0,
        'failure_std': float(np.std(failure_entropies)) if failure_entropies else 0,
    }
    
    # User entropy
    success_user_entropies = []
    failure_user_entropies = []
    for t in success_traj:
        ent = t.get('entropy_C_user', [])
        if ent:
            success_user_entropies.extend([float(e) for e in ent])
    for t in failure_traj:
        ent = t.get('entropy_C_user', [])
        if ent:
            failure_user_entropies.extend([float(e) for e in ent])
    
    stats['entropy_C_user'] = {
        'success_mean': float(np.mean(success_user_entropies)) if success_user_entropies else 0,
        'success_std': float(np.std(success_user_entropies)) if success_user_entropies else 0,
        'failure_mean': float(np.mean(failure_user_entropies)) if failure_user_entropies else 0,
        'failure_std': float(np.std(failure_user_entropies)) if failure_user_entropies else 0,
    }
    
    return stats


def plot_entropy_distribution(
    trajectories: List[Dict],
    output_path: str,
    task_name: str
):
    """绘制 entropy 分布对比图"""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    success_traj = [t for t in trajectories if t.get('_success', 0) == 1]
    failure_traj = [t for t in trajectories if t.get('_success', 0) == 0]
    
    success_entropies = []
    failure_entropies = []
    for t in success_traj:
        ent = t.get('entropy_C_assistant', [])
        if ent:
            success_entropies.extend(ent)
    for t in failure_traj:
        ent = t.get('entropy_C_assistant', [])
        if ent:
            failure_entropies.extend(ent)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 直方图
    ax = axes[0]
    if success_entropies:
        ax.hist(success_entropies, bins=50, alpha=0.6, color='blue', label='Success', density=True)
    if failure_entropies:
        ax.hist(failure_entropies, bins=50, alpha=0.6, color='red', label='Failure', density=True)
    ax.set_xlabel('Entropy (C)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'{task_name}\nAssistant Entropy Distribution', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 箱线图
    ax = axes[1]
    data_to_plot = []
    labels = []
    if success_entropies:
        data_to_plot.append(success_entropies)
        labels.append(f'Success (n={len(success_entropies)})')
    if failure_entropies:
        data_to_plot.append(failure_entropies)
        labels.append(f'Failure (n={len(failure_entropies)})')
    
    if data_to_plot:
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        colors_box = ['lightblue', 'lightcoral'][:len(data_to_plot)]
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
    
    ax.set_ylabel('Entropy (C)', fontsize=12)
    ax.set_title(f'{task_name}\nAssistant Entropy Box Plot', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Entropy Analysis for SciWorld, BabyAI, AlfWorld, WebShop"
    )
    parser.add_argument(
        '--input_sciworld', '-i_sciworld',
        type=str,
        default='/Data/wyh/datasets/Verl-Data/outputs/entropy_offline_minimax/entropy_offline_sciworld_20260316_160706/offline_results_20260316_160713.jsonl',
        help='Input sciworld entropy data'
    )
    parser.add_argument(
        '--input_babyai', '-i_babyai',
        type=str,
        default='/Data/wyh/datasets/Verl-Data/outputs/entropy_offline_minimax/entropy_offline_babyai_20260316_143138/offline_results_20260316_143145.jsonl',
        help='Input babyai entropy data'
    )
    parser.add_argument(
        '--input_alfworld', '-i_alfworld',
        type=str,
        default='',
        help='Input alfworld entropy data (auto-detected from entropy_offline_minimax dir)'
    )
    parser.add_argument(
        '--input_webshop', '-i_webshop',
        type=str,
        default='',
        help='Input webshop entropy data (auto-detected from entropy_offline_minimax dir)'
    )
    parser.add_argument(
        '--output_base', '-o',
        type=str,
        default='/Data/wyh/datasets/Verl-Data/outputs/entropy_analysis',
        help='Base output directory'
    )
    parser.add_argument(
        '--plot_samples', '-n',
        type=int,
        default=10,
        help='Number of sample trajectories to plot'
    )
    parser.add_argument(
        '--tasks', '-t',
        type=str,
        default='sciworld,babyai,alfworld,webshop',
        help='Comma-separated list of tasks to process'
    )

    args = parser.parse_args()
    
    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib is required for visualization")
        return

    # 自动检测最新结果文件
    base_dir = "/Data/wyh/datasets/Verl-Data/outputs/entropy_offline_minimax"
    if not args.input_alfworld:
        candidates = sorted([d for d in os.listdir(base_dir) if d.startswith("entropy_offline_alfworld_")], reverse=True)
        if candidates:
            subdir = os.path.join(base_dir, candidates[0])
            result_files = [f for f in os.listdir(subdir) if f.startswith("offline_results_") and f.endswith(".jsonl")]
            if result_files:
                args.input_alfworld = os.path.join(subdir, sorted(result_files)[-1])

    if not args.input_webshop:
        candidates = sorted([d for d in os.listdir(base_dir) if d.startswith("entropy_offline_webshop_")], reverse=True)
        if candidates:
            subdir = os.path.join(base_dir, candidates[0])
            result_files = [f for f in os.listdir(subdir) if f.startswith("offline_results_") and f.endswith(".jsonl")]
            if result_files:
                args.input_webshop = os.path.join(subdir, sorted(result_files)[-1])

    tasks_to_run = set(t.strip().lower() for t in args.tasks.split(','))
    print("\n" + "="*60)
    print("Processing SciWorld...")
    print("="*60)
    
    if os.path.exists(args.input_sciworld):
        sciworld_output_dir = os.path.join(args.output_base, 'sciworld')
        os.makedirs(sciworld_output_dir, exist_ok=True)
        
        trajectories_sciworld = load_trajectories(args.input_sciworld)
        
        print("\nGenerating visualizations...")
        plot_global_entropy_curves(
            trajectories_sciworld,
            os.path.join(sciworld_output_dir, 'entropy_curves_global.png'),
            'SciWorld'
        )
        
        plot_length_binned_entropy_curves(
            trajectories_sciworld,
            os.path.join(sciworld_output_dir, 'entropy_curves_by_length.png'),
            'SciWorld'
        )
        
        plot_entropy_distribution(
            trajectories_sciworld,
            os.path.join(sciworld_output_dir, 'entropy_distribution.png'),
            'SciWorld'
        )
        
        plot_detailed_trajectories(
            trajectories_sciworld,
            os.path.join(sciworld_output_dir, 'detailed_trajectories'),
            n_samples=args.plot_samples
        )
        
        # 统计摘要
        stats_sciworld = compute_statistics(trajectories_sciworld, 'sciworld')
        stats_path = os.path.join(sciworld_output_dir, 'statistics.json')
        with open(stats_path, 'w') as f:
            json.dump(stats_sciworld, f, indent=2)
        print(f"  Saved: {stats_path}")
        
        print(f"\nSciWorld Results:")
        print(f"  Total trajectories: {stats_sciworld['total_trajectories']}")
        print(f"  Success: {stats_sciworld['success_count']} ({stats_sciworld['success_rate']*100:.1f}%)")
        print(f"  Failure: {stats_sciworld['failure_count']}")
    else:
        print(f"Warning: SciWorld input file not found: {args.input_sciworld}")
    
    # 处理 babyai
    print("\n" + "="*60)
    print("Processing BabyAI...")
    print("="*60)
    
    if os.path.exists(args.input_babyai):
        babyai_output_dir = os.path.join(args.output_base, 'babyai')
        os.makedirs(babyai_output_dir, exist_ok=True)
        
        trajectories_babyai = load_trajectories(args.input_babyai)
        
        print("\nGenerating visualizations...")
        plot_global_entropy_curves(
            trajectories_babyai,
            os.path.join(babyai_output_dir, 'entropy_curves_global.png'),
            'BabyAI'
        )
        
        plot_length_binned_entropy_curves(
            trajectories_babyai,
            os.path.join(babyai_output_dir, 'entropy_curves_by_length.png'),
            'BabyAI'
        )
        
        plot_entropy_distribution(
            trajectories_babyai,
            os.path.join(babyai_output_dir, 'entropy_distribution.png'),
            'BabyAI'
        )
        
        plot_detailed_trajectories(
            trajectories_babyai,
            os.path.join(babyai_output_dir, 'detailed_trajectories'),
            n_samples=args.plot_samples
        )
        
        # 统计摘要
        stats_babyai = compute_statistics(trajectories_babyai, 'babyai')
        stats_path = os.path.join(babyai_output_dir, 'statistics.json')
        with open(stats_path, 'w') as f:
            json.dump(stats_babyai, f, indent=2)
        print(f"  Saved: {stats_path}")
        
        print(f"\nBabyAI Results:")
        print(f"  Total trajectories: {stats_babyai['total_trajectories']}")
        print(f"  Success: {stats_babyai['success_count']} ({stats_babyai['success_rate']*100:.1f}%)")
        print(f"  Failure: {stats_babyai['failure_count']}")
    else:
        print(f"Warning: BabyAI input file not found: {args.input_babyai}")

    # 处理 alfworld
    alfworld_path = args.input_alfworld or None
    print("\n" + "="*60)
    print("Processing AlfWorld...")
    print("="*60)

    if alfworld_path and os.path.exists(alfworld_path):
        alfworld_output_dir = os.path.join(args.output_base, 'alfworld')
        os.makedirs(alfworld_output_dir, exist_ok=True)
        trajectories_alfworld = load_trajectories(alfworld_path)
        print("\nGenerating visualizations...")
        plot_global_entropy_curves(
            trajectories_alfworld,
            os.path.join(alfworld_output_dir, 'entropy_curves_global.png'),
            'AlfWorld'
        )
        plot_length_binned_entropy_curves(
            trajectories_alfworld,
            os.path.join(alfworld_output_dir, 'entropy_curves_by_length.png'),
            'AlfWorld'
        )
        plot_entropy_distribution(
            trajectories_alfworld,
            os.path.join(alfworld_output_dir, 'entropy_distribution.png'),
            'AlfWorld'
        )
        plot_detailed_trajectories(
            trajectories_alfworld,
            os.path.join(alfworld_output_dir, 'detailed_trajectories'),
            n_samples=args.plot_samples
        )
        stats_alfworld = compute_statistics(trajectories_alfworld, 'alfworld')
        stats_path = os.path.join(alfworld_output_dir, 'statistics.json')
        with open(stats_path, 'w') as f:
            json.dump(stats_alfworld, f, indent=2)
        print(f"  Saved: {stats_path}")
        print(f"\nAlfWorld Results:")
        print(f"  Total trajectories: {stats_alfworld['total_trajectories']}")
        print(f"  Success: {stats_alfworld['success_count']} ({stats_alfworld['success_rate']*100:.1f}%)")
        print(f"  Failure: {stats_alfworld['failure_count']}")
    else:
        print(f"Warning: AlfWorld input file not found: {alfworld_path} (will be auto-detected after entropy calculation)")

    # 处理 webshop
    webshop_path = args.input_webshop or None
    print("\n" + "="*60)
    print("Processing WebShop...")
    print("="*60)

    if webshop_path and os.path.exists(webshop_path):
        webshop_output_dir = os.path.join(args.output_base, 'webshop')
        os.makedirs(webshop_output_dir, exist_ok=True)
        trajectories_webshop = load_trajectories(webshop_path)
        print("\nGenerating visualizations...")
        plot_global_entropy_curves(
            trajectories_webshop,
            os.path.join(webshop_output_dir, 'entropy_curves_global.png'),
            'WebShop'
        )
        plot_length_binned_entropy_curves(
            trajectories_webshop,
            os.path.join(webshop_output_dir, 'entropy_curves_by_length.png'),
            'WebShop'
        )
        plot_entropy_distribution(
            trajectories_webshop,
            os.path.join(webshop_output_dir, 'entropy_distribution.png'),
            'WebShop'
        )
        plot_detailed_trajectories(
            trajectories_webshop,
            os.path.join(webshop_output_dir, 'detailed_trajectories'),
            n_samples=args.plot_samples
        )
        stats_webshop = compute_statistics(trajectories_webshop, 'webshop')
        stats_path = os.path.join(webshop_output_dir, 'statistics.json')
        with open(stats_path, 'w') as f:
            json.dump(stats_webshop, f, indent=2)
        print(f"  Saved: {stats_path}")
        print(f"\nWebShop Results:")
        print(f"  Total trajectories: {stats_webshop['total_trajectories']}")
        print(f"  Success: {stats_webshop['success_count']} ({stats_webshop['success_rate']*100:.1f}%)")
        print(f"  Failure: {stats_webshop['failure_count']}")
    else:
        print(f"Warning: WebShop input file not found: {webshop_path} (will be auto-detected after entropy calculation)")

    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print(f"\nOutput directories:")
    print(f"  - {args.output_base}/sciworld/")
    print(f"  - {args.output_base}/babyai/")
    print(f"  - {args.output_base}/alfworld/")
    print(f"  - {args.output_base}/webshop/")


if __name__ == "__main__":
    main()
