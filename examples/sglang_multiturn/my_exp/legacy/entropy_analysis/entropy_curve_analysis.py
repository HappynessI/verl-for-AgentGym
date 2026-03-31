"""
Entropy Curve Analysis - 专注于 entropy 曲线本身和具体轨迹上的切分位置

目标：
1. 全局平均 entropy 曲线 (按 relative position 对齐)
2. 按长度分桶的平均 entropy 曲线
3. 重点轨迹细看 (success/failure 各选取代表样本)
"""

import json
import os
from typing import List, Dict, Tuple, Any
import numpy as np

# 可视化库
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available")

# =============================================================================
# 数据加载
# =============================================================================

def load_trajectories(jsonl_path: str) -> List[Dict]:
    """加载轨迹数据"""
    trajectories = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            trajectories.append(json.loads(line))
    return trajectories


def interpolate_to_relative_positions(
    entropy_values: List[float],
    n_points: int = 20
) -> np.ndarray:
    """
    将 entropy 曲线插值到统一的 relative position 网格上
    使得不同长度的轨迹可以在同一坐标系下比较
    """
    if not entropy_values or len(entropy_values) < 2:
        return np.zeros(n_points)
    
    # 原始相对位置
    n = len(entropy_values)
    original_positions = np.linspace(0, 1, n)
    
    # 目标网格
    target_positions = np.linspace(0, 1, n_points)
    
    # 插值
    interpolated = np.interp(target_positions, original_positions, entropy_values)
    return interpolated


# =============================================================================
# 可视化 1: 全局平均 Entropy 曲线
# =============================================================================

def plot_global_entropy_curves(
    trajectories: List[Dict],
    output_path: str
):
    """
    绘制全局平均 entropy 曲线
    按 relative position 对齐，区分 assistant/user 和 success/failure
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    
    # 分离 success 和 failure
    success_traj = [t for t in trajectories if t.get('success', 0) == 1]
    failure_traj = [t for t in trajectories if t.get('success', 0) == 0]
    
    print(f"Success: {len(success_traj)}, Failure: {len(failure_traj)}")
    
    # 收集所有 entropy 曲线并插值到统一网格
    n_points = 20
    rel_positions = np.linspace(0, 1, n_points)
    
    # Assistant entropy
    success_assistant_curves = []
    failure_assistant_curves = []
    
    # User entropy  
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
    
    # 计算平均值
    success_a_mean = np.mean(success_assistant_curves, axis=0) if success_assistant_curves else np.zeros(n_points)
    failure_a_mean = np.mean(failure_assistant_curves, axis=0) if failure_assistant_curves else np.zeros(n_points)
    success_u_mean = np.mean(success_user_curves, axis=0) if success_user_curves else np.zeros(n_points)
    failure_u_mean = np.mean(failure_user_curves, axis=0) if failure_user_curves else np.zeros(n_points)
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图: Assistant
    ax = axes[0]
    ax.plot(rel_positions, success_a_mean, 'b-', linewidth=2.5, label='Success', alpha=0.9)
    ax.plot(rel_positions, failure_a_mean, 'r-', linewidth=2.5, label='Failure', alpha=0.9)
    ax.fill_between(rel_positions, success_a_mean * 0.8, success_a_mean * 1.2, alpha=0.2, color='blue')
    ax.fill_between(rel_positions, failure_a_mean * 0.8, failure_a_mean * 1.2, alpha=0.2, color='red')
    
    ax.set_xlabel('Relative Position q', fontsize=12)
    ax.set_ylabel('Entropy (C) - Assistant', fontsize=12)
    ax.set_title('Assistant Entropy Curve\n(Success vs Failure)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    
    # 右图: User
    ax = axes[1]
    ax.plot(rel_positions, success_u_mean, 'b-', linewidth=2.5, label='Success', alpha=0.9)
    ax.plot(rel_positions, failure_u_mean, 'r-', linewidth=2.5, label='Failure', alpha=0.9)
    ax.fill_between(rel_positions, success_u_mean * 0.8, success_u_mean * 1.2, alpha=0.2, color='blue')
    ax.fill_between(rel_positions, failure_u_mean * 0.8, failure_u_mean * 1.2, alpha=0.2, color='red')
    
    ax.set_xlabel('Relative Position q', fontsize=12)
    ax.set_ylabel('Entropy (C) - User', fontsize=12)
    ax.set_title('User Entropy Curve\n(Success vs Failure)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved global entropy curves to {output_path}")


# =============================================================================
# 可视化 2: 按长度分桶的平均 Entropy 曲线
# =============================================================================

def plot_length_binned_entropy_curves(
    trajectories: List[Dict],
    output_path: str
):
    """
    按轨迹长度分桶后分别绘制平均 entropy 曲线
    控制长度混杂因素
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    
    # 分桶
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
            
        # Success/Failure 分离
        success_traj = [t for t in bucket_traj if t.get('success', 0) == 1]
        failure_traj = [t for t in bucket_traj if t.get('success', 0) == 0]
        
        # 计算 assistant 平均曲线
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
        
        # 上半部分: Assistant
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
        
        # 下半部分: User
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
    
    plt.suptitle('Entropy Curves by Trajectory Length Bucket\n(Controlling for Length Confounder)', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved length-binned entropy curves to {output_path}")


# =============================================================================
# 可视化 3: 重点轨迹细看
# =============================================================================

def plot_detailed_trajectory(
    traj: Dict,
    output_path: str,
    show_entropy_B: bool = True
):
    """
    绘制单条轨迹的详细分析图
    显示 entropy 曲线和策略切分点
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    
    # 获取数据
    entropy_A = traj.get('entropy_C_assistant', [])
    entropy_U = traj.get('entropy_C_user', [])
    entropy_B = traj.get('entropy_B_assistant', []) if show_entropy_B else []
    
    # Cut points
    topk_cuts = traj.get('cut_topk_peaks', [])
    topk_rel = traj.get('cut_topk_peaks_relative', [])
    topk_tokens = traj.get('cut_topk_peaks_tokens', [])
    
    cum_cuts = traj.get('cut_cumulative_info', [])
    cum_rel = traj.get('cut_cumulative_info_relative', [])
    cum_tokens = traj.get('cut_cumulative_info_tokens', [])
    
    n_turns = len(entropy_A)
    
    if n_turns == 0:
        return
    
    # X 轴: 使用相对位置 q (0 到 1)
    # Assistant: 0 到 1 均匀分布
    if n_turns > 1:
        rel_positions = np.linspace(0, 1, n_turns)
    else:
        rel_positions = np.array([0.5])
    
    n_user = len(entropy_U)
    if n_user > 1:
        user_rel = np.linspace(0, 1, n_user)
    else:
        user_rel = np.array([0.5]) if n_user == 1 else np.array([])
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # 上图: Assistant entropy
    ax = axes[0]
    ax.plot(rel_positions, entropy_A, '-o', linewidth=2, markersize=8, 
            label='Entropy C (Assistant)', color='steelblue', alpha=0.9)
    
    # 如果有 entropy B (action entropy)，也画出来
    if entropy_B and len(entropy_B) == n_turns:
        valid_B = [v if v is not None else 0 for v in entropy_B]
        ax.plot(rel_positions, valid_B, '-s', linewidth=1.5, markersize=5,
                label='Entropy B (Action)', color='forestgreen', alpha=0.6)
    
    # 标记 Top-K Peaks 切分点 - 使用 relative position
    for i, (cut_idx, cut_rel, cut_tokens) in enumerate(zip(topk_cuts, topk_rel, topk_tokens)):
        if cut_idx is not None and cut_idx < n_turns:
            q_pos = cut_rel if cut_rel is not None else cut_idx / max(n_turns - 1, 1)
            ax.axvline(x=q_pos, color='green', linestyle='--', linewidth=2, alpha=0.7)
            ax.plot(q_pos, entropy_A[cut_idx], '^', markersize=18, 
                   color='green', markeredgecolor='black', markeredgewidth=1,
                   label=f'Top-K Peak @ q={cut_rel:.2f}' if i == 0 else None)
    
    # 标记 Cumulative Info 切分点 - 使用 relative position
    for i, (cut_idx, cut_rel, cut_tokens) in enumerate(zip(cum_cuts, cum_rel, cum_tokens)):
        if cut_idx is not None and cut_idx < n_turns:
            q_pos = cut_rel if cut_rel is not None else cut_idx / max(n_turns - 1, 1)
            ax.axvline(x=q_pos, color='red', linestyle=':', linewidth=2, alpha=0.7)
            ax.plot(q_pos, entropy_A[cut_idx], 'D', markersize=14,
                   color='red', markeredgecolor='black', markeredgewidth=1,
                   label=f'Cumulative Info @ q={cut_rel:.2f}' if i == 0 else None)
    
    ax.set_ylabel('Entropy', fontsize=12)
    ax.set_title(f"Trajectory {traj.get('item_id', 'unknown')} - Assistant\n"
                f"Success: {traj.get('success', 0)}, Turns: {n_turns}, "
                f"Tokens: {traj.get('total_tokens', 0)} [x-axis: relative position q]",
                fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    
    # 下图: User entropy
    ax = axes[1]
    if n_user > 0:
        ax.plot(user_rel, entropy_U, '-s', linewidth=2, markersize=7,
                label='Entropy C (User)', color='coral', alpha=0.9)
    
    ax.set_xlabel('Relative Position q', fontsize=12)
    ax.set_ylabel('Entropy', fontsize=12)
    ax.set_title("User Entropy [x-axis: relative position q]", fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def select_representative_trajectories(
    trajectories: List[Dict],
    n_success: int = 5,
    n_failure: int = 5
) -> Tuple[List[Dict], List[Dict]]:
    """
    选取代表性轨迹样本
    覆盖短、中、长三种长度
    """
    success_traj = [t for t in trajectories if t.get('success', 0) == 1]
    failure_traj = [t for t in trajectories if t.get('success', 0) == 0]
    
    print(f"Total success: {len(success_traj)}, Total failure: {len(failure_traj)}")
    
    def select_samples(traj_list, n_samples, is_failure=False):
        if len(traj_list) <= n_samples:
            return traj_list
        
        # 按长度分桶选取
        if is_failure:
            # Failure 轨迹都比较长，按长度均匀选取
            traj_list_sorted = sorted(traj_list, key=lambda t: t.get('total_tokens', 0))
            # 选取短的、中等、长的各一些
            indices = np.linspace(0, len(traj_list_sorted)-1, min(n_samples, len(traj_list_sorted)))
            return [traj_list_sorted[int(i)] for i in indices]
        else:
            buckets = [
                ('short', lambda t: t.get('total_tokens', 0) < 1000),
                ('medium', lambda t: 1000 <= t.get('total_tokens', 0) < 2000),
                ('long', lambda t: t.get('total_tokens', 0) >= 2000),
            ]
            
            samples = []
            for bucket_name, filter_fn in buckets:
                bucket_traj = [t for t in traj_list if filter_fn(t)]
                # 每个桶取 1-2 个
                n_take = min(2, len(bucket_traj))
                if bucket_traj:
                    # 取 entropy 变化最大的
                    bucket_traj.sort(key=lambda t: np.std(t.get('entropy_C_assistant', [0])), reverse=True)
                    samples.extend(bucket_traj[:n_take])
            
            # 补足数量
            if len(samples) < n_samples:
                remaining = [t for t in traj_list if t not in samples]
                remaining.sort(key=lambda t: abs(t.get('total_tokens', 0) - 1500))
                samples.extend(remaining[:n_samples - len(samples)])
            
            return samples[:n_samples]
    
    success_samples = select_samples(success_traj, n_success, is_failure=False)
    failure_samples = select_samples(failure_traj, n_failure, is_failure=True)
    
    return success_samples, failure_samples


# =============================================================================
# 主函数
# =============================================================================

def main():
    # 路径配置
    input_path = "/Data/wyh/datasets/Verl-Data/outputs/entropy_analysis/processed_trajectories.jsonl"
    output_dir = "/Data/wyh/datasets/Verl-Data/outputs/entropy_analysis/visualizations"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    print("Loading trajectories...")
    trajectories = load_trajectories(input_path)
    print(f"Loaded {len(trajectories)} trajectories")
    
    # 1. 全局平均 entropy 曲线
    print("\n[1/3] Plotting global entropy curves...")
    plot_global_entropy_curves(
        trajectories,
        os.path.join(output_dir, "entropy_curves_global.png")
    )
    
    # 2. 按长度分桶的平均 entropy 曲线
    print("\n[2/3] Plotting length-binned entropy curves...")
    plot_length_binned_entropy_curves(
        trajectories,
        os.path.join(output_dir, "entropy_curves_by_length.png")
    )
    
    # 3. 重点轨迹细看
    print("\n[3/3] Selecting representative trajectories...")
    success_samples, failure_samples = select_representative_trajectories(
        trajectories, n_success=5, n_failure=5
    )
    
    detailed_dir = os.path.join(output_dir, "detailed_trajectories")
    os.makedirs(detailed_dir, exist_ok=True)
    
    print(f"Saving {len(success_samples)} success trajectories...")
    for i, traj in enumerate(success_samples):
        traj_id = traj.get('item_id', f'success_{i}')
        plot_detailed_trajectory(
            traj,
            os.path.join(detailed_dir, f"traj_{traj_id}.png"),
            show_entropy_B=True
        )
    
    print(f"Saving {len(failure_samples)} failure trajectories...")
    for i, traj in enumerate(failure_samples):
        traj_id = traj.get('item_id', f'failure_{i}')
        plot_detailed_trajectory(
            traj,
            os.path.join(detailed_dir, f"traj_{traj_id}.png"),
            show_entropy_B=True
        )
    
    # 输出摘要
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print(f"\nGenerated files:")
    print(f"  - {output_dir}/entropy_curves_global.png")
    print(f"  - {output_dir}/entropy_curves_by_length.png")
    print(f"  - {detailed_dir}/ (detailed trajectory plots)")
    
    # 输出选取的轨迹信息
    print("\n--- Selected Success Trajectories ---")
    for traj in success_samples:
        print(f"  {traj.get('item_id')}: {traj.get('total_tokens')} tokens, "
              f"{len(traj.get('entropy_C_assistant', []))} assistant turns")
    
    print("\n--- Selected Failure Trajectories ---")
    for traj in failure_samples:
        print(f"  {traj.get('item_id')}: {traj.get('total_tokens')} tokens, "
              f"{len(traj.get('entropy_C_assistant', []))} assistant turns")


if __name__ == "__main__":
    main()
