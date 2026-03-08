#!/usr/bin/env python3
"""
重新绘制熵分析图（纯英文标签，避免中文字体问题）
直接读取已有的 summary JSON，无需重跑交互。
"""
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_entropy(summary: dict, output_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    labels = {
        "A_first_token":  "A: First-token Entropy (Think start)",
        "B_action_token": "B: Action first-token Entropy (after [[)",
        "C_turn_mean":    "C: Mean Token Entropy per Turn",
    }
    colors = {
        "A_first_token":  "steelblue",
        "B_action_token": "coral",
        "C_turn_mean":    "seagreen",
    }

    for ax_idx, (subset_key, subset_label) in enumerate([
        ("all_episodes",  "All Episodes"),
        ("success_only",  "Success Episodes Only"),
    ]):
        ax = axes[ax_idx]
        subset = summary[subset_key]
        for metric_key, metric_label in labels.items():
            data = subset[metric_key]
            if not data:
                continue
            # JSON 序列化后 key 是字符串，排序时按整数排
            steps = sorted(data.keys(), key=lambda x: int(x))
            means = [data[s]["mean"] for s in steps]
            stds  = [data[s]["std"]  for s in steps]
            xs    = [int(s) + 1 for s in steps]   # 1-based display

            ax.plot(xs, means, marker='o', markersize=4,
                    label=metric_label, color=colors[metric_key])
            ax.fill_between(
                xs,
                [m - sd for m, sd in zip(means, stds)],
                [m + sd for m, sd in zip(means, stds)],
                alpha=0.15, color=colors[metric_key],
            )

        ax.set_xlabel("Turn Step (1-indexed)", fontsize=12)
        ax.set_ylabel("Token Entropy", fontsize=12)
        ax.set_title(
            f"TextCraft Token Entropy vs. Step\n"
            f"({subset_label}, total={summary['total_episodes']} episodes, "
            f"success={summary['success_episodes']})",
            fontsize=11,
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    summary_path = sys.argv[1] if len(sys.argv) > 1 else \
        "/Data/wyh/datasets/Verl-Data/outputs/entropy_analysis/entropy_summary_20260307_132808.json"
    output_path = sys.argv[2] if len(sys.argv) > 2 else \
        "/Data/wyh/datasets/Verl-Data/outputs/entropy_analysis/entropy_plot_374queries.png"

    with open(summary_path) as f:
        summary = json.load(f)

    plot_entropy(summary, output_path)
