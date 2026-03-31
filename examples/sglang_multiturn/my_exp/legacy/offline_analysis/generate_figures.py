"""
可视化脚本：step200 vs step460 离线对比
"""
import numpy as np
import pandas as pd
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = "/Data/wyh/datasets/Verl-Data/outputs/offline_analysisi"
SRC_DIR = "/Data/wyh/datasets/Verl-Data/outputs/textcraft_old_logits"
CODE_DIR = "/Data/wyh/verl/examples/sglang_multiturn/my_exp/offline_analysis"
os.makedirs(os.path.join(OUT_DIR, "figures"), exist_ok=True)

# ============ 加载数据 ============
df200 = pd.read_parquet(f"{SRC_DIR}/textcraft_trajectories_old_logprobs_step200.parquet")
df460 = pd.read_parquet(f"{SRC_DIR}/textcraft_trajectories_old_logprobs_step460.parquet")

with open(f"{SRC_DIR}/textcraft_trajectories_old_logprobs_step200_analysis.json") as f:
    j200 = json.load(f)
with open(f"{SRC_DIR}/textcraft_trajectories_old_logprobs_step460_analysis.json") as f:
    j460 = json.load(f)

# ============ 提取展平的 token 数组 ============
all_lp_200, all_lp_460 = [], []
for _, row in df200.iterrows():
    lps = row["sequence_old_logprobs"]
    msk = row["assistant_mask"]
    all_lp_200.extend([lps[i] for i in range(len(lps)) if msk[i] > 0.5])
for _, row in df460.iterrows():
    lps = row["sequence_old_logprobs"]
    msk = row["assistant_mask"]
    all_lp_460.extend([lps[i] for i in range(len(lps)) if msk[i] > 0.5])
all_lp_200 = np.array(all_lp_200)
all_lp_460 = np.array(all_lp_460)

# ============ 提取 per-sample 数据 ============
s200, s460 = [], []
for _, row in df200.iterrows():
    lps = row["sequence_old_logprobs"]
    msk = row["assistant_mask"]
    al = [lps[i] for i in range(len(lps)) if msk[i] > 0.5]
    s200.append({"success": int(row["success"]), "mean_lp": np.mean(al), "abs_lp": np.mean(np.abs(al))})
for _, row in df460.iterrows():
    lps = row["sequence_old_logprobs"]
    msk = row["assistant_mask"]
    al = [lps[i] for i in range(len(lps)) if msk[i] > 0.5]
    s460.append({"success": int(row["success"]), "mean_lp": np.mean(al), "abs_lp": np.mean(np.abs(al))})
df_s200 = pd.DataFrame(s200)
df_s460 = pd.DataFrame(s460)

# ============ 颜色设置 ============
C200 = "#2196F3"   # 蓝色
C460 = "#FF5722"   # 橙色
ALPHA = 0.65

# ========================================================
# FIG 1: Token-level logprob 分布直方图
# ========================================================
print("=== 生成 Fig 1: logprob 直方图 ===")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左：完整分布
ax = axes[0]
bins = np.linspace(-35, 0.5, 80)
ax.hist(all_lp_200, bins=bins, alpha=ALPHA, label="step200", color=C200, density=True, linewidth=0)
ax.hist(all_lp_460, bins=bins, alpha=ALPHA, label="step460", color=C460, density=True, linewidth=0)
ax.axvline(np.mean(all_lp_200), color=C200, linestyle="--", linewidth=1.5, label=f"step200 mean={np.mean(all_lp_200):.3f}")
ax.axvline(np.mean(all_lp_460), color=C460, linestyle="--", linewidth=1.5, label=f"step460 mean={np.mean(all_lp_460):.3f}")
ax.set_xlabel("Old Logprob", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("Assistant Token Logprob Distribution", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.set_xlim(-35, 1)
ax.grid(alpha=0.2)

# 右：[-10, 0] 放大
ax = axes[1]
bins = np.linspace(-10, 0.5, 60)
ax.hist(all_lp_200, bins=bins, alpha=ALPHA, label="step200", color=C200, density=True, linewidth=0)
ax.hist(all_lp_460, bins=bins, alpha=ALPHA, label="step460", color=C460, density=True, linewidth=0)
ax.axvline(np.mean(all_lp_200), color=C200, linestyle="--", linewidth=1.5)
ax.axvline(np.mean(all_lp_460), color=C460, linestyle="--", linewidth=1.5)
ax.set_xlabel("Old Logprob", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("Logprob Distribution [Zoom: -10, 0]", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "figures", "fig1_logprob_histogram.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  -> fig1_logprob_histogram.png")

# ========================================================
# FIG 2: Entropy 直方图 (使用已有 json 的 bucketized 数据)
# ========================================================
print("=== 生成 Fig 2: Entropy 分位数对比 ===")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

e200 = j200["assistant_tokens"]["entropy"]
e460 = j460["assistant_tokens"]["entropy"]

# 左：分位数对比
ax = axes[0]
keys = ["min", "p25", "p50", "p75", "p90", "p95", "p99", "max"]
x = np.arange(len(keys))
w = 0.35
v200 = [e200[k] for k in keys]
v460 = [e460[k] for k in keys]
ax.bar(x - w/2, v200, w, label="step200", color=C200, alpha=ALPHA)
ax.bar(x + w/2, v460, w, label="step460", color=C460, alpha=ALPHA)
ax.set_xticks(x)
ax.set_xticklabels(keys, fontsize=9)
ax.set_xlabel("Percentile", fontsize=11)
ax.set_ylabel("Entropy", fontsize=11)
ax.set_title("Entropy Percentile Comparison", fontsize=12, fontweight="bold")
ax.legend()
ax.grid(alpha=0.2, axis="y")

# 右：箱线图风格对比 (mean ± std)
ax = axes[1]
metrics = ["mean", "std"]
y200 = [e200["mean"], e200["std"]]
y460 = [e460["mean"], e460["std"]]
x = np.arange(len(metrics))
ax.bar(x - 0.2, y200, 0.35, label="step200", color=C200, alpha=ALPHA)
ax.bar(x + 0.2, y460, 0.35, label="step460", color=C460, alpha=ALPHA)
for i, (v, l) in enumerate(zip(y200, metrics)):
    ax.text(i - 0.2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9, color=C200)
for i, (v, l) in enumerate(zip(y460, metrics)):
    ax.text(i + 0.2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9, color=C460)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=10)
ax.set_ylabel("Entropy", fontsize=11)
ax.set_title("Entropy Mean & Std Comparison", fontsize=12, fontweight="bold")
ax.legend()
ax.grid(alpha=0.2, axis="y")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "figures", "fig2_entropy_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  -> fig2_entropy_comparison.png")

# ========================================================
# FIG 3: Turn-level entropy + logprob 曲线
# ========================================================
print("=== 生成 Fig 3: Turn-level 对比 ===")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左：entropy by turn
ax = axes[0]
turns = []
e200_means, e460_means = [], []
e200_stds, e460_stds = [], []
for t in range(1, 22):
    k = f"turn_{t}"
    if k in j200["entropy_by_turn_idx"] and k in j460["entropy_by_turn_idx"]:
        turns.append(t)
        e200_means.append(j200["entropy_by_turn_idx"][k]["mean"])
        e460_means.append(j460["entropy_by_turn_idx"][k]["mean"])
        e200_stds.append(j200["entropy_by_turn_idx"][k]["std"])
        e460_stds.append(j460["entropy_by_turn_idx"][k]["std"])

ax.plot(turns, e200_means, "o-", color=C200, linewidth=2, markersize=5, label="step200", alpha=ALPHA)
ax.plot(turns, e460_means, "s-", color=C460, linewidth=2, markersize=5, label="step460", alpha=ALPHA)
ax.fill_between(turns,
                np.array(e200_means) - np.array(e200_stds) * 0.3,
                np.array(e200_means) + np.array(e200_stds) * 0.3,
                color=C200, alpha=0.1)
ax.fill_between(turns,
                np.array(e460_means) - np.array(e460_stds) * 0.3,
                np.array(e460_means) + np.array(e460_stds) * 0.3,
                color=C460, alpha=0.1)
ax.set_xlabel("Turn Index", fontsize=11)
ax.set_ylabel("Mean Entropy", fontsize=11)
ax.set_title("Entropy by Turn Index", fontsize=12, fontweight="bold")
ax.legend()
ax.grid(alpha=0.2)
ax.set_xticks(turns)

# 右：logprob by turn
ax = axes[1]
lp200_means, lp460_means = [], []
lp200_stds, lp460_stds = [], []
lp200_pct, lp460_pct = [], []

with open(os.path.join(OUT_DIR, "step200_vs_step460_summary.json")) as f:
    summary = json.load(f)

for t in range(1, 22):
    k = f"turn_{t}"
    if k in summary["turn_level"]:
        d = summary["turn_level"][k]
        lp200_means.append(d["step200"]["mean"])
        lp460_means.append(d["step460"]["mean"])
        lp200_stds.append(d["step200"]["std"])
        lp460_stds.append(d["step460"]["std"])
        lp200_pct.append(d.get("step200_more_positive_pct", 0))

ax.plot(turns, lp200_means, "o-", color=C200, linewidth=2, markersize=5, label="step200", alpha=ALPHA)
ax.plot(turns, lp460_means, "s-", color=C460, linewidth=2, markersize=5, label="step460", alpha=ALPHA)
ax.fill_between(turns,
                np.array(lp200_means) - np.array(lp200_stds) * 0.15,
                np.array(lp200_means) + np.array(lp200_stds) * 0.15,
                color=C200, alpha=0.1)
ax.fill_between(turns,
                np.array(lp460_means) - np.array(lp460_stds) * 0.15,
                np.array(lp460_means) + np.array(lp460_stds) * 0.15,
                color=C460, alpha=0.1)
ax.set_xlabel("Turn Index", fontsize=11)
ax.set_ylabel("Mean Old Logprob", fontsize=11)
ax.set_title("Logprob by Turn Index", fontsize=12, fontweight="bold")
ax.legend()
ax.grid(alpha=0.2)
ax.set_xticks(turns)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "figures", "fig3_turn_level.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  -> fig3_turn_level.png")

# ========================================================
# FIG 4: Sample-level 散点图 + success/failure 分组箱线图
# ========================================================
print("=== 生成 Fig 4: Sample-level 对比 ===")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左：per-sample scatter (step200 vs step460)
ax = axes[0]
s200_s = df_s200[df_s200["success"] == 1]
s200_f = df_s200[df_s200["success"] == 0]
s460_s = df_s460[df_s460["success"] == 1]
s460_f = df_s460[df_s460["success"] == 0]

ax.scatter(s200_s["mean_lp"], s460_s["mean_lp"], alpha=0.25, s=10, color="green", label=f"success (n={len(s200_s)})")
ax.scatter(s200_f["mean_lp"], s460_f["mean_lp"], alpha=0.8, s=30, color="red", marker="x", label=f"failure (n={len(s200_f)})")
max_val = max(df_s200["mean_lp"].max(), df_s460["mean_lp"].max())
min_val = min(df_s200["mean_lp"].min(), df_s460["mean_lp"].min())
ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5, label="y=x line")
ax.set_xlabel("step200 Mean Logprob", fontsize=11)
ax.set_ylabel("step460 Mean Logprob", fontsize=11)
ax.set_title("Per-Sample Mean Logprob: step200 vs step460", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.2)

# 右：success/failure 分布对比（violin 或 箱线图）
ax = axes[1]
data_to_plot = []
labels = []
colors = []
positions = []
pos = 0

for name, df_s, col in [("step200\nsuccess", df_s200[df_s200["success"]==1], C200),
                          ("step200\nfailure", df_s200[df_s200["success"]==0], C200),
                          ("step460\nsuccess", df_s460[df_s460["success"]==1], C460),
                          ("step460\nfailure", df_s460[df_s460["success"]==0], C460)]:
    data_to_plot.append(df_s["mean_lp"].values)
    labels.append(name)
    colors.append(col)
    positions.append(pos)
    pos += 1

bp = ax.boxplot(data_to_plot, positions=range(len(data_to_plot)), patch_artist=True, widths=0.6)
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(ALPHA)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Mean Logprob per Sample", fontsize=11)
ax.set_title("Sample-Level Mean Logprob by Success/Failure", fontsize=12, fontweight="bold")
ax.grid(alpha=0.2, axis="y")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "figures", "fig4_sample_level.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  -> fig4_sample_level.png")

# ========================================================
# FIG 5: 置信度区间分布 + sharpness vs success
# ========================================================
print("=== 生成 Fig 5: Sharpness 分析 ===")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左：置信度区间条形图
ax = axes[0]
bins_info = [
    ("< -10", np.sum(all_lp_200 < -10) / len(all_lp_200) * 100, np.sum(all_lp_460 < -10) / len(all_lp_460) * 100),
    ("(-10, -5]", np.sum((all_lp_200 >= -10) & (all_lp_200 < -5)) / len(all_lp_200) * 100, np.sum((all_lp_460 >= -10) & (all_lp_460 < -5)) / len(all_lp_460) * 100),
    ("(-5, -1]", np.sum((all_lp_200 >= -5) & (all_lp_200 < -1)) / len(all_lp_200) * 100, np.sum((all_lp_460 >= -5) & (all_lp_460 < -1)) / len(all_lp_460) * 100),
    ("(-1, -0.01]", np.sum((all_lp_200 >= -1) & (all_lp_200 < -0.01)) / len(all_lp_200) * 100, np.sum((all_lp_460 >= -1) & (all_lp_460 < -0.01)) / len(all_lp_460) * 100),
    ("> -0.01", np.sum(all_lp_200 >= -0.01) / len(all_lp_200) * 100, np.sum(all_lp_460 >= -0.01) / len(all_lp_460) * 100),
]
bin_labels = [b[0] for b in bins_info]
v200_b = [b[1] for b in bins_info]
v460_b = [b[2] for b in bins_info]
x = np.arange(len(bin_labels))
w = 0.35
ax.bar(x - w/2, v200_b, w, label="step200", color=C200, alpha=ALPHA)
ax.bar(x + w/2, v460_b, w, label="step460", color=C460, alpha=ALPHA)
for i, (b200, b460) in enumerate(zip(v200_b, v460_b)):
    ax.text(i - w/2, b200 + 0.5, f"{b200:.1f}%", ha="center", va="bottom", fontsize=7, color=C200)
    ax.text(i + w/2, b460 + 0.5, f"{b460:.1f}%", ha="center", va="bottom", fontsize=7, color=C460)
ax.set_xticks(x)
ax.set_xticklabels(bin_labels, fontsize=9)
ax.set_ylabel("Token Percentage (%)", fontsize=11)
ax.set_title("Token Distribution by Logprob Range", fontsize=12, fontweight="bold")
ax.legend()
ax.grid(alpha=0.2, axis="y")

# 右：sharpness (|logprob|) vs success violin
ax = axes[1]
data_sharp = [
    df_s200[df_s200["success"]==1]["abs_lp"].values,
    df_s200[df_s200["success"]==0]["abs_lp"].values,
    df_s460[df_s460["success"]==1]["abs_lp"].values,
    df_s460[df_s460["success"]==0]["abs_lp"].values,
]
labels_sharp = ["200\nsuccess", "200\nfailure", "460\nsuccess", "460\nfailure"]
colors_sharp = [C200, C200, C460, C460]
bp2 = ax.boxplot(data_sharp, positions=range(4), patch_artist=True, widths=0.6)
for patch, color in zip(bp2["boxes"], colors_sharp):
    patch.set_facecolor(color)
    patch.set_alpha(ALPHA)
ax.set_xticks(range(4))
ax.set_xticklabels(labels_sharp, fontsize=9)
ax.set_ylabel("Mean |Logprob| (Sharpness)", fontsize=11)
ax.set_title("Sharpness by Checkpoint & Success", fontsize=12, fontweight="bold")
ax.grid(alpha=0.2, axis="y")

# 标注均值
for i, d in enumerate(data_sharp):
    ax.text(i, np.mean(d) + 0.03, f"μ={np.mean(d):.3f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "figures", "fig5_sharpness.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  -> fig5_sharpness.png")

# ========================================================
# FIG 6: 跨 checkpoint logprob_diff 分布
# ========================================================
print("=== 生成 Fig 6: Cross-checkpoint diff ===")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 加载 cross checkpoint 数据
df_merged = pd.read_csv(os.path.join(OUT_DIR, "step200_vs_step460_sample_comparison.csv"))

# 左：diff 直方图
ax = axes[0]
ax.hist(df_merged["logprob_diff"], bins=40, color=C200, alpha=ALPHA, edgecolor="white", linewidth=0.5)
ax.axvline(df_merged["logprob_diff"].mean(), color="red", linestyle="--", linewidth=2, label=f"mean={df_merged['logprob_diff'].mean():.3f}")
ax.axvline(df_merged["logprob_diff"].median(), color="green", linestyle=":", linewidth=2, label=f"median={df_merged['logprob_diff'].median():.3f}")
ax.set_xlabel("Logprob Diff (step200 - step460)", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("Per-Sample Logprob Diff Distribution\n(Positive = step200 more positive)", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.2)

# 右：diff 按 success 分组
ax = axes[1]
for success_val, label, color in [(1, "Success", "green"), (0, "Failure", "red")]:
    sub = df_merged[df_merged["success_200"] == success_val]
    ax.hist(sub["logprob_diff"], bins=20, alpha=0.6, label=f"{label} (n={len(sub)})", color=color, edgecolor="white", linewidth=0.5)
ax.axvline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)
ax.set_xlabel("Logprob Diff (step200 - step460)", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("Logprob Diff by Success/Failure", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "figures", "fig6_cross_checkpoint.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  -> fig6_cross_checkpoint.png")

# ========================================================
# FIG 7: Summary 热力图 - turn x metric
# ========================================================
print("=== 生成 Fig 7: Summary Heatmap ===")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 图7a: entropy diff by turn
ax = axes[0]
turn_nums = []
ent_diff = []
for t in range(1, 22):
    k = f"turn_{t}"
    if k in j200["entropy_by_turn_idx"] and k in j460["entropy_by_turn_idx"]:
        turn_nums.append(t)
        ent_diff.append(j200["entropy_by_turn_idx"][k]["mean"] - j460["entropy_by_turn_idx"][k]["mean"])
ax.bar(turn_nums, ent_diff, color=C200, alpha=ALPHA, edgecolor="white", linewidth=0.5)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Turn Index", fontsize=11)
ax.set_ylabel("Entropy Diff (step200 - step460)", fontsize=11)
ax.set_title("Entropy Diff by Turn\n(Positive = step200 more uncertain)", fontsize=10, fontweight="bold")
ax.set_xticks(turn_nums)
ax.grid(alpha=0.2, axis="y")

# 图7b: logprob diff by turn
ax = axes[1]
lp_diff = []
lp_pct = []
for t in range(1, 22):
    k = f"turn_{t}"
    if k in summary["turn_level"]:
        lp_diff.append(summary["turn_level"][k].get("diff_mean", 0))
        lp_pct.append(summary["turn_level"][k].get("step200_more_positive_pct", 0))
ax.bar(turn_nums, lp_diff, color=C200, alpha=ALPHA, edgecolor="white", linewidth=0.5)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Turn Index", fontsize=11)
ax.set_ylabel("Logprob Diff (step200 - step460)", fontsize=11)
ax.set_title("Logprob Diff by Turn\n(Positive = step200 more positive)", fontsize=10, fontweight="bold")
ax.set_xticks(turn_nums)
ax.grid(alpha=0.2, axis="y")

# 图7c: 汇总表格
ax = axes[2]
ax.axis("off")
summary_text = [
    "=== Key Metrics Summary ===",
    "",
    f"{'':20} {'step200':>10} {'step460':>10} {'diff':>10}",
    f"{'Token count':20} {len(all_lp_200):>10,} {len(all_lp_460):>10,} {'same':>10}",
    f"{'Mean logprob':20} {np.mean(all_lp_200):>10.4f} {np.mean(all_lp_460):>10.4f} {np.mean(all_lp_200)-np.mean(all_lp_460):>+10.4f}",
    f"{'Median logprob':20} {np.median(all_lp_200):>10.4f} {np.median(all_lp_460):>10.4f} {np.median(all_lp_200)-np.median(all_lp_460):>+10.4f}",
    f"{'Mean entropy':20} {e200['mean']:>10.4f} {e460['mean']:>10.4f} {e200['mean']-e460['mean']:>+10.4f}",
    f"{'Token >-0.01 (%)':20} {np.sum(all_lp_200>=-0.01)/len(all_lp_200)*100:>9.1f}% {np.sum(all_lp_460>=-0.01)/len(all_lp_460)*100:>9.1f}% {'':>10}",
    f"{'Token <-5 (%)':20} {np.sum(all_lp_200<-5)/len(all_lp_200)*100:>9.1f}% {np.sum(all_lp_460<-5)/len(all_lp_460)*100:>9.1f}% {'':>10}",
    f"{'':20} {'':>10} {'':>10} {'':>10}",
    f"{'Success entropy':20} {j200['entropy_by_success']['success']['entropy']['mean']:>10.4f} {j460['entropy_by_success']['success']['entropy']['mean']:>10.4f} {j200['entropy_by_success']['success']['entropy']['mean']-j460['entropy_by_success']['success']['entropy']['mean']:>+10.4f}",
    f"{'Failure entropy':20} {j200['entropy_by_success']['failure']['entropy']['mean']:>10.4f} {j460['entropy_by_success']['failure']['entropy']['mean']:>10.4f} {j200['entropy_by_success']['failure']['entropy']['mean']-j460['entropy_by_success']['failure']['entropy']['mean']:>+10.4f}",
    f"{'':20} {'':>10} {'':>10} {'':>10}",
    f"{'Samp. mean lp (s)':20} {df_s200[df_s200['success']==1]['mean_lp'].mean():>10.4f} {df_s460[df_s460['success']==1]['mean_lp'].mean():>10.4f} {df_s200[df_s200['success']==1]['mean_lp'].mean()-df_s460[df_s460['success']==1]['mean_lp'].mean():>+10.4f}",
    f"{'Samp. mean lp (f)':20} {df_s200[df_s200['success']==0]['mean_lp'].mean():>10.4f} {df_s460[df_s460['success']==0]['mean_lp'].mean():>10.4f} {df_s200[df_s200['success']==0]['mean_lp'].mean()-df_s460[df_s460['success']==0]['mean_lp'].mean():>+10.4f}",
]
ax.text(0.05, 0.95, "\n".join(summary_text), transform=ax.transAxes,
        fontsize=10, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))
ax.set_title("Summary Table", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "figures", "fig7_summary.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  -> fig7_summary.png")

print("\n=== 所有图表生成完成 ===")
print(f"图表目录: {os.path.join(OUT_DIR, 'figures')}")
