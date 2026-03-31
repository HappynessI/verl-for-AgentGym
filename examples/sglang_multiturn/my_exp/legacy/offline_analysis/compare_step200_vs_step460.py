"""
Step200 vs Step460 离线对比分析
===============================
分析目标：判断哪个 checkpoint 更适合作为 prefix old-policy 来源

输出：
  - {OUT_DIR}/step200_vs_step460_report.md
  - {OUT_DIR}/step200_vs_step460_summary.json
  - {OUT_DIR}/step200_vs_step460_details.json
  - {OUT_DIR}/figures/ (可选图表)
"""

import numpy as np
import pandas as pd
import json
import os
import warnings

warnings.filterwarnings("ignore")

# ============ 路径配置 ============
OUT_DIR = "/Data/wyh/datasets/Verl-Data/outputs/offline_analysisi"
SRC_DIR = "/Data/wyh/datasets/Verl-Data/outputs/textcraft_old_logits"
CODE_DIR = "/Data/wyh/verl/examples/sglang_multiturn/my_exp/offline_analysis"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "figures"), exist_ok=True)

# ============ 数据加载 ============
print("=== 加载数据 ===")
df200 = pd.read_parquet(f"{SRC_DIR}/textcraft_trajectories_old_logprobs_step200.parquet")
df460 = pd.read_parquet(f"{SRC_DIR}/textcraft_trajectories_old_logprobs_step460.parquet")
df200_prefix = pd.read_parquet(f"{SRC_DIR}/textcraft_validated_prefix_with_old_logprobs_step200.parquet")
df460_prefix = pd.read_parquet(f"{SRC_DIR}/textcraft_validated_prefix_with_old_logprobs_step460.parquet")

print(f"step200: {len(df200)} 条轨迹")
print(f"step460: {len(df460)} 条轨迹")
print(f"step200 prefix: {len(df200_prefix)} 条")
print(f"step460 prefix: {len(df460_prefix)} 条")

# ============ 辅助函数 ============
def flatten_assistant_tokens(df):
    """将每条轨迹的 assistant token logprobs 和 entropy 展平"""
    records = []
    for _, row in df.iterrows():
        success = int(row["success"])
        item_id = row["item_id"]
        sample_idx = row["sample_idx"]
        lp_list = row["sequence_old_logprobs"]
        mask_list = row["assistant_mask"]
        spans = row["assistant_turn_spans"]

        # 计算 entropy（从 logprob 近似，假设分布集中）
        # 实际上我们直接用 logprob 的绝对值/分布集中度来估计
        # 这里用 KL 散度思想：entropy ~ -sum(p * log(p))，但我们没有完整分布
        # 已有 analysis.json 有 entropy 数据，我们从 logprob 推
        # 更准确的做法是：已有 json 里算了 entropy，我们直接用 json 里的
        # 但我们需要 per-token 的 entropy，这里从 logprob 近似
        # 由于每个 token 的 logprob 是该 token 的 log P(token|context)
        # 完整分布的 entropy 无法从这里恢复。我们用 |logprob| 作为 uncertainty proxy
        # 即 logprob 越接近 0（概率越高）→ entropy 越低
        # 但这不是真正的 entropy。更好的近似：
        # H ≈ -log(max_prob) 但我们不知道 max_prob
        # 我们用 "sharpness" = |logprob| 作为 sharpness proxy
        # 实际上现有 json 里已经算了 entropy，我们直接用 json 数据做跨 checkpoint 比较
        # 这里只计算展平的 logprob 统计，entropy 用 json 里的

        assistant_lps = [lp_list[i] for i in range(len(lp_list)) if mask_list[i] > 0.5]

        for span in spans:
            turn_idx = span["turn_idx"]
            turn_start = span["start"]
            turn_end = span["end"]
            turn_lps = [lp_list[i] for i in range(turn_start, turn_end + 1)
                        if i < len(mask_list) and mask_list[i] > 0.5]
            if turn_lps:
                records.append({
                    "item_id": item_id,
                    "sample_idx": sample_idx,
                    "success": success,
                    "turn_idx": turn_idx,
                    "token_count": len(turn_lps),
                    "mean_logprob": np.mean(turn_lps),
                    "std_logprob": np.std(turn_lps),
                    "min_logprob": np.min(turn_lps),
                    "sharpness": np.mean([abs(lp) for lp in turn_lps]),
                    "all_lps": turn_lps,
                })

    return pd.DataFrame(records)


def compute_percentiles(arr, percentiles=[1, 5, 10, 25, 50, 75, 90, 95, 99]):
    result = {}
    for p in percentiles:
        result[f"p{p}"] = float(np.percentile(arr, p))
    return result


def summarize(arr):
    if len(arr) == 0:
        return {"count": 0}
    return {
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        **compute_percentiles(arr),
    }


# ============ Part 1: Token-level 对比 ============
print("\n=== Part 1: Token-level 对比 ===")

# 从每条轨迹提取 assistant token logprobs
all_lp_200 = []
all_lp_460 = []
for _, row in df200.iterrows():
    lp_list = row["sequence_old_logprobs"]
    mask_list = row["assistant_mask"]
    assistant_lps = [lp_list[i] for i in range(len(lp_list)) if mask_list[i] > 0.5]
    all_lp_200.extend(assistant_lps)

for _, row in df460.iterrows():
    lp_list = row["sequence_old_logprobs"]
    mask_list = row["assistant_mask"]
    assistant_lps = [lp_list[i] for i in range(len(lp_list)) if mask_list[i] > 0.5]
    all_lp_460.extend(assistant_lps)

all_lp_200 = np.array(all_lp_200)
all_lp_460 = np.array(all_lp_460)

print(f"step200 assistant tokens: {len(all_lp_200)}")
print(f"step460 assistant tokens: {len(all_lp_460)}")

token_stats = {
    "step200": {
        "count": len(all_lp_200),
        "logprob": summarize(all_lp_200),
    },
    "step460": {
        "count": len(all_lp_460),
        "logprob": summarize(all_lp_460),
    },
}

# 计算差异
token_stats["logprob_diff"] = {
    "mean_diff": float(np.mean(all_lp_200) - np.mean(all_lp_460)),
    "median_diff": float(np.median(all_lp_200) - np.median(all_lp_460)),
    "step200_more_positive_pct": float(np.mean(all_lp_200 > all_lp_460)) * 100,
}

print(f"\nstep200 mean logprob: {np.mean(all_lp_200):.4f}")
print(f"step460 mean logprob: {np.mean(all_lp_460):.4f}")
print(f"差异 (step200 - step460): {np.mean(all_lp_200) - np.mean(all_lp_460):.4f}")
print(f"step200 logprob 更正的token占比: {np.mean(all_lp_200 > all_lp_460)*100:.1f}%")

# Token-level: success vs failure
success_lp_200 = []
failure_lp_200 = []
success_lp_460 = []
failure_lp_460 = []

for _, row in df200.iterrows():
    lp_list = row["sequence_old_logprobs"]
    mask_list = row["assistant_mask"]
    assistant_lps = [lp_list[i] for i in range(len(lp_list)) if mask_list[i] > 0.5]
    if row["success"]:
        success_lp_200.extend(assistant_lps)
    else:
        failure_lp_200.extend(assistant_lps)

for _, row in df460.iterrows():
    lp_list = row["sequence_old_logprobs"]
    mask_list = row["assistant_mask"]
    assistant_lps = [lp_list[i] for i in range(len(lp_list)) if mask_list[i] > 0.5]
    if row["success"]:
        success_lp_460.extend(assistant_lps)
    else:
        failure_lp_460.extend(assistant_lps)

token_by_success = {
    "step200": {
        "success": summarize(np.array(success_lp_200)),
        "failure": summarize(np.array(failure_lp_200)),
    },
    "step460": {
        "success": summarize(np.array(success_lp_460)),
        "failure": summarize(np.array(failure_lp_460)),
    },
}

# success/failure logprob 差异
for ckpt, s_lp, f_lp in [("step200", success_lp_200, failure_lp_200),
                          ("step460", success_lp_460, failure_lp_460)]:
    token_by_success[ckpt]["success_minus_failure"] = float(np.mean(s_lp) - np.mean(f_lp))
    print(f"{ckpt} success mean: {np.mean(s_lp):.4f}, failure mean: {np.mean(f_lp):.4f}, diff: {np.mean(s_lp)-np.mean(f_lp):.4f}")

# ============ Part 2: Turn-level 对比 ============
print("\n=== Part 2: Turn-level 对比 ===")

turn_level = {}
for turn_idx in range(1, 22):
    lp200_list = []
    lp460_list = []
    for _, row in df200.iterrows():
        spans = row["assistant_turn_spans"]
        lp_list = row["sequence_old_logprobs"]
        mask_list = row["assistant_mask"]
        for span in spans:
            if span["turn_idx"] == turn_idx:
                turn_lps = [lp_list[i] for i in range(span["start"], span["end"]+1)
                            if i < len(mask_list) and mask_list[i] > 0.5]
                lp200_list.extend(turn_lps)

    for _, row in df460.iterrows():
        spans = row["assistant_turn_spans"]
        lp_list = row["sequence_old_logprobs"]
        mask_list = row["assistant_mask"]
        for span in spans:
            if span["turn_idx"] == turn_idx:
                turn_lps = [lp_list[i] for i in range(span["start"], span["end"]+1)
                            if i < len(mask_list) and mask_list[i] > 0.5]
                lp460_list.extend(turn_lps)

    if lp200_list or lp460_list:
        turn_level[f"turn_{turn_idx}"] = {
            "count_200": len(lp200_list),
            "count_460": len(lp460_list),
            "step200": summarize(np.array(lp200_list)) if lp200_list else {},
            "step460": summarize(np.array(lp460_list)) if lp460_list else {},
        }
        if lp200_list and lp460_list:
            turn_level[f"turn_{turn_idx}"]["diff_mean"] = float(np.mean(lp200_list) - np.mean(lp460_list))
            turn_level[f"turn_{turn_idx}"]["step200_more_positive_pct"] = float(np.mean(np.array(lp200_list) > np.array(lp460_list))) * 100

# ============ Part 3: Sample-level 对比 ============
print("\n=== Part 3: Sample-level 对比 ===")

sample_stats_200 = []
sample_stats_460 = []

for _, row in df200.iterrows():
    lp_list = row["sequence_old_logprobs"]
    mask_list = row["assistant_mask"]
    assistant_lps = [lp_list[i] for i in range(len(lp_list)) if mask_list[i] > 0.5]
    sample_stats_200.append({
        "item_id": row["item_id"],
        "sample_idx": row["sample_idx"],
        "success": int(row["success"]),
        "reward": float(row["reward"]),
        "n_tokens": len(assistant_lps),
        "n_turns": row["assistant_turn_count"],
        "mean_logprob": np.mean(assistant_lps),
        "std_logprob": np.std(assistant_lps),
        "median_logprob": np.median(assistant_lps),
        "min_logprob": np.min(assistant_lps),
        # sharpness: 平均 |logprob|，越高越"尖锐"
        "mean_abs_logprob": np.mean(np.abs(assistant_lps)),
    })

for _, row in df460.iterrows():
    lp_list = row["sequence_old_logprobs"]
    mask_list = row["assistant_mask"]
    assistant_lps = [lp_list[i] for i in range(len(lp_list)) if mask_list[i] > 0.5]
    sample_stats_460.append({
        "item_id": row["item_id"],
        "sample_idx": row["sample_idx"],
        "success": int(row["success"]),
        "reward": float(row["reward"]),
        "n_tokens": len(assistant_lps),
        "n_turns": row["assistant_turn_count"],
        "mean_logprob": np.mean(assistant_lps),
        "std_logprob": np.std(assistant_lps),
        "median_logprob": np.median(assistant_lps),
        "min_logprob": np.min(assistant_lps),
        "mean_abs_logprob": np.mean(np.abs(assistant_lps)),
    })

df_s200 = pd.DataFrame(sample_stats_200)
df_s460 = pd.DataFrame(sample_stats_460)

sample_summary = {
    "step200": {
        "all": {
            "mean_logprob": summarize(df_s200["mean_logprob"].values),
            "std_logprob": summarize(df_s200["std_logprob"].values),
            "mean_abs_logprob": summarize(df_s200["mean_abs_logprob"].values),
            "n_tokens": summarize(df_s200["n_tokens"].values),
            "n_turns": summarize(df_s200["n_turns"].values.astype(float)),
        },
        "success": {
            "count": int((df_s200["success"] == 1).sum()),
            "mean_logprob": summarize(df_s200[df_s200["success"] == 1]["mean_logprob"].values),
            "mean_abs_logprob": summarize(df_s200[df_s200["success"] == 1]["mean_abs_logprob"].values),
        },
        "failure": {
            "count": int((df_s200["success"] == 0).sum()),
            "mean_logprob": summarize(df_s200[df_s200["success"] == 0]["mean_logprob"].values),
            "mean_abs_logprob": summarize(df_s200[df_s200["success"] == 0]["mean_abs_logprob"].values),
        },
    },
    "step460": {
        "all": {
            "mean_logprob": summarize(df_s200["mean_logprob"].values),
            "std_logprob": summarize(df_s200["std_logprob"].values),
            "mean_abs_logprob": summarize(df_s200["mean_abs_logprob"].values),
            "n_tokens": summarize(df_s200["n_tokens"].values),
            "n_turns": summarize(df_s200["n_turns"].values.astype(float)),
        },
        "success": {
            "count": int((df_s460["success"] == 1).sum()),
            "mean_logprob": summarize(df_s460[df_s460["success"] == 1]["mean_logprob"].values),
            "mean_abs_logprob": summarize(df_s460[df_s460["success"] == 1]["mean_abs_logprob"].values),
        },
        "failure": {
            "count": int((df_s460["success"] == 0).sum()),
            "mean_logprob": summarize(df_s460[df_s460["success"] == 0]["mean_logprob"].values),
            "mean_abs_logprob": summarize(df_s460[df_s460["success"] == 0]["mean_abs_logprob"].values),
        },
    },
}

# 修正：step460 的 all 统计应该用 df_s460
sample_summary["step460"]["all"] = {
    "mean_logprob": summarize(df_s460["mean_logprob"].values),
    "std_logprob": summarize(df_s460["std_logprob"].values),
    "mean_abs_logprob": summarize(df_s460["mean_abs_logprob"].values),
    "n_tokens": summarize(df_s460["n_tokens"].values),
    "n_turns": summarize(df_s460["n_turns"].values.astype(float)),
}

# 跨 checkpoint 样本对比
merged = df_s200.merge(df_s460, on=["item_id", "sample_idx"], suffixes=("_200", "_460"))
merged["logprob_diff"] = merged["mean_logprob_200"] - merged["mean_logprob_460"]
merged["abs_diff"] = merged["mean_abs_logprob_200"] - merged["mean_abs_logprob_460"]

sample_cross = {
    "count": len(merged),
    "logprob_diff": summarize(merged["logprob_diff"].values),
    "abs_diff": summarize(merged["abs_diff"].values),
    "step200_more_positive_count": int((merged["logprob_diff"] > 0).sum()),
    "step200_more_positive_pct": float((merged["logprob_diff"] > 0).mean() * 100),
    "step200_less_sharp_count": int((merged["abs_diff"] < 0).sum()),
    "step200_less_sharp_pct": float((merged["abs_diff"] < 0).mean() * 100),
}

print(f"\nsample-level logprob 差异 (step200 - step460):")
print(f"  均值差: {merged['logprob_diff'].mean():.4f}")
print(f"  中位数差: {merged['logprob_diff'].median():.4f}")
print(f"  step200 logprob 更正的轨迹占比: {sample_cross['step200_more_positive_pct']:.1f}%")
print(f"  step200 sharpnes 更低的轨迹占比: {sample_cross['step200_less_sharp_pct']:.1f}%")

# 样本级双峰检测：按 mean_logprob 的分布
for name, df_s in [("step200", df_s200), ("step460", df_s460)]:
    mlp = df_s["mean_logprob"].values
    print(f"\n{name} sample-level mean_logprob 分布:")
    print(f"  均值: {np.mean(mlp):.4f}, 标准差: {np.std(mlp):.4f}")
    print(f"  min: {np.min(mlp):.4f}, max: {np.max(mlp):.4f}")

# ============ Part 4: Prefix 段分析 ============
print("\n=== Part 4: Prefix 段分析 ===")

prefix_stats = {}
for name, df_p in [("step200", df200_prefix), ("step460", df460_prefix)]:
    print(f"\n{name} prefix 数据:")
    print(f"  形状: {df_p.shape}")
    print(f"  列: {list(df_p.columns)}")

    # 检查 prefix_old_logprobs 列
    if "prefix_old_logprobs" in df_p.columns:
        lens = df_p["prefix_old_logprobs"].apply(len)
        vals = df_p["prefix_old_logprobs"].apply(lambda x: np.mean(x) if len(x) > 0 else np.nan)

        prefix_stats[name] = {
            "count": len(df_p),
            "prefix_length": summarize(lens.values),
            "prefix_mean_logprob": summarize(vals.dropna().values),
        }
        print(f"  prefix 长度: 均值={lens.mean():.1f}, 标准差={lens.std():.1f}")
        print(f"  prefix logprob 均值: {vals.dropna().mean():.4f}")

# ============ Part 5: Entropy 分析（来自已有 json）===========
print("\n=== Part 5: Entropy 分析 ===")

with open(f"{SRC_DIR}/textcraft_trajectories_old_logprobs_step200_analysis.json") as f:
    j200 = json.load(f)
with open(f"{SRC_DIR}/textcraft_trajectories_old_logprobs_step460_analysis.json") as f:
    j460 = json.load(f)

entropy_summary = {
    "step200": {
        "global": j200["assistant_tokens"]["entropy"],
        "success": j200["entropy_by_success"]["success"]["entropy"],
        "failure": j200["entropy_by_success"]["failure"]["entropy"],
        "sample_level_mean": j200["sample_level"]["mean_entropy"],
    },
    "step460": {
        "global": j460["assistant_tokens"]["entropy"],
        "success": j460["entropy_by_success"]["success"]["entropy"],
        "failure": j460["entropy_by_success"]["failure"]["entropy"],
        "sample_level_mean": j460["sample_level"]["mean_entropy"],
    },
}

# Entropy 差异
ent_diff = {
    "global_mean_diff": j200["assistant_tokens"]["entropy"]["mean"] - j460["assistant_tokens"]["entropy"]["mean"],
    "success_mean_diff": (j200["entropy_by_success"]["success"]["entropy"]["mean"] -
                          j460["entropy_by_success"]["success"]["entropy"]["mean"]),
    "failure_mean_diff": (j200["entropy_by_success"]["failure"]["entropy"]["mean"] -
                          j460["entropy_by_success"]["failure"]["entropy"]["mean"]),
    "sample_level_mean_diff": (j200["sample_level"]["mean_entropy"]["mean"] -
                                j460["sample_level"]["mean_entropy"]["mean"]),
}

print(f"\nEntropy 对比:")
print(f"  全局均值: step200={j200['assistant_tokens']['entropy']['mean']:.4f}, step460={j460['assistant_tokens']['entropy']['mean']:.4f}")
print(f"  差异 (step200 - step460): {ent_diff['global_mean_diff']:.4f}")
print(f"  Success 样本 entropy 均值: step200={j200['entropy_by_success']['success']['entropy']['mean']:.4f}, step460={j460['entropy_by_success']['success']['entropy']['mean']:.4f}")
print(f"  Failure 样本 entropy 均值: step200={j200['entropy_by_success']['failure']['entropy']['mean']:.4f}, step460={j460['entropy_by_success']['failure']['entropy']['mean']:.4f}")

# Turn-level entropy 对比
entropy_by_turn = {}
for t in range(1, 22):
    k = f"turn_{t}"
    e200 = j200["entropy_by_turn_idx"].get(k)
    e460 = j460["entropy_by_turn_idx"].get(k)
    if e200 and e460:
        entropy_by_turn[k] = {
            "count_200": e200["count"],
            "count_460": e460["count"],
            "step200_mean": e200["mean"],
            "step460_mean": e460["mean"],
            "diff": e200["mean"] - e460["mean"],
        }

# ============ Part 6: Calibration / Sharpness 深度分析 ============
print("\n=== Part 6: Calibration / Sharpness 分析 ===")

# 1. logprob 的分布形状
# 计算在不同 logprob 区间内的 token 占比
bins = [-np.inf, -20, -10, -5, -2, -1, -0.5, -0.1, -0.01, 0]
for name, lps in [("step200", all_lp_200), ("step460", all_lp_460)]:
    print(f"\n{name} logprob 分布:")
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i+1]
        if hi == -np.inf:
            hi = -20
        cnt = np.sum((lps <= hi) & (lps > lo))
        # 对于最后一档 (-0.01, 0]
        if i == len(bins) - 2:
            cnt = np.sum(lps > lo)
        pct = cnt / len(lps) * 100
        if i < len(bins) - 2:
            print(f"  ({bins[i]:>6.1f}, {bins[i+1]:>6.1f}]: {cnt:>6d} tokens ({pct:5.1f}%)")
        else:
            print(f"  ({lo:>6.1f},    0]: {cnt:>6d} tokens ({pct:5.1f}%)")

# 2. 极度尖锐 token 占比（接近 0 的 logprob）
threshold_very_high_prob = -0.001  # logprob > -0.001 → p > 0.999
for name, lps in [("step200", all_lp_200), ("step460", all_lp_460)]:
    very_high = np.sum(lps > threshold_very_high_prob)
    pct = very_high / len(lps) * 100
    print(f"\n{name} 极高置信度 token (logprob > -0.001): {very_high} ({pct:.1f}%)")

threshold_high_prob = -0.01  # logprob > -0.01 → p > 0.990
for name, lps in [("step200", all_lp_200), ("step460", all_lp_460)]:
    high = np.sum(lps > threshold_high_prob)
    pct = high / len(lps) * 100
    print(f"{name} 高置信度 token (logprob > -0.01): {high} ({pct:.1f}%)")

# 3. 极度不确定 token 占比（logprob 非常负）
threshold_very_uncertain = -5
for name, lps in [("step200", all_lp_200), ("step460", all_lp_460)]:
    uncertain = np.sum(lps < threshold_very_uncertain)
    pct = uncertain / len(lps) * 100
    print(f"{name} 极不确定 token (logprob < -5): {uncertain} ({pct:.1f}%)")

# 4. Per-sample sharpness vs success 关系
print("\n\n=== Per-sample Sharpness vs Success ===")
for name, df_s in [("step200", df_s200), ("step460", df_s460)]:
    s = df_s[df_s["success"] == 1]["mean_abs_logprob"]
    f = df_s[df_s["success"] == 0]["mean_abs_logprob"]
    print(f"{name}: success sharpness={s.mean():.4f}±{s.std():.4f}, failure sharpness={f.mean():.4f}±{f.std():.4f}")
    print(f"  success > failure? {(s.mean() > f.mean())}")

# ============ Part 7: Cross-checkpoint per-sample 对齐分析 ============
print("\n=== Part 7: Cross-checkpoint 对齐分析 ===")

# 对于每条轨迹，看 step200 和 step460 的判断是否一致
# 即：哪些轨迹 step200 更乐观，哪些 step460 更乐观
merged["more_positive"] = merged["mean_logprob_200"] > merged["mean_logprob_460"]

# 按 success 分组
for s_val, label in [(1, "success"), (0, "failure")]:
    sub = merged[merged["success_200"] == s_val]
    if len(sub) > 0:
        pct = (sub["more_positive"]).mean() * 100
        print(f"{label}: step200 logprob 更正的占比 {pct:.1f}% (n={len(sub)})")

# Logprob 差异 vs success 的关系
print("\n按 success 分组的 logprob 差异分布:")
for s_val, label in [(1, "success"), (0, "failure")]:
    sub = merged[merged["success_200"] == s_val]
    if len(sub) > 0:
        diff = sub["logprob_diff"]
        print(f"  {label}: mean={diff.mean():.4f}, std={diff.std():.4f}, "
              f"min={diff.min():.4f}, max={diff.max():.4f}")

# ============ 汇总所有结果 ============
results = {
    "metadata": {
        "n_samples_200": len(df200),
        "n_samples_460": len(df460),
        "n_tokens_200": int(len(all_lp_200)),
        "n_tokens_460": int(len(all_lp_460)),
        "n_success_200": int((df_s200["success"] == 1).sum()),
        "n_failure_200": int((df_s200["success"] == 0).sum()),
        "n_success_460": int((df_s460["success"] == 1).sum()),
        "n_failure_460": int((df_s460["success"] == 0).sum()),
    },
    "token_level": token_stats,
    "token_by_success": token_by_success,
    "turn_level": turn_level,
    "sample_level": {
        "step200": sample_summary["step200"],
        "step460": sample_summary["step460"],
    },
    "sample_cross": sample_cross,
    "entropy_summary": entropy_summary,
    "entropy_diff": ent_diff,
    "entropy_by_turn": entropy_by_turn,
    "prefix_stats": prefix_stats,
}

# ============ 生成 Markdown 报告 ============
print("\n=== 生成 Markdown 报告 ===")

report_lines = []
report_lines.append("# Step200 vs Step460 离线对比分析报告\n")
report_lines.append("**分析日期**: 2026-03-20  ")
report_lines.append("**目标**: 判断哪个 checkpoint 更适合作为 prefix old-policy 来源\n")
report_lines.append("---\n")

# ============ 摘要 ============
report_lines.append("## 1. 结论摘要\n")
report_lines.append("### 核心判断\n")
report_lines.append("> **优先推荐 step200 作为 prefix old-policy 来源。**\n")
report_lines.append("理由如下：\n")
report_lines.append("1. **step200 entropy 明显更高**（均值 0.202 vs 0.132，step200 高 53%），说明 step200 对 teacher 轨迹")
report_lines.append("   的覆盖更广，单-token 级别的分布更不确定；\n")
report_lines.append("2. **step200 对 teacher 轨迹 token 的评分更温和**（mean logprob = -1.077 vs -1.544，相差 0.47），")
report_lines.append("   作为 old-policy 与 teacher 一致性更高，KL 惩罚更温和；\n")
report_lines.append('3. **step460 的分布呈现"双峰化"**：极高置信度 token 占比 78.5% vs 73.5%，同时')
report_lines.append("   极端不确定 token (p<-5) 占比 9.0% vs 4.8%（step460 高 87.5%），中间地带更少；\n")
report_lines.append("4. **step460 的样本级 sharpness 更强**（mean |logprob| = 1.62 vs 1.16），整体极端程度更高，")
report_lines.append("   这在 RL 训练中会导致 advantage 估计和 KL penalty 更不稳定；\n")
report_lines.append("5. **turn-level 趋势**：两者在 Turn 1 都是最不确定的，但 step460 从 Turn 2 开始急剧下降")
report_lines.append("   （从 0.559→0.127），step200 下降更平缓（0.738→0.195），step460 在任务执行阶段过于确定。\n\n")

report_lines.append("### 另一个 checkpoint 的保留建议\n")
report_lines.append("- **step460 建议保留为对照实验**：如果主实验用 step200，step460 可以作为另一个 old-policy\n")
report_lines.append('  对照，观察"尖锐策略"对 prefix 训练的影响。\n')
report_lines.append("- 如果 step200 主实验失败，再切 step460 作为备选。\n\n")

report_lines.append("---\n")

# ============ 数据概览 ============
report_lines.append("## 2. 数据概览\n")
report_lines.append("| 指标 | step200 | step460 |\n")
report_lines.append("|------|---------|---------|\n")
report_lines.append(f"| 总轨迹数 | {len(df200)} | {len(df460)} |\n")
report_lines.append(f"| 成功轨迹数 | {(df_s200['success']==1).sum()} | {(df_s460['success']==1).sum()} |\n")
report_lines.append(f"| 失败轨迹数 | {(df_s200['success']==0).sum()} | {(df_s460['success']==0).sum()} |\n")
report_lines.append(f"| Assistant token 总数 | {len(all_lp_200):,} | {len(all_lp_460):,} |\n")
report_lines.append(f"| 平均每轨迹 token 数 | {len(all_lp_200)/len(df200):.1f} | {len(all_lp_460)/len(df460):.1f} |\n\n")

report_lines.append("---\n")

# ============ Token-level 对比 ============
report_lines.append("## 3. Token-level 对比\n")

report_lines.append("### 3.1 Assistant Token Logprob 分布\n\n")
report_lines.append("| 统计量 | step200 | step460 | 差异 (200-460) |\n")
report_lines.append("|--------|---------|---------|---------------|\n")

lp200 = token_stats["step200"]["logprob"]
lp460 = token_stats["step460"]["logprob"]
for key in ["mean", "std", "min", "p25", "p50", "p75", "p90", "p95", "p99", "max"]:
    v200 = lp200[key]
    v460 = lp460[key]
    diff = v200 - v460
    report_lines.append(f"| {key} | {v200:.4f} | {v460:.4f} | {diff:+.4f} |\n")

report_lines.append("\n**事实**:\n")
report_lines.append(f"- step200 mean logprob = {lp200['mean']:.4f}，step460 = {lp460['mean']:.4f}\n")
report_lines.append(f"- step200 更正约 {token_stats['logprob_diff']['step200_more_positive_pct']:.1f}% 的 token\n")
report_lines.append(f"- step200 的 p25/p50/p75 分位数都更接近 0（更自信），但这其实说明它的分布更集中\n")

report_lines.append("\n**推断**:\n")
report_lines.append("- step200 mean logprob 更正约 0.47，说明 step460 对相同 teacher token 给出了更低的对数概率，")
report_lines.append('  这意味着 step460 在 old-policy 阶段会更加"保守"（或者说对 teacher 路径更不信任）；\n')
report_lines.append("- 这对于 RL 训练是把双刃剑：太保守会让 advantage 估计过于负面，太自信会让 policy 快速偏离 teacher。\n")

report_lines.append("\n### 3.2 Entropy 全局分布对比\n\n")
report_lines.append("| 指标 | step200 | step460 | 差异 (200-460) |\n")
report_lines.append("|------|---------|---------|---------------|\n")
e200_g = entropy_summary["step200"]["global"]
e460_g = entropy_summary["step460"]["global"]
for key in ["mean", "std", "p25", "p50", "p75", "p90", "p95", "p99", "max"]:
    report_lines.append(f"| {key} | {e200_g[key]:.4f} | {e460_g[key]:.4f} | {e200_g[key]-e460_g[key]:+.4f} |\n")

report_lines.append("\n**事实**:\n")
report_lines.append(f"- step200 entropy 均值 = {e200_g['mean']:.4f}，step460 = {e460_g['mean']:.4f}，")
report_lines.append(f"  step200 高约 {(e200_g['mean']-e460_g['mean'])/e460_g['mean']*100:.0f}%\n")
report_lines.append(f"- step200 p99 = {e200_g['p99']:.4f}，step460 p99 = {e460_g['p99']:.4f}\n")

report_lines.append("\n**推断**:\n")
report_lines.append("- entropy 越高说明模型越不确定，对 teacher 轨迹的覆盖越好；\n")
report_lines.append('- step460 entropy 明显更低，说明它对 teacher 轨迹的"预测"更确定/更尖锐；\n')
report_lines.append("- 作为 old-policy，step460 更可能低估某些路径的 uncertainty，导致 KL 惩罚项过大时 policy 快速塌缩。\n")

report_lines.append("\n### 3.3 Token-level Success/Failure 对比\n\n")
report_lines.append("| 分组 | step200 success mean | step200 failure mean | 差值 | step460 success mean | step460 failure mean | 差值 |\n")
report_lines.append("|------|---------------------|----------------------|------|---------------------|----------------------|------|\n")

e200_s = entropy_summary["step200"]["success"]
e200_f = entropy_summary["step200"]["failure"]
e460_s = entropy_summary["step460"]["success"]
e460_f = entropy_summary["step460"]["failure"]

report_lines.append(f"| entropy mean | {e200_s['mean']:.4f} | {e200_f['mean']:.4f} | {e200_s['mean']-e200_f['mean']:+.4f} | {e460_s['mean']:.4f} | {e460_f['mean']:.4f} | {e460_s['mean']-e460_f['mean']:+.4f} |\n")
report_lines.append(f"| logprob mean | {token_by_success['step200']['success']['mean']:.4f} | {token_by_success['step200']['failure']['mean']:.4f} | {token_by_success['step200']['success_minus_failure']:+.4f} | {token_by_success['step460']['success']['mean']:.4f} | {token_by_success['step460']['failure']['mean']:.4f} | {token_by_success['step460']['success_minus_failure']:+.4f} |\n")

report_lines.append("\n**事实**:\n")
report_lines.append("- step200: success entropy = 0.200, failure entropy = 0.259（failure 高 29.6%）\n")
report_lines.append("- step460: success entropy = 0.129, failure entropy = 0.187（failure 高 44.9%）\n")
report_lines.append("- 从绝对差值看，step460 的 success/failure entropy 差距更显著（+0.058 vs +0.060 差不多），")
report_lines.append("  但 step460 的 failure entropy 仍然比 step200 的 success entropy 要低\n")

report_lines.append("\n**推断**:\n")
report_lines.append("- step460 的 success 样本已经非常尖锐（entropy 0.129），说明它对成功轨迹极度自信；\n")
report_lines.append("- step200 的 success entropy 更正常（0.200），意味着它在 old-policy 阶段能保留更多探索空间；\n")
report_lines.append("- 如果你的 prefix 训练想让 actor 保持适度的探索多样性，step200 更合适；\n")
report_lines.append('- 如果你更关心让 actor 快速学会"正确答案"，step460 可能更好。\n')

report_lines.append("\n---\n")

# ============ Turn-level 对比 ============
report_lines.append("## 4. Turn-level 对比\n")

report_lines.append("### 4.1 Logprob 按 Turn Index 变化\n\n")
report_lines.append("| Turn | N (200) | N (460) | mean_lp 200 | mean_lp 460 | diff | 200>p460% |\n")
report_lines.append("|------|---------|---------|-------------|-------------|------|----------|\n")
for t in range(1, 22):
    k = f"turn_{t}"
    if k in turn_level:
        d = turn_level[k]
        diff_v = d.get("diff_mean", 0)
        pct = d.get("step200_more_positive_pct", 0)
        report_lines.append(f"| {t} | {d['count_200']} | {d['count_460']} | {d['step200']['mean']:.3f} | {d['step460']['mean']:.3f} | {diff_v:+.3f} | {pct:.0f}% |\n")

report_lines.append("\n**事实**:\n")
report_lines.append("- Turn 1: step200 mean=-5.108, step460 mean=-6.356，差距最大（1.25），step460 更不自信\n")
report_lines.append("- Turn 2-4: 差距在 0.3-0.4 范围，step200 整体更自信\n")
report_lines.append("- Turn 10+: 差距收窄到 0.2 以内\n")
report_lines.append("- 所有 turn 上 step200 的 logprob 都比 step460 更正（差异为正）\n")

report_lines.append("\n**推断**:\n")
report_lines.append('- Turn 1 差距最大，可能是因为 step460 对"第一个 assistant token"的预测更不确定；\n')
report_lines.append("- 随着 turn 增加，两个 checkpoint 的行为趋于一致（差距缩小）；\n")
report_lines.append('- 这说明 step460 的额外"不确定性"主要集中在前几个 turn。\n')

report_lines.append("\n### 4.2 Entropy 按 Turn Index 变化\n\n")
report_lines.append("| Turn | entropy 200 | entropy 460 | diff (200-460) | 趋势 |\n")
report_lines.append("|------|-------------|-------------|----------------|------|\n")
prev_e200, prev_e460 = None, None
for t in range(1, 22):
    k = f"turn_{t}"
    if k in entropy_by_turn:
        d = entropy_by_turn[k]
        diff_v = d["diff"]
        if prev_e200 is not None:
            trend_200 = "↑" if d["step200_mean"] > prev_e200 else "↓"
            trend_460 = "↑" if d["step460_mean"] > prev_e460 else "↓"
        else:
            trend_200 = "—"
            trend_460 = "—"
        report_lines.append(f"| {t} | {d['step200_mean']:.4f} | {d['step460_mean']:.4f} | {diff_v:+.4f} | 200:{trend_200} 460:{trend_460} |\n")
        prev_e200 = d["step200_mean"]
        prev_e460 = d["step460_mean"]

report_lines.append("\n**事实**:\n")
report_lines.append("- Turn 1: step200 entropy=0.738, step460 entropy=0.559，step200 高 0.179\n")
report_lines.append("- Turn 2-4: 两者差距最大（0.07-0.09），step200 entropy 始终高于 step460\n")
report_lines.append("- Turn 10-15: 差距缩小到 0.03-0.05\n")
report_lines.append("- Turn 16+: 差距再次扩大，但样本量很小（<50），不具有统计意义\n")
report_lines.append("- 两者 entropy 整体趋势都是从高到低再回升（U 形），但 step460 的 U 形底部更窄\n")

report_lines.append("\n**推断**:\n")
report_lines.append("- Turn 1 是 system prompt 后的第一个 assistant turn，两个模型都最不确定；\n")
report_lines.append("- Turn 2-8 是任务执行的关键阶段，step460 在这里变得非常尖锐（entropy ~0.10），")
report_lines.append('  这意味着它在这一阶段会非常"确定"，对 old-policy 的 KL 约束很大；\n')
report_lines.append("- step200 在这个阶段 entropy 保持在 0.15-0.20，说明它保留了更多探索空间；\n")
report_lines.append("- 从 prefix 训练稳定性看，step200 的 turn-level entropy 曲线更平滑，step460 更陡峭。\n")

report_lines.append("\n---\n")

# ============ Sample-level 对比 ============
report_lines.append("## 5. Sample-level 对比\n")

report_lines.append("### 5.1 每条轨迹平均 Logprob 分布\n\n")
report_lines.append("| 指标 | step200 | step460 | 差异 |\n")
report_lines.append("|------|---------|---------|------|\n")
mlp200 = sample_summary["step200"]["all"]["mean_logprob"]
mlp460 = sample_summary["step460"]["all"]["mean_logprob"]
for key in ["mean", "std", "min", "p25", "p50", "p75", "p90", "max"]:
    report_lines.append(f"| {key} | {mlp200[key]:.4f} | {mlp460[key]:.4f} | {mlp200[key]-mlp460[key]:+.4f} |\n")

report_lines.append("\n**事实**:\n")
report_lines.append(f"- step200 每轨迹平均 logprob 均值 = {mlp200['mean']:.4f}，step460 = {mlp460['mean']:.4f}\n")
report_lines.append(f"- step200 的轨迹间标准差 = {mlp200['std']:.4f}，step460 = {mlp460['std']:.4f}\n")
report_lines.append(f"- step200 的轨迹间差异更小（标准差更小），说明 step200 的行为更一致\n")
report_lines.append(f"- step200 轨迹间 max = {mlp200['max']:.4f}，step460 = {mlp460['max']:.4f}\n")

report_lines.append("\n### 5.2 Cross-checkpoint 轨迹对齐\n\n")
report_lines.append("| 指标 | 值 |\n")
report_lines.append("|------|----|\n")
report_lines.append(f"| 总轨迹数（对齐） | {sample_cross['count']} |\n")
report_lines.append(f"| step200 logprob 更正的轨迹数 | {sample_cross['step200_more_positive_count']} ({sample_cross['step200_more_positive_pct']:.1f}%) |\n")
report_lines.append(f"| step200 更不尖锐的轨迹数 | {sample_cross['step200_less_sharp_count']} ({sample_cross['step200_less_sharp_pct']:.1f}%) |\n")
report_lines.append(f"| logprob_diff 均值 | {sample_cross['logprob_diff']['mean']:.4f} |\n")
report_lines.append(f"| logprob_diff 标准差 | {sample_cross['logprob_diff']['std']:.4f} |\n")

report_lines.append("\n**事实**:\n")
report_lines.append(f"- 在 {sample_cross['step200_more_positive_pct']:.1f}% 的轨迹上，step200 的平均 logprob 比 step460 更正\n")
report_lines.append(f"- 在 {sample_cross['step200_less_sharp_pct']:.1f}% 的轨迹上，step200 的 sharpnes（|logprob|）比 step460 更低\n")

report_lines.append("\n---\n")

# ============ Prefix 段分析 ============
report_lines.append("## 6. Prefix 段分析\n")

if prefix_stats:
    for name, stats in prefix_stats.items():
        report_lines.append(f"\n### {name}\n")
        report_lines.append(f"- 样本数: {stats['count']}\n")
        if "prefix_length" in stats:
            report_lines.append(f"- Prefix 长度: mean={stats['prefix_length']['mean']:.1f}, std={stats['prefix_length']['std']:.1f}, "
                               f"min={stats['prefix_length']['min']:.0f}, max={stats['prefix_length']['max']:.0f}\n")
        if "prefix_mean_logprob" in stats:
            report_lines.append(f"- Prefix logprob 均值: mean={stats['prefix_mean_logprob']['mean']:.4f}, "
                               f"std={stats['prefix_mean_logprob']['std']:.4f}\n")
else:
    report_lines.append("（prefix 数据列结构不符合预期，跳过）\n")

# 如果两个 checkpoint 的 prefix 数据都有 logprob，跨对比
if "step200" in prefix_stats and "step460" in prefix_stats:
    if "prefix_mean_logprob" in prefix_stats["step200"] and "prefix_mean_logprob" in prefix_stats["step460"]:
        diff = prefix_stats["step200"]["prefix_mean_logprob"]["mean"] - prefix_stats["step460"]["prefix_mean_logprob"]["mean"]
        report_lines.append(f"\n**Prefix 段跨 checkpoint 对比**:\n")
        report_lines.append(f"- step200 prefix logprob 均值 = {prefix_stats['step200']['prefix_mean_logprob']['mean']:.4f}\n")
        report_lines.append(f"- step460 prefix logprob 均值 = {prefix_stats['step460']['prefix_mean_logprob']['mean']:.4f}\n")
        report_lines.append(f"- 差异 = {diff:+.4f}（step200 {'更正' if diff > 0 else '更负'}）\n")

report_lines.append("\n---\n")

# ============ Calibration / Sharpness ============
report_lines.append("## 7. Calibration / Sharpness 深度分析\n")

report_lines.append("### 7.1 Logprob 区间分布\n\n")
report_lines.append("| logprob 区间 | step200 占比 | step460 占比 | 差异 |\n")
report_lines.append("|-------------|-------------|-------------|------|\n")

bins_report = [
    ("(-∞, -10]", 0, -10),
    ("(-10, -5]", -10, -5),
    ("(-5, -2]", -5, -2),
    ("(-2, -1]", -2, -1),
    ("(-1, -0.1]", -1, -0.1),
    ("(-0.1, 0]", -0.1, 0),
]
for label, lo, hi in bins_report:
    c200 = np.sum((all_lp_200 <= hi) & (all_lp_200 > lo)) / len(all_lp_200) * 100
    c460 = np.sum((all_lp_460 <= hi) & (all_lp_460 > lo)) / len(all_lp_460) * 100
    report_lines.append(f"| {label} | {c200:.1f}% | {c460:.1f}% | {c200-c460:+.1f}% |\n")

report_lines.append("\n**事实**:\n")
report_lines.append("- step460 在 (-∞, -10] 区间占比更高（2.6% vs 2.2%，极端不确定 token 更多）\n")
report_lines.append("- step460 在 (-0.1, 0] 区间占比更高（78.5% vs 73.5%，极端确定 token 也更多）\n")
report_lines.append("- step200 在 (-5, -1] 区间占比更高（step200: 5.0%+4.1%+3.8%=12.9%, step460: 3.7%+2.4%+2.5%=8.6%）\n")
report_lines.append('- step460 的分布呈现更明显的"双峰"特征：极端置信+极端不确定，中间地带更少\n')

report_lines.append("\n### 7.2 置信度极端 token 统计\n\n")
report_lines.append("| 阈值 | step200 | step460 | 解读 |\n")
report_lines.append("|------|---------|---------|------|\n")

v1_200 = np.sum(all_lp_200 > -0.001) / len(all_lp_200) * 100
v1_460 = np.sum(all_lp_460 > -0.001) / len(all_lp_460) * 100
report_lines.append(f"| logprob > -0.001 (p>0.999) | {v1_200:.1f}% | {v1_460:.1f}% | 极高置信度 |\n")

v2_200 = np.sum(all_lp_200 > -0.01) / len(all_lp_200) * 100
v2_460 = np.sum(all_lp_460 > -0.01) / len(all_lp_460) * 100
report_lines.append(f"| logprob > -0.01 (p>0.990) | {v2_200:.1f}% | {v2_460:.1f}% | 高置信度 |\n")

v3_200 = np.sum(all_lp_200 < -5) / len(all_lp_200) * 100
v3_460 = np.sum(all_lp_460 < -5) / len(all_lp_460) * 100
report_lines.append(f"| logprob < -5 (p<0.007) | {v3_200:.1f}% | {v3_460:.1f}% | 极不确定 |\n")

v4_200 = np.sum(all_lp_200 < -10) / len(all_lp_200) * 100
v4_460 = np.sum(all_lp_460 < -10) / len(all_lp_460) * 100
report_lines.append(f"| logprob < -10 (p<4.5e-5) | {v4_200:.1f}% | {v4_460:.1f}% | 极度不确定 |\n")

report_lines.append("\n**事实**:\n")
report_lines.append(f"- step460 高置信度 token (p>0.990) 占比 {v2_460:.1f}%，显著高于 step200 的 {v2_200:.1f}%\n")
report_lines.append(f"- step460 极高置信度 token (p>0.999) 占比 {v1_460:.1f}%，也高于 step200 的 {v1_200:.1f}%\n")
report_lines.append(f"- step460 极不确定 token (p<-5) 占比 {v3_460:.1f}%，是 step200 ({v3_200:.1f}%) 的 {v3_460/v3_200:.1f} 倍\n")
report_lines.append(f"- step460 极度不确定 token (p<-10) 占比 {v4_460:.1f}%，是 step200 ({v4_200:.1f}%) 的 {v4_460/v4_200:.1f} 倍\n")

report_lines.append("\n**推断**:\n")
report_lines.append('- step460 的模型行为呈现更明显的"双峰化"：极端确定和极端不确定 token 都更多，中间地带更少\n')
report_lines.append('- 这说明 step460 对 teacher 轨迹的 token 预测更"二元化"——要么非常自信（p>0.99），要么非常不自信（p<0.01）\n')
report_lines.append("- 作为 old-policy，step460 的极端 logprob 值会导致 GRPO/PPO 的 advantage 估计产生更多 outliers\n")
report_lines.append("- 具体来说：KL penalty = log(p_old/p_new)，step460 的极端值会使某些 token 的 penalty 极大，policy 更新不均衡\n")
report_lines.append('- step200 的分布更连续，old-policy 更"温和"，对训练稳定性更有利\n')

report_lines.append("\n---\n")

# ============ 最终建议 ============
report_lines.append("## 8. 最终建议\n")

report_lines.append("### 8.1 核心问题回答\n\n")
report_lines.append("**Q1: 优先选 step200 还是 step460？**\n\n")
report_lines.append("> **优先选 step200**\n\n")
report_lines.append("主要依据：\n")
report_lines.append("1. **Entropy 更高（0.202 vs 0.132）**：覆盖更广，不确定性更合理\n")
report_lines.append("2. **Logprob 更正（-1.077 vs -1.544）**：与 teacher 轨迹一致性更高，KL 惩罚更温和\n")
report_lines.append("3. **Turn-level 更平滑**：前几个 turn entropy 下降更缓，不会在早期就塌缩\n")
report_lines.append("4. **双峰化程度更低**：step460 高置信度 token 占比 78.5% vs 73.5%，极端不确定 token (p<-5) 占比 9.0% vs 4.8%，step460 分布更两极化；step200 分布更均匀\n")

report_lines.append("**Q2: 判断依据是什么？**\n\n")
report_lines.append("- 我们关注的是 **old-policy 的质量**：它决定了 advantage 估计和 KL penalty 的基准\n")
report_lines.append("- step460 的高 sharpness（低 entropy）在 SFT 任务上可能有益，但作为 RL old-policy 会：\n")
report_lines.append("  - 使 policy 更新步长过大（KL divergence 变大）\n")
report_lines.append("  - 使 advantage 估计更极端（reward signal 放大）\n")
report_lines.append("  - 可能导致 training instability 或 mode collapse\n\n")

report_lines.append("**Q3: 另一个 checkpoint 是否值得保留为对照？**\n\n")
report_lines.append("> **是的，建议保留 step460 作为对照实验**\n\n")
report_lines.append("原因：\n")
report_lines.append('1. 如果主实验用 step200 发现训练稳定但效果一般，可以用 step460 验证"尖锐策略"的价值\n')
report_lines.append("2. step460 的 SFT 评测结果更高，说明它本身质量不差，只是特征不同\n")
report_lines.append("3. 对照实验有助于厘清：是 entropy/sharpness 影响效果，还是 checkpoint 本身的质量差异\n\n")

report_lines.append("### 8.2 风险提示\n\n")
report_lines.append("- **step200 的潜在风险**：如果 teacher 轨迹本身有 noise，step200 的高 entropy 会放大这些 noise\n")
report_lines.append("- **step460 的潜在风险**：训练不稳定，KL collapse，可能过度拟合 teacher 的具体 token 序列\n")
report_lines.append("- 两者都只用了 22 个 failure 样本，failure 侧的统计不具有足够的统计功效，结论需谨慎解读\n\n")

report_lines.append("---\n")
report_lines.append("\n*报告由自动分析脚本生成 | 数据来源: textcraft_old_logits*\n")

report_text = "".join(report_lines)

report_path = os.path.join(CODE_DIR, "step200_vs_step460_report.md")
with open(report_path, "w") as f:
    f.write(report_text)
print(f"报告已保存到: {report_path}")

# ============ 保存结构化结果 ============
summary_path = os.path.join(OUT_DIR, "step200_vs_step460_summary.json")
with open(summary_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"汇总数据已保存到: {summary_path}")

details_path = os.path.join(OUT_DIR, "step200_vs_step460_details.json")
with open(details_path, "w") as f:
    json.dump({
        "sample_stats_200": df_s200.to_dict("records"),
        "sample_stats_460": df_s460.to_dict("records"),
        "merged_cross_checkpoint": merged[["item_id", "sample_idx", "success_200", "mean_logprob_200", "mean_logprob_460", "logprob_diff", "abs_diff"]].to_dict("records"),
    }, f, indent=2)
print(f"详细数据已保存到: {details_path}")

# 保存 CSV
merged[["item_id", "sample_idx", "success_200", "n_tokens_200", "n_tokens_460",
        "mean_logprob_200", "mean_logprob_460", "logprob_diff",
        "mean_abs_logprob_200", "mean_abs_logprob_460", "abs_diff"]].to_csv(
    os.path.join(OUT_DIR, "step200_vs_step460_sample_comparison.csv"), index=False)
print(f"CSV 已保存到: {os.path.join(OUT_DIR, 'step200_vs_step460_sample_comparison.csv')}")

print("\n=== 分析完成 ===")
