"""
分析脚本：对比两个 SFT checkpoint 的 old logprob / 熵分布

输入：
- sidecar parquet 文件（step200 vs step460）
- 原始 trajectories jsonl（用于计算 entropy）
- SFT model（用于计算 entropy）

输出：
- 统计报告（markdown）
- 统计摘要（json）

核心功能：
1. token 级统计：assistant token old logprob / entropy 均值/方差/分位数
2. turn 级统计：assistant turn 平均 old logprob / entropy
3. 样本级统计：每条轨迹的平均 old logprob / entropy
4. 按 success/failure 分别统计
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="分析 SFT checkpoint 的 old logprob / 熵分布"
    )
    parser.add_argument(
        "--sidecar_path",
        type=str,
        required=True,
        help="sidecar parquet 路径"
    )
    parser.add_argument(
        "--trajectories_path",
        type=str,
        default=None,
        help="原始 trajectories jsonl 路径（用于计算 entropy）"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="SFT model 路径（用于计算 entropy）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录，默认与 sidecar 同目录"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="设备"
    )
    return parser.parse_args()


def load_sidecar(path: str) -> pd.DataFrame:
    """加载 sidecar"""
    return pd.read_parquet(path)


def load_trajectories(path: str) -> pd.DataFrame:
    """加载原始 trajectories"""
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)


def get_statistics(arr: np.ndarray) -> Dict[str, float]:
    """计算统计信息"""
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def compute_entropy(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    计算每个位置的 entropy
    返回: (seq_len - 1,) 的 entropy 值
    """
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            use_cache=False
        )
        
        logits = outputs.logits[0]
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        
        return entropy[:-1]


def tokenize_conversations(tokenizer, conversations: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用 tokenizer 对 conversations 进行 tokenize
    必须和 sidecar 预处理脚本完全一致：
    1. apply_chat_template(..., tokenize=False)
    2. tokenizer(..., add_special_tokens=True)
    """
    if hasattr(tokenizer, 'apply_chat_template'):
        text = tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=False
        )
    else:
        # 备用方案：手写模板
        text = ""
        for msg in conversations:
            role = msg.get("role", "")
            content = msg.get("content", "")
            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        text += "<|im_end|>"
    
    # 必须用 add_special_tokens=True，和 sidecar 预处理一致
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    
    input_ids = encoded["input_ids"].squeeze(0)
    attention_mask = encoded["attention_mask"].squeeze(0)
    
    return input_ids, attention_mask


def analyze_sidecar(
    sidecar_df: pd.DataFrame,
    trajectories_df: Optional[pd.DataFrame] = None,
    model=None,
    tokenizer=None,
    device=None
) -> Dict[str, Any]:
    """分析 sidecar 的 old logprob / 熵分布"""
    results = {}
    
    # 准备数据
    all_old_logprobs = []
    assistant_mask_1d_list = []
    
    for idx, row in tqdm(sidecar_df.iterrows(), total=len(sidecar_df), desc="准备数据"):
        old_logprobs = np.array(row["sequence_old_logprobs"])
        assistant_mask = np.array(row["assistant_mask"])
        assistant_mask_1d = assistant_mask[1:]
        
        all_old_logprobs.append(old_logprobs)
        assistant_mask_1d_list.append(assistant_mask_1d)
    
    all_old_logprobs_flat = np.concatenate(all_old_logprobs)
    all_assistant_mask_flat = np.concatenate(assistant_mask_1d_list)
    
    # Token 级统计
    results["all_tokens"] = {"old_logprob": get_statistics(all_old_logprobs_flat)}
    
    assistant_old_logprobs = all_old_logprobs_flat[all_assistant_mask_flat > 0.5]
    results["assistant_tokens"] = {
        "old_logprob": get_statistics(assistant_old_logprobs),
        "count": len(assistant_old_logprobs)
    }
    
    # 按 success/failure 统计
    results["by_success"] = {}
    for label, df in [("success", sidecar_df[sidecar_df["success"] == 1]), ("failure", sidecar_df[sidecar_df["success"] == 0])]:
        if len(df) == 0:
            results["by_success"][label] = {"count": 0}
            continue
        
        all_logprobs = np.concatenate([np.array(row["sequence_old_logprobs"]) for _, row in df.iterrows()])
        all_mask = np.concatenate([np.array(row["assistant_mask"])[1:] for _, row in df.iterrows()])
        assistant_logprobs = all_logprobs[all_mask > 0.5]
        
        results["by_success"][label] = {
            "count": len(df),
            "assistant_token_count": len(assistant_logprobs),
            "assistant_old_logprob": get_statistics(assistant_logprobs)
        }
    
    # Turn 级统计
    turn_stats = []
    for idx, row in tqdm(sidecar_df.iterrows(), total=len(sidecar_df), desc="处理 turn"):
        assistant_turn_spans = row["assistant_turn_spans"]
        old_logprobs = np.array(row["sequence_old_logprobs"])
        assistant_mask_1d = np.array(row["assistant_mask"])[1:]
        
        for span in assistant_turn_spans:
            start, end, turn_idx = span["start"], span["end"], span["turn_idx"]
            lp_start, lp_end = max(0, start - 1), end - 1
            turn_logprobs = old_logprobs[lp_start:lp_end]
            
            if len(turn_logprobs) > 0:
                turn_stats.append({
                    "turn_idx": turn_idx,
                    "old_logprob_mean": np.mean(turn_logprobs),
                    "old_logprob_std": np.std(turn_logprobs)
                })
    
    turn_stats_df = pd.DataFrame(turn_stats)
    results["by_turn_idx"] = {}
    for turn_idx in sorted(turn_stats_df["turn_idx"].unique()):
        turn_df = turn_stats_df[turn_stats_df["turn_idx"] == turn_idx]
        results["by_turn_idx"][f"turn_{turn_idx}"] = {
            "count": len(turn_df),
            "mean_old_logprob": float(turn_df["old_logprob_mean"].mean()),
            "std_old_logprob": float(turn_df["old_logprob_mean"].std())
        }
    
    # 样本级统计
    sample_stats = []
    for idx, row in tqdm(sidecar_df.iterrows(), total=len(sidecar_df), desc="处理样本级"):
        old_logprobs = np.array(row["sequence_old_logprobs"])
        assistant_mask_1d = np.array(row["assistant_mask"])[1:]
        assistant_logprobs = old_logprobs[assistant_mask_1d > 0.5]
        
        if len(assistant_logprobs) > 0:
            sample_stats.append({
                "assistant_token_count": len(assistant_logprobs),
                "mean_old_logprob": np.mean(assistant_logprobs)
            })
    
    sample_stats_df = pd.DataFrame(sample_stats)
    results["sample_level"] = {
        "total_samples": len(sample_stats_df),
        "mean_old_logprob": get_statistics(sample_stats_df["mean_old_logprob"].values)
    }
    
    # 计算 entropy（如果提供了 trajectories 和 model）
    if trajectories_df is not None and model is not None and tokenizer is not None:
        print("\n计算 entropy（需要重新 forward）...")
        
        trajectories_index = trajectories_df.set_index(["item_id", "sample_idx"])
        
        all_entropies = []
        all_entropy_mask = []
        entropy_by_turn = defaultdict(list)
        entropy_by_sample = []
        entropy_by_success = {"success": [], "failure": []}
        
        for idx, row in tqdm(sidecar_df.iterrows(), total=len(sidecar_df), desc="计算 entropy"):
            key = (row["item_id"], row["sample_idx"])
            
            # key 不匹配时直接 fail-fast 报错
            if key not in trajectories_index.index:
                raise RuntimeError(
                    f"Trajectories key 缺失: {key}, "
                    f"sidecar 中有但 trajectories 中找不到对应的样本"
                )
            
            conversations = trajectories_index.loc[key]["conversations"]
            
            try:
                input_ids, attention_mask = tokenize_conversations(tokenizer, conversations)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                entropy = compute_entropy(model, tokenizer, input_ids, attention_mask, device)
                # 先转换为 float32，避免 bfloat16 不被 numpy 支持
                entropy = entropy.float().cpu().numpy()
            except Exception as e:
                raise RuntimeError(f"Entropy 计算失败: {key}, 错误: {e}")
            
            assistant_mask_row = np.array(row["assistant_mask"])
            assistant_mask_1d = assistant_mask_row[1:]
            
            # ========== fail-fast 长度一致性校验 ==========
            # 1. entropy 长度必须和 assistant_mask_1d 一致
            if len(entropy) != len(assistant_mask_1d):
                raise RuntimeError(
                    f"长度对齐失败: {key}, "
                    f"entropy_len={len(entropy)}, assistant_mask_1d_len={len(assistant_mask_1d)}, "
                    f"差异={abs(len(entropy) - len(assistant_mask_1d))}"
                )
            
            # 2. assistant_turn_spans 的 end-1 不能超过 entropy 长度
            for span in row["assistant_turn_spans"]:
                end = span["end"]
                if end - 1 >= len(entropy):
                    raise RuntimeError(
                        f"Turn span 越界: {key}, span={span}, "
                        f"lp_end={end-1} >= entropy_len={len(entropy)}"
                    )
            
            all_entropies.append(entropy)
            all_entropy_mask.append(assistant_mask_1d)
            
            # 按 turn 统计
            for span in row["assistant_turn_spans"]:
                start, end, turn_idx = span["start"], span["end"], span["turn_idx"]
                lp_start, lp_end = max(0, start - 1), end - 1
                turn_entropy = entropy[lp_start:lp_end]
                if len(turn_entropy) > 0:
                    entropy_by_turn[turn_idx].append(np.mean(turn_entropy))
            
            # 样本级
            assistant_entropy = entropy[assistant_mask_1d > 0.5]
            if len(assistant_entropy) > 0:
                entropy_by_sample.append(np.mean(assistant_entropy))
                success_label = "success" if row["success"] == 1 else "failure"
                entropy_by_success[success_label].extend(assistant_entropy.tolist())
        
        # 合并统计
        all_entropies_flat = np.concatenate(all_entropies)
        all_entropy_mask_flat = np.concatenate(all_entropy_mask)
        
        assistant_entropies = all_entropies_flat[all_entropy_mask_flat > 0.5]
        results["assistant_tokens"]["entropy"] = get_statistics(assistant_entropies)
        
        results["entropy_by_turn_idx"] = {}
        for turn_idx in sorted(entropy_by_turn.keys()):
            if entropy_by_turn[turn_idx]:
                results["entropy_by_turn_idx"][f"turn_{turn_idx}"] = {
                    "mean": float(np.mean(entropy_by_turn[turn_idx])),
                    "std": float(np.std(entropy_by_turn[turn_idx])),
                    "count": len(entropy_by_turn[turn_idx])
                }
        
        results["sample_level"]["mean_entropy"] = get_statistics(np.array(entropy_by_sample))
        
        results["entropy_by_success"] = {}
        for label in ["success", "failure"]:
            if entropy_by_success[label]:
                results["entropy_by_success"][label] = {
                    "count": len(entropy_by_success[label]),
                    "entropy": get_statistics(np.array(entropy_by_success[label]))
                }
        
        results["entropy_computed"] = True
    else:
        results["entropy_computed"] = False
        results["entropy_note"] = "需要同时提供 --trajectories_path 和 --model_path 才能计算 entropy"
    
    return results


def print_report(results: Dict[str, Any], title: str = ""):
    """打印报告"""
    print("\n" + "=" * 60)
    if title:
        print(f" {title}")
        print("=" * 60)
    
    print("\n### Token 级统计")
    print("\n所有 token old_logprob:")
    for k, v in results["all_tokens"]["old_logprob"].items():
        print(f"  {k}: {v:.4f}")
    
    print(f"\nAssistant token old_logprob (共 {results['assistant_tokens']['count']} 个):")
    for k, v in results["assistant_tokens"]["old_logprob"].items():
        print(f"  {k}: {v:.4f}")
    
    if "entropy" in results["assistant_tokens"]:
        print(f"\nAssistant token entropy:")
        for k, v in results["assistant_tokens"]["entropy"].items():
            print(f"  {k}: {v:.4f}")
    
    print("\n### 按 Success/Failure 统计")
    for label, stats in results["by_success"].items():
        print(f"\n{label} (共 {stats['count']} 个样本):")
        if 'assistant_old_logprob' in stats:
            for k, v in stats["assistant_old_logprob"].items():
                print(f"  old_logprob {k}: {v:.4f}")
        
        if "entropy_by_success" in results and label in results["entropy_by_success"]:
            ent_stats = results["entropy_by_success"][label]
            if 'entropy' in ent_stats:
                for k, v in ent_stats['entropy'].items():
                    print(f"  entropy {k}: {v:.4f}")
    
    print("\n### 按 Turn Index 统计")
    for turn_label, stats in results["by_turn_idx"].items():
        print(f"{turn_label}: 均值={stats['mean_old_logprob']:.4f}, std={stats['std_old_logprob']:.4f}, 样本数={stats['count']}")
    
    if "entropy_by_turn_idx" in results:
        print("\n### Entropy 按 Turn Index 统计")
        for turn_label, stats in results["entropy_by_turn_idx"].items():
            print(f"{turn_label}: 均值={stats['mean']:.4f}, std={stats['std']:.4f}, 样本数={stats['count']}")
    
    print("\n### 样本级统计")
    print(f"总样本数: {results['sample_level']['total_samples']}")
    print("每条轨迹 assistant token 平均 old_logprob:")
    for k, v in results["sample_level"]["mean_old_logprob"].items():
        print(f"  {k}: {v:.4f}")
    
    if "mean_entropy" in results["sample_level"]:
        print("\n每条轨迹 assistant token 平均 entropy:")
        for k, v in results["sample_level"]["mean_entropy"].items():
            print(f"  {k}: {v:.4f}")


def main():
    args = parse_args()
    
    print("=" * 60)
    print("SFT Old Logprob / 熵分布分析")
    print("=" * 60)
    print(f"Sidecar: {args.sidecar_path}")
    if args.trajectories_path:
        print(f"Trajectories: {args.trajectories_path}")
    if args.model_path:
        print(f"Model: {args.model_path}")
    print(f"Device: {args.device}")
    
    # 加载 sidecar
    print("\n加载 sidecar...")
    sidecar_df = load_sidecar(args.sidecar_path)
    print(f"总样本数: {len(sidecar_df)}")
    
    # 加载 trajectories（可选）
    trajectories_df = None
    if args.trajectories_path:
        print(f"\n加载 trajectories: {args.trajectories_path}")
        trajectories_df = load_trajectories(args.trajectories_path)
        print(f"Trajectories 样本数: {len(trajectories_df)}")
    
    # 加载模型（可选）
    model = None
    tokenizer = None
    device = None
    if args.model_path:
        print(f"\n加载模型 {args.model_path}...")
        device = torch.device(args.device)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, pad_token="<|endoftext|>")
        model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
        model = model.to(device)
        model.eval()
        print("模型加载完成")
    
    # 分析
    print("\n开始分析...")
    results = analyze_sidecar(sidecar_df, trajectories_df, model, tokenizer, device)
    
    # 打印报告
    title = Path(args.sidecar_path).stem
    print_report(results, title)
    
    # 保存结果
    output_dir = args.output_dir or str(Path(args.sidecar_path).parent)
    json_path = Path(output_dir) / f"{Path(args.sidecar_path).stem}_analysis.json"
    
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj
    
    results_serializable = convert_to_serializable(results)
    
    with open(json_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\n统计结果已保存到: {json_path}")
    
    # 生成 markdown 报告
    md_path = Path(output_dir) / f"{Path(args.sidecar_path).stem}_analysis.md"
    with open(md_path, 'w') as f:
        f.write(f"# {title} 分析报告\n\n")
        
        f.write("## Token 级统计\n\n")
        f.write("### 所有 Token\n\n")
        f.write("| 指标 | 值 |\n|------|------|\n")
        for k, v in results["all_tokens"]["old_logprob"].items():
            f.write(f"| {k} | {v:.4f} |\n")
        
        f.write("\n### Assistant Token\n\n")
        f.write(f"共 {results['assistant_tokens']['count']} 个\n\n")
        f.write("| 指标 | 值 |\n|------|------|\n")
        for k, v in results["assistant_tokens"]["old_logprob"].items():
            f.write(f"| old_logprob_{k} | {v:.4f} |\n")
        
        if "entropy" in results["assistant_tokens"]:
            for k, v in results["assistant_tokens"]["entropy"].items():
                f.write(f"| entropy_{k} | {v:.4f} |\n")
        
        f.write("\n## 按 Success/Failure 统计\n\n")
        for label, stats in results["by_success"].items():
            f.write(f"### {label} ({stats['count']} 个样本)\n\n")
            if 'assistant_old_logprob' in stats:
                f.write("| 指标 | 值 |\n|------|------|\n")
                for k, v in stats["assistant_old_logprob"].items():
                    f.write(f"| old_logprob_{k} | {v:.4f} |\n")
                f.write("\n")
            
            if "entropy_by_success" in results and label in results["entropy_by_success"]:
                ent_stats = results["entropy_by_success"][label]
                if 'entropy' in ent_stats:
                    for k, v in ent_stats['entropy'].items():
                        f.write(f"| entropy_{k} | {v:.4f} |\n")
                    f.write("\n")
        
        f.write("\n## 按 Turn Index 统计\n\n")
        f.write("| Turn | 均值(old_logprob) | 标准差 | 样本数 |\n|------|------|--------|--------|\n")
        for turn_label, stats in results["by_turn_idx"].items():
            f.write(f"| {turn_label} | {stats['mean_old_logprob']:.4f} | {stats['std_old_logprob']:.4f} | {stats['count']} |\n")
        
        if "entropy_by_turn_idx" in results:
            f.write("\n## Entropy 按 Turn Index 统计\n\n")
            f.write("| Turn | 均值(entropy) | 标准差 | 样本数 |\n|------|------|--------|--------|\n")
            for turn_label, stats in results["entropy_by_turn_idx"].items():
                f.write(f"| {turn_label} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['count']} |\n")
        
        f.write("\n## 样本级统计\n\n")
        f.write(f"总样本数: {results['sample_level']['total_samples']}\n\n")
        f.write("| 指标 | 值 |\n|------|------|\n")
        for k, v in results["sample_level"]["mean_old_logprob"].items():
            f.write(f"| old_logprob_{k} | {v:.4f} |\n")
        
        if "mean_entropy" in results["sample_level"]:
            for k, v in results["sample_level"]["mean_entropy"].items():
                f.write(f"| entropy_{k} | {v:.4f} |\n")
    
    print(f"Markdown 报告已保存到: {md_path}")


if __name__ == "__main__":
    main()
