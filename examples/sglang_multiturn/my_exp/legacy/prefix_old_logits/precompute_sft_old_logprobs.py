"""
预处理脚本：为 prefix RL 预计算 SFT model 的 old logprob

输入：prefix_history_canonicalized.parquet
模型：SFT model (从 teacher 轨迹训练得到的)
输出：带有 cached old logprob 的新 parquet 文件

核心逻辑：
1. 加载 SFT model 和 tokenizer
2. 对每条样本的 prompt 做 teacher-forced forward
3. 提取 assistant token 的 logprob（使用真实 tokenization 差分计算 mask）
4. 生成 assistant_mask 和 assistant_prefix_mask
5. 输出带有缓存字段的新 parquet

【关键设计】
- 采用 fail-fast 模式：任何样本处理失败都直接报错退出
- assistant_mask 通过真实 tokenization 差分推导，不依赖手写模板
- assistant_prefix_mask 定义为 prompt 中已有 assistant 消息的 token span
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(
        description="预计算 SFT model 对 assistant token 的 old logprob"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="输入的 parquet 文件路径"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="输出的 parquet 文件路径"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="SFT model 路径"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最大处理样本数，用于调试"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="设备"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="batch size"
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="是否启用 thinking"
    )
    return parser.parse_args()


def tokenize_messages(
    tokenizer,
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = False,
    enable_thinking: bool = False
) -> Tuple[torch.Tensor, int]:
    """
    使用真实的 apply_chat_template + tokenizer 对消息列表进行 tokenization
    
    返回： (input_ids, token_length)
    """
    if enable_thinking:
        # 启用 thinking 模式：手写模板
        text = ""
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
    else:
        # 标准 chat template
        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False
        )
    
    # 真实 tokenization
    tokens = tokenizer(
        text,
        add_special_tokens=True,
        return_tensors="pt"
    )
    input_ids = tokens.input_ids[0]
    token_length = len(input_ids)
    
    return input_ids, token_length


def compute_token_spans_by_diff(
    tokenizer,
    messages: List[Dict[str, str]],
    enable_thinking: bool = False
) -> List[Tuple[int, int, str]]:
    """
    通过真实 tokenization 差分计算每条消息的 token span
    
    核心思想：
    - 对前 1 条、前 2 条、前 3 条 ... 消息分别做 tokenization
    - 用相邻前缀的 token 长度差来确定每条消息的真实 span
    
    返回：List[(start_pos, end_pos, role), ...]
    - 每条消息的 (起始位置, 结束位置, role)
    - 位置是 token 索引，不是字符索引
    """
    if len(messages) == 0:
        return []
    
    spans = []
    cumulative_length = 0
    
    # 逐条增加消息，用差分确定每条消息的 span
    for i in range(len(messages)):
        # tokenize 到第 i+1 条消息
        _, current_length = tokenize_messages(
            tokenizer,
            messages[:i+1],
            add_generation_prompt=False,
            enable_thinking=enable_thinking
        )
        
        # 这条消息的 token 长度 = 当前长度 - 之前累积长度
        start_pos = cumulative_length
        end_pos = current_length
        role = messages[i]["role"]
        
        spans.append((start_pos, end_pos, role))
        cumulative_length = current_length
    
    return spans


def compute_masks_from_spans(
    spans: List[Tuple[int, int, str]],
    token_length: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    根据消息 spans 计算 assistant_mask 和 assistant_prefix_mask
    
    assistant_mask: 所有 assistant 消息的 token 位置
    assistant_prefix_mask: prompt 中已有 assistant 消息的 token 位置（即 prefix 段）
    
    注意：
    - assistant_prefix_mask 是 assistant_mask 的子集
    - prefix 段 = prompt 中已有的 assistant 消息（replay 前的）
    - continuation 段 = 后续 student 生成的（不在这个离线缓存里）
    """
    assistant_mask = torch.zeros(token_length, dtype=torch.float32, device=device)
    assistant_prefix_mask = torch.zeros(token_length, dtype=torch.float32, device=device)
    
    # 用于判断是否是 prefix（遇到第一个 user 消息后的 assistant 就是 continuation）
    # prefix 应该是 prompt 中已有的所有 assistant 消息
    # 根据 prompt 结构：[..., assistant, user, assistant, user]
    # 最后一个 user 之后的 assistant 才是 continuation
    
    # 找到 prompt 中最后一个 user 消息的位置
    # prompt 结构：system, user, assistant, user, assistant, user
    # prefix assistant 是最后那个 user 之前的 assistant
    last_user_idx = -1
    for i, (start, end, role) in enumerate(spans):
        if role == "user":
            last_user_idx = i
    
    # prefix assistant 是最后那个 user 之前的 assistant
    for i, (start, end, role) in enumerate(spans):
        if role == "assistant":
            # 如果这个 assistant 在最后一个 user 之前，就是 prefix
            if i > last_user_idx:
                # continuation (不在缓存范围内，因为这是 student 生成的)
                pass
            else:
                # prefix
                start_idx = max(0, start)
                end_idx = min(token_length, end)
                if start_idx < end_idx:
                    assistant_mask[start_idx:end_idx] = 1.0
                    assistant_prefix_mask[start_idx:end_idx] = 1.0
    
    return assistant_mask, assistant_prefix_mask


def compute_old_logprobs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    计算 old logprob
    
    使用 teacher-forced 方式：输入 input_ids，预测下一个 token
    返回每个位置的 log probability
    
    Causal LM 的预测关系：
    - logits[i] 预测的是 input_ids[i+1] 的 token
    - 所以 log_probs[i] 对应 input_ids[i+1] 的 log probability
    
    返回的 shape: (seq_len - 1,)
    - 对应 input_ids[1:] 位置的预测
    - 第一个 token (bos) 没有对应的预测
    """
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            use_cache=False
        )
        
        # logits: (batch, seq_len, vocab_size)
        logits = outputs.logits[0]  # (seq_len, vocab_size)
        
        # 计算 log probability
        log_probs = torch.log_softmax(logits, dim=-1)  # (seq_len, vocab_size)
        
        # 提取对应位置的 log prob
        # log_probs[i] 预测的是 input_ids[i+1]
        # 所以我们取 log_probs[:-1]，对应 input_ids[1:] 的预测
        # 返回: (seq_len - 1,)
        return log_probs[:-1]  # 去掉最后一个位置（没有有效的下一个 token）


def process_single_sample(
    model,
    tokenizer,
    row: pd.Series,
    device: torch.device,
    enable_thinking: bool = False
) -> Dict[str, Any]:
    """
    处理单条样本（fail-fast 模式）
    
    返回需要缓存的字段：
    - sequence_log_probs: 完整序列的 logprob (token_length - 1,)
    - assistant_mask: assistant token 位置 (token_length,)
    - assistant_prefix_mask: prefix 段 token 位置 (token_length,)
    - token_length: 序列长度
    - assistant_token_count: assistant token 数量
    - assistant_prefix_token_count: prefix assistant token 数量
    
    注意：如果任何步骤失败，直接抛出异常（fail-fast）
    """
    # 获取 prompt
    messages = row["prompt"].tolist() if hasattr(row["prompt"], 'tolist') else row["prompt"]
    
    # 获取 task_id 和 index 作为稳定 key
    task_id = row["extra_info"]["interaction_kwargs"]["task_id"]
    index = row["extra_info"]["index"]
    
    # ========== 步骤 1：用真实 tokenization 得到完整序列 ==========
    input_ids, token_length = tokenize_messages(
        tokenizer,
        messages,
        add_generation_prompt=False,
        enable_thinking=enable_thinking
    )
    input_ids = input_ids.to(device)
    
    # ========== 步骤 2：用真实 tokenization 差分计算每条消息的 span ==========
    spans = compute_token_spans_by_diff(
        tokenizer,
        messages,
        enable_thinking=enable_thinking
    )
    
    # 验证 span 的总长度是否和 token_length 一致
    if spans:
        last_span_end = spans[-1][1]
        if last_span_end != token_length:
            raise ValueError(
                f"Span mismatch: last span ends at {last_span_end}, "
                f"but tokenization gives {token_length}"
            )
    
    # ========== 步骤 3：根据 spans 计算 mask ==========
    attention_mask = torch.ones_like(input_ids)
    
    assistant_mask, assistant_prefix_mask = compute_masks_from_spans(
        spans, token_length, device
    )
    
    # ========== 步骤 4：计算 old logprobs ==========
    log_probs = compute_old_logprobs(model, input_ids, attention_mask)
    
    # ========== 步骤 5：提取计数 ==========
    assistant_token_count = int(assistant_mask.sum().item())
    assistant_prefix_token_count = int(assistant_prefix_mask.sum().item())
    
    # ========== 步骤 6：静态校验（fail-fast）==========
    # 校验 1: log_probs 长度 = token_length - 1
    if len(log_probs) != token_length - 1:
        raise ValueError(
            f"Static validation failed: len(log_probs)={len(log_probs)} != token_length-1={token_length-1}"
        )
    
    # 校验 2: assistant_mask 长度 = token_length
    if len(assistant_mask) != token_length:
        raise ValueError(
            f"Static validation failed: len(assistant_mask)={len(assistant_mask)} != token_length={token_length}"
        )
    
    # 校验 3: assistant_prefix_mask 长度 = token_length
    if len(assistant_prefix_mask) != token_length:
        raise ValueError(
            f"Static validation failed: len(assistant_prefix_mask)={len(assistant_prefix_mask)} != token_length={token_length}"
        )
    
    # 校验 4: assistant_prefix_mask 是 assistant_mask 的子集
    if not torch.all(assistant_prefix_mask <= assistant_mask):
        raise ValueError(
            "Static validation failed: assistant_prefix_mask must be subset of assistant_mask"
        )
    
    # 校验 5: assistant_prefix_token_count <= assistant_token_count
    if assistant_prefix_token_count > assistant_token_count:
        raise ValueError(
            f"Static validation failed: assistant_prefix_token_count={assistant_prefix_token_count} > assistant_token_count={assistant_token_count}"
        )
    
    # 校验 6: 至少有 assistant token
    if assistant_token_count == 0:
        raise ValueError(f"No assistant tokens found for task_id={task_id}, index={index}")
    
    # ========== 步骤 7：准备输出 ==========
    result = {
        "task_id": task_id,
        "index": index,
        "token_length": token_length,
        "log_prob_length": len(log_probs),  # = token_length - 1
        "assistant_token_count": assistant_token_count,
        "assistant_prefix_token_count": assistant_prefix_token_count,
        # 存储完整的 dense logprob（长度 = token_length - 1）
        # 转换为 float32 以兼容 numpy
        "sequence_log_probs": log_probs.float().cpu().numpy().tolist(),
        # 存储 assistant mask（完整长度 = token_length）
        "assistant_mask": assistant_mask.cpu().numpy().tolist(),
        # 存储 assistant prefix mask（完整长度 = token_length）
        "assistant_prefix_mask": assistant_prefix_mask.cpu().numpy().tolist(),
    }
    
    return result


def main():
    args = parse_args()
    
    print("=" * 60)
    print("SFT Old Logprob 预计算")
    print("=" * 60)
    print(f"输入: {args.input_path}")
    print(f"输出: {args.output_path}")
    print(f"模型: {args.model_path}")
    print(f"设备: {args.device}")
    print()
    
    # 加载数据
    print("加载数据...")
    df = pd.read_parquet(args.input_path)
    if args.max_samples is not None:
        df = df.head(args.max_samples)
    print(f"总样本数: {len(df)}")
    
    # 加载模型
    print("加载模型...")
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        pad_token="<|endoftext|>"
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model = model.to(device)
    model.eval()
    print("模型加载完成")
    
    # ========== 处理每个样本（fail-fast 模式）==========
    print("开始处理样本（fail-fast 模式）...")
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理中"):
        # fail-fast：任何异常都直接抛出，不跳过
        result = process_single_sample(
            model, tokenizer, row, device, args.enable_thinking
        )
        results.append(result)
    
    print(f"\n成功处理 {len(results)} / {len(df)} 个样本")
    
    if len(results) == 0:
        print("错误：没有成功处理任何样本")
        return
    
    # ========== 转换为 DataFrame 并与原数据对齐 ==========
    results_df = pd.DataFrame(results)
    
    # 按 task_id 和 index 合并，确保不会因为跳过导致错位
    # 从原数据中提取 key
    df_keys = df["extra_info"].apply(
        lambda x: (x["interaction_kwargs"]["task_id"], x["index"])
    ).tolist()
    results_df["merge_key"] = list(zip(results_df["task_id"], results_df["index"]))
    
    # 创建原数据的 key 列
    df_with_key = df.copy()
    df_with_key["merge_key"] = df_keys
    
    # 按 key 合并
    output_df = df_with_key.merge(
        results_df,
        on="merge_key",
        how="left",
        suffixes=("", "_cache")
    )
    
    # 检查是否有未匹配的样本
    unmatched = output_df["sequence_log_probs"].isna().sum()
    if unmatched > 0:
        raise RuntimeError(f"发现 {unmatched} 个未匹配的样本，合并失败")
    
    # ========== 额外静态校验（整个数据集层面）==========
    print("\n执行静态校验...")
    
    # 校验 1: sequence_log_probs 长度 = token_length - 1
    length_check = output_df.apply(
        lambda row: len(row["sequence_log_probs"]) == row["token_length"] - 1,
        axis=1
    )
    if not length_check.all():
        raise RuntimeError("静态校验失败：存在 sequence_log_probs 长度不匹配的样本")
    
    # 校验 2: assistant_mask 长度 = token_length
    mask_check = output_df.apply(
        lambda row: len(row["assistant_mask"]) == row["token_length"],
        axis=1
    )
    if not mask_check.all():
        raise RuntimeError("静态校验失败：存在 assistant_mask 长度不匹配的样本")
    
    # 校验 3: assistant_prefix_mask 长度 = token_length
    prefix_mask_check = output_df.apply(
        lambda row: len(row["assistant_prefix_mask"]) == row["token_length"],
        axis=1
    )
    if not prefix_mask_check.all():
        raise RuntimeError("静态校验失败：存在 assistant_prefix_mask 长度不匹配的样本")
    
    # 校验 4: assistant_prefix_mask 是 assistant_mask 的子集
    subset_check = output_df.apply(
        lambda row: all(
            p <= a for p, a in zip(row["assistant_prefix_mask"], row["assistant_mask"])
        ),
        axis=1
    )
    if not subset_check.all():
        raise RuntimeError("静态校验失败：存在 assistant_prefix_mask 不是 assistant_mask 子集的样本")
    
    # 校验 5: assistant_prefix_token_count <= assistant_token_count
    count_check = output_df.apply(
        lambda row: row["assistant_prefix_token_count"] <= row["assistant_token_count"],
        axis=1
    )
    if not count_check.all():
        raise RuntimeError("静态校验失败：存在 assistant_prefix_token_count > assistant_token_count 的样本")
    
    print("静态校验通过！")
    
    # ========== 保存 ==========
    # 删除临时合并列
    output_df = output_df.drop(columns=["merge_key"])
    
    print(f"保存到 {args.output_path}...")
    output_df.to_parquet(args.output_path, index=False)
    print("完成!")
    
    # ========== 打印统计 ==========
    print()
    print("=" * 60)
    print("统计信息")
    print("=" * 60)
    print(f"token_length 范围: {output_df['token_length'].min()} - {output_df['token_length'].max()}")
    print(f"assistant_token_count 范围: {output_df['assistant_token_count'].min()} - {output_df['assistant_token_count'].max()}")
    print(f"assistant_prefix_token_count 范围: {output_df['assistant_prefix_token_count'].min()} - {output_df['assistant_prefix_token_count'].max()}")
    print(f"log_prob_length 范围: {output_df['log_prob_length'].min()} - {output_df['log_prob_length'].max()}")
    
    # ========== 展示第一条样本 ==========
    print()
    print("=" * 60)
    print("第一条样本详情")
    print("=" * 60)
    row = output_df.iloc[0]
    print(f"task_id: {row['extra_info']['interaction_kwargs']['task_id']}")
    print(f"index: {row['extra_info']['index']}")
    print(f"token_length: {row['token_length']}")
    print(f"len(sequence_log_probs): {len(row['sequence_log_probs'])}")
    print(f"assistant_token_count: {row['assistant_token_count']}")
    print(f"assistant_prefix_token_count: {row['assistant_prefix_token_count']}")
    print(f"len(assistant_mask): {len(row['assistant_mask'])}")
    print(f"len(assistant_prefix_mask): {len(row['assistant_prefix_mask'])}")
    
    # 验证 mask 和 log_probs 的对齐关系
    print()
    print("Mask 和 Log-probs 对齐验证:")
    print(f"  len(assistant_mask) == token_length: {len(row['assistant_mask']) == row['token_length']}")
    print(f"  len(sequence_log_probs) == token_length - 1: {len(row['sequence_log_probs']) == row['token_length'] - 1}")
    print(f"  len(assistant_prefix_mask) == token_length: {len(row['assistant_prefix_mask']) == row['token_length']}")
    
    # 验证 subset 关系
    assistant_mask = np.array(row['assistant_mask'])
    assistant_prefix_mask = np.array(row['assistant_prefix_mask'])
    is_subset = all(p <= a for p, a in zip(assistant_prefix_mask, assistant_mask))
    print(f"  assistant_prefix_mask 是 assistant_mask 子集: {is_subset}")
    
    # 打印 assistant prefix token logprob 示例
    print()
    print("Assistant prefix token logprob 示例:")
    log_probs = np.array(row['sequence_log_probs'])
    # assistant_prefix_mask[1:] 对应 log_probs 位置
    prefix_indices = assistant_prefix_mask[1:] > 0.5
    if prefix_indices.sum() > 0:
        prefix_logprobs = log_probs[prefix_indices]
        print(f"  assistant_prefix_token_count: {int(prefix_indices.sum())}")
        print(f"  prefix logprob 前3个: {prefix_logprobs[:3]}")
    else:
        print("  无 assistant prefix token")


if __name__ == "__main__":
    main()
