"""
预处理脚本：为 prefix RL 预计算 SFT model 的 old logprob

【输入】
- 原始完整 teacher 轨迹: textcraft_trajectories.jsonl
  - 每条样本 = 一次完整的 teacher 尝试
  - item_id + sample_idx 构成唯一标识

【输出】
- sidecar parquet 文件，包含完整轨迹级的缓存
  - 后续可用于不同 prefix cut 策略

【核心设计】
1. 输入是原始 jsonl，不是已经切好的 prefix parquet
2. 缓存完整序列的 per-token scalar old logprob（不是整行 vocab）
3. 缓存 assistant_turn_spans，支持任意 cut 策略
4. 采用 fail-fast 模式：任何样本处理失败都直接报错退出
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
        description="预计算 SFT model 对完整 teacher 轨迹的 old logprob"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="输入的 jsonl 文件路径（原始 teacher 轨迹）"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="输出的 sidecar parquet 文件路径"
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


def tokenize_conversations(
    tokenizer,
    conversations: List[Dict[str, str]],
    enable_thinking: bool = False
) -> Tuple[torch.Tensor, int]:
    """
    使用真实的 apply_chat_template + tokenizer 对对话列表进行 tokenization
    
    返回： (input_ids, token_length)
    """
    # 构建文本
    if enable_thinking:
        # 启用 thinking 模式：手写模板
        text = ""
        for msg in conversations:
            role = msg["role"]
            content = msg.get("content", "")
            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    else:
        # 标准 chat template
        text = tokenizer.apply_chat_template(
            conversations,
            add_generation_prompt=False,
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
    conversations: List[Dict[str, str]],
    enable_thinking: bool = False
) -> List[Tuple[int, int, str, int]]:
    """
    通过 offset_mapping 计算每条消息的 token span
    
    返回：List[(start_pos, end_pos, role, turn_idx), ...]
    - 每条消息的 (起始位置, 结束位置, role, turn_idx)
    - turn_idx: 统计每个 role 的出现次数（第几个该 role 的消息）
    - 位置是 token 索引，不是字符索引
    """
    if len(conversations) == 0:
        return []
    
    import re
    
    # 使用 apply_chat_template 生成完整文本
    full_text = tokenizer.apply_chat_template(
        conversations,
        add_generation_prompt=False,
        tokenize=False
    )
    
    # tokenize 并获取 offset_mapping
    result = tokenizer(
        full_text,
        add_special_tokens=True,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    input_ids = result.input_ids[0]
    offset_mapping = result.offset_mapping[0]
    token_length = len(input_ids)
    
    # 找出所有 <|im_start|> 和 <|im_end|> 的字符位置
    start_pattern = re.compile(r'<\|im_start\|>(user|assistant|tool|system)')
    end_pattern = re.compile(r'<\|im_end\|>')
    
    start_matches = list(start_pattern.finditer(full_text))
    end_matches = list(end_pattern.finditer(full_text))
    
    if len(start_matches) != len(conversations) or len(end_matches) != len(conversations):
        raise ValueError(
            f"Mismatch: {len(start_matches)} start tags, {len(end_matches)} end tags, "
            f"but {len(conversations)} conversations"
        )
    
    spans = []
    role_counters = {"user": 0, "assistant": 0}
    
    for i, msg in enumerate(conversations):
        role = msg["role"]
        
        # 消息内容范围：start tag 结束后到 end tag 开始后（包括 <|im_end|>）
        start_char = start_matches[i].end()
        end_char = end_matches[i].end()  # 改为 .end() 以包含 <|im_end|>
        
        # 找到对应的 token 范围
        start_token = None
        end_token = None
        for t, (s, e) in enumerate(offset_mapping):
            if s is None:  # special token
                continue
            # 这个 token 与消息字符范围有交集
            if s < end_char and e > start_char:
                if start_token is None:
                    start_token = t
                end_token = t + 1  # end is exclusive
        
        if start_token is None or end_token is None:
            raise ValueError(f"Could not find token range for message {i}, role={role}")
        
        # 统计 turn_idx
        role_counters[role] += 1
        turn_idx = role_counters[role]
        
        spans.append((start_token, end_token, role, turn_idx))
    
    return spans


def compute_masks_and_turn_spans(
    spans: List[Tuple[int, int, str, int]],
    token_length: int,
    device: torch.device
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """
    根据消息 spans 计算 assistant_mask 和 assistant_turn_spans
    
    assistant_mask: 所有 assistant 消息的 token 位置 (token_length,)
    assistant_turn_spans: 每个 assistant turn 的信息，用于后续不同 cut 策略
    
    返回：
    - assistant_mask: (token_length,)
    - assistant_turn_spans: list of {start, end, turn_idx}
    """
    assistant_mask = torch.zeros(token_length, dtype=torch.float32, device=device)
    assistant_turn_spans = []
    
    for (start_pos, end_pos, role, turn_idx) in spans:
        if role == "assistant":
            start_idx = max(0, start_pos)
            end_idx = min(token_length, end_pos)
            if start_idx < end_idx:
                assistant_mask[start_idx:end_idx] = 1.0
                assistant_turn_spans.append({
                    "start": start_idx,
                    "end": end_idx,
                    "turn_idx": turn_idx
                })
    
    return assistant_mask, assistant_turn_spans


def compute_old_logprobs_scalar(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    计算 per-token scalar old logprob
    
    使用 teacher-forced 方式：输入 input_ids，预测下一个 token
    返回每个位置对真实下一个 token 的 log probability
    
    Causal LM 的预测关系：
    - logits[i] 预测的是 input_ids[i+1] 的 token
    - 所以 log_probs[i] 对应 input_ids[i+1] 的 log probability
    
    【关键修正】：现在返回的是 scalar logprob，不是整行 vocab！
    
    返回的 shape: (seq_len - 1,)
    - 对应 input_ids[1:] 位置的预测
    - 第一个 token (bos) 没有对应的预测
    - 每个元素是 scalar（对应真实 token 的 logprob）
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
        
        # 【关键修正】：提取真实 token 的 logprob（scalar），不是整行
        # log_probs[i] 预测的是 input_ids[i+1]
        # 所以我们取 input_ids[1:] 作为索引，取出对应的 scalar logprob
        target_tokens = input_ids[1:]  # (seq_len - 1,)
        scalar_logprobs = log_probs[:-1].gather(
            dim=1,
            index=target_tokens.unsqueeze(-1)
        ).squeeze(-1)  # (seq_len - 1,)
        
        return scalar_logprobs


def process_single_sample(
    model,
    tokenizer,
    sample: Dict[str, Any],
    device: torch.device,
    enable_thinking: bool = False
) -> Dict[str, Any]:
    """
    处理单条样本（fail-fast 模式）
    
    【重要】：这是基于完整 teacher 轨迹，不只是 prefix！
    
    返回需要缓存的字段：
    - sequence_old_logprobs: 完整序列的 per-token scalar logprob (token_length - 1,)
    - assistant_mask: assistant token 位置 (token_length,)
    - assistant_turn_spans: 每个 assistant turn 的起止位置（用于后续不同 cut）
    - token_length: 序列长度
    - assistant_token_count: assistant token 数量
    
    注意：如果任何步骤失败，直接抛出异常（fail-fast）
    """
    # 获取基本信息
    item_id = sample["item_id"]
    sample_idx = sample["sample_idx"]
    conversations = sample["conversations"]
    success = sample.get("success", 0)
    reward = sample.get("reward", 0.0)
    
    # ========== 步骤 1：用真实 tokenization 得到完整序列 ==========
    input_ids, token_length = tokenize_conversations(
        tokenizer,
        conversations,
        enable_thinking=enable_thinking
    )
    input_ids = input_ids.to(device)
    
    # ========== 步骤 2：用真实 tokenization 差分计算每条消息的 span ==========
    spans = compute_token_spans_by_diff(
        tokenizer,
        conversations,
        enable_thinking=enable_thinking
    )
    
    # 验证 span 的总长度是否和 token_length 一致
    # 注意：由于 tokenizer 会额外处理换行符等，span 可能比 token_length 少1-2个
    # 这里改为更宽松的校验：span 长度应该在 [token_length - 2, token_length] 范围内
    if spans:
        last_span_end = spans[-1][1]
        # 更宽松的校验：允许 ±2 的误差
        if abs(last_span_end - token_length) > 2:
            raise ValueError(
                f"Span mismatch: last span ends at {last_span_end}, "
                f"but tokenization gives {token_length} (diff > 2)"
            )
    
    # ========== 步骤 3：根据 spans 计算 mask 和 turn spans ==========
    attention_mask = torch.ones_like(input_ids)
    
    assistant_mask, assistant_turn_spans = compute_masks_and_turn_spans(
        spans, token_length, device
    )
    
    # ========== 步骤 4：计算 per-token scalar old logprobs ==========
    # 【关键修正】：现在返回的是 scalar logprob，不是整行 vocab
    scalar_logprobs = compute_old_logprobs_scalar(model, input_ids, attention_mask)
    
    # ========== 步骤 5：提取计数 ==========
    assistant_token_count = int(assistant_mask.sum().item())
    
    # ========== 步骤 6：静态校验（fail-fast）==========
    # 校验 1: scalar_logprobs 长度 = token_length - 1
    if len(scalar_logprobs) != token_length - 1:
        raise ValueError(
            f"Static validation failed: len(scalar_logprobs)={len(scalar_logprobs)} != token_length-1={token_length-1}"
        )
    
    # 校验 2: assistant_mask 长度 = token_length
    if len(assistant_mask) != token_length:
        raise ValueError(
            f"Static validation failed: len(assistant_mask)={len(assistant_mask)} != token_length={token_length}"
        )
    
    # 校验 3: assistant_turn_spans 不越界
    for span in assistant_turn_spans:
        if span["start"] < 0 or span["end"] > token_length:
            raise ValueError(
                f"Static validation failed: assistant_turn_span out of bounds: {span}, token_length={token_length}"
            )
        if span["start"] >= span["end"]:
            raise ValueError(
                f"Static validation failed: assistant_turn_span start >= end: {span}"
            )
    
    # 校验 4: assistant_turn_spans 之间不重叠
    for i in range(len(assistant_turn_spans)):
        for j in range(i + 1, len(assistant_turn_spans)):
            span_i = assistant_turn_spans[i]
            span_j = assistant_turn_spans[j]
            # 检查是否重叠
            if not (span_i["end"] <= span_j["start"] or span_j["end"] <= span_i["start"]):
                raise ValueError(
                    f"Static validation failed: assistant_turn_spans overlap: {span_i} and {span_j}"
                )
    
    # 校验 5: 至少有 assistant token
    if assistant_token_count == 0:
        raise ValueError(f"No assistant tokens found for item_id={item_id}, sample_idx={sample_idx}")
    
    # ========== 步骤 7：准备输出 ==========
    result = {
        "item_id": item_id,
        "sample_idx": sample_idx,
        "token_length": token_length,
        "old_logprob_length": len(scalar_logprobs),  # = token_length - 1
        "assistant_token_count": assistant_token_count,
        "assistant_turn_count": len(assistant_turn_spans),
        "success": success,
        "reward": reward,
        # 存储完整的 per-token scalar logprob（长度 = token_length - 1）
        # 转换为 float32 以兼容 numpy
        "sequence_old_logprobs": scalar_logprobs.float().cpu().numpy().tolist(),
        # 存储 assistant mask（完整长度 = token_length）
        "assistant_mask": assistant_mask.cpu().numpy().tolist(),
        # 存储 assistant turn spans（用于后续不同 cut 策略）
        "assistant_turn_spans": assistant_turn_spans,
    }
    
    return result


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """加载 jsonl 文件"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def main():
    args = parse_args()
    
    print("=" * 60)
    print("SFT Old Logprob 预计算（完整轨迹级）")
    print("=" * 60)
    print(f"输入: {args.input_path}")
    print(f"输出: {args.output_path}")
    print(f"模型: {args.model_path}")
    print(f"设备: {args.device}")
    print()
    
    # 加载数据
    print("加载数据...")
    samples = load_jsonl(args.input_path)
    if args.max_samples is not None:
        samples = samples[:args.max_samples]
    print(f"总样本数: {len(samples)}")
    
    # 打印输入数据结构
    if samples:
        print(f"\n输入数据结构示例:")
        print(f"  字段: {list(samples[0].keys())}")
        print(f"  item_id: {samples[0]['item_id']}")
        print(f"  sample_idx: {samples[0]['sample_idx']}")
        print(f"  conversations 长度: {len(samples[0]['conversations'])}")
        print(f"  success: {samples[0].get('success', 'N/A')}")
        print(f"  reward: {samples[0].get('reward', 'N/A')}")
    
    # 加载模型
    print("\n加载模型...")
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
    print("\n开始处理样本（fail-fast 模式）...")
    results = []
    
    for idx, sample in tqdm(enumerate(samples), total=len(samples), desc="处理中"):
        # fail-fast：任何异常都直接抛出，不跳过
        result = process_single_sample(
            model, tokenizer, sample, device, args.enable_thinking
        )
        results.append(result)
    
    print(f"\n成功处理 {len(results)} / {len(samples)} 个样本")
    
    if len(results) == 0:
        print("错误：没有成功处理任何样本")
        return
    
    # ========== 转换为 DataFrame ==========
    results_df = pd.DataFrame(results)
    
    # ========== 额外静态校验（整个数据集层面）==========
    print("\n执行静态校验...")
    
    # 校验 1: sequence_old_logprobs 长度 = token_length - 1
    length_check = results_df.apply(
        lambda row: len(row["sequence_old_logprobs"]) == row["token_length"] - 1,
        axis=1
    )
    if not length_check.all():
        raise RuntimeError("静态校验失败：存在 sequence_old_logprobs 长度不匹配的样本")
    
    # 校验 2: assistant_mask 长度 = token_length
    mask_check = results_df.apply(
        lambda row: len(row["assistant_mask"]) == row["token_length"],
        axis=1
    )
    if not mask_check.all():
        raise RuntimeError("静态校验失败：存在 assistant_mask 长度不匹配的样本")
    
    # 校验 3: assistant_turn_spans 不越界
    bounds_check = results_df.apply(
        lambda row: all(
            0 <= span["start"] < span["end"] <= row["token_length"]
            for span in row["assistant_turn_spans"]
        ),
        axis=1
    )
    if not bounds_check.all():
        raise RuntimeError("静态校验失败：存在 assistant_turn_spans 越界的样本")
    
    # 校验 4: assistant_turn_spans 不重叠
    def check_no_overlap(spans):
        for i in range(len(spans)):
            for j in range(i + 1, len(spans)):
                if not (spans[i]["end"] <= spans[j]["start"] or spans[j]["end"] <= spans[i]["start"]):
                    return False
        return True
    
    overlap_check = results_df["assistant_turn_spans"].apply(check_no_overlap)
    if not overlap_check.all():
        raise RuntimeError("静态校验失败：存在 assistant_turn_spans 重叠的样本")
    
    # 校验 5: 有唯一标识可用于后续 join
    unique_check = results_df[["item_id", "sample_idx"]].drop_duplicates().shape[0] == len(results_df)
    if not unique_check:
        raise RuntimeError("静态校验失败：存在重复的 (item_id, sample_idx) 组合")
    
    print("静态校验通过！")
    
    # ========== 保存 ==========
    print(f"\n保存到 {args.output_path}...")
    results_df.to_parquet(args.output_path, index=False)
    print("完成!")
    
    # ========== 打印统计 ==========
    print()
    print("=" * 60)
    print("统计信息")
    print("=" * 60)
    print(f"token_length 范围: {results_df['token_length'].min()} - {results_df['token_length'].max()}")
    print(f"assistant_token_count 范围: {results_df['assistant_token_count'].min()} - {results_df['assistant_token_count'].max()}")
    print(f"assistant_turn_count 范围: {results_df['assistant_turn_count'].min()} - {results_df['assistant_turn_count'].max()}")
    print(f"old_logprob_length 范围: {results_df['old_logprob_length'].min()} - {results_df['old_logprob_length'].max()}")
    print(f"成功样本数: {(results_df['success'] == 1).sum()}")
    print(f"失败样本数: {(results_df['success'] == 0).sum()}")
    
    # ========== 展示第一条样本 ==========
    print()
    print("=" * 60)
    print("第一条样本详情")
    print("=" * 60)
    row = results_df.iloc[0]
    print(f"item_id: {row['item_id']}")
    print(f"sample_idx: {row['sample_idx']}")
    print(f"token_length: {row['token_length']}")
    print(f"len(sequence_old_logprobs): {len(row['sequence_old_logprobs'])}")
    print(f"assistant_token_count: {row['assistant_token_count']}")
    print(f"assistant_turn_count: {row['assistant_turn_count']}")
    print(f"len(assistant_mask): {len(row['assistant_mask'])}")
    print(f"assistant_turn_spans: {row['assistant_turn_spans']}")
    
    # 验证 mask 和 log_probs 的对齐关系
    print()
    print("Mask 和 Log-probs 对齐验证:")
    print(f"  len(assistant_mask) == token_length: {len(row['assistant_mask']) == row['token_length']}")
    print(f"  len(sequence_old_logprobs) == token_length - 1: {len(row['sequence_old_logprobs']) == row['token_length'] - 1}")
    
    # 打印 assistant token logprob 示例
    print()
    print("Assistant token logprob 示例:")
    old_logprobs = np.array(row['sequence_old_logprobs'])
    assistant_mask = np.array(row['assistant_mask'])
    # assistant_mask[1:] 对应 log_probs 位置
    assistant_indices = assistant_mask[1:] > 0.5
    if assistant_indices.sum() > 0:
        assistant_logprobs = old_logprobs[assistant_indices]
        print(f"  assistant_token_count: {int(assistant_indices.sum())}")
        print(f"  assistant logprob 前5个: {assistant_logprobs[:5]}")
        print(f"  assistant logprob 均值: {assistant_logprobs.mean():.4f}")
    else:
        print("  无 assistant token")


if __name__ == "__main__":
    main()