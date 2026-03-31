"""
数据管道脚本：sidecar -> prefix 训练样本（对齐到截断 prompt）

功能：
1. 读取训练 parquet（prompt = 截断后的 K-turn 对话）
2. 读取完整 trajectory sidecar 和原始 jsonl
3. 用 item_id + cumcount(sample_idx) join
4. 用 tokenizer 重新 tokenize 完整轨迹，提取与 K-turn 完全对齐的 old_logprobs
5. 生成新的训练前数据文件

核心修复（根因）：之前的版本用 fixed_ratio 在 N-turn 坐标系下做 cut，
导致 parquet 的 prefix old_logprobs 与训练 prompt 的 assistant token 数系统性不一致。
新版本用训练 parquet 的 prompt 中 K 个 assistant turns 作为 ground truth，
重新 tokenize 完整轨迹，提取与 K-turn 完全对齐的 old_logprobs。

Join 映射：
- task_id (训练数据) -> item_id: "textcraft_{task_id}"
- sample_idx (训练数据) -> sample_idx (jsonl): 需要 cumcount per task
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="构建 prefix 训练数据 + old logprob（对齐到截断 prompt tokenization）"
    )
    parser.add_argument(
        "--prefix_data_path",
        type=str,
        required=True,
        help="prefix 训练数据 parquet 路径"
    )
    parser.add_argument(
        "--sidecar_path",
        type=str,
        required=True,
        help="完整 trajectory sidecar parquet 路径"
    )
    parser.add_argument(
        "--original_jsonl_path",
        type=str,
        required=True,
        help="原始 teacher 轨迹 jsonl 路径（用于字符级对齐）"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="输出的训练数据路径"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="模型路径（用于 tokenizer）"
    )
    return parser.parse_args()


def build_join_key(prefix_df: pd.DataFrame) -> pd.DataFrame:
    """
    从 prefix 数据构建 join key
    
    映射规则：
    - task_id (训练数据) -> item_id (sidecar): "textcraft_{task_id}"
    - index (训练数据) -> sample_idx (sidecar): 需要转换为每个 task 内的本地索引
    
    注意：prefix 的 index 是全局连续索引（如 0,1,2,3,4,5,6,7...），而 sidecar 的 
    sample_idx 是每个 task 内的本地索引（如 task 31: 0,1,2,3; task 32: 0,1,2,3）。
    需要转换为本地索引才能正确匹配。
    """
    prefix_df = prefix_df.copy()
    
    # 提取 task_id 和全局 index
    prefix_df["task_id"] = prefix_df["extra_info"].apply(
        lambda x: x["interaction_kwargs"]["task_id"]
    )
    prefix_df["global_index"] = prefix_df["extra_info"].apply(
        lambda x: x["index"]
    )
    
    # 构建 item_id: "textcraft_{task_id}"
    prefix_df["item_id"] = prefix_df["task_id"].apply(lambda x: f"textcraft_{x}")
    
    # 将全局 index 转换为每个 task 内的本地索引
    # 先按 task_id 排序，然后组内排名
    prefix_df = prefix_df.sort_values(["task_id", "global_index"])
    prefix_df["sample_idx"] = prefix_df.groupby("task_id").cumcount()
    
    return prefix_df


def _tokenize_conversations_for_alignment(
    conversations: List[Dict],
    tokenizer,
) -> Tuple[List[str], List[Tuple[int, int, str]]]:
    """
    Tokenize conversations and return assistant turn spans (content char offsets).

    Returns:
        full_text: concatenated text
        assistant_spans: list of (content_start_char, content_end_char, role) for each message
        role_start_chars: list of (role_start_char, role) for each message
    """
    import re

    full_text = ""
    assistant_spans = []
    role_start_chars = []

    for msg in conversations:
        role = msg.get("role", "")
        content = msg.get("content", "")
        role_text = f"<|im_start|>{role}\n"
        role_start_char = len(full_text) + len(role_text)
        role_start_chars.append((len(full_text), role))
        full_text += role_text + content + "<|im_end|>\n"
        if role == "assistant":
            assistant_spans.append((role_start_char, len(full_text) - len("<|im_end|>\n"), role))

    return full_text, assistant_spans, role_start_chars


def _extract_prefix_old_logprobs_aligned(
    truncated_prompt: List[Dict],
    full_conversation_assistant_spans_char: List[Tuple[int, int, str]],
    sequence_old_logprobs: List[float],
    full_conversation_text: str,
    tokenizer,
) -> Tuple[List[float], List[int], Tuple[int, int]]:
    """
    Extract prefix old_logprobs aligned to the TRUNCATED prompt tokenization.

    Root fix: Previously the build script computed prefix spans using the FULL
    trajectory's K-turn cut (in FULL tokenization coords), but the training
    parquet's `prompt` field only contains K assistant turns in the TRUNCATED
    tokenization. This caused systematic mismatch.

    Solution:
    1. Use the TRUNCATED prompt's K assistant turns as ground truth.
    2. Re-tokenize the full conversation with the SAME tokenizer (Qwen3).
    3. Find token positions of each truncated assistant turn by matching character spans.
    4. Extract old_logprobs for exactly those K turns.

    The key is: we use the same tokenizer for the full conversation as for the
    truncated prompt, so token positions are consistent.

    Args:
        truncated_prompt: the K-turn chat messages (ground truth for K)
        full_conversation_assistant_spans_char: assistant turn char spans from FULL conversation
        sequence_old_logprobs: old_logprobs from precompute (in FULL tokenization coords)
        full_conversation_text: full conversation text
        tokenizer: tokenizer to use (must be the same one used by precompute)

    Returns:
        prefix_old_logprobs: old_logprobs for K turns (in token coords)
        prefix_mask: 1 for assistant token positions
        prefix_token_span: (prefix_start_token, prefix_end_token) in FULL token coords
    """
    # Step 1: Tokenize full conversation with offset mapping
    tokens = tokenizer(
        full_conversation_text,
        add_special_tokens=True,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    input_ids = tokens.input_ids[0]
    offset_mapping = tokens.offset_mapping[0]
    token_length = len(input_ids)

    # Step 2: Find token positions for each assistant turn in FULL conversation
    full_assistant_token_spans = []
    for (char_start, char_end, role) in full_conversation_assistant_spans_char:
        if role != "assistant":
            continue
        start_token = None
        end_token = None
        for t, (cs, ce) in enumerate(offset_mapping.tolist()):
            if cs is None:  # special token
                continue
            if cs < char_end and ce > char_start:
                if start_token is None:
                    start_token = t
                end_token = t + 1  # exclusive
        if start_token is not None and end_token is not None:
            full_assistant_token_spans.append((start_token, end_token))

    # Step 3: The truncated prompt's K assistant turns ARE the first K
    # assistant turns of the full conversation (in order).
    # No greedy matching needed — just take the first K in order.
    if len(full_assistant_token_spans) < K:
        raise ValueError(
            f"Not enough assistant turns in full conversation ({len(full_assistant_token_spans)}) "
            f"for K={K} truncated turns. "
            f"The truncated parquet may have more assistant turns than the full conversation."
        )
    matched_token_spans = full_assistant_token_spans[:K]

    if not matched_token_spans:
        return [], [0], (0, 0)

    # Step 4: Compute prefix span in FULL token coords
    first_token = matched_token_spans[0][0]
    last_token = matched_token_spans[-1][1]

    # Step 5: Extract old_logprobs for the prefix span
    # Old logprobs are next-token coords: logprob[t] = log P(input_ids[t+1] | input_ids[:t+1])
    # Token span [first_token, last_token) maps to old_logprob span [first_token-1, last_token-1)
    lp_start = max(0, first_token - 1)
    lp_end = min(len(sequence_old_logprobs), last_token - 1)
    prefix_old_logprobs = list(sequence_old_logprobs[lp_start:lp_end])

    # Step 6: Compute prefix_mask: 1 for assistant token positions
    prefix_mask = [0] * (last_token - first_token)
    offset = first_token
    for (a_start, a_end) in matched_token_spans:
        a_start_clip = max(a_start, first_token)
        a_end_clip = min(a_end, last_token)
        if a_start_clip < a_end_clip:
            for pos in range(a_start_clip - offset, a_end_clip - offset):
                prefix_mask[pos] = 1

    # Validation
    if sum(prefix_mask) == 0:
        raise ValueError(
            f"prefix_mask is all zeros for token span [{first_token}, {last_token}). "
            f"Matched spans: {matched_token_spans}. "
            f"This means no assistant tokens were found in the prefix span."
        )

    return prefix_old_logprobs, prefix_mask, (first_token, last_token)


def main():
    args = parse_args()

    print("=" * 60)
    print("Sidecar -> Prefix 训练数据构建（对齐到截断 prompt）")
    print("=" * 60)
    print(f"Prefix 数据: {args.prefix_data_path}")
    print(f"Sidecar: {args.sidecar_path}")
    print(f"原始 jsonl: {args.original_jsonl_path}")
    print(f"输出: {args.output_path}")
    print()

    # ========== 1. 加载 tokenizer（必须，用于 token 级对齐）==========
    model_path = args.model_path or "/Data/public/Qwen3-1.7B"
    print(f"加载 tokenizer: {model_path}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("  ✓ tokenizer 加载成功")
    print()

    # ========== 2. 加载原始 jsonl（用于字符级对齐）==========
    print(f"加载原始轨迹 jsonl...")
    jsonl_conversations = {}  # (item_id, sample_idx) -> conversations
    with open(args.original_jsonl_path) as f:
        for line in f:
            data = json.loads(line)
            item_id = data.get("item_id", data.get("task_id", ""))
            sample_idx = data.get("sample_idx", 0)
            conversations = data.get("conversations", [])
            jsonl_conversations[(item_id, sample_idx)] = conversations
    print(f"  ✓ 加载 {len(jsonl_conversations)} 条原始轨迹")
    print()

    # ========== 3. 加载 parquet 数据并 join ==========
    print("加载数据...")
    prefix_df = pd.read_parquet(args.prefix_data_path)
    sidecar_df = pd.read_parquet(args.sidecar_path)

    print(f"Prefix 数据样本数: {len(prefix_df)}")
    print(f"Sidecar 样本数: {len(sidecar_df)}")

    print("\n构建 join key...")
    prefix_df = build_join_key(prefix_df)
    prefix_df = prefix_df.sort_values(["item_id", "sample_idx"]).copy()
    prefix_df["_join_sample_idx"] = prefix_df.groupby("item_id").cumcount()
    print(f"  ✓ 构建 item_id 并将 global sample_idx 转为 per-task cumcount 用于 join")

    print("\n执行 join...")
    merged_df = prefix_df.merge(
        sidecar_df,
        left_on=["item_id", "_join_sample_idx"],
        right_on=["item_id", "sample_idx"],
        how="left",
        suffixes=("", "_sidecar")
    )

    total = len(merged_df)
    matched = merged_df["sequence_old_logprobs"].notna().sum()
    print(f"Join 结果: {matched}/{total} 匹配 ({matched/total*100:.2f}%)")

    if matched != total:
        unmatched = merged_df[merged_df["sequence_old_logprobs"].isna()]
        print(f"\n未匹配的样本 ({len(unmatched)} 个):")
        for _, row in unmatched.head(5).iterrows():
            print(f"  item_id={row['item_id']}, sample_idx={row['sample_idx']}")
        raise RuntimeError(
            f"Join 覆盖率不是 100%: {matched}/{total}。"
            f"请检查输入数据和 join key 是否一致。"
        )
    print(f"✓ Join 覆盖率 100%: {matched}/{total}")
    print()

    # ========== 4. 逐样本计算（用 jsonl 字符级对齐）==========
    print("计算 prefix old logprobs（对齐到截断 prompt）...")

    new_prefix_lp = []
    new_prefix_mask = []
    new_prefix_span = []
    new_prefix_token_count = []
    new_assistant_turn_count = []
    errors = []

    for idx, (_, row) in enumerate(tqdm(merged_df.iterrows(), total=len(merged_df), desc="Processing")):
        item_id = row["item_id"]
        # _join_sample_idx 是 cumcount per item_id，与 jsonl 的 sample_idx 一致
        cumcount_idx = int(row["_join_sample_idx"])

        # 获取原始 jsonl 的 conversations（用 cumcount 作为 key）
        key = (item_id, cumcount_idx)
        if key not in jsonl_conversations:
            errors.append(f"[{idx}] item_id={item_id}, sample_idx={sample_idx}: jsonl 中找不到")
            new_prefix_lp.append([])
            new_prefix_mask.append([])
            new_prefix_span.append({"start": 0, "end": 0})
            new_prefix_token_count.append(0)
            new_assistant_turn_count.append(0)
            continue

        full_conversations = jsonl_conversations[key]

        # 获取截断后的 prompt（训练 parquet 的 prompt）
        prompt = row["prompt"]
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()

        # 用截断 prompt 的 assistant turn 数
        K = sum(1 for msg in prompt if msg.get("role") == "assistant")

        # 构建完整对话的文本（用于字符级匹配）
        full_text_built = ""
        full_assistant_spans = []
        for msg in full_conversations:
            role = msg.get("role", "")
            content = msg.get("content", "")
            role_text = f"<|im_start|>{role}\n"
            role_start_char = len(full_text_built) + len(role_text)
            full_text_built += role_text + content + "<|im_end|>\n"
            if role == "assistant":
                full_assistant_spans.append(
                    (role_start_char, len(full_text_built) - len("<|im_end|>\n"), role)
                )

        # 获取 sidecar 的 old_logprobs
        sequence_old_logprobs = row["sequence_old_logprobs"]
        token_length = int(row.get("token_length", len(sequence_old_logprobs) + 1))

        # 对齐提取
        try:
            prefix_old_logprobs, prefix_mask, span = _extract_prefix_old_logprobs_aligned(
                truncated_prompt=prompt,
                full_conversation_assistant_spans_char=full_assistant_spans,
                sequence_old_logprobs=sequence_old_logprobs,
                full_conversation_text=full_text_built,
                tokenizer=tokenizer,
            )
        except Exception as e:
            errors.append(f"[{idx}] item_id={item_id}, _join_idx={cumcount_idx}: {e}")
            prefix_old_logprobs = []
            prefix_mask = []
            span = (0, 0)

        prefix_start, prefix_end = span
        prefix_token_count = sum(prefix_mask)

        new_prefix_lp.append(prefix_old_logprobs)
        new_prefix_mask.append(prefix_mask)
        new_prefix_span.append({"start": prefix_start, "end": prefix_end})
        new_prefix_token_count.append(prefix_token_count)
        new_assistant_turn_count.append(K)

    merged_df["prefix_old_logprobs"] = new_prefix_lp
    merged_df["prefix_loss_mask"] = new_prefix_mask
    merged_df["assistant_prefix_span"] = new_prefix_span
    merged_df["prefix_token_count"] = new_prefix_token_count
    merged_df["assistant_turn_count"] = new_assistant_turn_count

    if errors:
        print(f"\n处理错误 ({len(errors)} 个):")
        for err in errors[:10]:
            print(f"  {err}")
        print("  ...")

    # ========== 5. 静态校验 ==========
    print("\n执行静态校验...")

    # 校验 1: 非空
    empty = (merged_df["prefix_old_logprobs"].apply(len) == 0).sum()
    print(f"  ✓ prefix_old_logprobs 非空: {len(merged_df) - empty}/{len(merged_df)}")
    if empty > 0:
        raise RuntimeError(f"静态校验失败：{empty} 个样本的 prefix_old_logprobs 为空")

    # 校验 2: prefix_mask 和 prefix_old_logprobs 长度一致
    len_mismatch = (merged_df.apply(
        lambda r: len(r["prefix_old_logprobs"]) != len(r["prefix_loss_mask"]), axis=1
    )).sum()
    if len_mismatch > 0:
        raise RuntimeError(f"静态校验失败：{len_mismatch} 个样本的 old_logprobs 和 mask 长度不一致")
    print(f"  ✓ old_logprobs 和 mask 长度一致: {len(merged_df)}/{len(merged_df)}")

    # 校验 3: prefix_mask 非零
    zero_mask = (merged_df["prefix_loss_mask"].apply(lambda x: sum(x) == 0)).sum()
    if zero_mask > 0:
        raise RuntimeError(f"静态校验失败：{zero_mask} 个样本的 prefix_loss_mask 全为零")
    print(f"  ✓ prefix_loss_mask 非零: {len(merged_df) - zero_mask}/{len(merged_df)}")

    # ========== 6. 【新增】Runtime 对齐校验 ==========
    if tokenizer is not None:
        print("\n执行 Runtime 对齐校验（用 tokenizer 验证截断 prompt 的 assistant token 数）...")

        runtime_token_counts = []
        mismatches = []

        for idx, (_, row) in enumerate(tqdm(merged_df.iterrows(), total=len(merged_df), desc="Runtime validation")):
            prompt = row["prompt"]
            if hasattr(prompt, "tolist"):
                prompt = prompt.tolist()

            full_text = tokenizer.apply_chat_template(
                prompt, add_generation_prompt=False, tokenize=False
            )
            tokens = tokenizer(full_text, add_special_tokens=True, return_tensors=None)
            token_ids = tokens.input_ids

            # 逐消息统计 assistant token 数
            import re
            start_matches = list(re.compile(r"<\|im_start\|>(user|assistant|tool|system)").finditer(full_text))
            end_matches = list(re.compile(r"<\|im_end\|>").finditer(full_text))

            offset_mapping = tokens.offset_mapping
            assistant_count = 0
            role_counters = {"user": 0, "assistant": 0}

            for i, msg in enumerate(prompt):
                role = msg.get("role", "")
                if role not in role_counters:
                    continue
                role_counters[role] += 1

                if role == "assistant":
                    start_char = start_matches[i].end()
                    end_char = end_matches[i].end()
                    for t, (s, e) in enumerate(offset_mapping):
                        if s is None:
                            continue
                        if s < end_char and e > start_char:
                            assistant_count += 1

            cached_count = int(row["prefix_token_count"])
            runtime_token_counts.append(assistant_count)

            if assistant_count != cached_count:
                mismatches.append(
                    f"item_id={row['item_id']}, sample_idx={row['sample_idx']}: "
                    f"cached={cached_count}, runtime={assistant_count}, diff={cached_count - assistant_count}"
                )

        merged_df["_runtime_prefix_token_count"] = runtime_token_counts

        mismatch_count = len(mismatches)
        print(f"  ✓ Runtime 对齐: {len(merged_df) - mismatch_count}/{len(merged_df)}")
        if mismatch_count > 0:
            print(f"  ✗ 不对齐 ({mismatch_count} 个):")
            for m in mismatches[:10]:
                print(f"    {m}")
            print("  ... 更多不匹配样本详见上述处理错误")
            raise RuntimeError(
                f"Runtime 对齐校验失败：{mismatch_count}/{len(merged_df)} 个样本的 "
                f"cached prefix_token_count 与 runtime tokenizer 结果不一致！"
                f"这意味着 parquet 构建的 prefix old_logprobs 仍然没有正确对齐到训练 prompt。"
            )
        else:
            print(f"  ✓ 所有 {len(merged_df)} 个样本的 cached 和 runtime 完全一致！")

        # 清理临时列
        merged_df.drop(columns=["_runtime_prefix_token_count"], inplace=True)

    print("\n静态校验通过！")

    # ========== 7. 输出 ==========
    cols_to_drop = [
        "task_id", "index",
        "token_length", "old_logprob_length", "assistant_token_count",
        "sequence_old_logprobs", "assistant_mask",
    ]
    cols_to_drop = [c for c in cols_to_drop if c in merged_df.columns]
    output_df = merged_df.drop(columns=cols_to_drop)

    print(f"\n保存到 {args.output_path}...")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    output_df.to_parquet(args.output_path, index=False)
    print("完成！")

    print("\n" + "=" * 60)
    print("统计信息")
    print("=" * 60)
    print(f"总样本数: {len(output_df)}")
    print(f"prefix_token_count 范围: {output_df['prefix_token_count'].min()} - {output_df['prefix_token_count'].max()}")
    print(f"assistant_turn_count (K from prompt) 范围: {output_df['assistant_turn_count'].min()} - {output_df['assistant_turn_count'].max()}")

    print("\n前 3 条样本详情:")
    for i in range(min(3, len(output_df))):
        row = output_df.iloc[i]
        print(f"\n  [{i}] item_id={row['item_id']}, sample_idx={row['sample_idx']}")
        print(f"      prefix_token_count: {row['prefix_token_count']}")
        print(f"      assistant_turn_count (prompt K): {row['assistant_turn_count']}")
        print(f"      prefix_old_logprobs 长度: {len(row['prefix_old_logprobs'])}")
        print(f"      prefix_loss_mask 长度: {len(row['prefix_loss_mask'])}")
        print(f"      prefix_loss_mask sum: {sum(row['prefix_loss_mask'])}")


if __name__ == "__main__":
    main()