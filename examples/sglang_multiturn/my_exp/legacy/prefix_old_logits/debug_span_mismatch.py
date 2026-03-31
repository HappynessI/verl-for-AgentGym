#!/usr/bin/env python3
"""调试脚本：分析 Span mismatch 问题"""

import json
import re
from transformers import AutoTokenizer

# 加载一个样本
INPUT_JSONL = "/Data/wyh/datasets/Sampling-Data/textcraft_MiniMax-M2.1_20260307_150412/textcraft_trajectories.jsonl"
MODEL_PATH = "/Data/wyh/datasets/Verl-Data/outputs/textcraft_sft/qwen3-1.7b-sft/global_step_200/huggingface"

# 加载数据
samples = []
with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 10:
            break
        samples.append(json.loads(line))

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    pad_token="<|endoftext|>"
)

# 分析第一个样本
sample = samples[0]
conversations = sample["conversations"]

print("=" * 60)
print("样本信息")
print("=" * 60)
print(f"item_id: {sample['item_id']}")
print(f"sample_idx: {sample['sample_idx']}")
print(f"conversations 长度: {len(conversations)}")
print()

# 使用 apply_chat_template 生成完整文本
full_text = tokenizer.apply_chat_template(
    conversations,
    add_generation_prompt=False,
    tokenize=False
)

print("=" * 60)
print("Full Text 长度分析")
print("=" * 60)
print(f"full_text 字符长度: {len(full_text)}")
print()

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

print(f"token_length: {token_length}")
print(f"offset_mapping 长度: {len(offset_mapping)}")
print()

# 找出所有 <|im_start|> 和 <|im_end|> 的字符位置
start_pattern = re.compile(r'<\|im_start\|>(user|assistant|tool|system)')
end_pattern = re.compile(r'<\|im_end\|>')

start_matches = list(start_pattern.finditer(full_text))
end_matches = list(end_pattern.finditer(full_text))

print(f"start_matches 数量: {len(start_matches)}")
print(f"end_matches 数量: {len(end_matches)}")
print(f"conversations 数量: {len(conversations)}")
print()

# 分析每个消息的span
print("=" * 60)
print("每个消息的 Span 分析")
print("=" * 60)

for i, msg in enumerate(conversations):
    role = msg.get("role", "unknown")
    content_preview = msg.get("content", "")[:50].replace("\n", "\\n")
    
    # 当前逻辑: start_char = start_matches[i].end(), end_char = end_matches[i].end()
    start_char_old = start_matches[i].end()
    end_char_old = end_matches[i].end()
    
    # 修正: start_char = start_matches[i].end(), end_char = end_matches[i].start() (不包括<|im_end|>)
    start_char_new = start_matches[i].end()
    end_char_new = end_matches[i].start()  # 改为 .start()
    
    # 找token范围 - 旧逻辑
    start_token_old = None
    end_token_old = None
    for t, (s, e) in enumerate(offset_mapping):
        if s is None:
            continue
        if s < end_char_old and e > start_char_old:
            if start_token_old is None:
                start_token_old = t
            end_token_old = t + 1
    
    # 找token范围 - 新逻辑
    start_token_new = None
    end_token_new = None
    for t, (s, e) in enumerate(offset_mapping):
        if s is None:
            continue
        if s < end_char_new and e > start_char_new:
            if start_token_new is None:
                start_token_new = t
            end_token_new = t + 1
    
    print(f"消息 {i}: role={role}")
    print(f"  内容预览: {content_preview}...")
    print(f"  start_matches[{i}].end() = {start_char_old}")
    print(f"  end_matches[{i}].end() = {end_char_old} (旧逻辑)")
    print(f"  end_matches[{i}].start() = {end_char_new} (新逻辑)")
    print(f"  Token范围 (旧): [{start_token_old}, {end_token_old})")
    print(f"  Token范围 (新): [{start_token_new}, {end_token_new})")
    print()

# 最后span分析
print("=" * 60)
print("最后Span分析")
print("=" * 60)
last_span_end_old = end_token_old
last_span_end_new = end_token_new

print(f"旧逻辑: last_span_end = {last_span_end_old}, token_length = {token_length}")
print(f"新逻辑: last_span_end = {last_span_end_new}, token_length = {token_length}")
print()

if last_span_end_old != token_length:
    print("BUG确认: 旧逻辑导致 Span mismatch!")
    print(f"  预期: {token_length}")
    print(f"  实际: {last_span_end_old}")
    print(f"  差异: {token_length - last_span_end_old}")
    
if last_span_end_new == token_length:
    print("新逻辑正确!")
