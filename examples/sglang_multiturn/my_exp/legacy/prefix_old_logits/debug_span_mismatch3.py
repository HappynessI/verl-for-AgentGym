#!/usr/bin/env python3
"""调试脚本：分析 Span mismatch 问题 - 深入分析"""

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
        if i >= 1:
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

# 使用 apply_chat_template 生成完整文本
full_text = tokenizer.apply_chat_template(
    conversations,
    add_generation_prompt=False,
    tokenize=False
)

# tokenize 带 special tokens
result = tokenizer(
    full_text,
    add_special_tokens=True,
    return_tensors="pt",
    return_offsets_mapping=True,
)
input_ids = result.input_ids[0]
offset_mapping = result.offset_mapping[0]
token_length = len(input_ids)

# 分析最后一个消息
last_msg = conversations[-1]
print(f"最后一个消息 role: {last_msg['role']}")
print(f"最后一个消息内容 (最后100字符): {repr(last_msg['content'][-100:])}")
print()

# 找最后一个 <|im_start|>user 和 <|im_end|>
start_pattern = re.compile(r'<\|im_start\|>(user|assistant|tool|system)')
end_pattern = re.compile(r'<\|im_end\|>')

start_matches = list(start_pattern.finditer(full_text))
end_matches = list(end_pattern.finditer(full_text))

# 最后一个消息对应的 index = len(conversations) - 1 = 8
last_idx = len(conversations) - 1
print(f"最后一个消息 index: {last_idx}")
print()

print("=" * 60)
print("分析最后一个消息的字符范围")
print("=" * 60)
start_char = start_matches[last_idx].end()
end_char_current = end_matches[last_idx].end()  # 当前代码逻辑
end_char_fixed = end_matches[last_idx].start()  # 另一种可能

print(f"start_char = start_matches[{last_idx}].end() = {start_char}")
print(f"end_char (当前) = end_matches[{last_idx}].end() = {end_char_current}")
print(f"end_char (另一种) = end_matches[{last_idx}].start() = {end_char_fixed}")
print()

# 分析当前逻辑找到的 token 范围
print("=" * 60)
print("当前逻辑 (end_char = end_matches[i].end())")
print("=" * 60)
start_token = None
end_token = None
for t, (s, e) in enumerate(offset_mapping):
    if s is None:
        continue
    if s < end_char_current and e > start_char:
        if start_token is None:
            start_token = t
        end_token = t + 1
        
print(f"start_token = {start_token}")
print(f"end_token = {end_token}")
print(f"span = [{start_token}, {end_token})")
print(f"span长度 = {end_token - start_token}")
print()

# 验证：打印每个 token 的范围，看最后几个
print("=" * 60)
print("最后10个 token 的 offset")
print("=" * 60)
for t in range(max(0, token_length - 10), token_length):
    s, e = offset_mapping[t].tolist()
    token_str = full_text[s:e] if s is not None else "None"
    print(f"  token {t}: offset=({s}, {e}), text={repr(token_str)}")

print()
print("=" * 60)
print("问题分析")
print("=" * 60)
print(f"full_text 字符长度: {len(full_text)}")
print(f"token_length: {token_length}")
print(f"最后一个 offset: {offset_mapping[-1].tolist()}")
print(f"最后一个 offset end: {offset_mapping[-1][1].item()}")
print()
print(f"span end_token = {end_token}")
print(f"期望 token_length = {token_length}")
print(f"差异 = {token_length - end_token}")
