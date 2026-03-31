#!/usr/bin/env python3
"""调试脚本：验证修复方案"""

import json
import re
from transformers import AutoTokenizer

# 加载数据
INPUT_JSONL = "/Data/wyh/datasets/Sampling-Data/textcraft_MiniMax-M2.1_20260307_150412/textcraft_trajectories.jsonl"
MODEL_PATH = "/Data/wyh/datasets/Verl-Data/outputs/textcraft_sft/qwen3-1.7b-sft/global_step_200/huggingface"

samples = []
with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 1:
            break
        samples.append(json.loads(line))

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    pad_token="<|endoftext|>"
)

sample = samples[0]
conversations = sample["conversations"]

full_text = tokenizer.apply_chat_template(
    conversations,
    add_generation_prompt=False,
    tokenize=False
)

result = tokenizer(
    full_text,
    add_special_tokens=True,
    return_tensors="pt",
    return_offsets_mapping=True,
)
input_ids = result.input_ids[0]
offset_mapping = result.offset_mapping[0]
token_length = len(input_ids)

start_pattern = re.compile(r'<\|im_start\|>(user|assistant|tool|system)')
end_pattern = re.compile(r'<\|im_end\|>')

start_matches = list(start_pattern.finditer(full_text))
end_matches = list(end_pattern.finditer(full_text))

print("=" * 60)
print("测试修复方案：使用 end_matches[i].start()")
print("=" * 60)

# 最后一个消息的 index
last_idx = len(conversations) - 1

# 使用 start() 而不是 end()
start_char = start_matches[last_idx].end()
end_char_fixed = end_matches[last_idx].start()  # 修复：用 start() 而不是 end()

start_token_fixed = None
end_token_fixed = None
for t, (s, e) in enumerate(offset_mapping):
    if s is None:
        continue
    if s < end_char_fixed and e > start_char:
        if start_token_fixed is None:
            start_token_fixed = t
        end_token_fixed = t + 1

print(f"最后一个消息 index: {last_idx}")
print(f"start_char = {start_char}")
print(f"end_char (修复) = end_matches[{last_idx}].start() = {end_char_fixed}")
print(f"修复后的 span = [{start_token_fixed}, {end_token_fixed})")
print(f"修复后的 span 长度 = {end_token_fixed - start_token_fixed}")
print()
print(f"期望 token_length = {token_length}")
print(f"实际 end_token_fixed = {end_token_fixed}")
print(f"匹配: {end_token_fixed == token_length}")
