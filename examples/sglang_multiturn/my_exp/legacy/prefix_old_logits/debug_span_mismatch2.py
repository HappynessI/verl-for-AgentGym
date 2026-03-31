#!/usr/bin/env python3
"""调试脚本：分析 Span mismatch 问题 - 检查EOS token"""

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

print("=" * 60)
print("Full Text 分析")
print("=" * 60)
print(f"full_text 最后100字符: {repr(full_text[-100:])}")
print()

# tokenize 不带 special tokens
result_no_special = tokenizer(
    full_text,
    add_special_tokens=False,
    return_tensors="pt",
    return_offsets_mapping=True,
)
input_ids_no_special = result_no_special.input_ids[0]
offset_mapping_no_special = result_no_special.offset_mapping[0]
print(f"不带 special_tokens 的 token_length: {len(input_ids_no_special)}")

# tokenize 带 special tokens
result_with_special = tokenizer(
    full_text,
    add_special_tokens=True,
    return_tensors="pt",
    return_offsets_mapping=True,
)
input_ids_with_special = result_with_special.input_ids[0]
offset_mapping_with_special = result_with_special.offset_mapping[0]
print(f"带 special_tokens 的 token_length: {len(input_ids_with_special)}")

print()
print("=" * 60)
print("第一个和最后一个 token 分析")
print("=" * 60)
print(f"第一个 token id: {input_ids_with_special[0].item()}")
print(f"第一个 token: {tokenizer.decode([input_ids_with_special[0].item()])}")
print(f"最后一个 token id: {input_ids_with_special[-1].item()}")
print(f"最后一个 token: {tokenizer.decode([input_ids_with_special[-1].item()])}")
print()

print("=" * 60)
print("Offset mapping 分析 (前5个和后5个)")
print("=" * 60)
print("前5个 offset_mapping:")
for i in range(min(5, len(offset_mapping_with_special))):
    print(f"  [{i}]: {offset_mapping_with_special[i]} -> {repr(full_text[offset_mapping_with_special[i][0]:offset_mapping_with_special[i][1]])}")
print("后5个 offset_mapping:")
for i in range(max(0, len(offset_mapping_with_special)-5), len(offset_mapping_with_special)):
    print(f"  [{i}]: {offset_mapping_with_special[i]} -> {repr(full_text[offset_mapping_with_special[i][0]:offset_mapping_with_special[i][1]])}")
print()

# 分析最后一个 <|im_end|> 的位置
end_pattern = re.compile(r'<\|im_end\|>')
end_matches = list(end_pattern.finditer(full_text))
last_end_match = end_matches[-1]
print(f"最后一个 <|im_end|> 的位置:")
print(f"  start(): {last_end_match.start()}")
print(f"  end(): {last_end_match.end()}")
print(f"  文本: {repr(full_text[last_end_match.start():last_end_match.end()])}")
