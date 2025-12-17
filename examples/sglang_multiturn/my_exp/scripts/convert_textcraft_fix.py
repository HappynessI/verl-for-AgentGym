#!/usr/bin/env python3
"""
修复TextCraft数据转换 - 正确处理AgentGym格式
"""

import json
import pyarrow as pa
import pyarrow.parquet as pq

def convert_agentgym_to_sft(agentgym_data):
    """
    AgentGym格式:
    [0] human: system prompt + "OK. I'll follow..."
    [1] gpt: "OK. I'll follow your instructions..."
    [2] human: 实际任务开始
    [3] gpt: 第一个回复
    ...
    
    SFT格式应该是:
    [0] system: system prompt (从第0条human提取)
    [1] user: 实际任务 (第2条human)
    [2] assistant: 第一个回复 (第3条gpt)
    ...
    """
    messages = []
    
    # 第一条human消息包含system prompt，提取它
    first_human = agentgym_data[0]['value']
    messages.append({
        'role': 'system',
        'content': first_human
    })
    
    # 跳过第1条（gpt确认）和第2条之前的内容，从第2条human开始
    for i in range(2, len(agentgym_data)):
        conv = agentgym_data[i]
        if conv['from'] == 'human':
            messages.append({
                'role': 'user',
                'content': conv['value']
            })
        elif conv['from'] == 'gpt':
            messages.append({
                'role': 'assistant',
                'content': conv['value']
            })
    
    return messages


input_file = '/Data/wyh/datasets/AgentGym-Data/AgentTraj-L/textcraft_train.json'
output_file = '/Data/wyh/datasets/Parquet-Data/textcraft/train.parquet'

print("读取数据...")
with open(input_file, 'r') as f:
    data = json.load(f)

print(f"总样本: {len(data)}")

# 转换
converted = []
for item in data:
    messages = convert_agentgym_to_sft(item['conversations'])
    converted.append({'messages': messages})
    
# 验证第一条
print(f"\n第一条数据验证 (前5条消息):")
for i, msg in enumerate(converted[0]['messages'][:5]):
    print(f"  [{i}] {msg['role']}: {msg['content'][:60]}...")

# 保存
message_struct = pa.struct([
    ('role', pa.string()),
    ('content', pa.string())
])
schema = pa.schema([
    ('messages', pa.list_(message_struct))
])

table = pa.Table.from_pydict(
    {'messages': [item['messages'] for item in converted]},
    schema=schema
)

pq.write_table(table, output_file)
print(f"\n保存到: {output_file}")
print(f"数据条数: {len(table)}")

