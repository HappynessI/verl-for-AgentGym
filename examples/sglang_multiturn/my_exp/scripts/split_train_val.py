#!/usr/bin/env python3
"""将训练集split为训练集和验证集"""
import pyarrow.parquet as pq
import pyarrow as pa
import random

# 读取原始训练数据
table = pq.read_table('/Data/wyh/datasets/Parquet-Data/textcraft/train.parquet')
data = [{'messages': table['messages'][i].as_py()} for i in range(len(table))]

print(f"原始数据: {len(data)} 条")

# 设置随机种子
random.seed(42)
random.shuffle(data)

# Split: 90% train, 10% val
split_idx = int(len(data) * 0.9)
train_data = data[:split_idx]
val_data = data[split_idx:]

print(f"训练集: {len(train_data)} 条")
print(f"验证集: {len(val_data)} 条")

# 定义schema
message_struct = pa.struct([
    ('role', pa.string()),
    ('content', pa.string())
])
schema = pa.schema([('messages', pa.list_(message_struct))])

# 保存训练集
train_table = pa.Table.from_pydict(
    {'messages': [item['messages'] for item in train_data]},
    schema=schema
)
pq.write_table(train_table, '/Data/wyh/datasets/Parquet-Data/textcraft/train_split.parquet')

# 保存验证集
val_table = pa.Table.from_pydict(
    {'messages': [item['messages'] for item in val_data]},
    schema=schema
)
pq.write_table(val_table, '/Data/wyh/datasets/Parquet-Data/textcraft/val_split.parquet')

print("\n保存完成:")
print("  训练集: /Data/wyh/datasets/Parquet-Data/textcraft/train_split.parquet")
print("  验证集: /Data/wyh/datasets/Parquet-Data/textcraft/val_split.parquet")

