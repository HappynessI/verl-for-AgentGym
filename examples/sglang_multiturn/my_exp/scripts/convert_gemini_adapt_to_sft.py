#!/usr/bin/env python3
"""
转换Gemini采样的ADaPT格式数据为SFT训练格式
从: textcraft_trajectories.jsonl (Gemini采样，ADaPT格式)
到: Parquet格式，只包含messages列
"""

import json
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import List, Dict

def convert_conversation(conversations: List[Dict]) -> List[Dict]:
    """
    转换Gemini采样的conversations到SFT格式
    
    Args:
        conversations: [{'role': 'user'/'assistant', 'content': '...'}, ...]
    
    Returns:
        [{'role': 'user'/'assistant', 'content': '...'}, ...]
    """
    messages = []
    
    for i, conv in enumerate(conversations):
        role = conv['role']
        content = conv['content']
        
        # Gemini采样的数据已经是标准的role格式（user/assistant）
        # 第一条user消息包含system prompt和任务描述
        if i == 0 and role == 'user':
            # 保持原样，第一条user消息作为system+user的组合
            messages.append({
                'role': 'user',
                'content': content
            })
        elif role in ['user', 'assistant']:
            messages.append({
                'role': role,
                'content': content
            })
        else:
            print(f"Warning: Unknown role '{role}', skipping")
    
    return messages


def main():
    # 输入输出路径
    input_file = '/Data/wyh/datasets/Sampling-Data/textcraft_次gemini-3-pro-preview_20251216_101619/textcraft_trajectories.jsonl'
    output_file = '/Data/wyh/datasets/Verl-Data/train/textcraft/train_gemini_adapt.parquet'
    
    print("="*80)
    print("转换 Gemini ADaPT 采样数据为 SFT 格式")
    print("="*80)
    print(f"输入: {input_file}")
    print(f"输出: {output_file}")
    print()
    
    # 创建输出目录
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 读取JSONL数据
    print("读取Gemini采样数据...")
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"✓ 读取 {len(data)} 条训练样本")
    
    # 只保留成功的样本
    success_data = [item for item in data if item.get('success') == 1]
    print(f"✓ 过滤后保留 {len(success_data)} 条成功样本 (success=1)")
    
    # 转换数据
    print("\n转换数据格式...")
    converted_data = []
    
    for i, item in enumerate(success_data):
        if 'conversations' not in item:
            print(f"Warning: 第{i}条数据缺少conversations字段，跳过")
            continue
        
        messages = convert_conversation(item['conversations'])
        
        # 检查messages格式是否符合SFT训练要求
        # 必须是 user-assistant 交替出现
        if len(messages) < 2:
            print(f"Warning: 第{i}条数据消息数少于2条，跳过")
            continue
        
        converted_data.append({'messages': messages})
        
        # 打印第一条作为示例
        if i == 0:
            print(f"\n示例（第1条数据）:")
            print(f"  原始对话数: {len(item['conversations'])}")
            print(f"  转换后消息数: {len(messages)}")
            print(f"  前3条消息:")
            for j, msg in enumerate(messages[:3]):
                content_preview = msg['content'][:80] if len(msg['content']) > 80 else msg['content']
                print(f"    [{j}] {msg['role']}: {content_preview}...")
    
    print(f"\n✓ 成功转换 {len(converted_data)} 条样本")
    
    # 保存为Parquet
    print("\n保存为Parquet格式...")
    
    # 构建PyArrow Table
    messages_list = [item['messages'] for item in converted_data]
    
    # 定义schema
    message_struct = pa.struct([
        ('role', pa.string()),
        ('content', pa.string())
    ])
    schema = pa.schema([
        ('messages', pa.list_(message_struct))
    ])
    
    # 创建Table
    table = pa.Table.from_pydict(
        {'messages': messages_list},
        schema=schema
    )
    
    # 写入Parquet
    pq.write_table(table, output_file)
    
    print(f"✓ 数据已保存到: {output_file}")
    
    # 验证
    print("\n验证生成的文件...")
    verify_table = pq.read_table(output_file)
    print(f"  行数: {len(verify_table)}")
    print(f"  列名: {verify_table.column_names}")
    
    # 读取第一条验证
    first_row = {col: verify_table[col][0].as_py() for col in verify_table.column_names}
    first_messages = first_row['messages']
    print(f"\n第一条数据验证:")
    print(f"  消息数: {len(first_messages)}")
    print(f"  第一条消息role: {first_messages[0]['role']}")
    print(f"  第一条消息content (前80字符): {first_messages[0]['content'][:80]}...")
    print(f"  第二条消息role: {first_messages[1]['role']}")
    print(f"  第二条消息content (前80字符): {first_messages[1]['content'][:80]}...")
    
    print("\n" + "="*80)
    print("✓ 转换完成！")
    print(f"✓ 原始样本: {len(data)} 条")
    print(f"✓ 成功样本: {len(success_data)} 条")
    print(f"✓ 最终输出: {len(converted_data)} 条")
    print("="*80)


if __name__ == "__main__":
    main()

