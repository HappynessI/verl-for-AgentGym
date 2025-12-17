#!/usr/bin/env python3
"""
转换AgentGym TextCraft数据为SFT训练格式
从: AgentGym-Data/AgentTraj-L/textcraft_train.json
到: Parquet格式，只包含messages列

格式转换:
- AgentGym: {'conversations': [{'from': 'human'/'gpt', 'value': '...'}], 'item_id': '...'}
- SFT目标: {'messages': [{'role': 'system'/'user'/'assistant', 'content': '...'}]}
"""

import json
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import List, Dict

def convert_conversation(agentgym_conv: List[Dict]) -> List[Dict]:
    """
    转换AgentGym格式的对话到SFT格式
    
    Args:
        agentgym_conv: [{'from': 'human'/'gpt', 'value': '...'}, ...]
    
    Returns:
        [{'role': 'system'/'user'/'assistant', 'content': '...'}, ...]
    """
    messages = []
    
    for i, conv in enumerate(agentgym_conv):
        from_role = conv['from']
        content = conv['value']
        
        # 映射角色
        if from_role == 'human':
            # 第一条human消息可能包含system prompt，需要特殊处理
            if i == 0 and content.startswith('You are given'):
                # 这是system prompt + 第一条user消息
                # AgentGym的格式通常是system prompt在第一条human消息中
                messages.append({
                    'role': 'system',
                    'content': content
                })
            else:
                messages.append({
                    'role': 'user',
                    'content': content
                })
        elif from_role == 'gpt':
            messages.append({
                'role': 'assistant',
                'content': content
            })
        else:
            print(f"Warning: Unknown role '{from_role}', skipping")
    
    return messages


def main():
    # 输入输出路径
    input_file = '/Data/wyh/datasets/AgentGym-Data/AgentTraj-L/textcraft_train.json'
    output_file = '/Data/wyh/datasets/Parquet-Data/textcraft/train.parquet'
    
    print("="*80)
    print("转换 TextCraft 训练数据为 SFT 格式")
    print("="*80)
    print(f"输入: {input_file}")
    print(f"输出: {output_file}")
    print()
    
    # 创建输出目录
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 读取AgentGym数据
    print("读取AgentGym数据...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"✓ 读取 {len(data)} 条训练样本")
    
    # 转换数据
    print("\n转换数据格式...")
    converted_data = []
    
    for i, item in enumerate(data):
        if 'conversations' not in item:
            print(f"Warning: 第{i}条数据缺少conversations字段，跳过")
            continue
        
        messages = convert_conversation(item['conversations'])
        converted_data.append({'messages': messages})
        
        # 打印第一条作为示例
        if i == 0:
            print(f"\n示例（第1条数据）:")
            print(f"  原始对话数: {len(item['conversations'])}")
            print(f"  转换后消息数: {len(messages)}")
            print(f"  前3条消息:")
            for j, msg in enumerate(messages[:3]):
                content_preview = msg['content'][:100] if len(msg['content']) > 100 else msg['content']
                print(f"    [{j}] {msg['role']}: {content_preview}...")
    
    print(f"\n✓ 成功转换 {len(converted_data)} 条样本")
    
    # 保存为Parquet
    print("\n保存为Parquet格式...")
    
    # 构建PyArrow Table
    # messages字段是一个list of struct
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
    print(f"  Schema: {verify_table.schema}")
    
    # 读取第一条验证
    first_row = {col: verify_table[col][0].as_py() for col in verify_table.column_names}
    first_messages = first_row['messages']
    print(f"\n第一条数据验证:")
    print(f"  消息数: {len(first_messages)}")
    print(f"  第一条消息role: {first_messages[0]['role']}")
    print(f"  第一条消息content (前100字符): {first_messages[0]['content'][:100]}...")
    
    print("\n" + "="*80)
    print("✓ 转换完成！")
    print("="*80)


if __name__ == "__main__":
    main()

