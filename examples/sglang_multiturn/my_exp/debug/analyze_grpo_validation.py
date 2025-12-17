#!/usr/bin/env python3
"""
分析GRPO validation日志中的生成质量
"""
import re
import sys
from collections import defaultdict

def analyze_log(log_path):
    """分析GRPO validation日志"""
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # 提取所有Raw assistant message
    # 格式: Step X - Raw assistant (len=XXX): content
    raw_messages = re.findall(
        r'Step \d+ - Raw assistant \(len=(\d+)\): (.+?)(?=\nWARNING:|\n\[36m|$)',
        content,
        re.DOTALL
    )
    
    print(f"总共找到 {len(raw_messages)} 条assistant消息")
    print("="*80)
    
    # 统计分析
    lengths = []
    has_action_tag = 0
    has_think_tag = 0
    truncated = 0
    complete_action = 0
    
    action_patterns = {
        'has_Action': 0,
        'has_craft_cmd': 0,
        'has_get_cmd': 0,
        'ends_incomplete': 0,
        'ends_with_think': 0,
    }
    
    # 详细分析前10条
    print("\n前10条消息详情:")
    print("-"*80)
    
    for i, (length, msg) in enumerate(raw_messages[:10]):
        length = int(length)
        lengths.append(length)
        
        # 检查特征
        has_action = 'Action:' in msg
        has_think = '<think>' in msg
        has_craft = re.search(r'craft\s+\d+', msg, re.IGNORECASE) is not None
        has_get = re.search(r'get\s+\d+', msg, re.IGNORECASE) is not None
        
        # 检查是否截断（以不完整的句子结尾）
        msg_end = msg[-100:] if len(msg) > 100 else msg
        is_truncated = not msg_end.strip().endswith(('.', '!', '?', '</think>', '\n'))
        
        if has_action:
            has_action_tag += 1
            action_patterns['has_Action'] += 1
        if has_think:
            has_think_tag += 1
        if is_truncated:
            truncated += 1
            action_patterns['ends_incomplete'] += 1
        if has_craft:
            action_patterns['has_craft_cmd'] += 1
        if has_get:
            action_patterns['has_get_cmd'] += 1
        if msg.strip().endswith('</think>'):
            action_patterns['ends_with_think'] += 1
        
        # 显示详情
        print(f"\n消息 {i+1} (长度={length}):")
        print(f"  有Action:: {has_action}")
        print(f"  有<think>: {has_think}")
        print(f"  有craft命令: {has_craft}")
        print(f"  有get命令: {has_get}")
        print(f"  是否截断: {is_truncated}")
        
        # 显示开头和结尾
        print(f"  开头: {msg[:150]}")
        print(f"  结尾: ...{msg[-150:]}")
    
    # 统计所有消息
    print("\n" + "="*80)
    print("整体统计:")
    print(f"  总消息数: {len(raw_messages)}")
    if lengths:
        print(f"  平均长度: {sum(lengths)/len(lengths):.1f} 字符")
        print(f"  最大长度: {max(lengths)}")
        print(f"  最小长度: {min(lengths)}")
    else:
        print("  没有找到任何消息！")
        return
    print(f"  包含Action:标签: {has_action_tag} ({has_action_tag/len(raw_messages)*100:.1f}%)")
    print(f"  包含<think>标签: {has_think_tag} ({has_think_tag/len(raw_messages)*100:.1f}%)")
    print(f"  被截断: {truncated} ({truncated/len(raw_messages)*100:.1f}%)")
    
    print("\n动作模式:")
    for pattern, count in action_patterns.items():
        print(f"  {pattern}: {count} ({count/len(raw_messages)*100:.1f}%)")
    
    # 提取action成功/失败的统计
    print("\n" + "="*80)
    print("Action提取结果:")
    
    extracted_actions = re.findall(r'\[TextCraft\] Extracted action: \'(.+?)\'', content)
    failed_extractions = content.count('Failed to extract valid action')
    
    total_attempts = len(extracted_actions) + failed_extractions
    print(f"  成功提取: {len(extracted_actions)} / {total_attempts} ({len(extracted_actions)/total_attempts*100:.1f}%)")
    print(f"  失败提取: {failed_extractions} / {total_attempts} ({failed_extractions/total_attempts*100:.1f}%)")
    
    # 显示提取的action类型分布
    action_types = defaultdict(int)
    for action in extracted_actions[:50]:  # 只看前50个
        if action.startswith('craft'):
            action_types['craft'] += 1
        elif action.startswith('get'):
            action_types['get'] += 1
        elif action == 'inventory':
            action_types['inventory'] += 1
        elif action == 'look':
            action_types['look'] += 1
        else:
            action_types['other'] += 1
    
    print("\n提取的action类型分布（前50个）:")
    for action_type, count in sorted(action_types.items(), key=lambda x: -x[1]):
        print(f"  {action_type}: {count}")
    
    # 任务成功率
    print("\n" + "="*80)
    print("任务成功率:")
    
    success_episodes = content.count('NON-ZERO REWARD')
    total_episodes = content.count('Step 1 - Env response')
    
    if total_episodes > 0:
        print(f"  成功episode: {success_episodes} / {total_episodes} ({success_episodes/total_episodes*100:.1f}%)")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python analyze_grpo_validation.py <log_path>")
        sys.exit(1)
    
    analyze_log(sys.argv[1])

