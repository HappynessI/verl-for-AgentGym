#!/usr/bin/env python3
"""
训练日志分析工具
用于快速分析GRPO训练日志，提取关键指标
"""

import re
import sys
from pathlib import Path


def analyze_log(log_file: str):
    """分析训练日志"""
    with open(log_file, 'r') as f:
        content = f.read()
    
    print(f"=== 分析日志: {Path(log_file).name} ===\n")
    
    # 1. Action提取统计
    extracted = content.count('Extracted action:')
    failed = content.count('Failed to extract valid action')
    
    if extracted + failed > 0:
        print(f"🎬 Action提取:")
        print(f"  成功: {extracted}")
        print(f"  失败: {failed}")
        print(f"  成功率: {extracted/(extracted+failed)*100:.1f}%\n")
    
    # 2. Reward统计
    rewards = re.findall(r"reward=([^,\s]+)", content)
    if rewards:
        reward_values = []
        for r in rewards:
            try:
                reward_values.append(float(r))
            except:
                pass
        
        if reward_values:
            nonzero = [r for r in reward_values if r != 0]
            print(f"🎯 Reward统计:")
            print(f"  总数: {len(reward_values)}")
            print(f"  非零: {len(nonzero)}")
            print(f"  成功率: {len(nonzero)/len(reward_values)*100:.1f}%")
            if nonzero:
                print(f"  平均reward: {sum(nonzero)/len(nonzero):.2f}\n")
    
    # 3. 生成长度统计
    lengths = re.findall(r'Raw assistant \(len=(\d+)\):', content)
    if len(lengths) >= 10:
        lengths = [int(l) for l in lengths]
        token_lengths = [l/1.2 for l in lengths]
        
        print(f"📏 生成长度统计（{len(lengths)}个样本）:")
        print(f"  平均: {sum(lengths)//len(lengths)} 字符 (~{int(sum(token_lengths)/len(token_lengths))} tokens)")
        print(f"  最小: {min(lengths)} 字符 (~{int(min(token_lengths))} tokens)")
        print(f"  最大: {max(lengths)} 字符 (~{int(max(token_lengths))} tokens)")
        
        over_1000 = sum(1 for t in token_lengths if t > 1000)
        over_2000 = sum(1 for t in token_lengths if t > 2000)
        print(f"  超过1000 tokens: {over_1000} ({over_1000/len(token_lengths)*100:.1f}%)")
        print(f"  超过2000 tokens: {over_2000} ({over_2000/len(token_lengths)*100:.1f}%)\n")
    
    # 4. Episode完成情况
    done_true = content.count('done=True')
    done_false = content.count('done=False')
    if done_true > 0:
        print(f"📈 Episode完成:")
        print(f"  完成: {done_true}")
        print(f"  未完成: {done_false}")
        print(f"  完成率: {done_true/(done_true+done_false)*100:.1f}%\n")
    
    # 5. 训练阶段
    if 'Begin training' in content:
        print("✅ 训练阶段: 已开始训练")
        
        # 查找训练指标
        clip_ratios = re.findall(r"'response_length/clip_ratio':\s*([\d.]+)", content)
        if clip_ratios:
            print(f"\n📊 Response Length Clip Ratio:")
            for i, cr in enumerate(clip_ratios[:5]):
                print(f"  Step {i}: {float(cr):.3f}")
    else:
        print("⏳ 训练阶段: Pre-training validation")
    
    # 6. 错误检查
    if 'TypeError' in content or 'Traceback' in content:
        print("\n❌ 发现错误!")
    else:
        print("\n✅ 无技术错误")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python analyze_training_log.py <log_file>")
        sys.exit(1)
    
    analyze_log(sys.argv[1])

