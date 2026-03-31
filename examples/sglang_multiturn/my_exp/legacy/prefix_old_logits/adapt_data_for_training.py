#!/usr/bin/env python3
"""
数据适配脚本：将 prefix old_logprobs 数据适配为训练代码需要的格式

训练代码期望的字段：
- assistant_prefix_old_log_probs: 列表的列表，每个样本一个列表
- prefix_loss_mask: 列表的列表（可选）

当前数据字段：
- prefix_old_logprobs: 列表
- prefix_loss_mask: 列表

这个脚本会：
1. 读取原始数据
2. 重命名字段为训练代码期望的名称
3. 保存为新的 parquet 文件
"""

import argparse
import pandas as pd
import os


def main():
    parser = argparse.ArgumentParser(description="Adapt prefix old_logprobs data for training")
    parser.add_argument("--input_path", type=str, required=True, help="Input parquet path")
    parser.add_argument("--output_path", type=str, required=True, help="Output parquet path")
    args = parser.parse_args()
    
    print(f"读取数据: {args.input_path}")
    df = pd.read_parquet(args.input_path)
    
    print(f"原始列: {df.columns.tolist()}")
    print(f"样本数: {len(df)}")
    
    # 检查必要字段
    if "prefix_old_logprobs" not in df.columns:
        raise ValueError(f"缺少必要字段: prefix_old_logprobs")
    
    # 重命名字段：prefix_old_logprobs -> assistant_prefix_old_log_probs
    # 训练代码从这个字段读取 cached SFT old logprobs
    df["assistant_prefix_old_log_probs"] = df["prefix_old_logprobs"]
    
    # 可选：添加 prefix_loss_mask（如果存在）
    if "prefix_loss_mask" in df.columns:
        # 保持原样，训练代码会根据需要使用
        pass
    
    # 添加 item_id 和 sample_idx（如果不存在）
    if "item_id" not in df.columns:
        # 从 extra_info 提取
        df["item_id"] = df["extra_info"].apply(
            lambda x: f"textcraft_{x['interaction_kwargs']['task_id']}"
            if x and "interaction_kwargs" in x and "task_id" in x.get("interaction_kwargs", {})
            else "unknown"
        )
    
    if "sample_idx" not in df.columns:
        # 从 extra_info 提取
        df["sample_idx"] = df["extra_info"].apply(
            lambda x: x.get("index", 0) if x else 0
        )
    
    # 打印统计信息
    print("\n数据统计:")
    print(f"  assistant_prefix_old_log_probs 长度范围: {df['assistant_prefix_old_log_probs'].apply(len).min()} - {df['assistant_prefix_old_log_probs'].apply(len).max()}")
    
    if "prefix_loss_mask" in df.columns:
        print(f"  prefix_loss_mask 长度范围: {df['prefix_loss_mask'].apply(len).min()} - {df['prefix_loss_mask'].apply(len).max()}")
    
    # 保存
    print(f"\n保存到: {args.output_path}")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    df.to_parquet(args.output_path)
    
    print("完成!")
    print(f"\n输出列: {df.columns.tolist()}")


if __name__ == "__main__":
    main()
