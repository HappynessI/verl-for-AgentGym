# Copyright 2024 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Create a simple multi-turn dataset for testing
"""
import argparse
import os
import pandas as pd

# -------------------------------------------------------------------------
# Linus 评注: 
# 这是一个典型的 ETL (Extract, Transform, Load) 脚本的"玩具版"。
# 它手动构造了数据，而不是从文件读取。
# 在生产环境中，数据生成逻辑应该与数据存储逻辑分离。
# -------------------------------------------------------------------------

def main():
    # 解析命令行参数
    # 这里的默认路径设为 '~/data/multiturn'，符合 Linux 文件系统惯例
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/multiturn", help="本地存储 Parquet 文件的路径")
    parser.add_argument("--hdfs_dir", default=None, help="可选：同步到 HDFS 分布式文件系统的路径")
    args = parser.parse_args()

    # 初始化对话列表
    # 数据结构：List[Dict]，每个 Dict 代表一个完整的 Episode（多轮交互）
    conversations = []

    # ---------------------------------------------------------------------
    # 场景 1: 基础问答 (General Knowledge)
    # 结构：System -> User -> Assistant -> User -> Assistant
    # ---------------------------------------------------------------------
    conversations.append(
        {
            "messages": [
                # System Prompt: 定义 Agent 的基本行为模式
                {"role": "system", "content": "You are a helpful assistant."},
                # Turn 1
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."},
                # Turn 2: 依赖上下文的问题 ("And what about Germany?")
                {"role": "user", "content": "And what about Germany?"},
                {"role": "assistant", "content": "The capital of Germany is Berlin."},
            ]
        }
    )

    # ---------------------------------------------------------------------
    # 场景 2: 解释性任务 (Explanation)
    # 展示长文本生成的即使
    # ---------------------------------------------------------------------
    conversations.append(
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Can you explain quantum computing?"},
                {
                    "role": "assistant",
                    "content": "Quantum computing is a type of computing that uses quantum-mechanical phenomena, "
                    "such as superposition and entanglement, to perform operations on data.",
                },
                {"role": "user", "content": "How is it different from classical computing?"},
                {
                    "role": "assistant",
                    "content": "Classical computing uses bits that are either 0 or 1, while quantum computing uses "
                    "quantum bits or qubits that can exist in multiple states simultaneously due to superposition.",
                },
            ]
        }
    )

    # ---------------------------------------------------------------------
    # 场景 3: 代码生成 (Code Generation)
    # 注意：这里演示了如何在 content 中嵌入 Markdown 代码块
    # ---------------------------------------------------------------------
    conversations.append(
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Write a simple Python function to calculate factorial."},
                {
                    "role": "assistant",
                    # 使用 Markdown 格式包裹代码，这对 LLM 训练至关重要
                    "content": (
                        "```python\ndef factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    else:\n        "
                        "return n * factorial(n-1)\n```\n\nThis is a recursive function to calculate the "
                        "factorial of a number."
                    ),
                },
                {"role": "user", "content": "Can you make it iterative instead?"},
                {
                    "role": "assistant",
                    "content": (
                        "```python\ndef factorial(n):\n    result = 1\n    for i in range(1, n+1):\n        "
                        "result *= i\n    return result\n```\n\nThis is an iterative version of the factorial function."
                    ),
                },
            ]
        }
    )

    # ---------------------------------------------------------------------
    # 数据集切分
    # 这是一个非常粗糙的切分方式（硬编码切分点）。
    # 在实际工程中，应该使用 sklearn.model_selection.train_test_split 或随机采样。
    # ---------------------------------------------------------------------
    train_data = conversations[:2]  # 前 2 个对话用于训练
    test_data = conversations[2:]   # 最后 1 个对话用于测试

    # 创建输出目录
    # os.path.expanduser 处理 '~' 符号，确保路径在任何用户环境下都有效
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    # ---------------------------------------------------------------------
    # 核心存储逻辑
    # 使用 Pandas 将内存中的 List[Dict] 转换为 DataFrame，并保存为 Parquet。
    # Parquet 是列式存储，比 CSV 更高效，且支持嵌套数据结构（如 list），
    # 是目前 AI 训练数据的主流格式。
    # ---------------------------------------------------------------------
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    train_df.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_df.to_parquet(os.path.join(local_dir, "test.parquet"))

    # ---------------------------------------------------------------------
    # HDFS 处理逻辑
    # 这是一个可选步骤，用于将数据同步到 Hadoop 集群。
    # 这种特定的导入方式 (verl.utils.hdfs_io) 表明这是特定框架 (VeRL) 的一部分。
    # ---------------------------------------------------------------------
    if args.hdfs_dir is not None:
        try:
            from verl.utils.hdfs_io import copy, makedirs

            makedirs(args.hdfs_dir)
            copy(src=local_dir, dst=args.hdfs_dir)
        except ImportError:
            # 这种错误处理是明智的，避免因为缺少可选依赖而导致整个脚本崩溃
            print("Warning: HDFS support not available. Skipping HDFS copy.")

    # 打印统计信息，好的 CLI 程序都应该给用户反馈
    print(f"Train dataset size: {len(train_df)}")
    print(f"Test dataset size: {len(test_df)}")
    print(f"Data saved to {local_dir}")


if __name__ == "__main__":
    main()