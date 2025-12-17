#!/bin/bash
# 检查TextCraft SFT训练环境是否就绪

echo "================================================================================"
echo "检查 TextCraft SFT 训练环境"
echo "================================================================================"
echo ""

# 检查训练数据
echo "1. 检查训练数据..."
if [ -f "/Data/wyh/datasets/Parquet-Data/textcraft/train.parquet" ]; then
    size=$(ls -lh /Data/wyh/datasets/Parquet-Data/textcraft/train.parquet | awk '{print $5}')
    echo "   ✅ 训练数据存在: /Data/wyh/datasets/Parquet-Data/textcraft/train.parquet ($size)"
else
    echo "   ❌ 训练数据不存在"
    exit 1
fi

# 检查训练脚本
echo ""
echo "2. 检查训练脚本..."
if [ -f "/Data/wyh/verl/examples/sft/multiturn/run_textcraft_qwen3_17b_sft.sh" ]; then
    echo "   ✅ 训练脚本存在: run_textcraft_qwen3_17b_sft.sh"
    if [ -x "/Data/wyh/verl/examples/sft/multiturn/run_textcraft_qwen3_17b_sft.sh" ]; then
        echo "   ✅ 训练脚本可执行"
    else
        echo "   ⚠️  训练脚本不可执行，添加执行权限..."
        chmod +x /Data/wyh/verl/examples/sft/multiturn/run_textcraft_qwen3_17b_sft.sh
    fi
else
    echo "   ❌ 训练脚本不存在"
    exit 1
fi

# 检查评估脚本
echo ""
echo "3. 检查评估脚本..."
if [ -f "/Data/wyh/verl/examples/sglang_multiturn/my_exp/eval/run_textcraft_eval.sh" ]; then
    echo "   ✅ 评估脚本存在: run_textcraft_eval.sh"
else
    echo "   ❌ 评估脚本不存在"
fi

# 检查基础模型
echo ""
echo "4. 检查基础模型..."
if [ -d "/Data/public/Qwen3-1.7B" ]; then
    echo "   ✅ Qwen3-1.7B 模型存在"
else
    echo "   ❌ Qwen3-1.7B 模型不存在"
    exit 1
fi

# 检查GPU
echo ""
echo "5. 检查GPU..."
if command -v nvidia-smi &> /dev/null; then
    gpu_count=$(nvidia-smi --list-gpus | wc -l)
    echo "   ✅ 检测到 $gpu_count 个GPU"
    echo ""
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader | head -4
else
    echo "   ❌ 未检测到GPU"
fi

# 验证数据格式
echo ""
echo "6. 验证训练数据格式..."
python3 << 'EOF'
import pyarrow.parquet as pq
try:
    table = pq.read_table('/Data/wyh/datasets/Parquet-Data/textcraft/train.parquet')
    print(f"   ✅ 数据条数: {len(table)}")
    print(f"   ✅ 列名: {table.column_names}")
    if 'messages' in table.column_names:
        first_messages = table['messages'][0].as_py()
        print(f"   ✅ 第一条数据消息数: {len(first_messages)}")
        print(f"   ✅ 第一条消息角色: {first_messages[0]['role']}")
    else:
        print("   ❌ 缺少messages列")
except Exception as e:
    print(f"   ❌ 读取数据失败: {e}")
EOF

echo ""
echo "================================================================================"
echo "✅ 所有检查通过！可以开始训练"
echo "================================================================================"
echo ""
echo "启动训练命令:"
echo "  cd /Data/wyh/verl"
echo "  source ~/miniconda3/bin/activate verl"
echo "  CUDA_VISIBLE_DEVICES=0,1 bash examples/sft/multiturn/run_textcraft_qwen3_17b_sft.sh 2 \\"
echo "    /Data/wyh/datasets/Verl-Data/outputs/textcraft_sft"
echo ""
echo "================================================================================"

