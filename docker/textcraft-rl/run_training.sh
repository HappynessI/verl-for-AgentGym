#!/bin/bash
# ============================================================
# TextCraft GRPO 训练启动脚本（Docker 容器内运行）
# ============================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  TextCraft GRPO 训练环境${NC}"
echo -e "${GREEN}============================================================${NC}"

# 检查 GPU
echo -e "\n${YELLOW}[1/5] 检查 GPU 状态...${NC}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo -e "${GREEN}✓ 检测到 ${NUM_GPUS} 个 GPU${NC}"

# 检查 verl 安装
echo -e "\n${YELLOW}[2/5] 检查 verl 安装...${NC}"
if [ -d "/workspace/verl" ]; then
    cd /workspace/verl
    pip install -e . --no-deps -q
    echo -e "${GREEN}✓ verl 已从本地目录安装 (开发模式)${NC}"
else
    echo -e "${RED}✗ 未找到 verl 目录，请检查挂载${NC}"
    exit 1
fi

# 检查数据
echo -e "\n${YELLOW}[3/5] 检查训练数据...${NC}"
TRAIN_DATA="/workspace/datasets/Verl-Data/train/textcraft/train.parquet"
if [ -f "$TRAIN_DATA" ]; then
    echo -e "${GREEN}✓ 训练数据存在: $TRAIN_DATA${NC}"
else
    echo -e "${RED}✗ 未找到训练数据: $TRAIN_DATA${NC}"
    exit 1
fi

# 检查模型
echo -e "\n${YELLOW}[4/5] 检查模型路径...${NC}"
MODEL_PATH="${MODEL_PATH:-/workspace/models/Qwen3-1.7B}"
if [ -d "$MODEL_PATH" ]; then
    echo -e "${GREEN}✓ 模型路径存在: $MODEL_PATH${NC}"
else
    echo -e "${YELLOW}⚠ 模型路径不存在: $MODEL_PATH${NC}"
    echo -e "${YELLOW}  请确保模型已挂载到正确位置${NC}"
fi

# 检查 wandb
echo -e "\n${YELLOW}[5/5] 检查 wandb 配置...${NC}"
if [ -n "$WANDB_API_KEY" ]; then
    echo -e "${GREEN}✓ WANDB_API_KEY 已设置${NC}"
else
    echo -e "${YELLOW}⚠ WANDB_API_KEY 未设置，日志将保存到本地${NC}"
fi

echo -e "\n${GREEN}============================================================${NC}"
echo -e "${GREEN}  环境检查完成，准备开始训练${NC}"
echo -e "${GREEN}============================================================${NC}"

# ============================================================
# 训练参数配置（可通过环境变量覆盖）
# ============================================================
NUM_EPOCHS=${NUM_EPOCHS:-50}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
LEARNING_RATE=${LEARNING_RATE:-5e-6}
ENTROPY_COEFF=${ENTROPY_COEFF:-0.01}
TEMPERATURE=${TEMPERATURE:-1.0}
ROLLOUT_N=${ROLLOUT_N:-8}

echo -e "\n${YELLOW}训练参数:${NC}"
echo "  NUM_EPOCHS: $NUM_EPOCHS"
echo "  TRAIN_BATCH_SIZE: $TRAIN_BATCH_SIZE"
echo "  LEARNING_RATE: $LEARNING_RATE"
echo "  ENTROPY_COEFF: $ENTROPY_COEFF"
echo "  TEMPERATURE: $TEMPERATURE"
echo "  ROLLOUT_N: $ROLLOUT_N"
echo "  MODEL_PATH: $MODEL_PATH"

# 确认是否开始训练
echo -e "\n${YELLOW}是否开始训练? [y/N]${NC}"
read -r confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "训练已取消"
    exit 0
fi

# 创建输出目录
OUTPUT_DIR="/workspace/outputs/textcraft_grpo_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR/logs"

# 运行训练
echo -e "\n${GREEN}开始训练...${NC}"
echo "输出目录: $OUTPUT_DIR"

cd /workspace/verl/examples/sglang_multiturn/my_exp/rl

# 设置环境变量并运行
export NUM_EPOCHS
export TRAIN_BATCH_SIZE
export LEARNING_RATE
export ENTROPY_COEFF
export TEMPERATURE
export ROLLOUT_N
export MODEL_PATH
export OUTPUT_DIR

bash run_textcraft_grpo_train.sh 2>&1 | tee "$OUTPUT_DIR/logs/train.log"

echo -e "\n${GREEN}训练完成！${NC}"
echo "输出目录: $OUTPUT_DIR"

