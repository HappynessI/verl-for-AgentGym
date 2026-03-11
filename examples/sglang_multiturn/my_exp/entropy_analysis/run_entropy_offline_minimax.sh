#!/bin/bash
# 离线熵分析脚本：支持 BabyAI 和 TextCraft 数据集
# 前置：vLLM 服务已启动（端口 8000）

set -e

# ========== 数据集选择 ==========
# 可选值: babyai, textcraft
DATASET=${DATASET:-"textcraft"}

# 根据数据集选择对应的轨迹目录
case "$DATASET" in
    babyai)
        TRAJ_DIR="/Data/wyh/datasets/Sampling-Data/babyai_MiniMax-M2.1_20260307_150356"
        ;;
    textcraft)
        TRAJ_DIR="/Data/wyh/datasets/Sampling-Data/textcraft_MiniMax-M2.1_20260307_150412"
        ;;
    *)
        echo "Error: Unknown dataset '$DATASET'. Supported: babyai, textcraft"
        exit 1
        ;;
esac

VLLM_URL="http://localhost:8001"
MODEL_NAME="qwen3"

# 基础输出目录
BASE_OUTPUT_DIR=${BASE_OUTPUT_DIR:-"/Data/wyh/datasets/Verl-Data/outputs"}

# 自动创建带时间戳的实验子目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${BASE_OUTPUT_DIR}/entropy_offline_${DATASET}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

MAX_SAMPLES=${MAX_SAMPLES:--1}      # -1 = 全部轨迹
CONCURRENCY=${CONCURRENCY:-8}       # 本地 vLLM，避免与训练任务竞争
MAX_TURNS=${MAX_TURNS:--1}          # -1 = 分析所有 turn
TOP_K=${TOP_K:-100}                  # top-k logprobs

echo "======================================================="
echo "  Offline Entropy: Qwen3-1.7B on ${DATASET} Trajectories"
echo "  Output dir   : $OUTPUT_DIR"
echo "  Traj dir    : $TRAJ_DIR"
echo "  vLLM URL    : $VLLM_URL"
echo "  Max samples : $MAX_SAMPLES  (-1=all)"
echo "  Max turns   : $MAX_TURNS    (-1=all turns)"
echo "  Concurrency : $CONCURRENCY"
echo "======================================================="

/home/wyh/miniconda3/envs/verl/bin/python \
    /Data/wyh/verl/examples/sglang_multiturn/my_exp/entropy_analysis/entropy_offline_minimax_traj.py \
    --traj_dir    "$TRAJ_DIR"    \
    --vllm_url    "$VLLM_URL"    \
    --model_name  "$MODEL_NAME"  \
    --output_dir  "$OUTPUT_DIR"   \
    --max_samples "$MAX_SAMPLES" \
    --concurrency "$CONCURRENCY" \
    --max_turns   "$MAX_TURNS"   \
    --top_k       "$TOP_K"       \
    --save_per_token_entropy
