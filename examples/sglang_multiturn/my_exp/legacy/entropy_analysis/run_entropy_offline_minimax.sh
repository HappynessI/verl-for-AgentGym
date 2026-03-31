#!/bin/bash
# 离线熵分析脚本：支持 BabyAI 和 TextCraft 数据集
# 本地 forward：直接加载 Qwen3-1.7B（不依赖 vLLM）

set -e

# ========== 数据集选择 ==========
# 可选值: babyai, textcraft, sciworld, alfworld, webshop
DATASET=${DATASET:-"textcraft"}

# 根据数据集选择对应的轨迹目录
case "$DATASET" in
    babyai)
        TRAJ_DIR="/Data/wyh/datasets/Sampling-Data/babyai_MiniMax-M2.1_20260307_150356"
        ;;
    textcraft)
        TRAJ_DIR="/Data/wyh/datasets/Sampling-Data/textcraft_MiniMax-M2.1_20260307_150412"
        ;;
    sciworld)
        TRAJ_DIR="/Data/wyh/datasets/Sampling-Data/sciworld_MiniMax-M2.1_20260307_150458"
        ;;
    alfworld)
        TRAJ_DIR="/Data/wyh/datasets/Sampling-Data/alfworld_MiniMax-M2.1_20260313_212024"
        ;;
    webshop)
        # 优先使用最新的 webshop 数据
        if [ -d "/Data/wyh/datasets/Sampling-Data/webshop_MiniMax-M2.1_20260314_132331" ]; then
            TRAJ_DIR="/Data/wyh/datasets/Sampling-Data/webshop_MiniMax-M2.1_20260314_132331"
        else
            TRAJ_DIR="/Data/wyh/datasets/Sampling-Data/webshop_MiniMax-M2.1_20260307_150431"
        fi
        ;;
    *)
        echo "Error: Unknown dataset '$DATASET'. Supported: babyai, textcraft, sciworld, alfworld, webshop"
        exit 1
        ;;
esac

MODEL_PATH=${MODEL_PATH:-"/Data/public/Qwen3-1.7B"}
TOKENIZER_NAME=${TOKENIZER_NAME:-"/Data/public/Qwen3-1.7B"}
CUDA_DEVICE=${CUDA_DEVICE:-0}

# 基础输出目录 (统一放在 entropy_offline_minimax 下)
BASE_OUTPUT_DIR=${BASE_OUTPUT_DIR:-"/Data/wyh/datasets/Verl-Data/outputs/entropy_offline_minimax"}

# 自动创建带时间戳的实验子目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${BASE_OUTPUT_DIR}/entropy_offline_${DATASET}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

MAX_SAMPLES=${MAX_SAMPLES:--1}      # -1 = 全部轨迹
MAX_TURNS=${MAX_TURNS:--1}          # -1 = 分析所有 turn
TOP_K=${TOP_K:-100}                  # top-k logprobs

echo "======================================================="
echo "  Offline Entropy: Qwen3-1.7B on ${DATASET} Trajectories"
echo "  Output dir   : $OUTPUT_DIR"
echo "  Traj dir    : $TRAJ_DIR"
echo "  Model path  : $MODEL_PATH"
echo "  Tokenizer   : $TOKENIZER_NAME"
echo "  CUDA device : $CUDA_DEVICE"
echo "  Max samples : $MAX_SAMPLES  (-1=all)"
echo "  Max turns   : $MAX_TURNS    (-1=all turns)"
echo "  Top-k       : $TOP_K"
echo "======================================================="

/home/wyh/miniconda3/envs/verl/bin/python \
    /Data/wyh/verl/examples/sglang_multiturn/my_exp/entropy_analysis/entropy_offline_minimax_traj.py \
    --traj_dir    "$TRAJ_DIR"    \
    --model_path  "$MODEL_PATH"  \
    --tokenizer_name "$TOKENIZER_NAME" \
    --cuda_device "$CUDA_DEVICE" \
    --output_dir  "$OUTPUT_DIR"   \
    --max_samples "$MAX_SAMPLES" \
    --max_turns   "$MAX_TURNS"   \
    --top_k       "$TOP_K"       \
    --save_per_token_entropy
