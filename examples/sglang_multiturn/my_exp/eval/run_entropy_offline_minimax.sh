#!/bin/bash
# 小模型（Qwen3-1.7B）在 MiniMax 轨迹上的离线熵分析
# 前置：vLLM 服务已启动（端口 8000）
# 不需要 textcraft 环境服务器

set -e

TRAJ_DIR="/Data/wyh/datasets/Sampling-Data/textcraft_MiniMax-M2.1_20260307_150412"
VLLM_URL="http://localhost:8000"
MODEL_NAME="qwen3"
OUTPUT_DIR="/Data/wyh/datasets/Verl-Data/outputs/entropy_offline_minimax_150412"

MAX_SAMPLES=${MAX_SAMPLES:--1}      # -1 = 全部 1496 条轨迹
CONCURRENCY=${CONCURRENCY:-32}      # 本地 vLLM，并发可以高一些
MAX_TURNS=${MAX_TURNS:-15}          # 每条轨迹最多分析前 N 个 turn
TOP_K=${TOP_K:-20}                  # top-k logprobs

mkdir -p "$OUTPUT_DIR"

echo "======================================================="
echo "  Offline Entropy: Qwen3-1.7B on Gemini Trajectories"
echo "  Traj dir   : $TRAJ_DIR"
echo "  vLLM URL   : $VLLM_URL"
echo "  Max samples: $MAX_SAMPLES  (-1=all 1496)"
echo "  Concurrency: $CONCURRENCY"
echo "======================================================="

/home/wyh/miniconda3/envs/verl/bin/python \
    /Data/wyh/verl/examples/sglang_multiturn/my_exp/eval/entropy_offline_minimax_traj.py \
    --traj_dir    "$TRAJ_DIR"    \
    --vllm_url    "$VLLM_URL"    \
    --model_name  "$MODEL_NAME"  \
    --output_dir  "$OUTPUT_DIR"  \
    --max_samples "$MAX_SAMPLES" \
    --concurrency "$CONCURRENCY" \
    --max_turns   "$MAX_TURNS"   \
    --top_k       "$TOP_K"
