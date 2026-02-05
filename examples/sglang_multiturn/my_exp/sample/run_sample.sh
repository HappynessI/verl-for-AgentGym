#!/bin/bash
# ========================================
# TextCraft 采样脚本 - Box Format
# 使用 Gemini API 与 TextCraft 环境交互采样轨迹
# ========================================

# API配置 - 中转API
export GEMINI_API_KEY="sk-7HWdZiaoPNzAxAaieSs5NwRyPjknm0Q2tvih2x2Pd4N04ZB3"

# 任务选择
TASK_NAME="textcraft"

# 环境服务器地址
ENV_SERVER_BASE="http://127.0.0.1:36001"

# 最大交互轮数
MAX_ROUND=25

# 数据容量
DATA_LEN=400

# Gemini API参数
BASE_URL="https://once.novai.su/v1"
MODEL="[次]gemini-3-pro-preview"
TEMPERATURE=1
TOP_P=1
MAX_TOKENS=16384

# 每个任务采样次数
NUM_SAMPLES=4

# 起始任务索引（用于断点恢复，例如从 textcraft_192 开始则设为 192）
START_IDX=0

# 数据文件
INFERENCE_FILE="/Data/wyh/datasets/AgentGym-RL-Data/train/textcraft_train.json"

# 指定输出目录（可选，用于断点恢复）
# OUTPUT_DIR=""

# 切换到脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "========================================"
echo "TextCraft Sampling (Box Format)"
echo "========================================"
echo "Model: ${MODEL}"
echo "Temperature: ${TEMPERATURE}, Top-p: ${TOP_P}"
echo "Max tokens: ${MAX_TOKENS}"
echo "Max rounds: ${MAX_ROUND}"
echo "Samples per task: ${NUM_SAMPLES}"
echo "Data length: ${DATA_LEN}"
echo "Start index: ${START_IDX}"
echo "========================================"

python api_sample.py \
    --api_key "${GEMINI_API_KEY}" \
    --base_url "${BASE_URL}" \
    --model "${MODEL}" \
    --task_name "${TASK_NAME}" \
    --env_server_base "${ENV_SERVER_BASE}" \
    --max_round "${MAX_ROUND}" \
    --data_len "${DATA_LEN}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --max_tokens "${MAX_TOKENS}" \
    --num_samples "${NUM_SAMPLES}" \
    --inference_file "${INFERENCE_FILE}" \
    --start_idx "${START_IDX}" \
    --seed 42 \
    ${OUTPUT_DIR:+--output_dir "${OUTPUT_DIR}"}

