#!/bin/bash
set -e

# 配置参数
VLLM_SERVER_URL="http://localhost:8001"
MODEL_NAME=${MODEL_NAME:-"qwen3"}
ALFWORLD_SERVER="http://127.0.0.1:36004"
DATA_PATH="/Data/wyh/datasets/Verl-Data/eval/alfworld/test.parquet"
OUTPUT_DIR="/Data/wyh/datasets/Verl-Data/outputs/alfworld_eval"

# 环境变量覆盖
MAX_SAMPLES=${MAX_SAMPLES:--1}
NUM_SAMPLES_PER_TASK=${NUM_SAMPLES_PER_TASK:-2}
CONCURRENCY=${CONCURRENCY:-8}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
MAX_ROUNDS=${MAX_ROUNDS:-20}

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/eval_service_${TIMESTAMP}.log"

python /Data/wyh/verl/examples/sglang_multiturn/my_exp/eval/eval_alfworld_vllm_server.py \
  --vllm_server_url "$VLLM_SERVER_URL" \
  --model_name "$MODEL_NAME" \
  --alfworld_server "$ALFWORLD_SERVER" \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --max_samples "$MAX_SAMPLES" \
  --num_samples_per_task "$NUM_SAMPLES_PER_TASK" \
  --concurrency "$CONCURRENCY" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --max_rounds "$MAX_ROUNDS" \
  2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "评估完成! 结果已保存至:" | tee -a "$LOG_FILE"
grep -A 2 "Output file" "$LOG_FILE" | tail -n 2 | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
