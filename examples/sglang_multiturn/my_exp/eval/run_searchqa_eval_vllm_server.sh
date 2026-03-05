#!/bin/bash
set -e

# 配置参数
VLLM_SERVER_URL="http://localhost:8000"
MODEL_NAME=${MODEL_NAME:-"qwen3"}          # vLLM中注册的模型名称
ENV_SERVER="http://127.0.0.1:36003"
DATA_PATH="/Data/wyh/datasets/Verl-Data/eval/searchqa/test.parquet"
OUTPUT_DIR="/Data/wyh/datasets/Verl-Data/outputs/searchqa_eval"

# 环境变量覆盖
MAX_SAMPLES=${MAX_SAMPLES:--1}
NUM_SAMPLES_PER_TASK=${NUM_SAMPLES_PER_TASK:-1}
CONCURRENCY=${CONCURRENCY:-16}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
MAX_ROUNDS=${MAX_ROUNDS:-25}

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/eval_service_${TIMESTAMP}.log"

python /Data/wyh/verl/examples/sglang_multiturn/my_exp/eval/eval_searchqa_vllm_server.py \
  --vllm_server_url "$VLLM_SERVER_URL" \
  --model_name "$MODEL_NAME" \
  --env_server "$ENV_SERVER" \
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

echo "评估完成! 结果: $OUTPUT_DIR"

