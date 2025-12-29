
set -e

# 配置参数
VLLM_SERVER_URL="http://localhost:8000"
TEXTCRAFT_SERVER="http://127.0.0.1:3222"
DATA_PATH="/agent_distill/Data/Verl-Data/eval/textcraft/test.parquet"
OUTPUT_DIR="/agent_distill/Data/Verl-Data/eval/textcraft/textcraft_eval"

# 环境变量覆盖
MAX_SAMPLES=${MAX_SAMPLES:--1}          # -1 means all samples
NUM_SAMPLES_PER_TASK=${NUM_SAMPLES_PER_TASK:-8}  # Number of samples per task
CONCURRENCY=${CONCURRENCY:-256}         # Concurrent requests to vLLM
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2000}  # Max tokens per generation
TEMPERATURE=${TEMPERATURE:-1.0}         # Sampling temperature
TOP_P=${TOP_P:-1.0}                     # Top-p sampling

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/eval_service_${TIMESTAMP}.log"




# 运行评估
python examples/sglang_multiturn/my_exp/eval/eval_textcraft_vllm_server.py \
  --vllm_server_url "$VLLM_SERVER_URL" \
  --textcraft_server "$TEXTCRAFT_SERVER" \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --max_samples "$MAX_SAMPLES" \
  --num_samples_per_task "$NUM_SAMPLES_PER_TASK" \
  --concurrency "$CONCURRENCY" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "评估完成! 结果已保存至:" | tee -a "$LOG_FILE"
grep -A 2 "Output file" "$LOG_FILE" | tail -n 2 | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
