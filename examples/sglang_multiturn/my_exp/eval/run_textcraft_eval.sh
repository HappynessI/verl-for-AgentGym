#!/bin/bash
# TextCraft评估脚本 - Qwen3-1.7B (ADaPT风格)
# 
# 使用方法:
#   全量测试: bash run_textcraft_eval.sh
#   指定样本数: MAX_SAMPLES=20 bash run_textcraft_eval.sh
#   每个任务采样多次: NUM_SAMPLES_PER_TASK=8 bash run_textcraft_eval.sh
#   使用其他GPU: CUDA_VISIBLE_DEVICES=3 bash run_textcraft_eval.sh
#
# ADaPT配置说明:
#   - 不使用chat template，直接拼接prompt (与ADaPT一致)
#   - Few-shot prompt包含2个完整示例
#   - 贪心解码 (temperature=0.0, do_sample=False)
#   - 单行输出 (stop at \n, max_new_tokens=150)

set -e

export CUDA_VISIBLE_DEVICES=3

# 允许通过环境变量覆盖模型路径（用于训练后自动评估）
MODEL_PATH="${MODEL_PATH:-/Data/public/Qwen3-1.7B}"
DATA_PATH="/Data/wyh/datasets/Verl-Data/eval/textcraft/test.parquet"
OUTPUT_DIR="/Data/wyh/datasets/Verl-Data/outputs/textcraft_eval"
TEXTCRAFT_SERVER="http://127.0.0.1:36003"
MAX_SAMPLES=${MAX_SAMPLES:--1}  # -1 means all samples
NUM_SAMPLES_PER_TASK=${NUM_SAMPLES_PER_TASK:-1}  # Number of samples per task (default: 1)

echo "评估配置:"
echo "  模型路径: $MODEL_PATH"
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo "  样本数: $MAX_SAMPLES"
echo "  每个任务采样次数: $NUM_SAMPLES_PER_TASK"
echo ""

# ADaPT风格参数配置
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-150}    # ADaPT: 150 (单行输出)
TEMPERATURE=0.3       # ADaPT: 0.0 (贪心解码)
TOP_P=0.95               # ADaPT: 1.0
# DO_SAMPLE=""             # ADaPT: 不采样 (空字符串=不传--do_sample)
DO_SAMPLE="--do_sample"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/eval_${TIMESTAMP}.log"

echo "================================================================================" | tee "$LOG_FILE"
echo "TextCraft评估 - Qwen3-1.7B (ADaPT风格)" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "GPU: $CUDA_VISIBLE_DEVICES" | tee -a "$LOG_FILE"
echo "模型: $MODEL_PATH" | tee -a "$LOG_FILE"
echo "数据: $DATA_PATH" | tee -a "$LOG_FILE"
echo "样本数: $MAX_SAMPLES (-1=全部)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "ADaPT参数配置:" | tee -a "$LOG_FILE"
echo "  max_new_tokens: $MAX_NEW_TOKENS (ADaPT: 150)" | tee -a "$LOG_FILE"
echo "  temperature: $TEMPERATURE (ADaPT: 0.0, 贪心解码)" | tee -a "$LOG_FILE"
echo "  top_p: $TOP_P (ADaPT: 1.0)" | tee -a "$LOG_FILE"
echo "  do_sample: ${DO_SAMPLE:-False} (ADaPT: False)" | tee -a "$LOG_FILE"
echo "  max_rounds: 40 (ADaPT默认)" | tee -a "$LOG_FILE"
echo "  stop_tokens: ['\\n'] (单行输出)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "日志: $LOG_FILE" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

cd /Data/wyh/verl
source ~/miniconda3/bin/activate verl

python examples/sglang_multiturn/my_exp/eval/eval_textcraft_qwen3_1.7b.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --textcraft_server "$TEXTCRAFT_SERVER" \
    --max_samples "$MAX_SAMPLES" \
    --num_samples_per_task "$NUM_SAMPLES_PER_TASK" \
    --max_rounds 40 \
    --max_length 8192 \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    $DO_SAMPLE \
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "评估完成，日志保存至: $LOG_FILE" | tee -a "$LOG_FILE"

