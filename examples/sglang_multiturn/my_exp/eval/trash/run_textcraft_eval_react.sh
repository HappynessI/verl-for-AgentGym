#!/bin/bash
# TextCraft评估脚本 - ADaPT风格
# 配置：贪婪解码，单行输出，few-shot examples

set -e

export CUDA_VISIBLE_DEVICES=2  # 使用GPU 2

MODEL_PATH="/Data/public/Qwen3-1.7B"
DATA_PATH="/Data/wyh/datasets/Verl-Data/eval/textcraft/test.parquet"
OUTPUT_DIR="/Data/wyh/datasets/Verl-Data/outputs/textcraft_eval"  # 保持和原来一致
TEXTCRAFT_SERVER="http://127.0.0.1:36002"
MAX_SAMPLES=${MAX_SAMPLES:-100}  # 默认测试100个样本

# ADaPT风格参数设置
MAX_NEW_TOKENS=150    # ADaPT: 短输出，单行action
TEMPERATURE=${TEMPERATURE:-0.0}          # ADaPT: 贪婪解码
TOP_P=${TOP_P:-1.0}                      # ADaPT: top_p=1.0
DO_SAMPLE=""                             # ADaPT: 关闭采样（贪婪）

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/eval_react_${TIMESTAMP}.log"

echo "TextCraft评估 - ADaPT风格配置" | tee "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "GPU: $CUDA_VISIBLE_DEVICES" | tee -a "$LOG_FILE"
echo "模型: $MODEL_PATH" | tee -a "$LOG_FILE"
echo "数据: $DATA_PATH" | tee -a "$LOG_FILE"
echo "样本数: $MAX_SAMPLES" | tee -a "$LOG_FILE"
echo "输出目录: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "ADaPT参数配置:" | tee -a "$LOG_FILE"
echo "  max_new_tokens: $MAX_NEW_TOKENS (单行输出)" | tee -a "$LOG_FILE"
echo "  temperature: $TEMPERATURE (贪婪解码)" | tee -a "$LOG_FILE"
echo "  top_p: $TOP_P" | tee -a "$LOG_FILE"
echo "  do_sample: False (贪婪)" | tee -a "$LOG_FILE"
echo "  max_rounds: 40" | tee -a "$LOG_FILE"
echo "  stop: ['\n'] (强制单行)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "ADaPT特性:" | tee -a "$LOG_FILE"
echo "  ✅ Few-shot examples" | tee -a "$LOG_FILE"
echo "  ✅ 单行action输出 (stop=['\n'])" | tee -a "$LOG_FILE"
echo "  ✅ 贪婪解码 (temperature=0)" | tee -a "$LOG_FILE"
echo "  ✅ 直接字符串拼接 (不使用chat_template)" | tee -a "$LOG_FILE"
echo "  ✅ 启用vLLM加速推理" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

cd /Data/wyh/verl
source ~/miniconda3/bin/activate verl

python examples/sglang_multiturn/my_exp/eval/eval_textcraft_qwen3_1.7b_react.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --textcraft_server "$TEXTCRAFT_SERVER" \
    --max_samples "$MAX_SAMPLES" \
    --max_rounds 40 \
    --max_length 8192 \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    $DO_SAMPLE \
    --use_vllm \
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "=" | tee -a "$LOG_FILE"
echo "评估完成！" | tee -a "$LOG_FILE"
echo "日志文件: $LOG_FILE" | tee -a "$LOG_FILE"
echo "=" | tee -a "$LOG_FILE"

