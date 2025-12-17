#!/bin/bash
# TextCraft评估脚本 - Qwen2.5-7B (复现AgentGym实验)
# 基于AgentGym的配置，使用vLLM推理

set -e

export CUDA_VISIBLE_DEVICES=3  # 使用GPU 2

# 模型路径 - 请修改为实际路径
MODEL_PATH="/Data/public/Qwen2.5-7B-Instruct"  
DATA_PATH="/Data/wyh/datasets/Verl-Data/eval/textcraft/test.parquet"
OUTPUT_DIR="/Data/wyh/datasets/Verl-Data/outputs/textcraft_eval/qwen2.5-7b"
TEXTCRAFT_SERVER="http://127.0.0.1:36003"
MAX_SAMPLES=${MAX_SAMPLES:-100}  # 默认测试100个样本

# AgentGym配置参数
MAX_NEW_TOKENS=512       # AgentGym默认max_length=8192，但新增tokens设置为512合理
TEMPERATURE=0.7          # 适度随机性
TOP_P=0.9               # 标准设置
DO_SAMPLE="--do_sample" # 启用采样
MAX_ROUNDS=50           # 最大轮数（AgentGym设置）

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/eval_qwen2.5_7b_${TIMESTAMP}.log"

echo "TextCraft评估 - Qwen2.5-7B (AgentGym配置)" | tee "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "GPU: $CUDA_VISIBLE_DEVICES" | tee -a "$LOG_FILE"
echo "模型: $MODEL_PATH" | tee -a "$LOG_FILE"
echo "数据: $DATA_PATH" | tee -a "$LOG_FILE"
echo "样本数: $MAX_SAMPLES" | tee -a "$LOG_FILE"
echo "输出目录: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "参数配置 (AgentGym风格):" | tee -a "$LOG_FILE"
echo "  max_new_tokens: $MAX_NEW_TOKENS" | tee -a "$LOG_FILE"
echo "  temperature: $TEMPERATURE" | tee -a "$LOG_FILE"
echo "  top_p: $TOP_P" | tee -a "$LOG_FILE"
echo "  do_sample: True" | tee -a "$LOG_FILE"
echo "  max_rounds: $MAX_ROUNDS" | tee -a "$LOG_FILE"
echo "  chat_template: chatml" | tee -a "$LOG_FILE"
echo "  action_format: react" | tee -a "$LOG_FILE"
echo "  seed: 42" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "关键设置:" | tee -a "$LOG_FILE"
echo "  ✓ 使用ReAct格式 (Thought + Action)" | tee -a "$LOG_FILE"
echo "  ✓ 应用chatml模板" | tee -a "$LOG_FILE"
echo "  ✓ 启用vLLM加速推理" | tee -a "$LOG_FILE"
echo "  ✓ 固定随机种子seed=42" | tee -a "$LOG_FILE"
echo "  ✓ 参考AgentGym配置" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

cd /Data/wyh/verl
source ~/miniconda3/bin/activate verl

python3 examples/sglang_multiturn/my_exp/eval/eval_textcraft_qwen2.5_7b.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --textcraft_server "$TEXTCRAFT_SERVER" \
    --max_samples "$MAX_SAMPLES" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --max_rounds "$MAX_ROUNDS" \
    --seed 42 \
    --use_vllm \
    $DO_SAMPLE \
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "评估完成！" | tee -a "$LOG_FILE"
echo "日志文件: $LOG_FILE" | tee -a "$LOG_FILE"

