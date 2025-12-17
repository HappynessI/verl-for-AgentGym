#!/bin/bash
# TextCraft评估脚本 - Qwen3-1.7B (vLLM高性能推理 + ADaPT风格)
# 
# 使用方法:
#   全量测试: bash run_textcraft_eval_vllm.sh
#   指定样本数: MAX_SAMPLES=20 bash run_textcraft_eval_vllm.sh
#   每个任务采样多次: NUM_SAMPLES_PER_TASK=8 bash run_textcraft_eval_vllm.sh
#   使用其他GPU: CUDA_VISIBLE_DEVICES=3 bash run_textcraft_eval_vllm.sh
#   调整GPU显存利用率: GPU_MEMORY_UTILIZATION=0.85 bash run_textcraft_eval_vllm.sh
#
# vLLM优势:
#   - PagedAttention: 更高效的内存管理
#   - 更快的推理速度（约2-3x）
#   - 更高的吞吐量
#
# vLLM GPU配置:
#   - gpu_memory_utilization: 0.9 (默认使用90%的GPU显存)
#   - 可通过环境变量 GPU_MEMORY_UTILIZATION 调整
#   - 脚本会在开始和结束时显示GPU状态
#
# ADaPT配置说明:
#   - 不使用chat template，直接拼接prompt (与ADaPT一致)
#   - Few-shot prompt包含2个完整示例
#   - 贪心解码 (temperature=0.0, do_sample=False)
#   - 单行输出 (stop at \n, max_new_tokens=150)

set -e

export CUDA_VISIBLE_DEVICES=2

# 允许通过环境变量覆盖模型路径（用于训练后自动评估）
MODEL_PATH="${MODEL_PATH:-/Data/public/Qwen3-1.7B}"
DATA_PATH="/Data/wyh/datasets/Verl-Data/eval/textcraft/test.parquet"
OUTPUT_DIR="/Data/wyh/datasets/Verl-Data/outputs/textcraft_eval"
TEXTCRAFT_SERVER="http://127.0.0.1:36004"
MAX_SAMPLES=${MAX_SAMPLES:--1}  # -1 means all samples
NUM_SAMPLES_PER_TASK=${NUM_SAMPLES_PER_TASK:-1}  # Number of samples per task (default: 1)

echo "评估配置 (vLLM版本):"
echo "  模型路径: $MODEL_PATH"
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo "  样本数: $MAX_SAMPLES"
echo "  每个任务采样次数: $NUM_SAMPLES_PER_TASK"
echo "  推理引擎: vLLM (高性能)"
echo ""

# ADaPT风格参数配置
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-150}    # ADaPT: 150 (单行输出)
TEMPERATURE=0.0       # ADaPT: 0.0 (贪心解码)
TOP_P=1.0                # ADaPT: 1.0
MAX_LENGTH=8192          # 上下文长度

# vLLM GPU配置
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}  # 使用90%的GPU显存

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/eval_vllm_${TIMESTAMP}.log"

echo "================================================================================" | tee "$LOG_FILE"
echo "TextCraft评估 - Qwen3-1.7B (vLLM + ADaPT风格)" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "GPU: $CUDA_VISIBLE_DEVICES" | tee -a "$LOG_FILE"
echo "模型: $MODEL_PATH" | tee -a "$LOG_FILE"
echo "数据: $DATA_PATH" | tee -a "$LOG_FILE"
echo "样本数: $MAX_SAMPLES (-1=全部)" | tee -a "$LOG_FILE"
echo "推理引擎: vLLM" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "vLLM GPU配置:" | tee -a "$LOG_FILE"
echo "  gpu_memory_utilization: $GPU_MEMORY_UTILIZATION (使用GPU显存比例)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "ADaPT参数配置:" | tee -a "$LOG_FILE"
echo "  max_new_tokens: $MAX_NEW_TOKENS (ADaPT: 150)" | tee -a "$LOG_FILE"
echo "  temperature: $TEMPERATURE (ADaPT: 0.0, 贪心解码)" | tee -a "$LOG_FILE"
echo "  top_p: $TOP_P (ADaPT: 1.0)" | tee -a "$LOG_FILE"
echo "  max_length: $MAX_LENGTH" | tee -a "$LOG_FILE"
echo "  max_rounds: 40 (ADaPT默认)" | tee -a "$LOG_FILE"
echo "  stop_tokens: ['\\n'] (单行输出)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "日志: $LOG_FILE" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 显示GPU初始状态
echo "GPU初始状态:" | tee -a "$LOG_FILE"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

cd /Data/wyh/verl
source ~/miniconda3/bin/activate verl

python examples/sglang_multiturn/my_exp/eval/eval_textcraft_qwen3_1.7b_vllm.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --textcraft_server "$TEXTCRAFT_SERVER" \
    --max_samples "$MAX_SAMPLES" \
    --num_samples_per_task "$NUM_SAMPLES_PER_TASK" \
    --max_rounds 40 \
    --max_length "$MAX_LENGTH" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "GPU最终状态:" | tee -a "$LOG_FILE"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "评估完成，日志保存至: $LOG_FILE" | tee -a "$LOG_FILE"


