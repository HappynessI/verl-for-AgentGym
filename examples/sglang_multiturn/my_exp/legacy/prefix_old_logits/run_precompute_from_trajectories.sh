#!/bin/bash
# 运行脚本：基于完整 teacher 轨迹预计算 SFT old logprob

set -e

# 配置
INPUT_JSONL="/Data/wyh/datasets/Sampling-Data/textcraft_MiniMax-M2.1_20260307_150412/textcraft_trajectories.jsonl"
OUTPUT_PARQUET="/Data/wyh/datasets/Sampling-Data/textcraft_MiniMax-M2.1_20260307_150412/textcraft_trajectories_old_logprobs.parquet"
MODEL_PATH="/Data/wyh/datasets/Verl-Data/outputs/textcraft_sft/qwen3-1.7b-sft/global_step_200/huggingface"
DEVICE="cuda"
ENABLE_THINKING=false

# 启用 thinking 时取消注释
# ENABLE_THINKING=true

echo "=========================================="
echo "SFT Old Logprob 预计算（完整轨迹级）"
echo "=========================================="
echo "输入: $INPUT_JSONL"
echo "输出: $OUTPUT_PARQUET"
echo "模型: $MODEL_PATH"
echo "设备: $DEVICE"
echo "思考模式: $ENABLE_THINKING"
echo ""

# 运行预处理
python3 /Data/wyh/verl/examples/sglang_multiturn/my_exp/prefix_old_logits/precompute_sft_old_logprobs_from_trajectories.py \
    --input_path "$INPUT_JSONL" \
    --output_path "$OUTPUT_PARQUET" \
    --model_path "$MODEL_PATH" \
    --device "$DEVICE" \
    $( [ "$ENABLE_THINKING" = "true" ] && echo "--enable_thinking" || true )

echo ""
echo "完成！输出文件: $OUTPUT_PARQUET"