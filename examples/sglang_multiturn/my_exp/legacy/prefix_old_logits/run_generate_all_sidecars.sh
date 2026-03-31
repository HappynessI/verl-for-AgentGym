#!/bin/bash
# 批处理脚本：分别用 step200 和 step460 生成 sidecar

set -e

# 路径配置
INPUT_JSONL="/Data/wyh/datasets/Sampling-Data/textcraft_MiniMax-M2.1_20260307_150412/textcraft_trajectories.jsonl"
OUTPUT_DIR="/Data/wyh/datasets/Sampling-Data/textcraft_MiniMax-M2.1_20260307_150412"

# Step 200
echo "=========================================="
echo "生成 step200 sidecar"
echo "=========================================="
python3 /Data/wyh/verl/examples/sglang_multiturn/my_exp/prefix_old_logits/precompute_sft_old_logprobs_from_trajectories.py \
    --input_path "$INPUT_JSONL" \
    --output_path "$OUTPUT_DIR/textcraft_trajectories_old_logprobs_step200.parquet" \
    --model_path "/Data/wyh/datasets/Verl-Data/outputs/textcraft_sft/qwen3-1.7b-sft/global_step_200/huggingface" \
    --device cuda

echo "step200 完成: $OUTPUT_DIR/textcraft_trajectories_old_logprobs_step200.parquet"

# Step 460
echo ""
echo "=========================================="
echo "生成 step460 sidecar"
echo "=========================================="
python3 /Data/wyh/verl/examples/sglang_multiturn/my_exp/prefix_old_logits/precompute_sft_old_logprobs_from_trajectories.py \
    --input_path "$INPUT_JSONL" \
    --output_path "$OUTPUT_DIR/textcraft_trajectories_old_logprobs_step460.parquet" \
    --model_path "/Data/wyh/datasets/Verl-Data/outputs/textcraft_sft/qwen3-1.7b-sft/global_step_460/huggingface" \
    --device cuda

echo "step460 完成: $OUTPUT_DIR/textcraft_trajectories_old_logprobs_step460.parquet"

echo ""
echo "=========================================="
echo "所有 sidecar 生成完成"
echo "=========================================="
ls -lh "$OUTPUT_DIR"/textcraft_trajectories_old_logprobs_step*.parquet