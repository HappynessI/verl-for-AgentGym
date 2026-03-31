#!/bin/bash
# Smoke test: 生成 step200 sidecar，batch_size=16

set -e

# 路径配置
INPUT_JSONL="/Data/wyh/datasets/Sampling-Data/textcraft_MiniMax-M2.1_20260307_150412/textcraft_trajectories.jsonl"
OUTPUT_DIR="/Data/wyh/datasets/Verl-Data/outputs/textcraft_old_logits"
MODEL_BASE="/Data/wyh/datasets/Verl-Data/outputs/textcraft_sft/qwen3-1.7b-sft"

# 使用3卡
export CUDA_VISIBLE_DEVICES=2,3,4

echo "=========================================="
echo "Smoke Test: 生成 step200 sidecar (batch_size=16)"
echo "=========================================="
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# 先用小量数据测试 (max_samples=10)
python3 /Data/wyh/verl/examples/sglang_multiturn/my_exp/prefix_old_logits/precompute_sft_old_logprobs_from_trajectories.py \
    --input_path "$INPUT_JSONL" \
    --output_path "$OUTPUT_DIR/smoke_test_step200_bs16.parquet" \
    --model_path "$MODEL_BASE/global_step_200/huggingface" \
    --device cuda \
    --batch_size 16 \
    --max_samples 10

echo ""
echo "Smoke test 完成!"
ls -lh "$OUTPUT_DIR"/smoke_test_step200_bs16.parquet
