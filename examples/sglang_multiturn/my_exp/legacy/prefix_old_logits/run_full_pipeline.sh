#!/bin/bash
# 批处理脚本：生成 step200 和 step460 的 sidecar，然后构建 prefix 训练数据

set -e

# GPU 配置
export CUDA_VISIBLE_DEVICES=4

# 路径配置
INPUT_JSONL="/Data/wyh/datasets/Sampling-Data/textcraft_MiniMax-M2.1_20260307_150412/textcraft_trajectories.jsonl"
OUTPUT_DIR="/Data/wyh/datasets/Sampling-Data/textcraft_MiniMax-M2.1_20260307_150412"
PREFIX_DATA="/Data/wyh/datasets/Verl-Data/train/textcraft/prefix-rl/textcraft_validated_prefix_history_canonicalized.parquet"
MODEL_BASE="/Data/wyh/datasets/Verl-Data/outputs/textcraft_sft/qwen3-1.7b-sft"
OLD_LOGITS_OUTPUT="/Data/wyh/datasets/Verl-Data/outputs/textcraft_old_logits"

# Step 200
echo "=========================================="
echo "Step 1/4: 生成 step200 sidecar"
echo "=========================================="
python3 /Data/wyh/verl/examples/sglang_multiturn/my_exp/prefix_old_logits/precompute_sft_old_logprobs_from_trajectories.py \
    --input_path "$INPUT_JSONL" \
    --output_path "$OLD_LOGITS_OUTPUT/textcraft_trajectories_old_logprobs_step200.parquet" \
    --model_path "$MODEL_BASE/global_step_200/huggingface" \
    --device cuda \
    --batch_size 64
echo "step200 sidecar 完成"

# Step 460
echo ""
echo "=========================================="
echo "Step 2/4: 生成 step460 sidecar"
echo "=========================================="
python3 /Data/wyh/verl/examples/sglang_multiturn/my_exp/prefix_old_logits/precompute_sft_old_logprobs_from_trajectories.py \
    --input_path "$INPUT_JSONL" \
    --output_path "$OLD_LOGITS_OUTPUT/textcraft_trajectories_old_logprobs_step460.parquet" \
    --model_path "$MODEL_BASE/global_step_460/huggingface" \
    --device cuda \
    --batch_size 64
echo "step460 sidecar 完成"

# 分析 step200
echo ""
echo "=========================================="
echo "Step 3/4: 分析 step200 old logprob + entropy"
echo "=========================================="
python3 /Data/wyh/verl/examples/sglang_multiturn/my_exp/prefix_old_logits/analyze_old_logprob_entropy.py \
    --sidecar_path "$OLD_LOGITS_OUTPUT/textcraft_trajectories_old_logprobs_step200.parquet" \
    --trajectories_path "$INPUT_JSONL" \
    --model_path "$MODEL_BASE/global_step_200/huggingface" \
    --device cuda
echo "step200 分析完成"

# 分析 step460
echo ""
echo "=========================================="
echo "Step 3/4: 分析 step460 old logprob + entropy"
echo "=========================================="
python3 /Data/wyh/verl/examples/sglang_multiturn/my_exp/prefix_old_logits/analyze_old_logprob_entropy.py \
    --sidecar_path "$OLD_LOGITS_OUTPUT/textcraft_trajectories_old_logprobs_step460.parquet" \
    --trajectories_path "$INPUT_JSONL" \
    --model_path "$MODEL_BASE/global_step_460/huggingface" \
    --device cuda
echo "step460 分析完成"

# 构建 step200 prefix 训练数据
echo ""
echo "=========================================="
echo "Step 4/4: 构建 step200 prefix 训练数据"
echo "=========================================="
python3 /Data/wyh/verl/examples/sglang_multiturn/my_exp/prefix_old_logits/build_prefix_old_logprob_dataset.py \
    --prefix_data_path "$PREFIX_DATA" \
    --sidecar_path "$OLD_LOGITS_OUTPUT/textcraft_trajectories_old_logprobs_step200.parquet" \
    --output_path "$OLD_LOGITS_OUTPUT/textcraft_validated_prefix_with_old_logprobs_step200.parquet" \
    --fixed_ratio 0.4
echo "step200 prefix 训练数据完成"

# 构建 step460 prefix 训练数据
echo ""
echo "=========================================="
echo "Step 4/4: 构建 step460 prefix 训练数据"
echo "=========================================="
python3 /Data/wyh/verl/examples/sglang_multiturn/my_exp/prefix_old_logits/build_prefix_old_logprob_dataset.py \
    --prefix_data_path "$PREFIX_DATA" \
    --sidecar_path "$OLD_LOGITS_OUTPUT/textcraft_trajectories_old_logprobs_step460.parquet" \
    --output_path "$OLD_LOGITS_OUTPUT/textcraft_validated_prefix_with_old_logprobs_step460.parquet" \
    --fixed_ratio 0.4
echo "step460 prefix 训练数据完成"

echo ""
echo "=========================================="
echo "所有任务完成!"
echo "=========================================="
echo "生成的文件:"
ls -lh "$OLD_LOGITS_OUTPUT"/textcraft_trajectories_old_logprobs_step*.parquet
ls -lh "$OLD_LOGITS_OUTPUT"/textcraft_validated_prefix_with_old_logprobs_step*.parquet