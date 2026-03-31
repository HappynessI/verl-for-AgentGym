#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)
DATA_ROOT="${REPO_ROOT}/data/textcraft/new_prefix_rl"

INPUT_PATH=${INPUT_PATH:-"${DATA_ROOT}/stage0_teacher/teacher_normalized.jsonl"}
OUTPUT_PATH=${OUTPUT_PATH:-"${DATA_ROOT}/stage5_old_logits/teacher_old_logprobs_step200.parquet"}
MODEL_PATH=${MODEL_PATH:-"/Data/wyh/datasets/Verl-Data/outputs/textcraft_sft/qwen3-1.7b-sft/global_step_200/huggingface"}
DEVICE=${DEVICE:-"cuda"}
BATCH_SIZE=${BATCH_SIZE:-64}
PRECOMPUTE_SCRIPT=${PRECOMPUTE_SCRIPT:-"${REPO_ROOT}/examples/sglang_multiturn/my_exp/legacy/prefix_old_logits/precompute_sft_old_logprobs_from_trajectories.py"}
PYTHON_BIN=${PYTHON_BIN:-"/home/wyh/miniconda3/envs/verl/bin/python"}

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES is not set. Set at most 2 GPU ids before running."
  exit 1
fi

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#GPU_IDS[@]} > 2 )); then
  echo "Refusing to run old-logprob job with more than 2 GPUs: ${CUDA_VISIBLE_DEVICES}"
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_PATH}")"

echo "Running old-logprob precompute with GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Input: ${INPUT_PATH}"
echo "Output: ${OUTPUT_PATH}"
echo "Model: ${MODEL_PATH}"

"${PYTHON_BIN}" "${PRECOMPUTE_SCRIPT}" \
  --input_path "${INPUT_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --model_path "${MODEL_PATH}" \
  --device "${DEVICE}" \
  --batch_size "${BATCH_SIZE}"
