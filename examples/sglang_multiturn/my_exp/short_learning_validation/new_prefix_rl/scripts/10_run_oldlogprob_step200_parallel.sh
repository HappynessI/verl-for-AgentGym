#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)
DATA_ROOT="${REPO_ROOT}/data/textcraft/new_prefix_rl"

PYTHON_BIN=${PYTHON_BIN:-/home/wyh/miniconda3/envs/verl/bin/python}
GPU0=${GPU0:-0}
GPU1=${GPU1:-1}
SHARD_DIR=${SHARD_DIR:-"${DATA_ROOT}/stage5_old_logits/shards"}
INPUT_JSONL=${INPUT_JSONL:-"${DATA_ROOT}/stage0_teacher/teacher_normalized.jsonl"}
MODEL_PATH=${MODEL_PATH:-"/Data/wyh/datasets/Verl-Data/outputs/textcraft_sft/qwen3-1.7b-sft/global_step_200/huggingface"}
FINAL_OUTPUT=${FINAL_OUTPUT:-"${DATA_ROOT}/stage5_old_logits/teacher_old_logprobs_step200.parquet"}

mkdir -p "${SHARD_DIR}"

echo "[1/4] Splitting teacher JSONL into 2 shards"
"${PYTHON_BIN}" "${SCRIPT_DIR}/08_split_teacher_jsonl.py" \
  --input-path "${INPUT_JSONL}" \
  --output-dir "${SHARD_DIR}" \
  --num-shards 2

echo "[2/4] Running shard0 on GPU ${GPU0}"
CUDA_VISIBLE_DEVICES=${GPU0} \
  INPUT_PATH="${SHARD_DIR}/teacher_normalized.shard0.jsonl" \
  OUTPUT_PATH="${SHARD_DIR}/teacher_old_logprobs_step200.shard0.parquet" \
  MODEL_PATH="${MODEL_PATH}" \
  BATCH_SIZE=64 \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash "${SCRIPT_DIR}/04_run_oldlogprob_step200.sh" &
PID0=$!

echo "[3/4] Running shard1 on GPU ${GPU1}"
CUDA_VISIBLE_DEVICES=${GPU1} \
  INPUT_PATH="${SHARD_DIR}/teacher_normalized.shard1.jsonl" \
  OUTPUT_PATH="${SHARD_DIR}/teacher_old_logprobs_step200.shard1.parquet" \
  MODEL_PATH="${MODEL_PATH}" \
  BATCH_SIZE=64 \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash "${SCRIPT_DIR}/04_run_oldlogprob_step200.sh" &
PID1=$!

wait ${PID0}
wait ${PID1}

echo "[4/4] Merging shard parquets"
"${PYTHON_BIN}" "${SCRIPT_DIR}/09_merge_oldlogprob_shards.py" \
  --input-dir "${SHARD_DIR}" \
  --output-path "${FINAL_OUTPUT}"

echo "Done: ${FINAL_OUTPUT}"
