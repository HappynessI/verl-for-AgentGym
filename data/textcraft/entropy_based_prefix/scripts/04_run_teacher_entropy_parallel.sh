#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

PYTHON_BIN=${PYTHON_BIN:-/home/wyh/miniconda3/envs/verl/bin/python}
INPUT_JSONL=${INPUT_JSONL:-"${ROOT_DIR}/../new_prefix_rl/stage0_teacher/teacher_normalized.jsonl"}
MODEL_PATH=${MODEL_PATH:-"/Data/wyh/datasets/Verl-Data/outputs/textcraft_sft/qwen3-1.7b-sft/global_step_200/huggingface"}
SHARD_DIR=${SHARD_DIR:-"${ROOT_DIR}/stage1_entropy/shards"}
FINAL_OUTPUT=${FINAL_OUTPUT:-"${ROOT_DIR}/stage1_entropy/textcraft_teacher_entropy_step200.parquet"}
MAX_SAMPLES=${MAX_SAMPLES:-}
MAX_BATCH_SAMPLES=${MAX_BATCH_SAMPLES:-8}
MAX_BATCH_TOKENS=${MAX_BATCH_TOKENS:-12288}
BALANCE_BY=${BALANCE_BY:-char_length}
SORT_BY_LENGTH=${SORT_BY_LENGTH:-1}

pick_gpus() {
  "${PYTHON_BIN}" - <<'PY'
import torch

gpu_stats = []
for idx in range(torch.cuda.device_count()):
    with torch.cuda.device(idx):
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    gpu_stats.append((free_bytes / total_bytes, free_bytes, idx))

gpu_stats.sort(reverse=True)
for _, free_bytes, idx in gpu_stats[:2]:
    print(idx)
PY
}

if [[ -z "${GPU0:-}" || -z "${GPU1:-}" ]]; then
  mapfile -t PICKED_GPUS < <(pick_gpus)
  GPU0=${GPU0:-${PICKED_GPUS[0]}}
  GPU1=${GPU1:-${PICKED_GPUS[1]}}
fi

run_entropy_shard() {
  local gpu=$1
  local input_path=$2
  local output_path=$3
  local manifest_path=$4

  CUDA_VISIBLE_DEVICES=${gpu} \
    "${PYTHON_BIN}" - \
    "${SCRIPT_DIR}/02_compute_teacher_entropy.py" \
    --input-path "${input_path}" \
    --output-path "${output_path}" \
    --manifest-path "${manifest_path}" \
    --device cuda \
    "${COMMON_ARGS[@]}" <<'PY'
import runpy
import sys
from pathlib import Path

script_path = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(script_path.parent))
sys.argv = [str(script_path), *sys.argv[2:]]
runpy.run_path(str(script_path), run_name="__main__")
PY
}

mkdir -p "${SHARD_DIR}"

SPLIT_ARGS=(
  --input-path "${INPUT_JSONL}"
  --output-dir "${SHARD_DIR}"
  --num-shards 2
  --balance-by "${BALANCE_BY}"
)
if [[ -n "${MAX_SAMPLES}" ]]; then
  SPLIT_ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

echo "[1/4] Split teacher JSONL into balanced shards"
"${PYTHON_BIN}" "${SCRIPT_DIR}/01_split_teacher_jsonl.py" "${SPLIT_ARGS[@]}"

COMMON_ARGS=(
  --model-path "${MODEL_PATH}"
  --max-batch-samples "${MAX_BATCH_SAMPLES}"
  --max-batch-tokens "${MAX_BATCH_TOKENS}"
)
if [[ "${SORT_BY_LENGTH}" == "1" ]]; then
  COMMON_ARGS+=(--sort-by-length)
fi

echo "[2/4] Run shard0 on GPU ${GPU0}"
run_entropy_shard \
  "${GPU0}" \
  "${SHARD_DIR}/teacher_normalized.shard0.jsonl" \
  "${SHARD_DIR}/textcraft_teacher_entropy_step200.shard0.parquet" \
  "${ROOT_DIR}/manifests/stage1_entropy_shard0_manifest.json" &
PID0=$!

echo "[3/4] Run shard1 on GPU ${GPU1}"
run_entropy_shard \
  "${GPU1}" \
  "${SHARD_DIR}/teacher_normalized.shard1.jsonl" \
  "${SHARD_DIR}/textcraft_teacher_entropy_step200.shard1.parquet" \
  "${ROOT_DIR}/manifests/stage1_entropy_shard1_manifest.json" &
PID1=$!

wait ${PID0}
wait ${PID1}

echo "[4/4] Merge entropy shards"
"${PYTHON_BIN}" "${SCRIPT_DIR}/03_merge_entropy_shards.py" \
  --input-dir "${SHARD_DIR}" \
  --output-path "${FINAL_OUTPUT}" \
  --manifest-path "${ROOT_DIR}/manifests/stage1_entropy_merge_manifest.json"

echo "Done: ${FINAL_OUTPUT}"
