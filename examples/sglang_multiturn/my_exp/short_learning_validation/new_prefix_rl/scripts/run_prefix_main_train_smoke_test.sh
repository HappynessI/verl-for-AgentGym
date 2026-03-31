#!/bin/bash
# Minimal training-level smoke test for the prefix-main GRPO experiment.

set -u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)
ROOT="${REPO_ROOT}/data/textcraft/new_prefix_rl"
PYTHON_BIN="/home/wyh/miniconda3/envs/verl/bin/python"
RUN_MAIN_SCRIPT="${REPO_ROOT}/examples/sglang_multiturn/my_exp/rl/run_textcraft_grpo_validated.sh"

MODEL_PATH="${MODEL_PATH:-/Data/public/Qwen3-1.7B}"
SOURCE_DATA_PATH="${SOURCE_DATA_PATH:-$ROOT/stage7_audit_release/textcraft_prefix_main_train_step200.audited.parquet}"
SMOKE_DATA_PATH="${SMOKE_DATA_PATH:-$ROOT/stage7_audit_release/textcraft_prefix_main_train_step200.smoke_train.parquet}"
TEXTCRAFT_SERVER="${TEXTCRAFT_SERVER:-http://127.0.0.1:36001}"

GPU_IDS="${GPU_IDS:-3}"
NUM_GPUS="${NUM_GPUS:-1}"

SMOKE_MAX_SAMPLES="${SMOKE_MAX_SAMPLES:-16}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
SAVE_FREQ="${SAVE_FREQ:-999}"
TEST_FREQ="${TEST_FREQ:--1}"
ROLLOUT_N="${ROLLOUT_N:-2}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.40}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-256}"
ROLLOUT_RESPONSE_LENGTH="${ROLLOUT_RESPONSE_LENGTH:-256}"
ROLLOUT_MAX_TOKENS="${ROLLOUT_MAX_TOKENS:-128}"
ROLLOUT_PROMPT_LENGTH="${ROLLOUT_PROMPT_LENGTH:-2048}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
PPO_MAX_TOKEN_LEN="${PPO_MAX_TOKEN_LEN:-4096}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-12}"
MAX_USER_TURNS="${MAX_USER_TURNS:-12}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-8}"
PREFIX_LOSS_WEIGHT="${PREFIX_LOSS_WEIGHT:-1.0}"
OPTIMIZE_PREFIX_TOKENS="${OPTIMIZE_PREFIX_TOKENS:-true}"
USE_KL_LOSS="${USE_KL_LOSS:-false}"
ENABLE_GRADIENT_CHECKPOINTING="${ENABLE_GRADIENT_CHECKPOINTING:-true}"
ENABLE_ACTIVATION_OFFLOAD="${ENABLE_ACTIVATION_OFFLOAD:-false}"
METRICS_CSV_FREQ="${METRICS_CSV_FREQ:-1}"
METRICS_CSV_FILENAME="${METRICS_CSV_FILENAME:-smoke_training_metrics.csv}"

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/prefix_main_train_smoke_test}"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/prefix_main_train_smoke_${TIMESTAMP}.log"

echo "================================================================================"
echo "  Prefix Main Training Smoke Test"
echo "  Time: $TIMESTAMP"
echo "  GPU_IDS: $GPU_IDS"
echo "  Source data: $SOURCE_DATA_PATH"
echo "  Smoke data: $SMOKE_DATA_PATH"
echo "  Log: $LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"

if [ ! -f "$SOURCE_DATA_PATH" ]; then
    echo "ERROR: Source parquet missing: $SOURCE_DATA_PATH" | tee -a "$LOG_FILE"
    exit 1
fi

echo "" | tee -a "$LOG_FILE"
echo "Python: $($PYTHON_BIN --version 2>&1)" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "Building smoke subset..." | tee -a "$LOG_FILE"
$PYTHON_BIN "$ROOT/scripts/13_build_train_smoke_subset.py" \
    --input-path "$SOURCE_DATA_PATH" \
    --output-path "$SMOKE_DATA_PATH" \
    --max-samples "$SMOKE_MAX_SAMPLES" \
    2>&1 | tee -a "$LOG_FILE"
SUBSET_EXIT=${PIPESTATUS[0]}
if [ $SUBSET_EXIT -ne 0 ]; then
    echo "ERROR: Smoke subset build failed (exit=$SUBSET_EXIT)" | tee -a "$LOG_FILE"
    exit $SUBSET_EXIT
fi

echo "" | tee -a "$LOG_FILE"
echo "Validating smoke parquet..." | tee -a "$LOG_FILE"
$PYTHON_BIN - <<'PY' "$SMOKE_DATA_PATH" 2>&1 | tee -a "$LOG_FILE"
import json
import sys
import pandas as pd

path = sys.argv[1]
df = pd.read_parquet(path)
required = [
    "sample_uid",
    "prompt",
    "extra_info",
    "prefix_actions",
    "assistant_prefix_old_log_probs",
    "prefix_mask",
    "prefix_token_count",
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")
if len(df) == 0:
    raise ValueError("Smoke parquet is empty")
print(json.dumps(
    {
        "rows": len(df),
        "unique_sample_uid": int(df["sample_uid"].nunique()),
        "min_prefix_token_count": int(df["prefix_token_count"].min()),
        "max_prefix_token_count": int(df["prefix_token_count"].max()),
    },
    indent=2,
    ensure_ascii=False,
))
PY
PARQUET_EXIT=${PIPESTATUS[0]}
if [ $PARQUET_EXIT -ne 0 ]; then
    echo "ERROR: Smoke parquet validation failed (exit=$PARQUET_EXIT)" | tee -a "$LOG_FILE"
    exit $PARQUET_EXIT
fi

echo "" | tee -a "$LOG_FILE"
echo "Cleaning Ray..." | tee -a "$LOG_FILE"
ray stop --force 2>/dev/null || true
sleep 3

echo "" | tee -a "$LOG_FILE"
echo "Checking TextCraft server..." | tee -a "$LOG_FILE"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$TEXTCRAFT_SERVER/" 2>/dev/null || echo "000")
if [ "$HTTP_CODE" != "200" ]; then
    echo "ERROR: TextCraft server unavailable (HTTP $HTTP_CODE): $TEXTCRAFT_SERVER" | tee -a "$LOG_FILE"
    exit 1
fi
echo "TextCraft server OK (HTTP $HTTP_CODE)" | tee -a "$LOG_FILE"

unset MASTER_ADDR
unset MASTER_PORT
unset WORLD_SIZE
unset RANK
unset LOCAL_RANK
unset LOCAL_WORLD_SIZE
unset TORCHELASTIC_RUN_ID
unset TORCHELASTIC_RESTART_COUNT
unset TORCHELASTIC_MAX_RESTARTS
unset RAY_ADDRESS
unset RAY_JOB_ID
unset RAY_NODE_ID
unset RAY_WORKER_CLASS

export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="${MASTER_PORT:-29517}"
export NCCL_SOCKET_IFNAME="lo"
export GLOO_SOCKET_IFNAME="lo"
export TOKENIZERS_PARALLELISM=false
export RAY_temp_dir="/tmp/ray_prefix_main_smoke_$$"
mkdir -p "$RAY_temp_dir"
export RAY_memory_usage_threshold=0.99
export RAY_memory_monitor_refresh_ms=0
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_CONFIGURE_LOGGING=0
export PYTHONWARNINGS=ignore
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export NCCL_CUMEM_ENABLE=0
export NCCL_IB_TIMEOUT=22
export NCCL_DEBUG=WARN
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_DISABLE_CUDA_GRAPH=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

echo "" | tee -a "$LOG_FILE"
echo "=== Smoke Config ===" | tee -a "$LOG_FILE"
echo "MODEL_PATH=$MODEL_PATH" | tee -a "$LOG_FILE"
echo "SMOKE_MAX_SAMPLES=$SMOKE_MAX_SAMPLES" | tee -a "$LOG_FILE"
echo "NUM_EPOCHS=$NUM_EPOCHS" | tee -a "$LOG_FILE"
echo "TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE" | tee -a "$LOG_FILE"
echo "PPO_MINI_BATCH_SIZE=$PPO_MINI_BATCH_SIZE" | tee -a "$LOG_FILE"
echo "MICRO_BATCH_SIZE=$MICRO_BATCH_SIZE" | tee -a "$LOG_FILE"
echo "ROLLOUT_N=$ROLLOUT_N" | tee -a "$LOG_FILE"
echo "MAX_RESPONSE_LENGTH=$MAX_RESPONSE_LENGTH" | tee -a "$LOG_FILE"
echo "MAX_MODEL_LEN=$MAX_MODEL_LEN" | tee -a "$LOG_FILE"
echo "GPU_MEMORY_UTIL=$GPU_MEMORY_UTIL" | tee -a "$LOG_FILE"
echo "OPTIMIZE_PREFIX_TOKENS=$OPTIMIZE_PREFIX_TOKENS" | tee -a "$LOG_FILE"
echo "USE_KL_LOSS=$USE_KL_LOSS" | tee -a "$LOG_FILE"
echo "ENABLE_GRADIENT_CHECKPOINTING=$ENABLE_GRADIENT_CHECKPOINTING" | tee -a "$LOG_FILE"
echo "ENABLE_ACTIVATION_OFFLOAD=$ENABLE_ACTIVATION_OFFLOAD" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "Starting prefix-main training smoke test..." | tee -a "$LOG_FILE"

cd "$REPO_ROOT"
MODEL_PATH="$MODEL_PATH" \
DATA_PATH="$SMOKE_DATA_PATH" \
OUTPUT_DIR="$OUTPUT_DIR" \
TEXTCRAFT_SERVER="$TEXTCRAFT_SERVER" \
GPU_IDS="$GPU_IDS" \
NUM_GPUS="$NUM_GPUS" \
NUM_EPOCHS="$NUM_EPOCHS" \
TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
PPO_MINI_BATCH_SIZE="$PPO_MINI_BATCH_SIZE" \
PPO_EPOCHS="$PPO_EPOCHS" \
MICRO_BATCH_SIZE="$MICRO_BATCH_SIZE" \
LEARNING_RATE="$LEARNING_RATE" \
SAVE_FREQ="$SAVE_FREQ" \
TEST_FREQ="$TEST_FREQ" \
ROLLOUT_N="$ROLLOUT_N" \
TEMPERATURE="$TEMPERATURE" \
TOP_P="$TOP_P" \
GPU_MEMORY_UTIL="$GPU_MEMORY_UTIL" \
MAX_PROMPT_LENGTH="$MAX_PROMPT_LENGTH" \
MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH" \
ROLLOUT_RESPONSE_LENGTH="$ROLLOUT_RESPONSE_LENGTH" \
ROLLOUT_MAX_TOKENS="$ROLLOUT_MAX_TOKENS" \
ROLLOUT_PROMPT_LENGTH="$ROLLOUT_PROMPT_LENGTH" \
MAX_MODEL_LEN="$MAX_MODEL_LEN" \
PPO_MAX_TOKEN_LEN="$PPO_MAX_TOKEN_LEN" \
MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS" \
MAX_NUM_SEQS="$MAX_NUM_SEQS" \
MAX_ASSISTANT_TURNS="$MAX_ASSISTANT_TURNS" \
MAX_USER_TURNS="$MAX_USER_TURNS" \
RAY_NUM_CPUS="$RAY_NUM_CPUS" \
PREFIX_LOSS_WEIGHT="$PREFIX_LOSS_WEIGHT" \
OPTIMIZE_PREFIX_TOKENS="$OPTIMIZE_PREFIX_TOKENS" \
USE_KL_LOSS="$USE_KL_LOSS" \
ENABLE_GRADIENT_CHECKPOINTING="$ENABLE_GRADIENT_CHECKPOINTING" \
ENABLE_ACTIVATION_OFFLOAD="$ENABLE_ACTIVATION_OFFLOAD" \
METRICS_CSV_FREQ="$METRICS_CSV_FREQ" \
METRICS_CSV_FILENAME="$METRICS_CSV_FILENAME" \
bash "$RUN_MAIN_SCRIPT" 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

rm -rf "$RAY_temp_dir" 2>/dev/null || true

echo "" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "  Run complete (exit=$EXIT_CODE). Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "=== Extracting Key Smoke Tags ===" | tee -a "$LOG_FILE"
$PYTHON_BIN - <<'PY' "$LOG_FILE" 2>&1 | tee -a "$LOG_FILE"
import sys

log_path = sys.argv[1]
tags = [
    "数据校验通过",
    "optimize_prefix_tokens: true",
    "[CHECK_2 PASS]",
    "actor/prefix_loss",
    "actor/continuation_loss",
    "actor/prefix_ppo_kl",
    "actor/prefix_loss_weight",
    "Training Progress",
    "global_step",
    "prefix_mask is required",
    "Missing 'assistant_prefix_old_log_probs'",
    "OutOfMemoryError",
    "CUDA out of memory",
    "RayTaskError",
]
found = {tag: [] for tag in tags}
with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        for tag in tags:
            if tag in line:
                found[tag].append(line.rstrip())

for tag, lines in found.items():
    if lines:
        print(f"\n=== {tag} ===")
        for line in lines[:15]:
            print(line)
        if len(lines) > 15:
            print(f"... ({len(lines)} matches, showing first 15)")
PY

exit $EXIT_CODE
