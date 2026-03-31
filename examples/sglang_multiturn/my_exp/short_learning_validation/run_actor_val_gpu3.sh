#!/bin/bash
# =============================================================================
# Lightweight Actor-Validation Run
#
# Goals:
#   1. Verify trainer → actor prefix tensor delivery (MATERIALIZE_SUMMARY)
#   2. Verify actor receives correct prefix tensors (DEBUG_1054)
#   3. Verify actor gather / prefix PPO loss path starts executing
#
# Key fix: Ref model init is now SKIPPED when use_kl_loss=false AND
#          use_kl_in_reward=false, eliminating compute_ref_log_prob OOM.
#
# Config: GPU3, batch=2, micro=1, response=256, gpu_util=0.35
# =============================================================================

# NOTE: No `set -e` here — we need to capture exit codes even when Python crashes.
# Use explicit error checking instead.
set -u

cd /Data/wyh/verl

# GPU selection: GPU7 (index 7) — free, for non-zero reward validation
export CUDA_VISIBLE_DEVICES=7
# Raise Ray memory threshold so we don't get killed prematurely (node has ~485 GB RAM)
export RAY_memory_usage_threshold=0.99
# Disable Ray memory monitor to avoid killing workers due to node RAM pressure
export RAY_memory_monitor_refresh_ms=0
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_CONFIGURE_LOGGING=0
export PYTHONWARNINGS=ignore
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0

# Paths
MODEL_PATH="/Data/public/Qwen3-1.7B"
CONFIG_FILE="/Data/wyh/verl/examples/sglang_multiturn/my_exp/short_learning_validation/actor_val_lightweight_config.yaml"
OUTPUT_DIR="/Data/wyh/datasets/Verl-Data/outputs/textcraft_grpo_actor_val_lightweight"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/actor_val_gpu7_${TIMESTAMP}.log"

echo "================================================================================"
echo "  ACTOR-VALIDATION RUN: GPU7, lightweight, goal-binding fix"
echo "  Time: $TIMESTAMP"
echo "  GPU: ${CUDA_VISIBLE_DEVICES}"
echo "  Config: ${CONFIG_FILE}"
echo "  Log: $LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"

source ~/miniconda3/bin/activate verl

echo "Python: $(python --version)" | tee -a "$LOG_FILE"

# Validate parquet
echo ""
echo "Validating parquet..." | tee -a "$LOG_FILE"
python3 -c "
import pandas as pd
df = pd.read_parquet('/Data/wyh/datasets/Verl-Data/outputs/textcraft_old_logits/active/textcraft_validated_prefix_history_canonicalized_with_prefix_old_logprobs_step200_v2.parquet')
print(f'  Samples: {len(df)}')
print(f'  Columns: {list(df.columns)}')
if 'prefix_token_count' in df.columns:
    ptc = df['prefix_token_count']
    print(f'  prefix_token_count: min={ptc.min()}, max={ptc.max()}, mean={ptc.mean():.1f}')
if 'assistant_prefix_old_log_probs' in df.columns:
    lens = df['assistant_prefix_old_log_probs'].apply(lambda x: len(x))
    print(f'  old_logprobs len: min={lens.min()}, max={lens.max()}')
print('  Parquet OK')
" 2>&1 | tee -a "$LOG_FILE"
PARQUET_OK=$?
if [ $PARQUET_OK -ne 0 ]; then
    echo "ERROR: Parquet validation failed (exit=$PARQUET_OK)" | tee -a "$LOG_FILE"
    exit $PARQUET_OK
fi

# Clean up Ray from previous runs
echo ""
echo "Cleaning Ray..." | tee -a "$LOG_FILE"
ray stop --force 2>/dev/null || true
sleep 3

# Verify TextCraft server
echo ""
echo "Verifying TextCraft server..." | tee -a "$LOG_FILE"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:36001/ 2>/dev/null || echo "000")
if [ "$HTTP_CODE" = "200" ]; then
    echo "  TextCraft server is running (HTTP $HTTP_CODE)" | tee -a "$LOG_FILE"
else
    echo "  WARNING: TextCraft server returned HTTP $HTTP_CODE" | tee -a "$LOG_FILE"
fi

echo ""
echo "Starting actor validation run..." | tee -a "$LOG_FILE"

# ===== Clean environment variables =====
unset MASTER_ADDR
unset MASTER_PORT
unset WORLD_SIZE
unset RANK
unset LOCAL_RANK
unset LOCAL_WORLD_SIZE
unset TORCHELASTIC_RUN_ID
unset TORCHELASTIC_RESTART_COUNT
unset TORCHELASTIC_MAX_RESTARTS
unset NCCL_TIMEOUT
unset NCCL_DEBUG
unset NCCL_IB_TIMEOUT
unset NCCL_SOCKET_IFNAME
unset NCCL_NET
unset GLOO_SOCKET_IFNAME
unset GLOO_TRANSPORT
unset RAY_ADDRESS
unset RAY_JOB_ID
unset RAY_NODE_ID
unset RAY_WORKER_CLASS

# Set explicit loopback for all distributed communication
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="29504"
export NCCL_SOCKET_IFNAME="lo"
export GLOO_SOCKET_IFNAME="lo"
export TOKENIZERS_PARALLELISM=false

# Ray temp dir
export RAY_temp_dir="/tmp/ray_actor_val_$$"
mkdir -p "$RAY_temp_dir"

# Disable NCCL memory stats
export NCCL_CUMEM_ENABLE=0
export NCCL_IB_TIMEOUT=22
export NCCL_DEBUG=WARN

# vLLM
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_DISABLE_CUDA_GRAPH=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

# Print key environment
echo "=== Environment ===" | tee -a "$LOG_FILE"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" | tee -a "$LOG_FILE"
echo "RAY_memory_usage_threshold=${RAY_memory_usage_threshold}" | tee -a "$LOG_FILE"
echo "MASTER_ADDR=${MASTER_ADDR}" | tee -a "$LOG_FILE"
echo "MASTER_PORT=${MASTER_PORT}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ===== Launch with recipe.wyh_exp.main_train =====
# This registers turn_full_trajectory advantage estimator before training starts.
cd /Data/wyh/verl
python3 -m recipe.wyh_exp.main_train \
    --config-path="$(dirname ${CONFIG_FILE})" \
    --config-name="$(basename ${CONFIG_FILE} .yaml)" \
    2>&1 | tee -a "$LOG_FILE"

# Capture exit code — this is the REAL exit code from Python
EXIT_CODE=${PIPESTATUS[0]}

# Cleanup
rm -rf "$RAY_temp_dir" 2>/dev/null || true

echo "" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "  Run complete (exit=$EXIT_CODE). Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"

# Extract critical debug tags
echo ""
echo "=== Extracting debug tags ===" | tee -a "$LOG_FILE"
python3 - <<'PY' "$LOG_FILE" 2>&1 | tee -a "$LOG_FILE"
import sys, re
log_path = sys.argv[1]
tags = [
    "MATERIALIZE_SUMMARY", "DEBUG_1592", "DEBUG_1054", "DEBUG_PREFIX_TOKEN_COUNT",
    "CHECK_1", "CHECK_2", "DEBUG_RESTORE",
    "[REF_POLICY]", "use_reference_policy",
    "OutOfMemoryError", "ActorUnavailableError", "RayTaskError",
    "FSDP_REF.*Skipping ref", "Engine_REF.*Skipping ref",
    "Run complete", "Training Progress",
    "Skipping ref model init",
    "compute_ref_log_prob.*called but",
]
found = {t: [] for t in tags}
try:
    with open(log_path) as f:
        for line in f:
            for tag in tags:
                if tag in line:
                    found[tag].append(line.rstrip())
except Exception as e:
    print(f"Error reading log: {e}")
    import traceback; traceback.print_exc()

for tag, lines in found.items():
    if lines:
        print(f"\n=== {tag} ===")
        for l in lines[:20]:
            print(l)
        if len(lines) > 20:
            print(f"  ... ({len(lines)} total matches, showing first 20)")
PY

echo ""
echo "Debug tag extraction done."
echo "Final exit code: $EXIT_CODE"

# Properly propagate exit code
exit $EXIT_CODE
