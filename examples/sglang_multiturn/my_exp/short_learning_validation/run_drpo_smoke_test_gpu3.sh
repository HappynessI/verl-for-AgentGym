#!/bin/bash
# ============================================================
# DRPO Smoke Test Run for TextCraft
# ============================================================
#
# Validates the end-to-end DRPO training pipeline on TextCraft:
#   1. Parquet data loading (train.parquet)
#   2. Multi-turn rollout with TextCraft environment interaction
#   3. DRPO loss computation (compute_policy_loss_drpo in core_algos.py)
#   4. uid / seq_level_rewards grouping
#   5. KL constraint (delta/beta) and length-weighting (tau/Lambda)
#
# Key settings:
#   - 1 GPU (GPU3)
#   - batch=1, micro=1, response=128
#   - n=2 rollouts per prompt (enough for DRPO grouping)
#   - val_before_train=false, no validation during run
#
# Expected success markers in log:
#   - "MICROBATCH STEP"             (actor update micro-batch executes)
#   - "actor/pg_loss"               (DRPO pg_loss computed)
#   - "actor/ppo_kl"                (KL penalty computed)
#   - "actor/kl_loss"               (ref KL loss computed)
#   - "Training Progress"           (epoch loop running)
# ============================================================

# NOTE: No `set -e` — we need to capture exit codes even when Python crashes.
set -u

cd /Data/wyh/verl

# GPU selection: GPU1 (index 1) — lowest utilization card for DRPO smoke test
export CUDA_VISIBLE_DEVICES=1
# Raise Ray memory threshold so we don't get killed prematurely
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
CONFIG_FILE="/Data/wyh/verl/examples/sglang_multiturn/my_exp/short_learning_validation/drpo_smoke_test_config.yaml"
OUTPUT_DIR="/Data/wyh/datasets/Verl-Data/outputs/textcraft_drpo_smoke_test_gpu1"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/drpo_smoke_gpu1_${TIMESTAMP}.log"

echo "================================================================================"
echo "  DRPO SMOKE TEST: TextCraft on GPU1"
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
df = pd.read_parquet('/Data/wyh/datasets/Verl-Data/train/textcraft/train.parquet')
print(f'  Samples: {len(df)}')
print(f'  Columns: {list(df.columns)}')
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
echo "Starting DRPO smoke test run..." | tee -a "$LOG_FILE"

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
export MASTER_PORT="29506"
export NCCL_SOCKET_IFNAME="lo"
export GLOO_SOCKET_IFNAME="lo"
export TOKENIZERS_PARALLELISM=false

# Ray temp dir
export RAY_temp_dir="/tmp/ray_drpo_$$"
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

# ===== Launch DRPO smoke test =====
cd /Data/wyh/verl
python3 -m verl.trainer.main_ppo \
    --config-path="$(dirname ${CONFIG_FILE})" \
    --config-name="$(basename ${CONFIG_FILE} .yaml)" \
    2>&1 | tee -a "$LOG_FILE"

# Capture exit code
EXIT_CODE=${PIPESTATUS[0]}

# Cleanup
rm -rf "$RAY_temp_dir" 2>/dev/null || true

echo "" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "  Run complete (exit=$EXIT_CODE). Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"

# Extract critical DRPO-related debug tags
echo ""
echo "=== Extracting DRPO-specific debug tags ===" | tee -a "$LOG_FILE"
python3 - <<'PY' "$LOG_FILE" 2>&1 | tee -a "$LOG_FILE"
import sys, re
log_path = sys.argv[1]
# DRPO-specific tags
tags = [
    # DRPO loss computation
    "MICROBATCH STEP",
    "actor/pg_loss",
    "actor/ppo_kl",
    "actor/kl_loss",
    "actor/entropy_loss",
    "actor/grad_norm",
    # DRPO algorithm
    "DRPO",
    "compute_policy_loss_drpo",
    # Data flow
    "seq_level_rewards",
    "uid",
    "n_rollouts",
    # Multi-turn rollout
    "generate_sequences",
    "TextCraftInteraction",
    "multi_turn",
    # Environment
    "textcraft",
    "env_server_base",
    # KL / ref model
    "compute_ref_log_prob",
    "use_reference_policy",
    "Skipping ref",
    # Errors / OOMs
    "OutOfMemoryError",
    "ActorUnavailableError",
    "RayTaskError",
    "CUDA error",
    # Completion
    "Run complete",
    "Training Progress",
    "epoch",
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
        for l in lines[:15]:
            print(l)
        if len(lines) > 15:
            print(f"  ... ({len(lines)} total matches, showing first 15)")
PY

echo ""
echo "Debug tag extraction done."
echo "Final exit code: $EXIT_CODE"

# Properly propagate exit code
exit $EXIT_CODE
