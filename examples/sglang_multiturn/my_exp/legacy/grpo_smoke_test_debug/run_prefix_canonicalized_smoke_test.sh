#!/bin/bash
set -e

# ==================== Smoke Test: Canonicalized Prefix GRPO ====================
# Fixed for distributed initialization timeout issue

# Activate correct conda environment
source /home/wyh/miniconda3/etc/profile.d/conda.sh
conda activate verl

CONFIG_FILE="/Data/wyh/verl/examples/sglang_multiturn/my_exp/grpo_smoke_test_debug/smoke_test_config.yaml"
MODEL_PATH="/Data/public/Qwen3-1.7B"

GPU_ID="2"
NUM_GPUS=1

OUTPUT_DIR="/Data/wyh/datasets/Verl-Data/outputs/textcraft_grpo_prefix_smoke_test_canonicalized"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/train_canonicalized_prefix_smoke_test_$(date +%Y%m%d_%H%M%S).log"

echo "================================================================================"
echo "  Smoke Test: Canonicalized Prefix GRPO (Fixed Init)"
echo "================================================================================"
echo "  Config: ${CONFIG_FILE}"
echo "  Model: ${MODEL_PATH}"
echo "  GPU: ${GPU_ID}"
echo "  Log: ${LOG_FILE}"
echo ""

# Validate parquet
echo "Validating parquet..."
python3 -c "
import pandas as pd
df = pd.read_parquet('/Data/wyh/datasets/Verl-Data/outputs/textcraft_old_logits/active/textcraft_validated_prefix_history_canonicalized_with_prefix_old_logprobs_step200_v2.parquet')
print(f'  Samples: {len(df)}')
print(f'  Columns: {list(df.columns)}')
if 'prefix_token_count' in df.columns:
    ptc = df['prefix_token_count']
    print(f'  prefix_token_count: min={ptc.min()}, max={ptc.max()}, mean={ptc.mean():.1f}')
    print(f'  All ptc > 0: {(ptc > 0).all()}')
if 'assistant_prefix_old_log_probs' in df.columns:
    lens = df['assistant_prefix_old_log_probs'].apply(lambda x: len(x))
    print(f'  old_logprobs len: min={lens.min()}, max={lens.max()}')
print('  ✓ Parquet validation passed')
" || { echo "  ✗ Parquet validation FAILED"; exit 1; }

echo ""
echo "Cleaning Ray..."
ray stop --force 2>/dev/null || true
sleep 5

echo "Starting Smoke Test..."
echo ""

# ===== CRITICAL FIX: Clean up all potentially conflicting environment variables =====
# These are the variables that can cause distributed init to connect to wrong addresses

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
export MASTER_PORT="29501"
export NCCL_SOCKET_IFNAME="lo"
export GLOO_SOCKET_IFNAME="lo"

# Disable CUDA graphs to avoid issues with forking
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export TOKENIZERS_PARALLELISM=false

# Ray specific
export RAY_temp_dir="/tmp/ray_$$"
mkdir -p "$RAY_temp_dir"

# Disable NCCL memory stats to avoid issues
export NCCL_CUMEM_ENABLE=0
export NCCL_IB_TIMEOUT=22
export NCCL_DEBUG=WARN

# Disable vLLM optimizations that can cause issues
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Ray init
export RAY_DEDUP_LOGS=0

# Print environment for debugging
echo "=== Environment Before Launch ==="
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
echo "GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME}"
echo ""

# Launch with torchrun
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=127.0.0.1:29501 \
    verl/trainer/main_ppo.py \
    --config-path="$(dirname ${CONFIG_FILE})" \
    --config-name="$(basename ${CONFIG_FILE} .yaml)" \
    2>&1 | tee "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}

# Cleanup
rm -rf "$RAY_temp_dir" 2>/dev/null || true

echo ""
echo "================================================================================"
echo "  Smoke Test Finished (Exit Code: ${EXIT_CODE})"
echo "  Log: ${LOG_FILE}"
echo "================================================================================"

exit ${EXIT_CODE}
