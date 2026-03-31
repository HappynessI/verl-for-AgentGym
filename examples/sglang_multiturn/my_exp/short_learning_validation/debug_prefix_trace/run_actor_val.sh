#!/bin/bash
# =============================================================================
# Lightweight actor-path validation
#
# Goals:
#   1. Verify trainer → actor prefix tensor delivery (MATERIALIZE_SUMMARY)
#   2. Verify actor receives correct prefix tensors (DEBUG_1054)
#   3. Verify actor gather / prefix PPO loss path starts executing
#
# Strategy: Kill ref policy to remove compute_ref_log_prob OOM blocker:
#   - actor.use_kl_loss=false  → ref policy worker NOT spawned
#   - algorithm.use_kl_in_reward=false → ref policy worker NOT spawned
#   → compute_ref_log_prob() is NEVER called → no ref OOM
#
# Config: GPU1, batch=2, micro=1, n=1, steps=1, short seqs
# =============================================================================
set -euo pipefail

cd /Data/wyh/verl

# GPU selection
export CUDA_VISIBLE_DEVICES=0
export RAY_memory_usage_threshold="${RAY_MEMORY_THRESHOLD:-0.998}"
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_CONFIGURE_LOGGING=0
export PYTHONWARNINGS=ignore
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

# Paths
MODEL_PATH="/Data/public/Qwen3-1.7B"
DATA_PATH="/Data/wyh/datasets/Verl-Data/outputs/textcraft_old_logits/active/textcraft_validated_prefix_history_canonicalized_with_prefix_old_logprobs_step200_v2.parquet"
OUTPUT_DIR="/Data/wyh/verl/examples/sglang_multiturn/my_exp/short_learning_validation/debug_prefix_trace"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/actor_val_${TIMESTAMP}.log"

echo "================================================================================"
echo "  ACTOR-VALIDATION RUN: lightweight, ref-policy-killed"
echo "  Time: $TIMESTAMP"
echo "  GPU: ${CUDA_VISIBLE_DEVICES}"
echo "  Log: $LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"

source ~/miniconda3/bin/activate verl

echo "Python: $(python --version)" | tee -a "$LOG_FILE"

echo ""
echo "Starting actor validation run..." | tee -a "$LOG_FILE"

# NOTE: temperature=0.0 requires do_sample=false
python3 -m recipe.wyh_exp.main_train \
    algorithm.adv_estimator=turn_full_trajectory \
    algorithm.optimize_prefix_tokens=true \
    algorithm.use_kl_in_reward=false \
    data.train_files=$DATA_PATH \
    data.val_files=$DATA_PATH \
    data.train_batch_size=2 \
    data.val_batch_size=2 \
    data.max_prompt_length=2048 \
    data.max_response_length=256 \
    "+data.apply_chat_template_kwargs.enable_thinking=true" \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.enable_activation_offload=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=3072 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=0.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.prompt_length=1024 \
    actor_rollout_ref.rollout.response_length=128 \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.max_num_batched_tokens=3072 \
    actor_rollout_ref.rollout.max_num_seqs=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.92 \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.free_cache_engine=true \
    actor_rollout_ref.rollout.multi_turn.enable=false \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=false \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_training_steps=1 \
    trainer.val_before_train=false \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.project_name=actor_val \
    trainer.experiment_name=actor_val_run \
    2>&1 | tee -a "$LOG_FILE"

# Save raw exit code BEFORE any pipe/tee manipulation
PIPE_STATUS=${PIPESTATUS[0]}
EXIT_CODE=${PIPE_STATUS:-$?}

echo ""
echo "================================================================================" | tee -a "$LOG_FILE"
echo "  Run complete (raw_exit=$PIPE_STATUS, reported_exit=$EXIT_CODE). Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"

# Extract critical debug tags
python3 - <<'PY' "$LOG_FILE"
import sys, re
log_path = sys.argv[1]
tags = ["MATERIALIZE_SUMMARY", "DEBUG_1592", "DEBUG_1054", "DEBUG_PREFIX_TOKEN_COUNT",
        "CHECK_1", "CHECK_2", "DEBUG_RESTORE",
        "ref policy NOT spawned", "use_reference_policy",
        "OutOfMemoryError", "ActorUnavailableError", "RayTaskError",
        "Run complete", "Training Progress"]

found = {t: [] for t in tags}
try:
    with open(log_path) as f:
        for line in f:
            for tag in tags:
                if tag in line:
                    found[tag].append(line.rstrip())
except Exception as e:
    print(f"Error reading log: {e}")

for tag, lines in found.items():
    if lines:
        print(f"\n=== {tag} ===")
        for l in lines:
            print(l)
PY

echo "Debug tags extracted. Done."
exit $EXIT_CODE
