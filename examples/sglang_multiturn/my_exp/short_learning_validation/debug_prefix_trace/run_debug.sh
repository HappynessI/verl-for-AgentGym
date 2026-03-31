#!/bin/bash
# =============================================================================
# Debug run: trace prefix length / dtype / shape at every pipeline layer
#
# Goals:
#   1. Confirm expand crash on ragged arrays (DEBUG_EXPAND_*)
#   2. Confirm dtype/shape at restore (DEBUG_RESTORE)
#   3. Confirm dtype/shape at ray_trainer:1592 (DEBUG_1592)
#   4. Confirm all 1054-related variables (DEBUG_1054)
#   5. Confirm prefix_token_count availability (DEBUG_PREFIX_TOKEN_COUNT)
#
# Config: GPU0, batch=2, n=1, total_steps=1, tiny model + short seqs
# =============================================================================
set -e

cd /Data/wyh/verl

# GPU selection:
# - If DEBUG_GPU_ID is set, use it.
# - Else respect externally provided CUDA_VISIBLE_DEVICES.
# - Else default to 3 (historical default for this script).
if [[ -n "${DEBUG_GPU_ID:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${DEBUG_GPU_ID}"
else
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
fi
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
LOG_FILE="$LOG_DIR/debug_run_${TIMESTAMP}.log"

echo "================================================================================"
echo "  DEBUG RUN: prefix trace"
echo "  Time: $TIMESTAMP"
echo "  GPU: ${CUDA_VISIBLE_DEVICES}"
echo "  Log: $LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"

source ~/miniconda3/bin/activate verl

echo "Python: $(python --version)" | tee -a "$LOG_FILE"

echo ""
echo "Starting debug run..." | tee -a "$LOG_FILE"

# NOTE: temperature=0.0 requires do_sample=false (not do_sample=true)
python3 -m recipe.wyh_exp.main_train \
    algorithm.adv_estimator=turn_full_trajectory \
    algorithm.optimize_prefix_tokens=true \
    data.train_files=$DATA_PATH \
    data.val_files=$DATA_PATH \
    data.train_batch_size=2 \
    data.val_batch_size=2 \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    "+data.apply_chat_template_kwargs.enable_thinking=true" \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.enable_activation_offload=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=0.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.prompt_length=2048 \
    actor_rollout_ref.rollout.response_length=512 \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.rollout.max_num_seqs=32 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
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
    trainer.project_name=debug_prefix \
    trainer.experiment_name=debug_prefix_trace \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=$?

echo ""
echo "================================================================================" | tee -a "$LOG_FILE"
echo "  Run complete (exit=$EXIT_CODE). Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"

# Extract debug tags
grep -E "DEBUG_EXPAND_BEFORE|DEBUG_EXPAND_AFTER|DEBUG_RESTORE|DEBUG_1592|DEBUG_1054|DEBUG_PREFIX_TOKEN_COUNT" "$LOG_FILE" > "${LOG_FILE%.log}_debug_tags.log" 2>/dev/null || echo "No debug tags found"
echo "Debug tags extracted to: ${LOG_FILE%.log}_debug_tags.log"
