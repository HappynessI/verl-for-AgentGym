#!/bin/bash
# GRPO Training script for Webshop on 4xL20
# 使用所有内存优化：FSDP、gradient checkpointing、activation offloading

set -x

# Ensure ulimit is set for Ray
ulimit -n 65535

# Configuration
MODEL_PATH=${MODEL_PATH:-"/Data/public/Qwen3-8B"}
DATA_PATH=${DATA_PATH:-"/Data/wyh/datasets/Verl-Data/train/webshop_small/train.parquet"}
OUTPUT_DIR=${OUTPUT_DIR:-"/Data/wyh/datasets/Verl-Data/outputs/webshop_grpo"}
WEBSHOP_SERVER=${WEBSHOP_SERVER:-"http://127.0.0.1:36003"}

# Training hyperparameters
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}  # Per GPU
LEARNING_RATE=${LEARNING_RATE:-5e-7}
NUM_EPOCHS=${NUM_EPOCHS:-15}
NUM_GPUS=${NUM_GPUS:-4}
ROLLOUT_N=${ROLLOUT_N:-4}  # 每个prompt采样次数

# GPU Configuration
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES="0,1,2,3"  # 默认使用0-3卡
fi
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$OUTPUT_DIR/logs"
LOG_FILE="$LOG_DIR/train_${TIMESTAMP}.log"

# Create directories
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

# Ensure webshop server is running
echo "Checking Webshop server at $WEBSHOP_SERVER..."
if ! curl -s $WEBSHOP_SERVER/ > /dev/null; then
    echo "ERROR: Webshop server is not running!"
    echo "Please start it first:"
    echo "  conda activate webshop"
    echo "  cd /Data/wyh/AgentGym-RL/AgentGym/agentenv-webshop"
    echo "  python -m uvicorn agentenv_webshop:app --host 0.0.0.0 --port 36003"
    exit 1
fi
echo "✓ Webshop server is running"

# Get project directory (verl root)
# 使用绝对路径
CONFIG_PATH="/Data/wyh/verl/examples/sglang_multiturn/config"

echo "================================"
echo "Webshop GRPO Training"
echo "================================"
echo "Model: $MODEL_PATH"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "Log File: $LOG_FILE"
echo "Webshop Server: $WEBSHOP_SERVER"
echo "GPUs (CUDA_VISIBLE_DEVICES): $CUDA_VISIBLE_DEVICES"
echo "Number of GPUs: $NUM_GPUS"
echo "Train Batch Size: $TRAIN_BATCH_SIZE"
echo "Micro Batch Size per GPU: $MICRO_BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Total Epochs: $NUM_EPOCHS"
echo "Rollout N (samples per prompt): $ROLLOUT_N"
echo "================================"
echo ""

# Run training
python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='webshop_grpo_train' \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_PATH \
    data.val_files=$DATA_PATH \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=8 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=25 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=25 \
    actor_rollout_ref.rollout.agent.num_workers=4 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.3 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.total_epochs=$NUM_EPOCHS \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.project_name=webshop_grpo \
    trainer.experiment_name=qwen3-8b_webshop_grpo_${TIMESTAMP} \
    trainer.save_freq=5 \
    trainer.test_freq=3 \
    2>&1 | tee "$LOG_FILE"

# Capture the exit status
PIPELINE_STATUS=${PIPESTATUS[0]}

echo ""
echo "================================"
if [ $PIPELINE_STATUS -eq 0 ]; then
    echo "Training Complete!"
else
    echo "Training Failed with exit code: $PIPELINE_STATUS"
fi
echo "================================"
echo "Results saved to: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo ""

exit $PIPELINE_STATUS

