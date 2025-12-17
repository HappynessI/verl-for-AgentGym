#!/bin/bash
# GRPO Training TEST script for Webshop
# 小规模测试：1 epoch, 少量数据

set -x

# Ensure ulimit is set for Ray
ulimit -n 65535

# Configuration
MODEL_PATH=${MODEL_PATH:-"/Data/public/Qwen3-1.7B"}
DATA_PATH=${DATA_PATH:-"/Data/wyh/datasets/Verl-Data/train/webshop_small/train.parquet"}
OUTPUT_DIR=${OUTPUT_DIR:-"/Data/wyh/datasets/Verl-Data/outputs/webshop_grpo_test"}
WEBSHOP_SERVER=${WEBSHOP_SERVER:-"http://127.0.0.1:36001"}

# Test parameters - 小规模
TRAIN_BATCH_SIZE=8  # 小batch size for quick test
MICRO_BATCH_SIZE=2
LEARNING_RATE=5e-7
NUM_EPOCHS=1  # 只训练1个epoch
NUM_GPUS=4
ROLLOUT_N=4  # 每个prompt只采样2次

# GPU Configuration
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES="4,5,6,7"  # 使用4-7卡避免冲突
fi
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$OUTPUT_DIR/logs"
LOG_FILE="$LOG_DIR/test_${TIMESTAMP}.log"

# Create directories
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

# Ensure webshop server is running
echo "Checking Webshop server at $WEBSHOP_SERVER..."
if ! curl -s $WEBSHOP_SERVER/ > /dev/null; then
    echo "ERROR: Webshop server is not running!"
    exit 1
fi
echo "✓ Webshop server is running"

# Get project directory (verl root)
# 方式1: 使用绝对路径 (推荐)
CONFIG_PATH="/Data/wyh/verl/examples/sglang_multiturn/config"

# 方式2: 从脚本位置计算 (如果需要移植性)
# SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# PROJECT_DIR="$( cd "$SCRIPT_DIR/../../../.." && pwd )"
# CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

echo "================================"
echo "Webshop GRPO Training TEST"
echo "================================"
echo "Model: $MODEL_PATH"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "Log File: $LOG_FILE"
echo "Webshop Server: $WEBSHOP_SERVER"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Train Batch Size: $TRAIN_BATCH_SIZE"
echo "Micro Batch Size: $MICRO_BATCH_SIZE"
echo "Epochs: $NUM_EPOCHS (TEST)"
echo "Rollout N: $ROLLOUT_N (TEST)"
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
    data.val_batch_size=4 \
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
    trainer.project_name=webshop_grpo_test \
    trainer.experiment_name=qwen3-8b_webshop_grpo_test_${TIMESTAMP} \
    trainer.save_freq=10 \
    trainer.test_freq=1 \
    2>&1 | tee "$LOG_FILE"

# Capture the exit status
PIPELINE_STATUS=${PIPESTATUS[0]}

echo ""
echo "================================"
if [ $PIPELINE_STATUS -eq 0 ]; then
    echo "✓ Test Training Complete!"
    echo "Check logs at: $LOG_FILE"
    echo "Check output at: $OUTPUT_DIR"
else
    echo "✗ Test Training Failed with exit code: $PIPELINE_STATUS"
    echo "Check logs at: $LOG_FILE"
fi
echo "================================"

exit $PIPELINE_STATUS

