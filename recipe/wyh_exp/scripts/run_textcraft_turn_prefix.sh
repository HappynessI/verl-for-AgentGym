#!/bin/bash
# Turn-based Prefix RL 训练脚本 - TextCraft 环境
# 支持两种训练模式：
#   - full_trajectory: 整个轨迹作为小模型自己的轨迹进行 RL
#   - prefix_guided: Prefix 作为参考，只对 rollout 部分进行 RL

set -e

# ============================================================================
# 基础配置
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")")"
RECIPE_DIR="${PROJECT_ROOT}/recipe/wyh_exp"

# 环境变量
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

# ============================================================================
# 训练配置（可通过命令行覆盖）
# ============================================================================
# 模型配置
MODEL_PATH="${MODEL_PATH:-/path/to/your/model}"
MODEL_NAME="${MODEL_NAME:-qwen3-1.7b}"

# 数据配置
TRAIN_DATA="${TRAIN_DATA:-/path/to/train.parquet}"
VAL_DATA="${VAL_DATA:-/path/to/val.parquet}"

# 训练模式: full_trajectory 或 prefix_guided
TRAINING_MODE="${TRAINING_MODE:-full_trajectory}"

# Prefix 配置（仅 prefix_guided 模式）
PREFIX_STRATEGY="${PREFIX_STRATEGY:-random}"
MIN_PREFIX_TURNS="${MIN_PREFIX_TURNS:-1}"
MAX_PREFIX_TURNS="${MAX_PREFIX_TURNS:-5}"

# 训练参数
BATCH_SIZE="${BATCH_SIZE:-32}"
ROLLOUT_N="${ROLLOUT_N:-4}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"

# GPU 配置
N_GPUS="${N_GPUS:-8}"

# 输出目录
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/wyh_exp/${MODEL_NAME}_${TRAINING_MODE}}"

# WandB 配置
WANDB_PROJECT="${WANDB_PROJECT:-wyh_exp}"
WANDB_NAME="${WANDB_NAME:-${MODEL_NAME}_${TRAINING_MODE}}"

# ============================================================================
# 创建输出目录
# ============================================================================
mkdir -p "${OUTPUT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"

# ============================================================================
# 启动训练
# ============================================================================
echo "=========================================="
echo "Starting Turn-based Prefix RL Training"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Training Mode: ${TRAINING_MODE}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Rollout N: ${ROLLOUT_N}"
echo "GPUs: ${N_GPUS}"
echo "=========================================="

python3 -m verl.trainer.main_ppo \
    --config-path="${RECIPE_DIR}/config" \
    --config-name="base_config" \
    \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${VAL_DATA}" \
    data.mode="${TRAINING_MODE}" \
    data.prefix_strategy="${PREFIX_STRATEGY}" \
    data.min_prefix_turns="${MIN_PREFIX_TURNS}" \
    data.max_prefix_turns="${MAX_PREFIX_TURNS}" \
    data.train_batch_size="${BATCH_SIZE}" \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr="${LEARNING_RATE}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    \
    algorithm.adv_estimator="${TRAINING_MODE/prefix_guided/turn_prefix_guided}" \
    \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.project_name="${WANDB_PROJECT}" \
    trainer.experiment_name="${WANDB_NAME}" \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    \
    2>&1 | tee "${OUTPUT_DIR}/train.log"

echo "Training completed! Output saved to: ${OUTPUT_DIR}"

