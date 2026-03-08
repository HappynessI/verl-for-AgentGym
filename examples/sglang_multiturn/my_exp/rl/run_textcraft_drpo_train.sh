#!/bin/bash
set -x

# TextCraft DRPO training script - 移植到verl官方目录
# 基于 /Data/wyh/DRPO/scripts/train/run_drpo_qwen3_1.7b_textcraft.sh

# ==================== 配置参数 ====================

# 模型路径
MODEL_PATH="/Data/public/Qwen3-1.7B"

# 数据路径
DATA_PATH="/Data/wyh/datasets/Verl-Data/train/textcraft/train.parquet"

# 输出目录
OUTPUT_DIR=${OUTPUT_DIR:-"/Data/wyh/datasets/Verl-Data/outputs/textcraft_drpo"}

# GPU配置 (8卡144GB)
GPU_IDS=${GPU_IDS:-"0,1,2,3,4,5,6,7"}
NUM_GPUS=${NUM_GPUS:-8}

# 训练超参数 (适配8卡144GB)
NUM_EPOCHS=${NUM_EPOCHS:-200}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}    # 144GB可设更大
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}      # 144GB可增大
LEARNING_RATE=${LEARNING_RATE:-2e-6}
PPO_MAX_TOKEN_LEN=${PPO_MAX_TOKEN_LEN:-36864}  # 144GB可用更大
SAVE_FREQ=${SAVE_FREQ:-100}                   # 每100个epoch保存
TEST_FREQ=${TEST_FREQ:-20}

# DRPO超参数 (参考DRPO论文)
DELTA=${DELTA:-1e-4}
BETA=${BETA:-1e3}
TAU=${TAU:-10}
LAMBDA=${LAMBDA:-0.1}
PPO_KL_TYPE=${PPO_KL_TYPE:-"kl"}

# vLLM配置 (适配144GB)
ROLLOUT_N=${ROLLOUT_N:-8}                   # 144GB可增大
TEMPERATURE=${TEMPERATURE:-0.6}
GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL:-0.85}      # 144GB可用更高
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1024}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}

# 实验名称
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME="drpo-qwen3-1.7b-textcraft-${TIMESTAMP}"

# 日志目录
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_${TIMESTAMP}.log"

# ==================== 验证GPU配置 ====================
GPU_COUNT=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
if [ "$GPU_COUNT" -ne "$NUM_GPUS" ]; then
    echo "错误: GPU_IDS中的GPU数量($GPU_COUNT)与NUM_GPUS($NUM_GPUS)不一致！"
    exit 1
fi

# ==================== 打印配置 ====================
echo "================================================================================" | tee "$LOG_FILE"
echo "  TextCraft DRPO训练 - Qwen3-1.7B (verl版本)" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "  模型路径: $MODEL_PATH" | tee -a "$LOG_FILE"
echo "  数据路径: $DATA_PATH" | tee -a "$LOG_FILE"
echo "  GPU IDs: $GPU_IDS" | tee -a "$LOG_FILE"
echo "  GPU数量: $NUM_GPUS" | tee -a "$LOG_FILE"
echo "  Epochs: $NUM_EPOCHS" | tee -a "$LOG_FILE"
echo "  Batch Size: $TRAIN_BATCH_SIZE" | tee -a "$LOG_FILE"
echo "  Learning Rate: $LEARNING_RATE" | tee -a "$LOG_FILE"
echo "  DRPO: delta=$DELTA, beta=$BETA, tau=$TAU, Lambda=$LAMBDA" | tee -a "$LOG_FILE"
echo "  vLLM: n=$ROLLOUT_N, temp=$TEMPERATURE, gpu_mem=$GPU_MEMORY_UTIL" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 切换到工作目录
cd /Data/wyh/verl
source ~/miniconda3/bin/activate verl

# Ray配置
export RAY_DEDUP_LOGS=0
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# 日志级别控制
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_CONFIGURE_LOGGING=0
export PYTHONWARNINGS=ignore
export RAY_DEDUP_LOGS=1

# ==================== 启动训练 ====================
python3 -m verl.trainer.main_ppo \
    --config-path='/Data/wyh/verl/examples/sglang_multiturn/config' \
    --config-name='textcraft_grpo_train' \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_PATH \
    data.val_files=$DATA_PATH \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=4 \
    data.max_prompt_length=2048 \
    data.max_response_length=16384 \
    '+data.apply_chat_template_kwargs.enable_thinking=True' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    '+actor_rollout_ref.actor.use_max_seq_len=True' \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN \
    actor_rollout_ref.actor.ppo_epochs=1 \
    '+actor_rollout_ref.ref.enable=True' \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    '+actor_rollout_ref.actor.ppo_kl_type=kl' \
    actor_rollout_ref.actor.delta=$DELTA \
    actor_rollout_ref.actor.beta=$BETA \
    actor_rollout_ref.actor.tau=$TAU \
    actor_rollout_ref.actor.Lambda=$LAMBDA \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    '++actor_rollout_ref.actor.policy_loss.loss_type=drpo' \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    '++actor_rollout_ref.actor.fsdp_config.param_offload=True' \
    '++actor_rollout_ref.actor.fsdp_config.grad_offload=False' \
    '++actor_rollout_ref.actor.fsdp_config.optimizer_offload=False' \
    '++actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16' \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTIL \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
    actor_rollout_ref.rollout.max_num_seqs=$MAX_NUM_SEQS \
    actor_rollout_ref.rollout.prompt_length=4096 \
    actor_rollout_ref.rollout.response_length=16384 \
    actor_rollout_ref.rollout.max_model_len=20480 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    '++actor_rollout_ref.ref.fsdp_config.param_offload=True' \
    '++actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16' \
    trainer.critic_warmup=0 \
    'trainer.logger=[console,wandb]' \
    trainer.project_name=textcraft_drpo \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.balance_batch=False \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.total_training_steps=null \
    trainer.total_epochs=$NUM_EPOCHS \
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
echo "训练完成！" | tee -a "$LOG_FILE"
echo "日志文件: $LOG_FILE" | tee -a "$LOG_FILE"
echo "检查点目录: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
