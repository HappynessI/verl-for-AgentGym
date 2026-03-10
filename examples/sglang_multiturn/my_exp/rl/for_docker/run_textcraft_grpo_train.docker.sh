set -e

# ============================================================
# TextCraft GRPO 训练脚本 (Docker 容器内版本)
#
# 说明：此脚本用于 docker 容器内部
# 容器内路径映射：
#   - /workspace/verl   <-- VERL_CODE_DIR (宿主机路径在 .env 中配置)
#   - /workspace/datasets <-- DATASETS_DIR
#   - /workspace/models  <-- MODELS_DIR
#   - /workspace/outputs <-- OUTPUTS_DIR
# ============================================================

# ==================== 配置参数 ====================

# -------------------- 模型和数据 (使用容器内路径) --------------------
MODEL_PATH=${MODEL_PATH:-"/workspace/models/Qwen3-1.7B"}
DATA_PATH=${DATA_PATH:-"/workspace/datasets/Verl-Data/train/textcraft/train.parquet"}
OUTPUT_DIR=${OUTPUT_DIR:-"/workspace/outputs/textcraft_grpo"}

# -------------------- GPU配置 --------------------
GPU_IDS=${GPU_IDS:-"0,1,2,3"}  # 默认使用4卡
NUM_GPUS=${NUM_GPUS:-4}

# -------------------- 训练超参数 --------------------
NUM_EPOCHS=${NUM_EPOCHS:-100}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}
LEARNING_RATE=${LEARNING_RATE:-5e-6}
SAVE_FREQ=${SAVE_FREQ:-50}
TEST_FREQ=${TEST_FREQ:-10}

# -------------------- Rollout Correction（TIS / MIS）--------------------
ROLLOUT_IS=${ROLLOUT_IS:-"none"}
ROLLOUT_IS_THRESHOLD=${ROLLOUT_IS_THRESHOLD:-2.0}
ROLLOUT_RS=${ROLLOUT_RS:-"none"}
ROLLOUT_RS_THRESHOLD=${ROLLOUT_RS_THRESHOLD:-2.0}
ROLLOUT_RS_THRESHOLD_LOWER=${ROLLOUT_RS_THRESHOLD_LOWER:-0.2}

# -------------------- vLLM Rollout配置 --------------------
ROLLOUT_N=${ROLLOUT_N:-8}
TEMPERATURE=${TEMPERATURE:-0.8}
TOP_P=${TOP_P:-0.95}
GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL:-0.7}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
ENFORCE_EAGER=${ENFORCE_EAGER:-false}
FREE_CACHE_ENGINE=${FREE_CACHE_ENGINE:-true}
CALCULATE_LOG_PROBS=${CALCULATE_LOG_PROBS:-true}

# -------------------- vLLM Validation配置 --------------------
VAL_TEMPERATURE=${VAL_TEMPERATURE:-1.0}
VAL_TOP_P=${VAL_TOP_P:-1.0}
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-false}
VAL_N=${VAL_N:-1}

# -------------------- Token长度限制 --------------------
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
ROLLOUT_PROMPT_LENGTH=${ROLLOUT_PROMPT_LENGTH:-4096}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
PPO_MAX_TOKEN_LEN=${PPO_MAX_TOKEN_LEN:-8192}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}

# -------------------- 环境服务器 --------------------
TEXTCRAFT_SERVER=${TEXTCRAFT_SERVER:-"http://127.0.0.1:36001"}

# 实验名称
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME="textcraft_grpo_${TIMESTAMP}"

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
echo "  TextCraft GRPO训练 - Docker 容器内版本" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "【模型和数据】" | tee -a "$LOG_FILE"
echo "  模型路径: $MODEL_PATH" | tee -a "$LOG_FILE"
echo "  训练数据: $DATA_PATH" | tee -a "$LOG_FILE"
echo "  输出目录: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "【GPU配置】" | tee -a "$LOG_FILE"
echo "  GPU IDs: $GPU_IDS" | tee -a "$LOG_FILE"
echo "  GPU数量: $NUM_GPUS" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "【训练超参数】" | tee -a "$LOG_FILE"
echo "  Epochs: $NUM_EPOCHS" | tee -a "$LOG_FILE"
echo "  全局Batch Size: $TRAIN_BATCH_SIZE" | tee -a "$LOG_FILE"
echo "  每GPU Micro Batch: $MICRO_BATCH_SIZE" | tee -a "$LOG_FILE"
echo "  Learning Rate: $LEARNING_RATE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "【vLLM配置】" | tee -a "$LOG_FILE"
echo "  采样数量 (N): $ROLLOUT_N" | tee -a "$LOG_FILE"
echo "  Temperature: $TEMPERATURE" | tee -a "$LOG_FILE"
echo "  GPU内存利用率: $GPU_MEMORY_UTIL" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "【环境服务器】" | tee -a "$LOG_FILE"
echo "  TextCraft Server: $TEXTCRAFT_SERVER" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "【实验信息】" | tee -a "$LOG_FILE"
echo "  实验名称: $EXPERIMENT_NAME" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ==================== 检查环境 ====================
echo "检查TextCraft服务器..." | tee -a "$LOG_FILE"
SERVER_RESPONSE=$(curl -sf "$TEXTCRAFT_SERVER/" 2>&1)
if [[ "$SERVER_RESPONSE" == *"TextCraft"* ]]; then
    echo "✓ TextCraft服务器正常运行" | tee -a "$LOG_FILE"
else
    echo "警告: TextCraft服务器 ($TEXTCRAFT_SERVER) 未运行！" | tee -a "$LOG_FILE"
    echo "请确认 docker-compose 中 textcraft-server 服务已启动" | tee -a "$LOG_FILE"
    exit 1
fi
echo "" | tee -a "$LOG_FILE"

# ==================== 切换到工作目录 (容器内路径) ====================
cd /workspace/verl
echo "当前工作目录: $(pwd)" | tee -a "$LOG_FILE"
echo "verl 包位置: $(python -c 'import verl; print(verl.__file__)')" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ==================== 启动训练 ====================
echo "开始GRPO训练..." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

export RAY_DEDUP_LOGS=0
export CUDA_VISIBLE_DEVICES=$GPU_IDS

export VLLM_LOGGING_LEVEL=WARNING
export VLLM_CONFIGURE_LOGGING=0
export PYTHONWARNINGS=ignore
export RAY_DEDUP_LOGS=1

# Rollout Correction 参数
ROLLOUT_CORR_ARGS=""
if [ "$ROLLOUT_IS" != "none" ] || [ "$ROLLOUT_RS" != "none" ]; then
    ROLLOUT_CORR_ARGS="algorithm.rollout_correction.bypass_mode=true \
    algorithm.rollout_correction.use_policy_gradient=true"
    if [ "$ROLLOUT_IS" != "none" ]; then
        ROLLOUT_CORR_ARGS="$ROLLOUT_CORR_ARGS \
    algorithm.rollout_correction.rollout_is=$ROLLOUT_IS \
    algorithm.rollout_correction.rollout_is_threshold=$ROLLOUT_IS_THRESHOLD"
    fi
    if [ "$ROLLOUT_RS" != "none" ]; then
        ROLLOUT_CORR_ARGS="$ROLLOUT_CORR_ARGS \
    algorithm.rollout_correction.rollout_rs=$ROLLOUT_RS \
    algorithm.rollout_correction.rollout_rs_threshold=$ROLLOUT_RS_THRESHOLD \
    algorithm.rollout_correction.rollout_rs_threshold_lower=$ROLLOUT_RS_THRESHOLD_LOWER"
    fi
fi

python3 -m verl.trainer.main_ppo \
    --config-path='/workspace/verl/examples/sglang_multiturn/config' \
    --config-name='textcraft_grpo_train.docker' \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_PATH \
    data.val_files=$DATA_PATH \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=4 \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    '+data.apply_chat_template_kwargs.enable_thinking=True' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.top_p=$TOP_P \
    actor_rollout_ref.rollout.prompt_length=$ROLLOUT_PROMPT_LENGTH \
    actor_rollout_ref.rollout.response_length=$MAX_RESPONSE_LENGTH \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
    actor_rollout_ref.rollout.max_num_seqs=$MAX_NUM_SEQS \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTIL \
    actor_rollout_ref.rollout.enforce_eager=$ENFORCE_EAGER \
    actor_rollout_ref.rollout.free_cache_engine=$FREE_CACHE_ENGINE \
    actor_rollout_ref.rollout.calculate_log_probs=$CALCULATE_LOG_PROBS \
    actor_rollout_ref.rollout.val_kwargs.temperature=$VAL_TEMPERATURE \
    actor_rollout_ref.rollout.val_kwargs.top_p=$VAL_TOP_P \
    actor_rollout_ref.rollout.val_kwargs.do_sample=$VAL_DO_SAMPLE \
    actor_rollout_ref.rollout.val_kwargs.n=$VAL_N \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.total_epochs=$NUM_EPOCHS \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.project_name=textcraft_grpo \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.resume_mode=disable \
    $ROLLOUT_CORR_ARGS \
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
echo "训练完成！" | tee -a "$LOG_FILE"
echo "日志文件: $LOG_FILE" | tee -a "$LOG_FILE"
echo "检查点目录: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
