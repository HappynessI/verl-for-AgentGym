set -e

# ==================== 配置参数 ====================

# -------------------- 模型和数据 --------------------
MODEL_PATH=${MODEL_PATH:-"/Data/public/Qwen3-1.7B"}
DATA_PATH=${DATA_PATH:-"/Data/wyh/datasets/Verl-Data/train/textcraft/train.parquet"}
OUTPUT_DIR=${OUTPUT_DIR:-"/Data/wyh/datasets/Verl-Data/outputs/textcraft_grpo_8x144gb"}

# -------------------- GPU配置 (8卡144GB) --------------------
GPU_IDS=${GPU_IDS:-"0,1,2,3,4,5,6,7"}  # 8卡144GB
NUM_GPUS=${NUM_GPUS:-8}

# -------------------- 训练超参数 (适配8卡144GB) --------------------
NUM_EPOCHS=${NUM_EPOCHS:-200}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}    # 全局batch size（8卡144GB可设更大）
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}      # 每张GPU的micro batch
LEARNING_RATE=${LEARNING_RATE:-5e-6}
SAVE_FREQ=${SAVE_FREQ:-100}                   # 每N个epoch保存checkpoint
TEST_FREQ=${TEST_FREQ:-50}                   # 每N个epoch进行validation
PPO_MAX_TOKEN_LEN=${PPO_MAX_TOKEN_LEN:-24576}

# -------------------- GRPO 配置 --------------------
CLIP_RATIO=${CLIP_RATIO:-0.2}                # GRPO clip ratio

# -------------------- vLLM Rollout配置（训练采样）--------------------
ROLLOUT_N=${ROLLOUT_N:-8}                   # 每个prompt采样数量（8卡可用更大）
TEMPERATURE=${TEMPERATURE:-0.8}
TOP_P=${TOP_P:-0.95}
GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL:-0.8}     # 144GB显存可用更高利用率
MAX_NUM_SEQS=${MAX_NUM_SEQS:-128}           # 更大并发
ENFORCE_EAGER=${ENFORCE_EAGER:-false}       # 144GB可用PagedAttention
FREE_CACHE_ENGINE=${FREE_CACHE_ENGINE:-true}
CALCULATE_LOG_PROBS=${CALCULATE_LOG_PROBS:-true}

# -------------------- vLLM Validation配置 --------------------
VAL_TEMPERATURE=${VAL_TEMPERATURE:-1.0}
VAL_TOP_P=${VAL_TOP_P:-1.0}
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-false}
VAL_N=${VAL_N:-1}

# -------------------- Token长度限制 --------------------
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
ROLLOUT_PROMPT_LENGTH=${ROLLOUT_PROMPT_LENGTH:-16384}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-20480}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}

# -------------------- 环境服务器 --------------------
TEXTCRAFT_SERVER=${TEXTCRAFT_SERVER:-"http://127.0.0.1:36001"}

# 实验名称
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME="textcraft_grpo_8x144gb_${TIMESTAMP}"

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
echo "  TextCraft GRPO训练 - Qwen3-1.7B (8卡144GB)" | tee -a "$LOG_FILE"
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
echo "  梯度累积步数: $((TRAIN_BATCH_SIZE / (NUM_GPUS * MICRO_BATCH_SIZE)))" | tee -a "$LOG_FILE"
echo "  Learning Rate: $LEARNING_RATE" | tee -a "$LOG_FILE"
echo "  Clip Ratio: $CLIP_RATIO" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "【vLLM Rollout配置】" | tee -a "$LOG_FILE"
echo "  采样数量 (N): $ROLLOUT_N" | tee -a "$LOG_FILE"
echo "  Temperature: $TEMPERATURE" | tee -a "$LOG_FILE"
echo "  GPU内存利用率: $GPU_MEMORY_UTIL" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "【Token长度限制】" | tee -a "$LOG_FILE"
echo "  Max Response Length: $MAX_RESPONSE_LENGTH" | tee -a "$LOG_FILE"
echo "  Max Model Len: $MAX_MODEL_LEN" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "【实验信息】" | tee -a "$LOG_FILE"
echo "  实验名称: $EXPERIMENT_NAME" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ==================== 检查环境 ====================
echo "检查TextCraft服务器..." | tee -a "$LOG_FILE"
SERVER_RESPONSE=$(curl -s "$TEXTCRAFT_SERVER/" 2>&1)
if [[ "$SERVER_RESPONSE" == *"TextCraft"* ]]; then
    echo "✓ TextCraft服务器正常运行" | tee -a "$LOG_FILE"
else
    echo "警告: TextCraft服务器未运行！" | tee -a "$LOG_FILE"
    exit 1
fi
echo "" | tee -a "$LOG_FILE"

# ==================== 切换到工作目录 ====================
cd /Data/wyh/verl
source ~/miniconda3/bin/activate verl

# ==================== 启动训练 ====================
echo "开始GRPO训练..." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

export RAY_DEDUP_LOGS=0
export CUDA_VISIBLE_DEVICES=$GPU_IDS

export VLLM_LOGGING_LEVEL=WARNING
export VLLM_CONFIGURE_LOGGING=0
export PYTHONWARNINGS=ignore
export RAY_DEDUP_LOGS=1

python3 -m verl.trainer.main_ppo \
    --config-path='/Data/wyh/verl/examples/sglang_multiturn/config' \
    --config-name='textcraft_grpo_train' \
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
    actor_rollout_ref.actor.clip_ratio=$CLIP_RATIO \
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
    trainer.project_name=textcraft_grpo_8x144gb \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.resume_mode=disable \
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "训练完成！日志: $LOG_FILE" | tee -a "$LOG_FILE"
