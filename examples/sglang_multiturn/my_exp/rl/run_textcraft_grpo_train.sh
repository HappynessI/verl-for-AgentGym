set -e

# ==================== 配置参数 ====================

# -------------------- 模型和数据 --------------------
MODEL_PATH=${MODEL_PATH:-"/Data/public/Qwen3-1.7B"}
DATA_PATH=${DATA_PATH:-"/Data/wyh/datasets/Verl-Data/train/textcraft/train.parquet"}
OUTPUT_DIR=${OUTPUT_DIR:-"/Data/wyh/datasets/Verl-Data/outputs/textcraft_grpo"}

# -------------------- GPU配置 --------------------
GPU_IDS=${GPU_IDS:-"5,6"}    # 使用的GPU编号
NUM_GPUS=${NUM_GPUS:-2}      # GPU数量（必须与GPU_IDS一致）

# -------------------- 训练超参数 --------------------
NUM_EPOCHS=${NUM_EPOCHS:-10}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}    # 全局batch size
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}     # 每张GPU的micro batch
LEARNING_RATE=${LEARNING_RATE:-5e-6}
ENTROPY_COEFF=${ENTROPY_COEFF:-0.05}        # 熵奖励系数（防止策略坍缩，增加探索）
SAVE_FREQ=${SAVE_FREQ:-50}                   # 每N个epoch保存checkpoint
TEST_FREQ=${TEST_FREQ:-10}                   # 每N个epoch进行validation

# -------------------- vLLM Rollout配置（训练采样）--------------------
ROLLOUT_N=${ROLLOUT_N:-8}                   # 每个prompt采样数量（GRPO需要>1）
TEMPERATURE=${TEMPERATURE:-1.0}             # 采样温度（建议0.5-1.0以探索）
TOP_P=${TOP_P:-1.0}                         # Nucleus采样参数
GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL:-0.6}     # vLLM GPU内存利用率
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}           # vLLM最大并发序列数
ENFORCE_EAGER=${ENFORCE_EAGER:-true}        # 使用eager模式（更灵活）
FREE_CACHE_ENGINE=${FREE_CACHE_ENGINE:-true} # 释放KV cache（节省内存）

# -------------------- vLLM Validation配置 --------------------
VAL_TEMPERATURE=${VAL_TEMPERATURE:-1.0}
VAL_TOP_P=${VAL_TOP_P:-1.0}
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-true}
VAL_N=${VAL_N:-1}                           # validation时每个prompt采样数量

# -------------------- Token长度限制 --------------------
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}          # 第一轮任务描述最大长度
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-9216}      # episode累积response总长度 (增大10%)
ROLLOUT_PROMPT_LENGTH=${ROLLOUT_PROMPT_LENGTH:-9216}  # rollout最大prompt长度 (匹配response长度)
MAX_MODEL_LEN=${MAX_MODEL_LEN:-12288}                 # vLLM最大序列长度 (增大以容纳更长response)
PPO_MAX_TOKEN_LEN=${PPO_MAX_TOKEN_LEN:-14336}         # PPO训练最大token长度 (相应增大)
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-12288} # vLLM批处理最大token数

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
# 计算GPU_IDS中的GPU数量
GPU_COUNT=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
if [ "$GPU_COUNT" -ne "$NUM_GPUS" ]; then
    echo "错误: GPU_IDS中的GPU数量($GPU_COUNT)与NUM_GPUS($NUM_GPUS)不一致！"
    echo "GPU_IDS=$GPU_IDS"
    echo "请确保NUM_GPUS与GPU_IDS中逗号分隔的GPU数量一致"
    exit 1
fi

# ==================== 打印配置 ====================
echo "================================================================================" | tee "$LOG_FILE"
echo "  TextCraft GRPO训练 - Qwen3-1.7B + 思考模式 (使用vLLM推理后端)" | tee -a "$LOG_FILE"
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
echo "  Entropy Coeff: $ENTROPY_COEFF (熵奖励系数)" | tee -a "$LOG_FILE"
echo "  Save Freq: 每 $SAVE_FREQ epochs" | tee -a "$LOG_FILE"
echo "  Test Freq: 每 $TEST_FREQ epochs" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "【vLLM Rollout配置】（训练时策略探索）" | tee -a "$LOG_FILE"
echo "  后端: vLLM (内嵌模式)" | tee -a "$LOG_FILE"
echo "  采样数量 (N): $ROLLOUT_N" | tee -a "$LOG_FILE"
echo "  Temperature: $TEMPERATURE" | tee -a "$LOG_FILE"
echo "  Top-P: $TOP_P" | tee -a "$LOG_FILE"
echo "  GPU内存利用率: $GPU_MEMORY_UTIL" | tee -a "$LOG_FILE"
echo "  最大并发序列: $MAX_NUM_SEQS" | tee -a "$LOG_FILE"
echo "  Eager模式: $ENFORCE_EAGER" | tee -a "$LOG_FILE"
echo "  释放Cache: $FREE_CACHE_ENGINE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "【vLLM Validation配置】（验证时策略评估）" | tee -a "$LOG_FILE"
echo "  Temperature: $VAL_TEMPERATURE" | tee -a "$LOG_FILE"
echo "  Top-P: $VAL_TOP_P" | tee -a "$LOG_FILE"
echo "  Do Sample: $VAL_DO_SAMPLE" | tee -a "$LOG_FILE"
echo "  采样数量 (N): $VAL_N" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "【Token长度限制】" | tee -a "$LOG_FILE"
echo "  Max Prompt Length: $MAX_PROMPT_LENGTH" | tee -a "$LOG_FILE"
echo "  Max Response Length: $MAX_RESPONSE_LENGTH" | tee -a "$LOG_FILE"
echo "  Rollout Prompt Length: $ROLLOUT_PROMPT_LENGTH" | tee -a "$LOG_FILE"
echo "  vLLM Max Model Len: $MAX_MODEL_LEN" | tee -a "$LOG_FILE"
echo "  PPO Max Token Len: $PPO_MAX_TOKEN_LEN" | tee -a "$LOG_FILE"
echo "  Max Batched Tokens: $MAX_NUM_BATCHED_TOKENS" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "【环境服务器】" | tee -a "$LOG_FILE"
echo "  TextCraft Server: $TEXTCRAFT_SERVER" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "【实验信息】" | tee -a "$LOG_FILE"
echo "  实验名称: $EXPERIMENT_NAME" | tee -a "$LOG_FILE"
echo "  日志文件: $LOG_FILE" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ==================== 检查环境 ====================
echo "检查TextCraft服务器..." | tee -a "$LOG_FILE"
SERVER_RESPONSE=$(curl -s "$TEXTCRAFT_SERVER/" 2>&1)
if [[ "$SERVER_RESPONSE" == *"TextCraft"* ]]; then
    echo "✓ TextCraft服务器正常运行" | tee -a "$LOG_FILE"
else
    echo "警告: TextCraft服务器 ($TEXTCRAFT_SERVER) 未运行！" | tee -a "$LOG_FILE"
    echo "请先启动服务器：" | tee -a "$LOG_FILE"
    echo "  cd /Data/wyh/AgentGym-RL/AgentGym/agentenv-textcraft" | tee -a "$LOG_FILE"
    echo "  textcraft --host 0.0.0.0 --port 36001" | tee -a "$LOG_FILE"
    exit 1
fi
echo "" | tee -a "$LOG_FILE"

# ==================== 切换到工作目录 ====================
cd /Data/wyh/verl
source ~/miniconda3/bin/activate verl

echo "激活verl环境" | tee -a "$LOG_FILE"
echo "Python版本: $(python --version)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ==================== 启动训练 ====================
echo "开始GRPO训练..." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Ray配置
export RAY_DEDUP_LOGS=0
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# ==================== 日志级别控制 ====================
# 关闭冗余的 DEBUG 日志，只保留重要信息
export VLLM_LOGGING_LEVEL=WARNING        # vLLM日志级别: WARNING(警告)/ERROR(错误)/INFO(信息)
export VLLM_CONFIGURE_LOGGING=0          # 禁用vLLM默认日志配置
export PYTHONWARNINGS=ignore             # 忽略Python警告信息
export RAY_DEDUP_LOGS=1                  # Ray日志去重(改为1以减少重复)

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
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
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
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
echo "训练完成！" | tee -a "$LOG_FILE"
echo "日志文件: $LOG_FILE" | tee -a "$LOG_FILE"
echo "检查点目录: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"

