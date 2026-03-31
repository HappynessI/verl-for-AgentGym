set -e

# ==================== 配置参数 ====================

# -------------------- 模型和数据 --------------------
MODEL_PATH=${MODEL_PATH:-"/Data/public/Qwen3-1.7B"}
DATA_PATH=${DATA_PATH:-"/Data/wyh/datasets/Verl-Data/train/textcraft/train.parquet"}
OUTPUT_DIR=${OUTPUT_DIR:-"/Data/wyh/datasets/Verl-Data/outputs/textcraft_grpo"}

# -------------------- GPU配置 --------------------
GPU_IDS=${GPU_IDS:-"0,1"}       # 使用的GPU编号（2卡H200配置）
NUM_GPUS=${NUM_GPUS:-2}         # GPU数量（必须与GPU_IDS一致）

# -------------------- 训练超参数 --------------------
# GRPO batch 设计核心原则：ppo_mini_batch_size = train_batch_size × rollout_n
# 这样每个 rollout batch 产生的数据尽量整体参与一次更新，避免切得太碎
# 当前配置：TRAIN_BATCH_SIZE=8, ROLLOUT_N=8, ppo_mini_batch_size=64
# 这样 8 个 prompt × 8 次采样 = 64 个样本，刚好组成一个 mini-batch
# 梯度累积 = 64 / (2 GPU × 8 micro_batch) = 4 步
NUM_EPOCHS=${NUM_EPOCHS:-100}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}        # 全局batch size（配合ROLLOUT_N=8，使 mini_batch=64）
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-32}        # 每张GPU的micro batch（2卡H200可用更大值）
LEARNING_RATE=${LEARNING_RATE:-3e-6}            # 学习率
SAVE_FREQ=${SAVE_FREQ:-500}                    # 每N个epoch保存checkpoint
TEST_FREQ=${TEST_FREQ:-500}                    # 每N个epoch进行validation

# -------------------- Rollout Correction（TIS / MIS）--------------------
# TIS: Truncated Importance Sampling（截断IS，加权但不丢弃）
# MIS: Masked Importance Sampling（拒绝采样，超阈值的token/序列直接mask）
# 两者可同时开启，也可单独使用；设为 "none" 则关闭对应功能
# 注：当前配置关闭 MIS/TIS，使用纯 GRPO（更稳定）
ROLLOUT_IS=${ROLLOUT_IS:-"none"}        # 关闭 TIS
ROLLOUT_IS_THRESHOLD=${ROLLOUT_IS_THRESHOLD:-2.0}   # TIS 截断阈值
ROLLOUT_RS=${ROLLOUT_RS:-"none"}        # 关闭 MIS
ROLLOUT_RS_THRESHOLD=${ROLLOUT_RS_THRESHOLD:-2.0}   # MIS 上阈值
ROLLOUT_RS_THRESHOLD_LOWER=${ROLLOUT_RS_THRESHOLD_LOWER:-0.2} # MIS 下阈值（降为0.2保守）

# -------------------- vLLM Rollout配置（训练采样）--------------------
ROLLOUT_N=${ROLLOUT_N:-8}                   # 每个prompt采样数量（GRPO需要>1，半显存建议先3）
TEMPERATURE=${TEMPERATURE:-1.0}             # 采样温度（更稳妥）
TOP_P=${TOP_P:-1.0}                         # Nucleus采样参数（更稳妥）
GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL:-0.85}     # vLLM GPU内存利用率（2卡H200显存充足，可提高）
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}            # vLLM最大并发序列数
ENFORCE_EAGER=${ENFORCE_EAGER:-true}        # 使用eager模式（更灵活）
FREE_CACHE_ENGINE=${FREE_CACHE_ENGINE:-true} # 释放KV cache（节省内存）
CALCULATE_LOG_PROBS=${CALCULATE_LOG_PROBS:-true}  # Rollout Correction 依赖 rollout_log_probs

# -------------------- vLLM Validation配置 --------------------
VAL_TEMPERATURE=${VAL_TEMPERATURE:-1.0}
VAL_TOP_P=${VAL_TOP_P:-1.0}
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-false}  # 关闭采样（验证结果更稳定可比较）
VAL_N=${VAL_N:-1}                           # validation时每个prompt采样数量

# -------------------- Token长度限制 --------------------
# 关键设计原则：
# - MAX_RESPONSE_LENGTH: 整条 episode 的累计 response 上限（用于 PPO 训练数据裁剪）
# - ROLLOUT_RESPONSE_LENGTH: 单轮 rollout 时 assistant 生成上限（控制 vLLM 生成）
# - ROLLOUT_PROMPT_LENGTH: rollout 时 prompt 的长度上限
# 这三者需要分开，否则单轮生成会被放得过长，容易复现之前的坏模式
#
# 基于采样数据统计设计：
# - 单轮 assistant：均值 143.6，最大 4020，P95 在 400~570 → ROLLOUT_RESPONSE_LENGTH=1024
# - 任务总输出：均值 1041，最大 7566，大部分 600~1500 → MAX_RESPONSE_LENGTH=4096
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}            # 第一轮任务描述最大长度
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}         # episode累积response总长度（覆盖大部分复杂任务）
ROLLOUT_RESPONSE_LENGTH=${ROLLOUT_RESPONSE_LENGTH:-1024} # 单轮assistant生成上限（覆盖P95=400~570，有余量）
ROLLOUT_PROMPT_LENGTH=${ROLLOUT_PROMPT_LENGTH:-4096}     # rollout最大prompt长度（增大以支持更长上下文）
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}                    # vLLM最大序列长度（支持更长上下文）
PPO_MAX_TOKEN_LEN=${PPO_MAX_TOKEN_LEN:-8192}             # PPO训练最大token长度（支持更长序列训练）
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}  # vLLM批处理最大token数（提升吞吐量）

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
echo "  Save Freq: 每 $SAVE_FREQ epochs" | tee -a "$LOG_FILE"
echo "  Test Freq: 每 $TEST_FREQ epochs" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "【Rollout Correction (TIS / MIS)】" | tee -a "$LOG_FILE"
echo "  TIS (rollout_is): $ROLLOUT_IS  阈值: $ROLLOUT_IS_THRESHOLD" | tee -a "$LOG_FILE"
echo "  MIS (rollout_rs): $ROLLOUT_RS  上阈值: $ROLLOUT_RS_THRESHOLD  下阈值: $ROLLOUT_RS_THRESHOLD_LOWER" | tee -a "$LOG_FILE"
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
echo "  计算rollout_log_probs: $CALCULATE_LOG_PROBS" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "【vLLM Validation配置】（验证时策略评估）" | tee -a "$LOG_FILE"
echo "  Temperature: $VAL_TEMPERATURE" | tee -a "$LOG_FILE"
echo "  Top-P: $VAL_TOP_P" | tee -a "$LOG_FILE"
echo "  Do Sample: $VAL_DO_SAMPLE" | tee -a "$LOG_FILE"
echo "  采样数量 (N): $VAL_N" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "【Token长度限制】" | tee -a "$LOG_FILE"
echo "  Max Prompt Length: $MAX_PROMPT_LENGTH" | tee -a "$LOG_FILE"
echo "  Max Response Length: $MAX_RESPONSE_LENGTH (episode累计response上限)" | tee -a "$LOG_FILE"
echo "  Rollout Response Length: $ROLLOUT_RESPONSE_LENGTH (单轮assistant生成上限)" | tee -a "$LOG_FILE"
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

# ==================== 构建 Rollout Correction 参数 ====================
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
    actor_rollout_ref.model.enable_activation_offload=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.actor.ppo_mini_batch_size=$((TRAIN_BATCH_SIZE * ROLLOUT_N)) \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.top_p=$TOP_P \
    actor_rollout_ref.rollout.prompt_length=$ROLLOUT_PROMPT_LENGTH \
    actor_rollout_ref.rollout.response_length=$ROLLOUT_RESPONSE_LENGTH \
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

