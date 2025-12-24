#!/bin/bash
# TextCraft GRPO训练脚本 - Qwen3-1.7B
# 
# 使用方法：
#   bash run_textcraft_grpo_train.sh  # 使用默认参数（2张GPU: 0,1）
#   GPU_IDS="2,3" NUM_GPUS=2 bash run_textcraft_grpo_train.sh  # 自定义GPU
#   ROLLOUT_N=16 TEMPERATURE=0.5 bash run_textcraft_grpo_train.sh  # 自定义其他参数
# 
# 主要可配置参数（通过环境变量）：
#   GPU_IDS="0,1"          - 使用的GPU编号（逗号分隔）
#   NUM_GPUS=2             - GPU数量（必须与GPU_IDS中的数量一致）
#   NUM_EPOCHS=10          - 训练轮数
#   TRAIN_BATCH_SIZE=64    - 全局训练batch size（梯度累积 = TRAIN_BATCH_SIZE / (NUM_GPUS * MICRO_BATCH_SIZE)）
#   MICRO_BATCH_SIZE=2     - 每张GPU的micro batch size
#   LEARNING_RATE=1e-6     - 学习率
#   ROLLOUT_N=8            - 每个prompt采样的响应数量
#   TEMPERATURE=0.3        - 采样温度
#   TOP_P=0.9              - nucleus采样参数
#   SAVE_FREQ=5            - checkpoint保存频率（每N个epoch）
#   TEST_FREQ=2            - validation频率（每N个epoch）
#   MAX_RESPONSE_LENGTH=4096 - 整个episode累积的response token总长度
#   MAX_MODEL_LEN=10240    - vLLM模型处理的总长度上限
# 
# 重要参数说明：
# - max_response_length (data): 整个episode所有轮次累积的response token总长度上限（设置为4096）
# - response_length (rollout): 同max_response_length，控制episode何时terminated
# 
# 关键决策记录：
# 1. 使用vLLM而非SGLang：vLLM更成熟稳定，社区支持更好
# 2. response_length设为4096：
#    - 原因：vLLM/SGLang的max_tokens参数在multi-turn场景下无法可靠限制单次生成长度
#    - 实测：设置max_tokens=512，但实际生成平均~870 tokens（超出70%）
#    - 解决：增加response_length到4096，降低clip_ratio，允许模型充分生成<think>推理
# 3. validation和eval的差异可接受：
#    - eval用transformers，max_new_tokens=512可以严格限制
#    - GRPO validation用vLLM，无法严格限制单次生成，但通过response_length控制总长度
#    - 重点是训练效果，不是严格对齐eval的生成长度

set -e

# ==================== 配置参数 ====================
# 模型和数据路径
MODEL_PATH=${MODEL_PATH:-"/Data/public/Qwen3-1.7B"}
DATA_PATH=${DATA_PATH:-"/Data/wyh/datasets/Verl-Data/train/textcraft/train.parquet"}
OUTPUT_DIR=${OUTPUT_DIR:-"/Data/wyh/datasets/Verl-Data/outputs/textcraft_grpo"}

# GPU配置
GPU_IDS=6,7 # 使用的GPU编号
NUM_GPUS=${NUM_GPUS:-2}    # GPU数量（必须与GPU_IDS一致）

# 训练参数
NUM_EPOCHS=${NUM_EPOCHS:-10}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}  # 全局batch size，梯度累积步数 = 64/(2*2)=16
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}   # 每张GPU的micro batch
LEARNING_RATE=${LEARNING_RATE:-1e-6}

# Rollout采样参数
ROLLOUT_N=${ROLLOUT_N:-8}  # 每个prompt采样的响应数量
TEMPERATURE=${TEMPERATURE:-0.3}  # 采样温度
TOP_P=${TOP_P:-0.9}  # nucleus采样参数

# Validation参数
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.3}
VAL_TOP_P=${VAL_TOP_P:-0.9}
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-true}
VAL_N=${VAL_N:-1}  # validation时每个prompt采样数量

# 训练控制参数
SAVE_FREQ=${SAVE_FREQ:-5}  # 每N个epoch保存一次checkpoint
TEST_FREQ=${TEST_FREQ:-2}  # 每N个epoch进行一次validation

# Token长度参数
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}  # data层：第一轮任务描述的最大长度
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}  # data层：整个episode累积的response token总长度

ROLLOUT_PROMPT_LENGTH=${ROLLOUT_PROMPT_LENGTH:-8192}  # rollout层：最大prompt长度（包含多轮历史）
# ROLLOUT_RESPONSE_LENGTH已删除：默认继承data.max_response_length

MAX_MODEL_LEN=${MAX_MODEL_LEN:-10240}  # vLLM: prompt+response的总长度上限
PPO_MAX_TOKEN_LEN=${PPO_MAX_TOKEN_LEN:-12288}  # PPO训练：每个GPU的最大token长度
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-10240}  # vLLM: 批处理的最大token数

# TextCraft服务器
TEXTCRAFT_SERVER=${TEXTCRAFT_SERVER:-"http://127.0.0.1:36002"}

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
echo "============================================" | tee "$LOG_FILE"
echo "TextCraft GRPO训练 - Qwen3-1.7B" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
echo "模型: $MODEL_PATH" | tee -a "$LOG_FILE"
echo "数据: $DATA_PATH" | tee -a "$LOG_FILE"
echo "输出: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "训练参数:" | tee -a "$LOG_FILE"
echo "  GPU IDs: $GPU_IDS" | tee -a "$LOG_FILE"
echo "  GPU数量: $NUM_GPUS" | tee -a "$LOG_FILE"
echo "  Epochs: $NUM_EPOCHS" | tee -a "$LOG_FILE"
echo "  全局Batch Size: $TRAIN_BATCH_SIZE" | tee -a "$LOG_FILE"
echo "  每GPU Micro Batch: $MICRO_BATCH_SIZE" | tee -a "$LOG_FILE"
echo "  梯度累积步数: $((TRAIN_BATCH_SIZE / (NUM_GPUS * MICRO_BATCH_SIZE)))" | tee -a "$LOG_FILE"
echo "  Learning Rate: $LEARNING_RATE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Rollout采样参数:" | tee -a "$LOG_FILE"
echo "  N (每prompt采样数): $ROLLOUT_N" | tee -a "$LOG_FILE"
echo "  Temperature: $TEMPERATURE" | tee -a "$LOG_FILE"
echo "  Top-P: $TOP_P" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Validation参数:" | tee -a "$LOG_FILE"
echo "  Temperature: $VAL_TEMPERATURE" | tee -a "$LOG_FILE"
echo "  Top-P: $VAL_TOP_P" | tee -a "$LOG_FILE"
echo "  Do Sample: $VAL_DO_SAMPLE" | tee -a "$LOG_FILE"
echo "  N (每prompt采样数): $VAL_N" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "训练控制:" | tee -a "$LOG_FILE"
echo "  Save Freq: 每 $SAVE_FREQ epochs" | tee -a "$LOG_FILE"
echo "  Test Freq: 每 $TEST_FREQ epochs" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Token长度限制:" | tee -a "$LOG_FILE"
echo "  Max Prompt Length: $MAX_PROMPT_LENGTH" | tee -a "$LOG_FILE"
echo "  Max Response Length: $MAX_RESPONSE_LENGTH (data & rollout共用)" | tee -a "$LOG_FILE"
echo "  Rollout Prompt Length: $ROLLOUT_PROMPT_LENGTH" | tee -a "$LOG_FILE"
echo "  Max Model Len (vLLM): $MAX_MODEL_LEN" | tee -a "$LOG_FILE"
echo "  PPO Max Token Len: $PPO_MAX_TOKEN_LEN" | tee -a "$LOG_FILE"
echo "  Max Batched Tokens: $MAX_NUM_BATCHED_TOKENS" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "TextCraft Server: $TEXTCRAFT_SERVER" | tee -a "$LOG_FILE"
echo "实验名称: $EXPERIMENT_NAME" | tee -a "$LOG_FILE"
echo "日志文件: $LOG_FILE" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
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
    echo "  textcraft --host 0.0.0.0 --port 36002" | tee -a "$LOG_FILE"
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
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.top_p=$TOP_P \
    actor_rollout_ref.rollout.prompt_length=$ROLLOUT_PROMPT_LENGTH \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
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

