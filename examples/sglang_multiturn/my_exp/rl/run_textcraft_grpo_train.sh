#!/bin/bash
# TextCraft GRPO训练脚本 - Qwen3-1.7B
# 使用4xL20 GPU
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

# 训练参数
NUM_EPOCHS=${NUM_EPOCHS:-10}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
NUM_GPUS=4

# TextCraft服务器
TEXTCRAFT_SERVER=${TEXTCRAFT_SERVER:-"http://127.0.0.1:36002"}

# 实验名称
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME="textcraft_grpo_${TIMESTAMP}"

# 日志目录
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_${TIMESTAMP}.log"

# ==================== 打印配置 ====================
echo "============================================" | tee "$LOG_FILE"
echo "TextCraft GRPO训练 - Qwen3-1.7B" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
echo "模型: $MODEL_PATH" | tee -a "$LOG_FILE"
echo "数据: $DATA_PATH" | tee -a "$LOG_FILE"
echo "输出: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Epochs: $NUM_EPOCHS" | tee -a "$LOG_FILE"
echo "Batch Size: $TRAIN_BATCH_SIZE" | tee -a "$LOG_FILE"
echo "Micro Batch: $MICRO_BATCH_SIZE" | tee -a "$LOG_FILE"
echo "Learning Rate: $LEARNING_RATE" | tee -a "$LOG_FILE"
echo "GPUs: $NUM_GPUS" | tee -a "$LOG_FILE"
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
export CUDA_VISIBLE_DEVICES=4,5,6,7

python3 -m verl.trainer.main_ppo \
    --config-path='/Data/wyh/verl/examples/sglang_multiturn/config' \
    --config-name='textcraft_grpo_train' \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_PATH \
    data.val_files=$DATA_PATH \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=4 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.total_epochs=$NUM_EPOCHS \
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

