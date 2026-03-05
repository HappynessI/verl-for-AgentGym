#!/bin/bash
# =============================================================================
# wyh_exp Turn-Prefix RL 训练脚本
# 支持两类实验：
#   1. GRPO baseline (ADV_ESTIMATOR=grpo, DATA_PATH=baseline_train.parquet)
#   2. Prefix-RL 主方法 (ADV_ESTIMATOR=turn_full_trajectory/turn_prefix_guided/...)
#
# 可用的优势估计器（ADV_ESTIMATOR）：
#   grpo                            — 原生 GRPO，无 prefix 处理，用于 baseline
#   turn_full_trajectory            — 方式A：全轨迹 GRPO（prefix+rollout 均参与）
#   turn_full_trajectory_with_turn_bonus — 方式A + turn 级奖励加成
#   turn_prefix_guided              — 方式B：prefix 部分梯度为 0，只训练 rollout 部分
#   turn_prefix_guided_dr           — 方式B 的 Dr.GRPO 变体（不除 std，只减均值）
#
# 关于 prefix 比例：
#   prefix 轮数已在数据准备阶段固定，当前 prefix_train.parquet 使用 random 策略
#   (min=1, max=all turns-1)。如需 fixed prefix 长度，需重新生成数据集：
#     python prepare_prefix_data.py --prefix_strategy=fixed --fixed_prefix_turns=N
#
# 使用示例：
#   # GRPO baseline
#   ADV_ESTIMATOR=grpo DATA_PATH=/Data/wyh/datasets/wyh_exp_prefix/baseline_train.parquet \
#     bash recipe/wyh_exp/scripts/run_test_train.sh
#
#   # Prefix-RL 主方法
#   ADV_ESTIMATOR=turn_full_trajectory \
#     bash recipe/wyh_exp/scripts/run_test_train.sh
#
#   # turn_prefix_guided（方式B）
#   ADV_ESTIMATOR=turn_prefix_guided \
#     bash recipe/wyh_exp/scripts/run_test_train.sh
#
# 前置条件：
#   conda activate agentenv-textcraft && textcraft --host 0.0.0.0 --port 36001
#   conda activate verl
#   cd /Data/wyh/verl
# =============================================================================
set -e

# ==================== 配置参数 ====================

# -------------------- 模型和数据 --------------------
MODEL_PATH=${MODEL_PATH:-"/Data/public/Qwen3-1.7B"}

# GRPO baseline：使用 baseline_train.parquet（无 prefix，从头 rollout）
# Prefix-RL：  使用 prefix_train.parquet（包含 Gemini prefix 轮次）
DATA_PATH=${DATA_PATH:-"/Data/wyh/datasets/wyh_exp_prefix/prefix_train.parquet"}
VAL_DATA_PATH=${VAL_DATA_PATH:-"/Data/wyh/datasets/wyh_exp_prefix/baseline_train.parquet"}
OUTPUT_DIR=${OUTPUT_DIR:-"/Data/wyh/datasets/Verl-Data/outputs/wyh_exp_textcraft"}

# -------------------- GPU配置 --------------------
GPU_IDS=${GPU_IDS:-"3,4"}
NUM_GPUS=${NUM_GPUS:-2}

# -------------------- 核心参数：优势估计器 --------------------
ADV_ESTIMATOR=${ADV_ESTIMATOR:-"turn_full_trajectory"}
# 可选: grpo | turn_full_trajectory | turn_full_trajectory_with_turn_bonus
#       turn_prefix_guided | turn_prefix_guided_dr

# -------------------- Thinking 模式 --------------------
# 保持 True
# clip ratio 问题通过增大 response_length 解决，而非关闭 thinking
# 两种模式产出不可比，统一使用 True
ENABLE_THINKING=${ENABLE_THINKING:-"true"}

# -------------------- 训练超参数 --------------------
NUM_EPOCHS=${NUM_EPOCHS:-100}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}
LEARNING_RATE=${LEARNING_RATE:-5e-6}
SAVE_FREQ=${SAVE_FREQ:-100}
TEST_FREQ=${TEST_FREQ:-50}

# -------------------- vLLM Rollout配置 --------------------
ROLLOUT_N=${ROLLOUT_N:-8}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-0.95}
GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL:-0.7}
ENFORCE_EAGER=${ENFORCE_EAGER:-true}
FREE_CACHE_ENGINE=${FREE_CACHE_ENGINE:-true}

# -------------------- Validation 配置 --------------------
# 重要：VAL_MAX_TOKENS 必须足够大，否则 thinking 模型无法完成任务导致验证严重低估
# 建议 ≥ 4096（与 MAX_RESPONSE_LENGTH 保持同量级）
VAL_TEMPERATURE=${VAL_TEMPERATURE:-1.0}
VAL_TOP_P=${VAL_TOP_P:-0.95}
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-true}
VAL_N=${VAL_N:-1}
VAL_MAX_TOKENS=${VAL_MAX_TOKENS:-8192}  # 验证时保留足够空间生成完整轨迹

# -------------------- Token 长度配置 --------------------
# Qwen3-1.7B + 2×L20(46GB) 容量评估：
#   模型权重 ≈ 3.5GB (BF16)
#   KV cache per token = 2 × 28 layers × 8 KV heads × 128 head_dim × 2 bytes = 112KB
#   单 GPU KV 容量 = 46GB × 0.7 / 112KB ≈ 301,000 tokens
#
# thinking=True 时 response 长度分析（上次实验实测）：
#   早期: ~36 turns × 137 tokens/turn ≈ 4944 tokens → 9216 勉强OK
#   后期: ~14 turns × 488 tokens/turn ≈ 6832 tokens → 超过 9216 导致 clip ratio 爆炸
#   最坏: ~20 turns × 800 tokens/turn ≈ 16000 tokens
#
#   结论：response_length=16384 覆盖 95%+ 正常训练场景，彻底解决 clip ratio 问题
#   max_model_len = 4096(prompt) + 16384(response) = 20480
#
# GRPO baseline prompt 实测 755 tokens，Prefix-RL prompt 实测 838-1017 tokens，
# 均远小于 4096，prompt 预算充裕
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}  # 解决 thinking=true 的 clip ratio 问题
MAX_MODEL_LEN=${MAX_MODEL_LEN:-20480}              # prompt(4096) + response(16384) = 20480
ROLLOUT_PROMPT_LENGTH=${ROLLOUT_PROMPT_LENGTH:-4096}
PPO_MAX_TOKEN_LEN=${PPO_MAX_TOKEN_LEN:-16384}      # actor update 每 GPU 最大 token 数
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-20480}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-128}                  # 从 256 降至 128，避免大 context 下 OOM

# -------------------- 环境服务器 --------------------
TEXTCRAFT_SERVER=${TEXTCRAFT_SERVER:-"http://127.0.0.1:36001"}

# 实验名称
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME="wyh_${ADV_ESTIMATOR}_${TIMESTAMP}"

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
echo "  wyh_exp Turn-Prefix RL 训练" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "【核心配置】" | tee -a "$LOG_FILE"
echo "  优势估计器:     $ADV_ESTIMATOR" | tee -a "$LOG_FILE"
echo "  Thinking 模式:  $ENABLE_THINKING" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "【模型和数据】" | tee -a "$LOG_FILE"
echo "  模型路径:       $MODEL_PATH" | tee -a "$LOG_FILE"
echo "  训练数据:       $DATA_PATH" | tee -a "$LOG_FILE"
echo "  验证数据:       $VAL_DATA_PATH" | tee -a "$LOG_FILE"
echo "  输出目录:       $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "【GPU配置】" | tee -a "$LOG_FILE"
echo "  GPU IDs: $GPU_IDS ($NUM_GPUS 卡)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "【Token 长度】" | tee -a "$LOG_FILE"
echo "  Max Prompt Length:   $MAX_PROMPT_LENGTH" | tee -a "$LOG_FILE"
echo "  Max Response Length: $MAX_RESPONSE_LENGTH" | tee -a "$LOG_FILE"
echo "  Max Model Len:       $MAX_MODEL_LEN" | tee -a "$LOG_FILE"
echo "  Val Max Tokens:      $VAL_MAX_TOKENS" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "【训练超参数】" | tee -a "$LOG_FILE"
echo "  Epochs: $NUM_EPOCHS | Batch: $TRAIN_BATCH_SIZE | LR: $LEARNING_RATE" | tee -a "$LOG_FILE"
echo "  Rollout N: $ROLLOUT_N | Temperature: $TEMPERATURE" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ==================== 检查环境 ====================
echo "检查TextCraft服务器..." | tee -a "$LOG_FILE"
SERVER_RESPONSE=$(curl -s "$TEXTCRAFT_SERVER/" 2>&1)
if [[ "$SERVER_RESPONSE" == *"TextCraft"* ]]; then
    echo "✓ TextCraft服务器正常运行" | tee -a "$LOG_FILE"
else
    echo "✗ TextCraft服务器 ($TEXTCRAFT_SERVER) 未运行！" | tee -a "$LOG_FILE"
    echo "请先启动：conda activate agentenv-textcraft && textcraft --host 0.0.0.0 --port 36001" | tee -a "$LOG_FILE"
    exit 1
fi

# ==================== 切换到工作目录 ====================
cd /Data/wyh/verl
source ~/miniconda3/bin/activate verl

export PYTHONPATH="/Data/wyh/verl:${PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

echo "激活verl环境, Python: $(python --version)" | tee -a "$LOG_FILE"

# 检查自定义优势估计器注册
python3 -c "
import recipe.wyh_exp
from verl.trainer.ppo.core_algos import ADV_ESTIMATOR_REGISTRY
assert '${ADV_ESTIMATOR}' in ADV_ESTIMATOR_REGISTRY, \
    f'${ADV_ESTIMATOR} not found, available: {list(ADV_ESTIMATOR_REGISTRY.keys())}'
print('✓ ${ADV_ESTIMATOR} 已注册')
print('  可用估计器:', list(ADV_ESTIMATOR_REGISTRY.keys()))
" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "开始训练..." | tee -a "$LOG_FILE"

# ==================== 环境变量 ====================
export RAY_DEDUP_LOGS=1
export CUDA_VISIBLE_DEVICES=$GPU_IDS
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_CONFIGURE_LOGGING=0
export PYTHONWARNINGS=ignore

# ==================== 启动训练 ====================
python3 -m recipe.wyh_exp.main_train \
    algorithm.adv_estimator=$ADV_ESTIMATOR \
    data.train_files=$DATA_PATH \
    data.val_files=$VAL_DATA_PATH \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=4 \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    "+data.apply_chat_template_kwargs.enable_thinking=$ENABLE_THINKING" \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.enable_activation_offload=true \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
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
    actor_rollout_ref.rollout.val_kwargs.temperature=$VAL_TEMPERATURE \
    actor_rollout_ref.rollout.val_kwargs.top_p=$VAL_TOP_P \
    actor_rollout_ref.rollout.val_kwargs.do_sample=$VAL_DO_SAMPLE \
    actor_rollout_ref.rollout.val_kwargs.n=$VAL_N \
    actor_rollout_ref.rollout.val_kwargs.max_tokens=$VAL_MAX_TOKENS \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.total_epochs=$NUM_EPOCHS \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.project_name=wyh_exp \
    trainer.experiment_name=$EXPERIMENT_NAME \
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
echo "训练完成！" | tee -a "$LOG_FILE"
echo "日志文件: $LOG_FILE" | tee -a "$LOG_FILE"
echo "检查点目录: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
