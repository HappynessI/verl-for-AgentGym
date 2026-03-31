set -e

# ==================== TextCraft Prefix RL (Validated) 训练脚本 ====================
# Debug 测试版本 - 使用 Qwen3-0.6B 单卡
#
# 用途：验证主实验能否跑通
# - 模型: Qwen3-0.6B (更小，更快)
# - GPU: 2,3 (双卡)
# - 参数: 尽可能小

# ==================== 配置参数 ====================

# 模型和数据
MODEL_PATH=${MODEL_PATH:-"/Data/public/Qwen3-0.6B"}
DATA_PATH=${DATA_PATH:-"/Data/wyh/datasets/Verl-Data/train/textcraft/prefix-rl/textcraft_validated_prefix_history_canonicalized.parquet"}
VAL_DATA_PATH=${VAL_DATA_PATH:-"/Data/wyh/datasets/Verl-Data/train/textcraft/train.parquet"}

# Debug 配置
DEBUG_MODE=${DEBUG_MODE:-1}
DEBUG_MAX_SAMPLES=${DEBUG_MAX_SAMPLES:-4}
DEBUG_PREFLIGHT_ONLY=${DEBUG_PREFLIGHT_ONLY:-0}
DEBUG_VAL_DATA_PATH=${DEBUG_VAL_DATA_PATH:-"/Data/wyh/datasets/Verl-Data/train/textcraft/train.parquet"}
DEBUG_VAL_MAX_SAMPLES=${DEBUG_VAL_MAX_SAMPLES:-16}

# GPU配置 - 使用双卡 GPU 0,1 (2,3 被占用)
GPU_IDS=${GPU_IDS:-"0,1"}
NUM_GPUS=${NUM_GPUS:-2}

# 训练超参数 - 尽可能小
NUM_EPOCHS=${NUM_EPOCHS:-1}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
LEARNING_RATE=${LEARNING_RATE:-5e-6}
SAVE_FREQ=${SAVE_FREQ:-1}
TEST_FREQ=${TEST_FREQ:-1}

# Rollout配置
ROLLOUT_N=${ROLLOUT_N:-1}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-0.95}
GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL:-0.25}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
ENFORCE_EAGER=${ENFORCE_EAGER:-true}
FREE_CACHE_ENGINE=${FREE_CACHE_ENGINE:-true}
CALCULATE_LOG_PROBS=${CALCULATE_LOG_PROBS:-true}

# vLLM Validation配置
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.3}
VAL_TOP_P=${VAL_TOP_P:-0.9}
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-true}
VAL_N=${VAL_N:-1}

# Token长度限制 - 需要足够大以容纳 prompt + response
# 注意：prefix_actions 会显著增加 prompt 长度
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
ROLLOUT_PROMPT_LENGTH=${ROLLOUT_PROMPT_LENGTH:-4096}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}  # 增加到 8K，因为 prefix_actions 会显著增加 prompt 长度
PPO_MAX_TOKEN_LEN=${PPO_MAX_TOKEN_LEN:-8192}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-4096}

# 环境服务器
TEXTCRAFT_SERVER=${TEXTCRAFT_SERVER:-"http://127.0.0.1:36001"}

# ==================== Debug 模式覆盖配置 ====================
if [ "$DEBUG_MODE" = "1" ]; then
    echo ""
    echo "============================================"
    echo "  ⚠️  DEBUG MODE - 0.6B 单卡测试 ⚠️"
    echo "  验证主实验能否跑通"
    echo "============================================"
    echo ""
    
    # Debug 模式专用目录 - 放在同一个 debug 目录下
    OUTPUT_DIR="/Data/wyh/datasets/Verl-Data/outputs/textcraft_grpo_validated_debug"
    
    echo "Debug 配置:"
    echo "  MODEL_PATH=$MODEL_PATH"
    echo "  GPU_IDS=$GPU_IDS (双卡)"
    echo "  GPU_MEMORY_UTIL=$GPU_MEMORY_UTIL"
    echo "  NUM_EPOCHS=$NUM_EPOCHS"
    echo "  TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE"
    echo "  MICRO_BATCH_SIZE=$MICRO_BATCH_SIZE"
    echo "  ROLLOUT_N=$ROLLOUT_N"
    echo "  MAX_NUM_SEQS=$MAX_NUM_SEQS"
    echo "  MAX_RESPONSE_LENGTH=$MAX_RESPONSE_LENGTH"
    echo "  MAX_MODEL_LEN=$MAX_MODEL_LEN"
    echo ""
    
    # 生成 debug 数据子集
    DEBUG_DATA_DIR="/Data/wyh/datasets/Verl-Data/train/textcraft/prefix-rl/debug"
    mkdir -p "$DEBUG_DATA_DIR"
    DEBUG_DATA_PATH="${DEBUG_DATA_DIR}/debug_subset_${DEBUG_MAX_SAMPLES}.parquet"
    
    echo "生成 Debug 数据子集 (max_samples=$DEBUG_MAX_SAMPLES)..."
    cd /Data/wyh/verl
    source ~/miniconda3/bin/activate verl
    
    python3 -c "
import sys
sys.path.insert(0, '/Data/wyh/verl/examples/sglang_multiturn/my_exp/rl/debug')
from debug_data_utils import create_debug_subset
create_debug_subset(
    '$DATA_PATH',
    '$DEBUG_DATA_PATH',
    $DEBUG_MAX_SAMPLES
)
"
    
    # 更新数据路径
    DATA_PATH="$DEBUG_DATA_PATH"
    echo "Debug Train 数据路径: $DATA_PATH"
    echo ""
    
    # 生成 debug val 数据子集（不需要 prefix_actions，只需要随机采样）
    DEBUG_VAL_OUTPUT_PATH="${DEBUG_DATA_DIR}/debug_val_subset_${DEBUG_VAL_MAX_SAMPLES}.parquet"
    
    echo "生成 Debug Val 数据子集 (max_samples=$DEBUG_VAL_MAX_SAMPLES)..."
    python3 -c "
import sys
sys.path.insert(0, '/Data/wyh/verl/examples/sglang_multiturn/my_exp/rl/debug')
from debug_data_utils import create_debug_val_subset
create_debug_val_subset(
    '$DEBUG_VAL_DATA_PATH',
    '$DEBUG_VAL_OUTPUT_PATH',
    $DEBUG_VAL_MAX_SAMPLES
)
"
    
    # 设置 val 数据路径
    VAL_DATA_PATH="$DEBUG_VAL_OUTPUT_PATH"
    echo "Debug Val 数据路径: $VAL_DATA_PATH"
    echo ""
fi

# 实验名称
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME="textcraft_0.6b_debug_${TIMESTAMP}"

# 日志目录
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_${TIMESTAMP}.log"

# ==================== 验证GPU配置 ====================
GPU_COUNT=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
if [ "$GPU_COUNT" -ne "$NUM_GPUS" ]; then
    echo "错误: GPU数量不匹配！"
    exit 1
fi

# ==================== 打印配置 ====================
echo "================================================================================" | tee "$LOG_FILE"
echo "  TextCraft GRPO训练 - Prefix RL (Validated) - Qwen3-0.6B DEBUG" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 打印模式信息
if [ "$DEBUG_MODE" = "1" ]; then
    echo "【DEBUG MODE - 0.6B 单卡测试】" | tee -a "$LOG_FILE"
    echo "  Debug 模式: 启用" | tee -a "$LOG_FILE"
    if [ "$DEBUG_PREFLIGHT_ONLY" = "1" ]; then
        echo "  Preflight 模式: 启用 (仅验证链路，不完整训练)" | tee -a "$LOG_FILE"
    fi
    echo "  Debug 样本数: $DEBUG_MAX_SAMPLES" | tee -a "$LOG_FILE"
    echo "  Debug 数据文件: $DATA_PATH" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
fi

echo "【模型和数据】" | tee -a "$LOG_FILE"
echo "  模型路径: $MODEL_PATH" | tee -a "$LOG_FILE"
echo "  训练数据: $DATA_PATH" | tee -a "$LOG_FILE"
echo "  输出目录: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "【关键设计】" | tee -a "$LOG_FILE"
echo "  - 使用 validated prefix 数据" | tee -a "$LOG_FILE"
echo "  - prefix_actions 通过 extra_info 传递" | tee -a "$LOG_FILE"
echo "  - continuation_messages 仅作为 debug/reference" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ==================== 检查环境 ====================
echo "检查TextCraft服务器..." | tee -a "$LOG_FILE"
SERVER_RESPONSE=$(curl -s "$TEXTCRAFT_SERVER/" 2>&1)
if [[ "$SERVER_RESPONSE" == *"TextCraft"* ]]; then
    echo "✓ TextCraft服务器正常运行" | tee -a "$LOG_FILE"
else
    echo "警告: TextCraft服务器未运行！" | tee -a "$LOG_FILE"
    echo "请先启动服务器" | tee -a "$LOG_FILE"
    exit 1
fi
echo "" | tee -a "$LOG_FILE"

# ==================== 切换到工作目录 ====================
cd /Data/wyh/verl
source ~/miniconda3/bin/activate verl

echo "激活verl环境" | tee -a "$LOG_FILE"
echo "Python版本: $(python --version)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ==================== 启用 debug 日志级别 ====================
export VERL_LOGGING_LEVEL=DEBUG
export VERL_DEBUG_MODE=1
echo "已启用 VERL_LOGGING_LEVEL=DEBUG" | tee -a "$LOG_FILE"
echo "已启用 VERL_DEBUG_MODE=1" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ==================== Preflight 模式 ====================
if [ "$DEBUG_PREFLIGHT_ONLY" = "1" ]; then
    echo "============================================" | tee -a "$LOG_FILE"
    echo "  运行 Preflight Smoke Test" | tee -a "$LOG_FILE"
    echo "============================================" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    
    # 运行 preflight 脚本
    python3 /Data/wyh/verl/examples/sglang_multiturn/my_exp/rl/debug/preflight_test.py \
        --data_path "$DATA_PATH" \
        --model_path "$MODEL_PATH" \
        --textcraft_server "$TEXTCRAFT_SERVER" \
        2>&1 | tee -a "$LOG_FILE"
    
    EXIT_CODE=${PIPESTATUS[0]}
    if [ $EXIT_CODE -ne 0 ]; then
        echo "" | tee -a "$LOG_FILE"
        echo "============================================" | tee -a "$LOG_FILE"
        echo "  ❌ Preflight 测试失败 (exit code: $EXIT_CODE)" | tee -a "$LOG_FILE"
        echo "============================================" | tee -a "$LOG_FILE"
        exit $EXIT_CODE
    fi
    
    echo "" | tee -a "$LOG_FILE"
    echo "============================================" | tee -a "$LOG_FILE"
    echo "  ✓ Preflight 测试通过" | tee -a "$LOG_FILE"
    echo "============================================" | tee -a "$LOG_FILE"
    exit 0
fi

# ==================== 启动训练 ====================
echo "开始GRPO训练 (Prefix RL - Validated - 0.6B DEBUG)..." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Ray配置
export RAY_DEDUP_LOGS=0
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# 日志级别
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_CONFIGURE_LOGGING=0
export PYTHONWARNINGS=ignore
export RAY_DEDUP_LOGS=1
export WANDB_MODE=offline  # 禁用 wandb，避免收尾异常

python3 -m verl.trainer.main_ppo \
    --config-path='/Data/wyh/verl/examples/sglang_multiturn/config' \
    --config-name='textcraft_grpo_train' \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_PATH \
    data.val_files=$VAL_DATA_PATH \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=2 \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.dataloader_num_workers=0 \
    '+data.apply_chat_template_kwargs.enable_thinking=True' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.enable_activation_offload=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
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
    trainer.project_name=textcraft_grpo_validated \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.resume_mode=disable \
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
echo "训练完成！" | tee -a "$LOG_FILE"
echo "日志文件: $LOG_FILE" | tee -a "$LOG_FILE"
echo "检查点目录: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
