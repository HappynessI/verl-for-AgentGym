#!/bin/bash
set -eo pipefail

# ==================== TextCraft Prefix RL (Validated) 训练脚本 ====================
# 
# 第一版主实验配置：
# - 数据: textcraft_fixed_ratio_validated.jsonl 转换后的 parquet
# - 策略: fixed_ratio_0.4
# - 样本: 只使用 validated 类别
# 
# 关键设计：
# - prefix_actions 通过 extra_info.interaction_kwargs 传递给 interaction 层
# - interaction 层 replay prefix_actions 后，student 从 cut state 继续 rollout
# - continuation_messages 仅作为 debug/reference，不参与训练监督
#
# Debug 模式支持：
# - DEBUG_MODE=1 启用 debug 模式，使用小规模配置
# - DEBUG_PREFLIGHT_ONLY=1 仅运行 preflight smoke test
# - 支持 --debug 和 --preflight 命令行参数

# ==================== 配置参数 ====================

# -------------------- 模型和数据 --------------------
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)
MODEL_PATH=${MODEL_PATH:-"/Data/public/Qwen3-1.7B"}
ENABLE_GRADIENT_CHECKPOINTING=${ENABLE_GRADIENT_CHECKPOINTING:-false}
ENABLE_ACTIVATION_OFFLOAD=${ENABLE_ACTIVATION_OFFLOAD:-false}
# 使用 repo 内重建并审计通过的主实验数据
DATA_PATH=${DATA_PATH:-"${REPO_ROOT}/data/textcraft/new_prefix_rl/stage7_audit_release/textcraft_prefix_main_train_step200.audited.parquet"}

# ==================== Debug 模��配置 ====================
DEBUG_MODE=${DEBUG_MODE:-0}
DEBUG_MAX_SAMPLES=${DEBUG_MAX_SAMPLES:-16}
DEBUG_PREFLIGHT_ONLY=${DEBUG_PREFLIGHT_ONLY:-0}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            DEBUG_MODE=1
            shift
            ;;
        --preflight)
            DEBUG_PREFLIGHT_ONLY=1
            DEBUG_MODE=1
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# -------------------- GPU配置 --------------------
GPU_IDS=${GPU_IDS:-"0,2"}  # 使用的GPU编号
NUM_GPUS=${NUM_GPUS:-2}      # GPU数量
SAVE_FREQ_IS_SET=${SAVE_FREQ+x}
TEST_FREQ_IS_SET=${TEST_FREQ+x}

# -------------------- 训练超参数（与 grpo_train 保持一致）--------------------
NUM_EPOCHS=${NUM_EPOCHS:-30}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
PPO_EPOCHS=${PPO_EPOCHS:-2}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
SAVE_FREQ=${SAVE_FREQ:-200}
TEST_FREQ=${TEST_FREQ:--1}

# -------------------- Prefix Optimization 配置 --------------------
# 主实验开关：
# - optimize_prefix_tokens=true: 主实验默认开启
# - 如需 continuation-only baseline，请显式传 OPTIMIZE_PREFIX_TOKENS=false
OPTIMIZE_PREFIX_TOKENS=${OPTIMIZE_PREFIX_TOKENS:-true}
# Prefix loss 权重
PREFIX_LOSS_WEIGHT=${PREFIX_LOSS_WEIGHT:-1.0}
PREFIX_LOSS_MODE=${PREFIX_LOSS_MODE:-split}
PREFIX_ADVANTAGE_MODE=${PREFIX_ADVANTAGE_MODE:-constant}
PREFIX_ADVANTAGE_CONSTANT=${PREFIX_ADVANTAGE_CONSTANT:-1.0}
PREFIX_CLIP_RATIO=${PREFIX_CLIP_RATIO:-null}
PREFIX_CLIP_RATIO_LOW=${PREFIX_CLIP_RATIO_LOW:-null}
PREFIX_CLIP_RATIO_HIGH=${PREFIX_CLIP_RATIO_HIGH:-null}
PREFIX_CLIP_RATIO_C=${PREFIX_CLIP_RATIO_C:-null}

# -------------------- KL 配置（主实验不使用 KL）--------------------
# 明确设置为 false，确保不使用 KL / reference
USE_KL_LOSS=${USE_KL_LOSS:-false}                   

# -------------------- Rollout配置 --------------------
ROLLOUT_N=${ROLLOUT_N:-8}                   # 每个prompt采样数量
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL:-0.85}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1024}
ENFORCE_EAGER=${ENFORCE_EAGER:-true}        
FREE_CACHE_ENGINE=${FREE_CACHE_ENGINE:-true}

# -------------------- vLLM Validation配置 --------------------
VAL_TEMPERATURE=${VAL_TEMPERATURE:-1.0}
VAL_TOP_P=${VAL_TOP_P:-1.0}
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-false}
VAL_N=${VAL_N:-1}

# -------------------- Token长度限制 --------------------
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}          
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-10240}
ROLLOUT_RESPONSE_LENGTH=${ROLLOUT_RESPONSE_LENGTH:-10240}
ROLLOUT_MAX_TOKENS=${ROLLOUT_MAX_TOKENS:-512}
ROLLOUT_PROMPT_LENGTH=${ROLLOUT_PROMPT_LENGTH:-2048}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
PPO_MAX_TOKEN_LEN=${PPO_MAX_TOKEN_LEN:-16384}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
MAX_ASSISTANT_TURNS=${MAX_ASSISTANT_TURNS:-30}
MAX_USER_TURNS=${MAX_USER_TURNS:-30}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-8}
METRICS_CSV_FREQ=${METRICS_CSV_FREQ:-50}
METRICS_CSV_FILENAME=${METRICS_CSV_FILENAME:-training_metrics.csv}

# -------------------- 环境服务器 --------------------
TEXTCRAFT_SERVER=${TEXTCRAFT_SERVER:-"http://127.0.0.1:36001"}

# ==================== Debug 模式覆盖配置 ====================
if [ "$DEBUG_MODE" = "1" ]; then
    echo ""
    echo "============================================"
    echo "  ⚠️  DEBUG MODE ENABLED ⚠️"
    echo "  使用小规模配置快速验证训练链路"
    echo "============================================"
    echo ""
    
    # Debug 模式专用目录
    OUTPUT_DIR="${OUTPUT_DIR:-/Data/wyh/datasets/Verl-Data/outputs/textcraft_grpo_validated}_debug"
    
    # 覆盖为小规模配置
    NUM_EPOCHS=1
    TRAIN_BATCH_SIZE=4
    PPO_MINI_BATCH_SIZE=4
    MICRO_BATCH_SIZE=2
    ROLLOUT_N=2
    if [ -z "$SAVE_FREQ_IS_SET" ]; then
        SAVE_FREQ=1
    fi
    if [ -z "$TEST_FREQ_IS_SET" ]; then
        TEST_FREQ=1
    fi
    GPU_MEMORY_UTIL=0.5
    MAX_NUM_SEQS=16
    MAX_RESPONSE_LENGTH=512
    ROLLOUT_RESPONSE_LENGTH=512
    MAX_NUM_BATCHED_TOKENS=4096
    
    echo "Debug 配置:"
    echo "  NUM_EPOCHS=$NUM_EPOCHS"
    echo "  TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE"
    echo "  MICRO_BATCH_SIZE=$MICRO_BATCH_SIZE"
    echo "  ROLLOUT_N=$ROLLOUT_N"
    echo "  GPU_MEMORY_UTIL=$GPU_MEMORY_UTIL"
    echo "  MAX_NUM_SEQS=$MAX_NUM_SEQS"
    echo "  MAX_RESPONSE_LENGTH=$MAX_RESPONSE_LENGTH"
    echo "  MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS"
    echo ""
    
    # 生成 debug 数据子集
    DEBUG_DATA_DIR="/Data/wyh/datasets/Verl-Data/train/textcraft/prefix-rl/debug"
    mkdir -p "$DEBUG_DATA_DIR"
    DEBUG_DATA_PATH="${DEBUG_DATA_DIR}/debug_subset_${DEBUG_MAX_SAMPLES}.parquet"
    
    echo "生成 Debug 数据子集 (max_samples=$DEBUG_MAX_SAMPLES)..."
    cd "$REPO_ROOT"
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
    echo "Debug 数据路径: $DATA_PATH"
    echo ""
fi

# 默认输出目录（如果未定义）
OUTPUT_DIR=${OUTPUT_DIR:-"/Data/wyh/datasets/Verl-Data/outputs/textcraft_grpo_validated"}

# 实验名称
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME="textcraft_grpo_validated_${TIMESTAMP}"

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
echo "  TextCraft GRPO训练 - Prefix RL (Validated) - Qwen3-1.7B" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 打印模式信息
if [ "$DEBUG_MODE" = "1" ]; then
    echo "【DEBUG MODE】" | tee -a "$LOG_FILE"
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
echo "  gradient_checkpointing: $ENABLE_GRADIENT_CHECKPOINTING" | tee -a "$LOG_FILE"
echo "  activation_offload: $ENABLE_ACTIVATION_OFFLOAD" | tee -a "$LOG_FILE"
echo "  训练数据: $DATA_PATH" | tee -a "$LOG_FILE"
echo "  输出目录: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "【关键设计】" | tee -a "$LOG_FILE"
echo "  - 使用 validated prefix 数据" | tee -a "$LOG_FILE"
echo "  - prefix_actions 通过 extra_info 传递" | tee -a "$LOG_FILE"
echo "  - continuation_messages 仅作为 debug/reference" | tee -a "$LOG_FILE"
echo "  - optimize_prefix_tokens: $OPTIMIZE_PREFIX_TOKENS" | tee -a "$LOG_FILE"
echo "  - prefix_loss_weight: $PREFIX_LOSS_WEIGHT" | tee -a "$LOG_FILE"
echo "  - prefix_loss_mode: $PREFIX_LOSS_MODE" | tee -a "$LOG_FILE"
echo "  - prefix_advantage_mode: $PREFIX_ADVANTAGE_MODE" | tee -a "$LOG_FILE"
echo "  - prefix_advantage_constant: $PREFIX_ADVANTAGE_CONSTANT" | tee -a "$LOG_FILE"
echo "  - prefix_clip_ratio: $PREFIX_CLIP_RATIO" | tee -a "$LOG_FILE"
echo "  - prefix_clip_ratio_low: $PREFIX_CLIP_RATIO_LOW" | tee -a "$LOG_FILE"
echo "  - prefix_clip_ratio_high: $PREFIX_CLIP_RATIO_HIGH" | tee -a "$LOG_FILE"
echo "  - prefix_clip_ratio_c: $PREFIX_CLIP_RATIO_C" | tee -a "$LOG_FILE"
echo "  - use_kl_loss: $USE_KL_LOSS (主实验关闭)" | tee -a "$LOG_FILE"
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

# ==================== 数据集校验 ====================
echo "校验训练数据..." | tee -a "$LOG_FILE"
export DATA_PATH
export OPTIMIZE_PREFIX_TOKENS
python3 - <<'PY' 2>&1 | tee -a "$LOG_FILE"
import os
import sys

import pandas as pd

data_path = os.environ["DATA_PATH"]
optimize_prefix_tokens = os.environ["OPTIMIZE_PREFIX_TOKENS"].lower() == "true"

if not os.path.exists(data_path):
    raise FileNotFoundError(f"训练数据不存在: {data_path}")

df = pd.read_parquet(data_path)
columns = list(df.columns)
print(f"  样本数: {len(df)}")
print(f"  列名: {columns}")

# 这个主实验脚本默认要求和 smoke test 一致的数据结构。
required_columns = ["data_source", "prompt", "reward_model", "extra_info"]
required_prefix_columns = ["assistant_prefix_old_log_probs", "prefix_token_count", "prefix_mask", "assistant_prefix_span"]
missing = [col for col in required_columns + required_prefix_columns if col not in columns]
if missing:
    raise ValueError(
        "当前主实验脚本要求使用包含 prefix sidecar 列的 parquet。"
        f" 缺失列: {missing}. data_path={data_path}"
    )

first_row = df.iloc[0].to_dict() if len(df) else {}
extra_info = first_row.get("extra_info", {})
interaction_kwargs = extra_info.get("interaction_kwargs", {}) if isinstance(extra_info, dict) else {}
prefix_actions = interaction_kwargs.get("prefix_actions", None)
has_prefix_actions = prefix_actions is not None and hasattr(prefix_actions, "__len__") and len(prefix_actions) > 0

if not has_prefix_actions:
    raise ValueError(
        "extra_info.interaction_kwargs.prefix_actions 缺失或为空，"
        "interaction replay 无法工作。"
    )

prefix_token_count = df["prefix_token_count"]
if (prefix_token_count <= 0).any():
    bad = int((prefix_token_count <= 0).sum())
    raise ValueError(f"prefix_token_count 中存在非正样本: {bad}")

mode = "prefix optimization" if optimize_prefix_tokens else "baseline"
print(f"  模式: {mode}")
print(f"  prefix_token_count: min={prefix_token_count.min()}, max={prefix_token_count.max()}, mean={prefix_token_count.mean():.1f}")
print(f"  样例 prefix_actions 数量: {len(prefix_actions)}")
print("  数据校验通过")
PY
echo "" | tee -a "$LOG_FILE"

# ==================== 启用 debug 日志级别 ====================
if [ "$DEBUG_MODE" = "1" ]; then
    export VERL_LOGGING_LEVEL=DEBUG
    echo "已启用 VERL_LOGGING_LEVEL=DEBUG" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
fi

# ==================== Preflight 模式 ====================
if [ "$DEBUG_PREFLIGHT_ONLY" = "1" ]; then
    echo "============================================" | tee -a "$LOG_FILE"
    echo "  运行 Preflight Smoke Test" | tee -a "$LOG_FILE"
    echo "============================================" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    
    # 运行 preflight 脚本
    python3 "$REPO_ROOT/examples/sglang_multiturn/my_exp/rl/debug/preflight_test.py" \
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
echo "开始GRPO训练 (Prefix RL - Validated)..." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Ray配置
export RAY_DEDUP_LOGS=0
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# 日志级别
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_CONFIGURE_LOGGING=0
export PYTHONWARNINGS=ignore
export RAY_DEDUP_LOGS=1

python3 -m verl.trainer.main_ppo \
    --config-path="$REPO_ROOT/examples/sglang_multiturn/config" \
    --config-name='textcraft_grpo_train' \
    hydra.searchpath=[file://${REPO_ROOT}/verl/trainer/config,file://${REPO_ROOT}/examples/sglang_multiturn/config] \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_PATH \
    data.val_files=$DATA_PATH \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=4 \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    '+data.apply_chat_template_kwargs.enable_thinking=True' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=$ENABLE_GRADIENT_CHECKPOINTING \
    actor_rollout_ref.model.enable_activation_offload=$ENABLE_ACTIVATION_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.actor.ppo_epochs=$PPO_EPOCHS \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN \
    actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.top_p=$TOP_P \
    actor_rollout_ref.rollout.prompt_length=$ROLLOUT_PROMPT_LENGTH \
    actor_rollout_ref.rollout.response_length=$ROLLOUT_RESPONSE_LENGTH \
    actor_rollout_ref.rollout.max_tokens=$ROLLOUT_MAX_TOKENS \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
    actor_rollout_ref.rollout.max_num_seqs=$MAX_NUM_SEQS \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTIL \
    actor_rollout_ref.rollout.enforce_eager=$ENFORCE_EAGER \
    actor_rollout_ref.rollout.free_cache_engine=$FREE_CACHE_ENGINE \
    actor_rollout_ref.rollout.val_kwargs.temperature=$VAL_TEMPERATURE \
    actor_rollout_ref.rollout.val_kwargs.top_p=$VAL_TOP_P \
    actor_rollout_ref.rollout.val_kwargs.do_sample=$VAL_DO_SAMPLE \
    actor_rollout_ref.rollout.val_kwargs.n=$VAL_N \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$MAX_ASSISTANT_TURNS \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$MAX_USER_TURNS \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="$REPO_ROOT/examples/sglang_multiturn/config/interaction_config/textcraft_interaction.yaml" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    ray_kwargs.ray_init.num_cpus=$RAY_NUM_CPUS \
    trainer.total_epochs=$NUM_EPOCHS \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.val_before_train=false \
    trainer.default_local_dir=$OUTPUT_DIR \
    +trainer.metrics_csv_freq=$METRICS_CSV_FREQ \
    +trainer.metrics_csv_filename="$METRICS_CSV_FILENAME" \
    trainer.project_name=textcraft_grpo_validated \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.resume_mode=disable \
    algorithm.optimize_prefix_tokens=$OPTIMIZE_PREFIX_TOKENS \
    algorithm.prefix_loss_weight=$PREFIX_LOSS_WEIGHT \
    algorithm.prefix_loss_mode=$PREFIX_LOSS_MODE \
    algorithm.prefix_advantage_mode=$PREFIX_ADVANTAGE_MODE \
    algorithm.prefix_advantage_constant=$PREFIX_ADVANTAGE_CONSTANT \
    actor_rollout_ref.actor.prefix_clip_ratio=$PREFIX_CLIP_RATIO \
    actor_rollout_ref.actor.prefix_clip_ratio_low=$PREFIX_CLIP_RATIO_LOW \
    actor_rollout_ref.actor.prefix_clip_ratio_high=$PREFIX_CLIP_RATIO_HIGH \
    actor_rollout_ref.actor.prefix_clip_ratio_c=$PREFIX_CLIP_RATIO_C \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo "" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
if [ $EXIT_CODE -eq 0 ]; then
    echo "训练完成！" | tee -a "$LOG_FILE"
else
    echo "训练失败！(exit code: $EXIT_CODE)" | tee -a "$LOG_FILE"
fi
echo "日志文件: $LOG_FILE" | tee -a "$LOG_FILE"
echo "检查点目录: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
exit $EXIT_CODE
