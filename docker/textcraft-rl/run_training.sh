#!/bin/bash
# ============================================================
# TextCraft RL 训练启动脚本（在 textcraft-rl 容器内运行）
#
# 使用方法：
#   # GRPO Baseline（默认 Qwen3-1.7B）
#   bash run_training.sh
#
#   # Prefix-RL，指定优势估计器和模型
#   EXPERIMENT=prefix_full MODEL_NAME=Qwen3-4B bash run_training.sh
#
#   # 不交互直接运行（适合批量提交）
#   AUTO_CONFIRM=1 bash run_training.sh
#
# 支持的 EXPERIMENT 类型：
#   grpo_baseline       —— GRPO 基线（ADV_ESTIMATOR=grpo）
#   prefix_full         —— Prefix-RL 全轨迹（ADV_ESTIMATOR=turn_full_trajectory）
#   prefix_guided       —— Prefix-RL 前缀引导（ADV_ESTIMATOR=turn_prefix_guided）
#   prefix_guided_dr    —— Prefix-RL + Dr.GRPO（ADV_ESTIMATOR=turn_prefix_guided_dr）
#   sft                 —— 监督微调（待实现）
# ============================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_step()  { echo -e "\n${BLUE}>>> $*${NC}"; }

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  TextCraft RL 训练环境 (verl + vLLM)${NC}"
echo -e "${GREEN}============================================================${NC}"

# ============================================================
# 实验类型 -> 训练参数映射
# ============================================================
EXPERIMENT=${EXPERIMENT:-grpo_baseline}
MODEL_NAME=${MODEL_NAME:-Qwen3-1.7B}
MODEL_PATH=${MODEL_PATH:-/workspace/models/${MODEL_NAME}}

case "$EXPERIMENT" in
    grpo_baseline)
        ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo}
        ;;
    prefix_full)
        ADV_ESTIMATOR=${ADV_ESTIMATOR:-turn_full_trajectory}
        ;;
    prefix_guided)
        ADV_ESTIMATOR=${ADV_ESTIMATOR:-turn_prefix_guided}
        ;;
    prefix_guided_dr)
        ADV_ESTIMATOR=${ADV_ESTIMATOR:-turn_prefix_guided_dr}
        ;;
    sft)
        log_error "SFT 实验脚本尚未实现，请参考 recipe/wyh_exp/scripts/ 目录"
        exit 1
        ;;
    *)
        log_error "未知的 EXPERIMENT 类型: $EXPERIMENT"
        log_error "支持: grpo_baseline | prefix_full | prefix_guided | prefix_guided_dr | sft"
        exit 1
        ;;
esac

# ============================================================
# 1. 检查 GPU
# ============================================================
log_step "检查 GPU 状态"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
log_info "检测到 ${NUM_GPUS} 个 GPU"

# ============================================================
# 2. 检查 verl 安装
# ============================================================
log_step "检查 verl 安装"
if [ -d "/workspace/verl" ]; then
    cd /workspace/verl
    pip install -e . --no-deps -q
    log_info "verl 已从挂载目录安装（开发模式，代码修改即生效）"
else
    log_error "未找到 /workspace/verl —— 请确认 VERL_CODE_DIR 已正确挂载"
    exit 1
fi

# ============================================================
# 3. 检查训练脚本
# ============================================================
TRAIN_SCRIPT="/workspace/verl/recipe/wyh_exp/scripts/run_test_train.sh"
log_step "检查训练脚本"
if [ ! -f "$TRAIN_SCRIPT" ]; then
    log_error "未找到训练脚本: $TRAIN_SCRIPT"
    exit 1
fi
log_info "训练脚本: $TRAIN_SCRIPT"

# ============================================================
# 4. 检查数据
# ============================================================
log_step "检查训练数据"
DATA_ROOT="${DATASETS_DIR:-/workspace/datasets}/Verl-Data"
TRAIN_DATA="${DATA_ROOT}/train/textcraft/train.parquet"
if [ -f "$TRAIN_DATA" ]; then
    log_info "训练数据: $TRAIN_DATA"
else
    log_warn "未找到训练数据: $TRAIN_DATA"
    log_warn "请确认数据路径是否正确（DATASETS_DIR=${DATASETS_DIR:-/workspace/datasets}）"
fi

# ============================================================
# 5. 检查模型
# ============================================================
log_step "检查模型路径"
if [ -d "$MODEL_PATH" ]; then
    log_info "模型路径: $MODEL_PATH"
else
    log_warn "模型路径不存在: $MODEL_PATH"
    log_warn "请确认模型已挂载（MODELS_DIR 中包含 ${MODEL_NAME}）"
fi

# ============================================================
# 6. 检查 TextCraft 服务器
# ============================================================
log_step "检查 TextCraft 环境服务器"
TEXTCRAFT_SERVER_URL=${TEXTCRAFT_SERVER_URL:-http://127.0.0.1:36001}
if curl -sf "$TEXTCRAFT_SERVER_URL" >/dev/null 2>&1; then
    log_info "TextCraft 服务器已就绪: $TEXTCRAFT_SERVER_URL"
else
    log_warn "TextCraft 服务器未响应: $TEXTCRAFT_SERVER_URL"
    log_warn "请确认 textcraft-server 容器已启动"
fi

# ============================================================
# 检查 wandb
# ============================================================
if [ -n "$WANDB_API_KEY" ]; then
    log_info "WANDB_API_KEY 已设置"
else
    log_warn "WANDB_API_KEY 未设置，日志将保存到本地"
fi

# ============================================================
# 打印实验参数
# ============================================================
echo -e "\n${BLUE}=== 实验配置 ===${NC}"
echo "  EXPERIMENT:       ${EXPERIMENT}"
echo "  ADV_ESTIMATOR:    ${ADV_ESTIMATOR}"
echo "  MODEL_NAME:       ${MODEL_NAME}"
echo "  MODEL_PATH:       ${MODEL_PATH}"
echo "  NUM_EPOCHS:       ${NUM_EPOCHS:-100}"
echo "  TRAIN_BATCH_SIZE: ${TRAIN_BATCH_SIZE:-128}"
echo "  LEARNING_RATE:    ${LEARNING_RATE:-1e-6}"
echo "  ROLLOUT_N:        ${ROLLOUT_N:-8}"
echo "  ENABLE_THINKING:  ${ENABLE_THINKING:-true}"
echo "  ENTROPY_COEFF:    ${ENTROPY_COEFF:-0.0}"
echo ""

# ============================================================
# 确认
# ============================================================
if [ "${AUTO_CONFIRM:-0}" != "1" ]; then
    echo -e "${YELLOW}是否开始训练? [y/N]${NC}"
    read -r confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "训练已取消"
        exit 0
    fi
fi

# ============================================================
# 创建输出目录
# ============================================================
EXP_TAG="${EXPERIMENT}_${MODEL_NAME}_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUTS_DIR:-/workspace/outputs}/${EXP_TAG}"
mkdir -p "${OUTPUT_DIR}/logs"
log_info "输出目录: ${OUTPUT_DIR}"

# ============================================================
# 启动训练（委托给 run_test_train.sh）
# ============================================================
log_step "开始训练"

cd /workspace/verl

# 将参数导出为环境变量，run_test_train.sh 通过 ${VAR:-default} 读取
export ADV_ESTIMATOR
export MODEL_PATH
export NUM_EPOCHS
export TRAIN_BATCH_SIZE
export LEARNING_RATE
export ROLLOUT_N
export ENABLE_THINKING
export ENTROPY_COEFF=${ENTROPY_COEFF:-0.0}   # 默认关闭 entropy bonus
export OUTPUT_DIR
export TEXTCRAFT_SERVER_URL

bash "$TRAIN_SCRIPT" 2>&1 | tee "${OUTPUT_DIR}/logs/train.log"

log_info "训练完成！输出目录: ${OUTPUT_DIR}"
