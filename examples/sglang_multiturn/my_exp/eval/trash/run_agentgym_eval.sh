#!/bin/bash
# 统一的AgentGym环境评估启动脚本
# 支持切换不同环境：webshop, babyai, alfworld, sciworld, sqlgym, textcraft, searchqa

set -x

# ========== 环境选择 ==========
# 必填参数：指定要评估的环境
ENV=${ENV:-""}

# ========== 模型和数据配置 ==========
MODEL_PATH=${MODEL_PATH:-"/Data/public/Qwen3-8B"}
DATA_PATH=${DATA_PATH:-""}  # 必须指定
OUTPUT_BASE_DIR=${OUTPUT_BASE_DIR:-"/Data/wyh/datasets/Verl-Data/outputs"}
GPU_ID=${GPU_ID:-6}

# ========== 环境服务器配置 ==========
# 如果不指定，使用默认端口
ENV_SERVER=${ENV_SERVER:-""}

# ========== 评估参数 ==========
MAX_ROUNDS=${MAX_ROUNDS:-30}
MAX_SAMPLES=${MAX_SAMPLES:-""}  # 空表示全部样本
MAX_LENGTH=${MAX_LENGTH:-4096}

# ========== 环境默认端口配置 ==========
declare -A DEFAULT_PORTS
DEFAULT_PORTS=(
    ["webshop"]="36003"
    ["babyai"]="36001"
    ["alfworld"]="36002"
    ["sciworld"]="36004"
    ["sqlgym"]="36005"
    ["textcraft"]="36006"
    ["searchqa"]="36007"
)

# ========== 环境默认数据路径 ==========
declare -A DEFAULT_DATA_PATHS
DEFAULT_DATA_PATHS=(
    ["webshop"]="/Data/wyh/datasets/Verl-Data/webshop/train.parquet"
    ["babyai"]="/Data/wyh/datasets/Verl-Data/babyai/train.parquet"
    ["alfworld"]="/Data/wyh/datasets/Verl-Data/alfworld/train.parquet"
    ["sciworld"]="/Data/wyh/datasets/Verl-Data/sciworld/train.parquet"
    ["sqlgym"]="/Data/wyh/datasets/Verl-Data/sqlgym/train.parquet"
    ["textcraft"]="/Data/wyh/datasets/Verl-Data/textcraft/train.parquet"
    ["searchqa"]="/Data/wyh/datasets/Verl-Data/searchqa/train.parquet"
)

# ========== 参数验证 ==========
if [ -z "$ENV" ]; then
    echo "ERROR: Environment not specified!"
    echo "Usage: ENV=<env_name> bash $0"
    echo "Available environments: webshop, babyai, alfworld, sciworld, sqlgym, textcraft, searchqa"
    echo ""
    echo "Examples:"
    echo "  ENV=webshop bash $0"
    echo "  ENV=babyai MAX_SAMPLES=10 bash $0"
    echo "  ENV=alfworld GPU_ID=7 bash $0"
    exit 1
fi

if [[ ! " ${!DEFAULT_PORTS[@]} " =~ " ${ENV} " ]]; then
    echo "ERROR: Unknown environment '$ENV'"
    echo "Available environments: ${!DEFAULT_PORTS[@]}"
    exit 1
fi

# 设置默认数据路径（如果未指定）
if [ -z "$DATA_PATH" ]; then
    DATA_PATH="${DEFAULT_DATA_PATHS[$ENV]}"
    echo "Using default data path: $DATA_PATH"
fi

# 设置默认服务器地址（如果未指定）
if [ -z "$ENV_SERVER" ]; then
    ENV_SERVER="http://127.0.0.1:${DEFAULT_PORTS[$ENV]}"
    echo "Using default server: $ENV_SERVER"
fi

# 检查数据文件是否存在
if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Data file not found: $DATA_PATH"
    echo "Please specify DATA_PATH or create the data file first"
    exit 1
fi

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "Setting CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# ========== 生成日志文件路径 ==========
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="$OUTPUT_BASE_DIR/${ENV}_eval"
LOG_DIR="$OUTPUT_DIR/logs"
LOG_FILE="$LOG_DIR/eval_${TIMESTAMP}.log"

# 创建输出和日志目录
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

# ========== 检查环境服务器 ==========
echo "Checking $ENV server at $ENV_SERVER..."
if ! curl -s $ENV_SERVER/ > /dev/null; then
    echo "ERROR: $ENV server is not running!"
    echo "Please start it first. Example commands:"
    case $ENV in
        webshop)
            echo "  cd /Data/wyh/AgentGym-RL/AgentGym/agentenv-webshop"
            echo "  python -m uvicorn agentenv_webshop:app --host 0.0.0.0 --port 36003"
            ;;
        babyai)
            echo "  cd /Data/wyh/AgentGym-RL/AgentGym/agentenv-babyai"
            echo "  python -m uvicorn agentenv_babyai.server:app --host 0.0.0.0 --port 36001"
            ;;
        alfworld)
            echo "  cd /Data/wyh/AgentGym-RL/AgentGym/agentenv-alfworld"
            echo "  python -m uvicorn agentenv_alfworld.server:app --host 0.0.0.0 --port 36002"
            ;;
        sciworld)
            echo "  cd /Data/wyh/AgentGym-RL/AgentGym/agentenv-sciworld"
            echo "  python -m uvicorn agentenv_sciworld.server:app --host 0.0.0.0 --port 36004"
            ;;
        sqlgym)
            echo "  cd /Data/wyh/AgentGym-RL/AgentGym/agentenv-sqlgym"
            echo "  python -m uvicorn agentenv_sqlgym.server:app --host 0.0.0.0 --port 36005"
            ;;
        textcraft)
            echo "  cd /Data/wyh/AgentGym-RL/AgentGym/agentenv-textcraft"
            echo "  python -m uvicorn agentenv_textcraft.server:app --host 0.0.0.0 --port 36006"
            ;;
        searchqa)
            echo "  cd /Data/wyh/AgentGym-RL/AgentGym/agentenv-searchqa"
            echo "  python -m uvicorn agentenv_searchqa.server:app --host 0.0.0.0 --port 36007"
            ;;
    esac
    exit 1
fi
echo "✓ $ENV server is running"

echo "========================================"
echo "AgentGym $ENV Evaluation"
echo "========================================"
echo "Environment: $ENV"
echo "Model: $MODEL_PATH"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "Log File: $LOG_FILE"
echo "Server: $ENV_SERVER"
echo "GPU ID: $GPU_ID"
echo "Max Rounds: $MAX_ROUNDS"
if [ -n "$MAX_SAMPLES" ]; then
    echo "Max Samples: $MAX_SAMPLES"
else
    echo "Max Samples: All"
fi
echo "========================================"
echo ""

# ========== 构建评估命令 ==========
CMD="python3 examples/sglang_multiturn/eval_agentgym_environments.py \
    --env=$ENV \
    --model_path=$MODEL_PATH \
    --data_path=$DATA_PATH \
    --output_dir=$OUTPUT_BASE_DIR \
    --env_server=$ENV_SERVER \
    --max_rounds=$MAX_ROUNDS \
    --max_length=$MAX_LENGTH"

# 添加max_samples参数（如果指定）
if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max_samples=$MAX_SAMPLES"
fi

# ========== 运行评估 ==========
eval $CMD 2>&1 | tee "$LOG_FILE"

# 捕获pipeline的退出状态
PIPELINE_STATUS=${PIPESTATUS[0]}

echo ""
echo "========================================"
if [ $PIPELINE_STATUS -eq 0 ]; then
    echo "Evaluation Complete!"
else
    echo "Evaluation Failed with exit code: $PIPELINE_STATUS"
fi
echo "========================================"
echo "Results saved to: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"

# 退出时使用python命令的状态码
exit $PIPELINE_STATUS

