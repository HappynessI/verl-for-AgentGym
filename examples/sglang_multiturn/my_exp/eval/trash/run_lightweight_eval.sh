#!/bin/bash
# Lightweight evaluation script for Webshop
# 参考AgentGym-RL设计，只加载1份模型，不依赖PPO框架

set -x

# Configuration
MODEL_PATH=${MODEL_PATH:-"/Data/public/Qwen3-8B"}
DATA_PATH=${DATA_PATH:-"/Data/wyh/datasets/Verl-Data/webshop/train.parquet"}
OUTPUT_DIR=${OUTPUT_DIR:-"/Data/wyh/datasets/Verl-Data/outputs/webshop_eval_lightweight"}
WEBSHOP_SERVER=${WEBSHOP_SERVER:-"http://127.0.0.1:36001"}
GPU_ID=${GPU_ID:-2}
MAX_ROUNDS=${MAX_ROUNDS:-25}
MAX_SAMPLES=${MAX_SAMPLES:-""}  # Empty means all samples
TEMPERATURE=${TEMPERATURE:-0.6}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}

# Set CUDA_VISIBLE_DEVICES before any Python imports
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "Setting CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$OUTPUT_DIR/logs"
LOG_FILE="$LOG_DIR/eval_${TIMESTAMP}.log"

# Create output and log directories
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

# Check Webshop server
echo "Checking Webshop server at $WEBSHOP_SERVER..."
if ! curl -s $WEBSHOP_SERVER/ > /dev/null; then
    echo "ERROR: Webshop server is not running!"
    echo "Please start it first:"
    echo "  conda activate webshop"
    echo "  cd /Data/wyh/AgentGym-RL/AgentGym/agentenv-webshop"
    echo "  python -m uvicorn agentenv_webshop:app --host 0.0.0.0 --port 36003"
    exit 1
fi
echo "✓ Webshop server is running"

echo "================================"
echo "Webshop Lightweight Evaluation"
echo "================================"
echo "Model: $MODEL_PATH"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "Log File: $LOG_FILE"
echo "Webshop Server: $WEBSHOP_SERVER"
echo "GPU ID: $GPU_ID"
echo "Max Rounds: $MAX_ROUNDS"
if [ -n "$MAX_SAMPLES" ]; then
    echo "Max Samples: $MAX_SAMPLES"
else
    echo "Max Samples: All"
fi
echo "Temperature: $TEMPERATURE"
echo "Max New Tokens: $MAX_NEW_TOKENS"
echo "================================"
echo ""

# Build command (GPU is controlled by CUDA_VISIBLE_DEVICES environment variable)
# 使用脚本所在目录的eval_webshop_lightweight.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMD="python3 $SCRIPT_DIR/eval_webshop_lightweight.py \
    --model_path=$MODEL_PATH \
    --data_path=$DATA_PATH \
    --output_dir=$OUTPUT_DIR \
    --webshop_server=$WEBSHOP_SERVER \
    --max_rounds=$MAX_ROUNDS"

# Add max_samples if specified
if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max_samples=$MAX_SAMPLES"
fi

# Run evaluation (redirect stdout and stderr to log file while still displaying in terminal)
eval $CMD 2>&1 | tee "$LOG_FILE"

# Capture the exit status of the pipeline
PIPELINE_STATUS=${PIPESTATUS[0]}

echo ""
echo "================================"
if [ $PIPELINE_STATUS -eq 0 ]; then
    echo "Evaluation Complete!"
else
    echo "Evaluation Failed with exit code: $PIPELINE_STATUS"
fi
echo "================================"
echo "Results saved to: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"

# Exit with the same status as the python command
exit $PIPELINE_STATUS

