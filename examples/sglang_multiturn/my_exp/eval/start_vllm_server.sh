
set -e

# 配置参数
MODEL_PATH="/models/Qwen3-4B-Thinking-2507"
PORT=8000
TP_SIZE=1  
DP_SIZE=8
MAX_MODEL_LEN=32768
GPU_MEM_UTIL=0.95
HOST="0.0.0.0"
LOG_FILE="/agent_distill/logs/vllm_server_$(date +%Y%m%d_%H%M%S).log"

echo "🚀 启动vLLM服务 [$(date)]"
echo "模型路径: $MODEL_PATH"
echo "张量并行: $TP_SIZE"
echo "最大上下文: $MAX_MODEL_LEN"
echo "GPU显存利用率: $GPU_MEM_UTIL"
echo "服务地址: http://$HOST:$PORT"
echo "日志文件: $LOG_FILE"
echo "----------------------------------------"

# 设置CUDA_VISIBLE_DEVICES为所有卡
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 启动vLLM服务 (后台运行)
nohup vllm serve "$MODEL_PATH" \
  --served-model-name "qwen3" \
  --tensor-parallel-size $TP_SIZE \
  --data-parallel-size $DP_SIZE     \
  --max-model-len $MAX_MODEL_LEN \
  --gpu-memory-utilization $GPU_MEM_UTIL \
  --host $HOST \
  --port $PORT \
  --disable-log-requests \
  --max-num-seqs 128 \
  --enable-chunked-prefill \
  --max-log-len 0 \
  > "$LOG_FILE" 2>&1 &

SERVER_PID=$!
