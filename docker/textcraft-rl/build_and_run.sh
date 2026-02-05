#!/bin/bash
# ============================================================
# 一键构建和启动 TextCraft RL Docker 环境
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  TextCraft RL Docker 环境部署"
echo "============================================================"

# 检查 docker 是否安装
if ! command -v docker &> /dev/null; then
    echo "错误: docker 未安装"
    exit 1
fi

# 检查 docker-compose 是否安装
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "错误: docker-compose 未安装"
    exit 1
fi

# 检查 nvidia-docker
if ! docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
    echo "警告: nvidia-docker 可能未正确配置"
    echo "请确保安装了 nvidia-container-toolkit"
fi

echo ""
echo "[1/3] 构建 Docker 镜像..."
echo ""

# 使用 docker compose（新版本）或 docker-compose（旧版本）
if docker compose version &> /dev/null 2>&1; then
    docker compose build
else
    docker-compose build
fi

echo ""
echo "[2/3] 启动容器..."
echo ""

if docker compose version &> /dev/null 2>&1; then
    docker compose up -d
else
    docker-compose up -d
fi

echo ""
echo "[3/3] 验证环境..."
echo ""

# 等待容器启动
sleep 3

# 检查容器状态
if docker ps | grep -q "textcraft-rl"; then
    echo "✓ 容器已启动"
else
    echo "✗ 容器启动失败"
    docker logs textcraft-rl
    exit 1
fi

# 检查 GPU 访问
echo ""
echo "GPU 状态:"
docker exec textcraft-rl nvidia-smi --query-gpu=name,memory.total --format=csv

# 检查 Python 环境
echo ""
echo "Python 环境:"
docker exec textcraft-rl python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
docker exec textcraft-rl python -c "import vllm; print(f'vLLM: {vllm.__version__}')"

echo ""
echo "============================================================"
echo "  部署完成！"
echo "============================================================"
echo ""
echo "进入容器:"
echo "  docker exec -it textcraft-rl bash"
echo ""
echo "运行训练:"
echo "  docker exec -it textcraft-rl bash /workspace/verl/docker/textcraft-rl/run_training.sh"
echo ""
echo "查看日志:"
echo "  docker logs -f textcraft-rl"
echo ""
echo "停止容器:"
echo "  docker-compose down"
echo ""

