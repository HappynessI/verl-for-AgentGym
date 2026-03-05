# ============================================================
# TextCraft RL 训练环境 Docker 镜像
# 职责：运行 SFT / GRPO / Prefix-RL 训练
# 基于 vllm/vllm-openai:v0.11.0 (CUDA 12.8 + PyTorch 2.8 + vLLM 0.11.0)
# ============================================================

FROM vllm/vllm-openai:v0.11.0

LABEL maintainer="wyh"
LABEL description="verl + vLLM RL Training Environment for TextCraft experiments"

ENV DEBIAN_FRONTEND=noninteractive
ENV MAX_JOBS=32
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV PIP_ROOT_USER_ACTION=ignore
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    numactl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# cuDNN（某些 FSDP 操作依赖）
RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get -y install cudnn && \
    rm cuda-keyring_1.1-1_all.deb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# ============================================================
# Python 依赖（从 requirements.txt 统一安装，避免重复定义）
# ============================================================
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# ============================================================
# verl 安装策略：
#   - 默认：启动时从挂载的本地目录安装（开发模式，修改代码即生效）
#   - 回退：若未挂载，从 GitHub 安装固定版本
# ============================================================
RUN cat > /entrypoint.sh << 'ENTRYPOINT_EOF'
#!/bin/bash
set -e

# 安装 verl
if [ -d "/workspace/verl" ]; then
    echo "[entrypoint] Installing verl from mounted directory (editable mode)..."
    cd /workspace/verl && pip install -e . --no-deps -q
else
    echo "[entrypoint] verl not mounted, installing from GitHub..."
    pip install --no-cache-dir git+https://github.com/volcengine/verl.git -q
fi

exec "$@"
ENTRYPOINT_EOF
RUN chmod +x /entrypoint.sh

# CUDA 环境变量
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 工作目录结构
RUN mkdir -p /workspace/data /workspace/outputs /workspace/checkpoints

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
