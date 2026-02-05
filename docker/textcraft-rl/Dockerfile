# ============================================================
# TextCraft RL 训练环境 Docker 配置
# 基于 vllm/vllm-openai:v0.11.0
# CUDA 12.8 + PyTorch 2.8 + vLLM 0.11.0
# ============================================================

FROM vllm/vllm-openai:v0.11.0

LABEL maintainer="wyh"
LABEL description="TextCraft GRPO RL Training Environment"

# 环境变量配置
ENV DEBIAN_FRONTEND=noninteractive
ENV MAX_JOBS=32
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV PIP_ROOT_USER_ACTION=ignore
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV WANDB_MODE=online

# 设置工作目录
WORKDIR /workspace

# 更新系统并安装基础依赖
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

# 安装 pybind11（flash-attn 依赖）
RUN pip install --no-cache-dir pybind11

# 安装 cudnn（某些操作需要）
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get -y install cudnn && \
    rm cuda-keyring_1.1-1_all.deb && \
    apt-get clean

# 升级 pip
RUN pip install --upgrade pip

# ============================================================
# 核心依赖安装
# ============================================================

# 安装 flash-attention
RUN pip install --no-cache-dir flash_attn==2.8.1

# 安装 transformers 和 tokenizers（最新版）
RUN pip install --no-cache-dir --upgrade transformers==4.56.1 tokenizers

# 安装 RL 训练核心依赖
RUN pip install --no-cache-dir \
    accelerate==1.12.0 \
    datasets \
    peft==0.18.0 \
    trl \
    liger-kernel

# 安装分布式训练和日志依赖
RUN pip install --no-cache-dir \
    ray[default]==2.52.1 \
    wandb==0.23.1 \
    hydra-core==1.3.2 \
    codetiming \
    tensordict

# 安装数据处理依赖
RUN pip install --no-cache-dir \
    pandas==2.3.3 \
    numpy \
    scipy \
    pyarrow

# 安装其他工具
RUN pip install --no-cache-dir \
    mathruler \
    pylatexenc \
    qwen_vl_utils \
    blobfile \
    dill \
    hf-transfer

# ============================================================
# 安装 verl
# ============================================================
# 方式1: 从 GitHub 安装（推荐用于生产环境）
# RUN pip install --no-cache-dir git+https://github.com/volcengine/verl.git@v0.7.0

# 方式2: 挂载本地 verl 代码（开发模式，需要在运行时挂载）
# 创建安装脚本，容器启动时执行
RUN echo '#!/bin/bash\n\
if [ -d "/workspace/verl" ]; then\n\
    echo "Installing verl from mounted directory..."\n\
    cd /workspace/verl && pip install -e . --no-deps\n\
else\n\
    echo "Installing verl from GitHub..."\n\
    pip install --no-cache-dir git+https://github.com/volcengine/verl.git\n\
fi\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

# ============================================================
# 环境配置
# ============================================================

# 设置 Python 符号链接
RUN ln -sf /usr/bin/python3 /usr/bin/python 2>/dev/null || true

# 设置 CUDA 环境变量
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 创建数据和输出目录
RUN mkdir -p /workspace/data /workspace/outputs /workspace/checkpoints

# 设置入口点
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]

