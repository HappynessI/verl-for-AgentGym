# TextCraft GRPO RL 训练 Docker 环境

## 环境配置

| 组件 | 版本 |
|------|------|
| Python | 3.12 |
| PyTorch | 2.8.0 |
| CUDA | 12.8 |
| vLLM | 0.11.0 |
| flash-attn | 2.8.1 |
| transformers | 4.56.1 |
| verl | 0.7.0 |

## 快速开始

### 1. 构建镜像

```bash
cd /Data/wyh/verl/docker/textcraft-rl
docker-compose build
```

### 2. 配置环境变量（可选）

```bash
# 创建 .env 文件
cat > .env << EOF
WANDB_API_KEY=your_wandb_api_key
HF_TOKEN=your_huggingface_token
EOF
```

### 3. 启动容器

```bash
docker-compose up -d
```

### 4. 进入容器

```bash
docker exec -it textcraft-rl bash
```

### 5. 运行训练

在容器内运行：

```bash
# 方式1: 使用封装脚本（交互式）
bash /workspace/verl/docker/textcraft-rl/run_training.sh

# 方式2: 直接运行训练脚本
cd /workspace/verl/examples/sglang_multiturn/my_exp/rl
bash run_textcraft_grpo_train.sh
```

### 6. 自定义训练参数

通过环境变量覆盖默认参数：

```bash
# 设置 50 个 epoch，学习率 5e-6
NUM_EPOCHS=50 LEARNING_RATE=5e-6 bash run_textcraft_grpo_train.sh

# 或者在 docker-compose 中设置
docker-compose run -e NUM_EPOCHS=50 -e LEARNING_RATE=5e-6 textcraft-rl bash run_training.sh
```

## 目录映射

| 容器内路径 | 宿主机路径 | 说明 |
|-----------|-----------|------|
| `/workspace/verl` | `/Data/wyh/verl` | verl 代码 |
| `/workspace/datasets` | `/Data/wyh/datasets` | 数据集 |
| `/workspace/models` | `/Data/public` | 预训练模型 |
| `/workspace/outputs` | `/Data/wyh/datasets/Verl-Data/outputs` | 训练输出 |

## 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `NUM_EPOCHS` | 50 | 训练轮数 |
| `TRAIN_BATCH_SIZE` | 64 | 全局 batch size |
| `LEARNING_RATE` | 5e-6 | 学习率 |
| `ENTROPY_COEFF` | 0.01 | 熵奖励系数 |
| `TEMPERATURE` | 1.0 | 采样温度 |
| `ROLLOUT_N` | 8 | 每个 prompt 采样数 |
| `MODEL_PATH` | `/workspace/models/Qwen3-1.7B` | 模型路径 |

## 迁移到其他服务器

### 方式1: 导出/导入镜像

```bash
# 在当前服务器上导出镜像
docker save textcraft-rl:latest | gzip > textcraft-rl.tar.gz

# 传输到目标服务器
scp textcraft-rl.tar.gz user@target_server:/path/to/

# 在目标服务器上导入镜像
docker load < textcraft-rl.tar.gz
```

### 方式2: 在目标服务器重新构建

```bash
# 复制 docker 配置文件
scp -r /Data/wyh/verl/docker/textcraft-rl user@target_server:/path/to/verl/docker/

# 在目标服务器上构建
cd /path/to/verl/docker/textcraft-rl
docker-compose build
```

## 常见问题

### 1. GPU 内存不足 (OOM)

调整以下参数：
```bash
GPU_MEMORY_UTIL=0.5 TRAIN_BATCH_SIZE=32 bash run_textcraft_grpo_train.sh
```

### 2. wandb 无法连接

```bash
# 离线模式
export WANDB_MODE=offline
```

### 3. 共享内存不足

确保 docker-compose.yml 中设置了足够的 shm_size：
```yaml
shm_size: '64gb'
```

## 监控训练

```bash
# 查看 wandb 日志
# 访问 https://wandb.ai/your_entity/textcraft_grpo

# 查看本地日志
tail -f /workspace/outputs/textcraft_grpo/logs/train_*.log

# 监控 GPU 使用
watch -n 1 nvidia-smi
```

