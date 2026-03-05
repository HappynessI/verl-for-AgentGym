# TextCraft RL 实验 Docker 环境

本目录包含用于 TextCraft 环境下强化学习实验（SFT / GRPO / Prefix-RL）的完整 Docker 配置。

> **需要跑哪些实验？** → 见 [EXPERIMENTS.md](EXPERIMENTS.md)

## 目录结构

```
docker/textcraft-rl/
├── Dockerfile                  # 训练容器（verl + vLLM + GPU）
├── Dockerfile.textcraft-server # TextCraft 游戏服务器容器（无 GPU）
├── docker-compose.yml          # 多服务编排（训练 + 游戏服务器）
├── .env.example                # 路径变量模板 → 复制为 .env 并修改
├── requirements.txt            # 训练容器的额外 Python 依赖
├── run_training.sh             # 训练启动脚本（在容器内运行）
└── build_and_run.sh            # 一键构建并启动脚本
```

## 快速开始

### 1. 配置路径

```bash
cd docker/textcraft-rl
cp .env.example .env
vim .env   # 按当前服务器实际路径修改
```

`.env` 中的必填变量：

| 变量 | 说明 | 示例 |
|------|------|------|
| `VERL_CODE_DIR` | verl 代码目录 | `/data/wyh/verl` |
| `DATASETS_DIR` | 数据集目录 | `/data/wyh/datasets` |
| `MODELS_DIR` | 模型目录（只读挂载） | `/data/public` |
| `OUTPUTS_DIR` | 训练输出目录 | `/data/wyh/outputs` |
| `HF_CACHE_DIR` | HuggingFace 缓存 | `/data/cache/huggingface` |
| `WANDB_API_KEY` | wandb API 密钥 | `xxxxxxxx` |

### 2. 构建并启动

```bash
# 构建两个镜像并启动服务
docker-compose up -d --build

# 查看服务状态
docker-compose ps

# 查看 TextCraft 服务器日志
docker-compose logs -f textcraft-server
```

### 3. 进入容器并开始训练

```bash
docker exec -it textcraft-rl bash

# 容器内运行（交互确认模式）
bash /workspace/verl/docker/textcraft-rl/run_training.sh

# 非交互模式（批量提交）
AUTO_CONFIRM=1 bash /workspace/verl/docker/textcraft-rl/run_training.sh
```

## 实验类型

通过 `EXPERIMENT` 变量选择实验类型（在 `.env` 或命令行传入）：

| EXPERIMENT | 优势估计器 | 说明 |
|---|---|---|
| `grpo_baseline` | `grpo` | GRPO 基线（默认） |
| `prefix_full` | `turn_full_trajectory` | Prefix-RL 全轨迹优势 |
| `prefix_guided` | `turn_prefix_guided` | Prefix-RL 前缀引导优势 |
| `prefix_guided_dr` | `turn_prefix_guided_dr` | Prefix-RL + Dr.GRPO |

```bash
# 示例：运行 Prefix-RL 实验，使用 Qwen3-4B 模型
EXPERIMENT=prefix_full MODEL_NAME=Qwen3-4B AUTO_CONFIRM=1 \
  docker exec textcraft-rl bash /workspace/verl/docker/textcraft-rl/run_training.sh
```

## 支持的模型

在 `MODELS_DIR` 下应包含以下子目录：

```
MODELS_DIR/
├── Qwen3-0.6B/
├── Qwen3-1.7B/   ← 默认
├── Qwen3-4B/
└── Qwen3-8B/
```

通过 `MODEL_NAME` 变量指定：`MODEL_NAME=Qwen3-4B`


## 服务说明

### textcraft-server（游戏服务器）

- 基于 `python:3.11-slim`，无 GPU 需求
- 自动从 GitHub 安装 `agentenv-textcraft`
- 默认端口：`36001`（可通过 `TEXTCRAFT_PORT` 修改）
- 训练容器通过 `TEXTCRAFT_SERVER_URL=http://127.0.0.1:36001` 访问

### textcraft-rl（训练容器）

- 基于 `vllm/vllm-openai:v0.11.0`（CUDA 12.8 + PyTorch 2.8 + vLLM 0.11.0）
- 启动时自动从挂载的 `VERL_CODE_DIR` 安装 verl（开发模式）
- 使用 `network_mode: host` + `shm_size: 64gb` 支持多 GPU 训练

## 常见问题

**Q: TextCraft 服务器启动失败？**
```bash
docker-compose logs textcraft-server
# 检查端口是否被占用
netstat -tlnp | grep 36001
```

**Q: 训练容器找不到数据？**
- 检查 `DATASETS_DIR` 和 `OUTPUTS_DIR` 是否正确配置
- 确认 `train.parquet` 路径：`DATASETS_DIR/Verl-Data/train/textcraft/train.parquet`

**Q: CUDA OOM？**
- 减小 `TRAIN_BATCH_SIZE`（默认 128）
- 减小 `ROLLOUT_N`（默认 8）
- 检查 `MAX_NUM_SEQS`（在 `run_test_train.sh` 中配置，默认 128）

**Q: 在新服务器部署时路径不同？**
- 只需修改 `.env` 中的路径变量，无需改动任何脚本
