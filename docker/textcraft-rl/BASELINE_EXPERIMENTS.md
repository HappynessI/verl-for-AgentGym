# TextCraft RL 实验 - 8x144GB 资源需求

## 概述

需要在 **8卡 144GB** 机器上跑以下 3 个 baseline 实验：

| 实验 | 算法 | 脚本 | 状态 |
|------|------|------|------|
| GRPO | GRPO 基线 | `run_textcraft_grpo_8x144gb.sh` | ✅ 本地已跑 |
| GRPO + MIS | GRPO + Multi-turn Importance Sampling | `run_textcraft_grpo_mis.sh` | ⏳ 需师兄跑 |
| GRPO + TIS | GRPO + Token-level IS with Segment grouping | `run_textcraft_grpo_tis.sh` | ⏳ 需师兄跑 |
| DRPO | Decoupled Reward Policy Optimization | `run_textcraft_drpo_train.sh` | ⏳ 需师兄跑 |

> **注意**：GRPO baseline 你本地已跑，MIS/TIS/DRPO 需要借助师兄的 8x144GB 资源。

---

## 关键超参数配置 (8x144GB)

### 公共参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `NUM_GPUS` | 8 | GPU 数量 |
| `GPU_IDS` | 0,1,2,3,4,5,6,7 | GPU 编号 |
| `TRAIN_BATCH_SIZE` | 128 | 全局 batch size |
| `MICRO_BATCH_SIZE` | 2 | 每 GPU micro batch |
| `NUM_EPOCHS` | 100 | 训练轮数 |
| `LEARNING_RATE` | 5e-6 | 学习率 |
| `SAVE_FREQ` | 100 | 保存 checkpoint 频率 (epoch) |
| `TEST_FREQ` | 20 | 验证频率 (epoch) |
| `ROLLOUT_N` | 8 | 每个 prompt 采样次数 |
| `TEMPERATURE` | 0.8 | 采样温度 |
| `GPU_MEMORY_UTIL` | 0.8~0.85 | vLLM 显存利用率 |
| `MAX_NUM_SEQS` | 128~1024 | vLLM 最大并发序列数 |
| `ENFORCE_EAGER` | false | 144GB 可用 PagedAttention |

### 各实验特有参数

#### GRPO (Baseline)

| 参数 | 值 | 说明 |
|------|-----|------|
| `ROLLOUT_IS` | none | 关闭 IS |
| `ROLLOUT_RS` | none | 关闭 MIS |

#### GRPO + MIS

| 参数 | 值 | 说明 |
|------|-----|------|
| `ROLLOUT_IS` | sequence | 开启 IS (Importance Sampling) |
| `ROLLOUT_IS_THRESHOLD` | 2.0 | IS 截断阈值 |
| `ROLLOUT_RS` | sequence | 开启 MIS (Multi-turn Importance Sampling) |
| `ROLLOUT_RS_THRESHOLD` | 2.0 | MIS 上阈值 (超过该值 mask) |
| `ROLLOUT_RS_THRESHOLD_LOWER` | 0.2 | MIS 下阈值 |

#### GRPO + TIS

| 参数 | 值 | 说明 |
|------|-----|------|
| `ROLLOUT_IS` | sequence | 开启 TIS (Token-level IS with Segment grouping) |
| `ROLLOUT_IS_THRESHOLD` | 2.0 | TIS 截断阈值 |
| `ROLLOUT_RS` | none | 关闭 MIS |

#### DRPO

| 参数 | 值 | 说明 |
|------|-----|------|
| `DELTA` | 1e-4 | KL 约束阈值 |
| `BETA` | 1e3 | KL 惩罚系数 |
| `TAU` | 10 | 温度参数 |
| `LAMBDA` | 0.1 | 长度权重 |
| `PPO_KL_TYPE` | kl | KL 类型 |

---

## 运行命令

```bash
# 1. 启动 Docker 环境
cd /Data/wyh/verl/docker/textcraft-rl
cp .env.example .env
# 编辑 .env 确认路径配置

# 指定 8 卡
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 docker-compose up -d

# 2. 进入容器
docker exec -it textcraft-rl bash

# 3. 运行实验
cd /workspace/verl

# GRPO (Baseline)
EXPERIMENT=grpo_baseline bash docker/textcraft-rl/run_training.sh

# GRPO + MIS
EXPERIMENT=grpo_mis bash docker/textcraft-rl/run_training.sh

# GRPO + TIS
EXPERIMENT=grpo_tis bash docker/textcraft-rl/run_training.sh

# DRPO
EXPERIMENT=drpo bash docker/textcraft-rl/run_training.sh
```

---

## 预计资源占用

| 项目 | 占用 |
|------|------|
| 模型权重 (Qwen3-1.7B) | ~3.4GB × 8 |
| vLLM KV Cache | ~80GB (0.85 util) |
| 训练 optimizer states | ~10GB |
| 梯度 & 激活值 | ~20GB |
| **总显存** | ~140GB / 144GB |

---

## 输出目录

- Checkpoints: `/Data/wyh/datasets/Verl-Data/outputs/textcraft_grpo_mis/`
- Logs: `.../textcraft_grpo_mis/logs/`
- WandB: `textcraft_grpo_mis` project
