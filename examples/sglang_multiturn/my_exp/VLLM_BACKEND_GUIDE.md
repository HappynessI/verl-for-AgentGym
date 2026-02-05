# vLLM 推理后端使用指南

## 概述

本文档说明 TextCraft 项目中 vLLM 推理后端的两种使用模式：
1. **训练模式**：vLLM 集成在训练框架内
2. **评估模式**：vLLM 作为独立服务器运行

---

## 1. 训练模式：vLLM 集成推理

### 脚本位置
```bash
verl/examples/sglang_multiturn/my_exp/rl/run_textcraft_grpo_train.sh
```

### 工作原理
- vLLM 引擎**内嵌**在训练框架中（verl）
- 训练过程中自动启动和管理 vLLM
- 模型权重实时更新，用于 Rollout 采样
- 无需手动启动 vLLM 服务器

### 架构图
```
GRPO训练框架
├── Actor模型 (训练中，权重更新)
├── vLLM引擎 (Rollout推理)
│   └── 使用当前Actor权重
├── Reference模型 (冻结)
└── TextCraft环境服务器 (独立)
```

### 关键配置参数

#### vLLM Rollout 配置（训练采样）
```bash
ROLLOUT_N=8                    # 每个prompt采样数量（GRPO需要>1）
TEMPERATURE=1.0                # 采样温度（建议0.5-1.0）
TOP_P=1.0                      # Nucleus采样
GPU_MEMORY_UTIL=0.6            # vLLM GPU内存利用率
MAX_NUM_SEQS=256               # 最大并发序列数
ENFORCE_EAGER=true             # 使用eager模式
FREE_CACHE_ENGINE=true         # 释放KV cache
```

#### vLLM Validation 配置
```bash
VAL_TEMPERATURE=0.6            # validation温度
VAL_TOP_P=0.95
VAL_N=1                        # validation采样数
```

### 使用方法
```bash
# 基本使用
bash run_textcraft_grpo_train.sh

# 自定义参数
GPU_IDS="2,3" NUM_GPUS=2 TEMPERATURE=0.7 bash run_textcraft_grpo_train.sh
```

### 配置文件
```yaml
# config/textcraft_grpo_train.yaml
actor_rollout_ref:
  rollout:
    name: vllm  # 使用vLLM后端
    temperature: 1.0
    gpu_memory_utilization: 0.6
    enforce_eager: True
    free_cache_engine: True
```

---

## 2. 评估模式：vLLM 独立服务器

### 脚本位置
```bash
verl/examples/sglang_multiturn/my_exp/eval/run_textcraft_eval_vllm_server.sh
```

### 工作原理
- vLLM 作为**独立 HTTP 服务器**运行
- 评估脚本通过 HTTP API 调用推理
- 模型权重加载一次，不更新
- 需要手动启动 vLLM 服务器

### 架构图
```
评估客户端 (eval script)
    ↓ HTTP
vLLM服务器 (http://localhost:8000)
    ├── /v1/chat/completions
    └── 加载的模型（只读）
    ↓
TextCraft环境服务器 (独立)
```

### 关键配置参数

```bash
VLLM_SERVER_URL="http://localhost:8000"
TEMPERATURE=1.0                # 采样温度
TOP_P=1.0                      # Nucleus采样
NUM_SAMPLES_PER_TASK=8         # 每个任务采样数（用于pass@k）
CONCURRENCY=256                # 并发请求数
MAX_NEW_TOKENS=2000            # 每次生成最大token数
```

### 使用步骤

#### 步骤1: 启动 vLLM 服务器
```bash
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/your/model \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 10240
```

#### 步骤2: 测试服务器
```bash
curl http://localhost:8000/v1/models
```

#### 步骤3: 运行评估
```bash
bash run_textcraft_eval_vllm_server.sh

# 或自定义参数
TEMPERATURE=0.7 NUM_SAMPLES_PER_TASK=16 bash run_textcraft_eval_vllm_server.sh
```

---

## 3. 两种模式对比

| 特性 | 训练模式（内嵌） | 评估模式（独立服务器） |
|------|-----------------|---------------------|
| **vLLM位置** | 集成在训练框架内 | 独立HTTP服务器 |
| **模型更新** | 实时更新（每个epoch） | 不更新（加载一次） |
| **启动方式** | 自动启动 | 手动启动 |
| **通信方式** | 内部调用 | HTTP API |
| **适用场景** | RL训练 | 模型评估/推理 |
| **GPU内存** | 需与训练共享 | 独占GPU |
| **并发控制** | 框架管理 | HTTP请求数控制 |
| **灵活性** | 绑定训练流程 | 独立部署，易扩展 |

---

## 4. 采样参数建议

### 训练时（Rollout）
- **Temperature**: 0.5-1.0（需要探索多样性）
- **Top-P**: 0.9-1.0
- **采样数 (N)**: 4-8（GRPO需要多个样本）
- **目的**: 策略探索，发现更好的行为

### 训练时（Validation）
- **Temperature**: 0.3-0.6（适度采样）
- **Top-P**: 0.9-0.95
- **采样数 (N)**: 1（快速验证）
- **目的**: 监控训练进度

### 评估时
- **Temperature**: 1.0（多样性评估）
- **Top-P**: 1.0
- **采样数 (N)**: 8-16（计算pass@k）
- **目的**: 全面评估模型能力

### 推理时
- **Temperature**: 0.0-0.1（贪婪策略）
- **Top-P**: 1.0
- **采样数 (N)**: 1（单次推理）
- **目的**: 获得最优输出

---

## 5. 常见问题

### Q1: 训练时可以连接外部vLLM服务器吗？
**A**: 理论上可以，但不推荐。训练需要频繁更新模型权重，内嵌模式更高效。

### Q2: 评估时vLLM服务器OOM怎么办？
**A**: 调整以下参数：
```bash
--gpu-memory-utilization 0.8  # 降低内存占用
--max-model-len 8192          # 降低最大序列长度
--tensor-parallel-size 2      # 使用多GPU
```

### Q3: 训练时vLLM占用太多内存？
**A**: 调整训练脚本参数：
```bash
GPU_MEMORY_UTIL=0.5           # 降低vLLM内存占用
FREE_CACHE_ENGINE=true        # 释放KV cache
MICRO_BATCH_SIZE=1            # 降低训练batch size
```

### Q4: 如何选择temperature？
**A**: 参考第4节的建议。关键原则：
- 训练时需要探索 → 高temperature (0.7-1.0)
- 评估时需要多样性 → 中等temperature (0.5-1.0)
- 推理时需要稳定 → 低temperature (0.0-0.1)

---

## 6. 快速参考

### 启动训练
```bash
cd /Data/wyh/verl/examples/sglang_multiturn/my_exp/rl
bash run_textcraft_grpo_train.sh
```

### 启动评估（需先启动vLLM服务器）
```bash
# 终端1: 启动vLLM服务器
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/checkpoint \
    --port 8000

# 终端2: 运行评估
cd /Data/wyh/verl/examples/sglang_multiturn/my_exp/eval
bash run_textcraft_eval_vllm_server.sh
```

### 监控训练日志
```bash
tail -f /Data/wyh/datasets/Verl-Data/outputs/textcraft_grpo/logs/train_*.log
```

---

## 7. 相关文件

- **训练脚本**: `rl/run_textcraft_grpo_train.sh`
- **训练配置**: `config/textcraft_grpo_train.yaml`
- **评估脚本**: `eval/run_textcraft_eval_vllm_server.sh`
- **评估代码**: `eval/eval_textcraft_vllm_server.py`
- **训练指南**: `rl/GRPO_TRAINING_GUIDE.md`

---

**最后更新**: 2026-01-28

