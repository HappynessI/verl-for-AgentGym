# wyh_exp: Turn-based Prefix RL

基于 verl 框架的 Turn-based Prefix 强化学习实验模块。

## 核心思想

在多轮 Agent 交互场景中，利用大模型的参考轨迹作为 prefix，让小模型从不同轮次继续 rollout。

### 两种训练模式

| 模式 | 描述 | Prefix 处理 | 适用场景 |
|------|------|------------|---------|
| **Full-Trajectory RL** | 整个轨迹作为小模型自己的轨迹进行 RL | Prefix 也计算梯度 | 希望小模型完全学习大模型行为 |
| **Prefix-Guided RL** | Prefix 作为参考，只对 rollout 部分进行 RL | Prefix 不计算梯度 | 希望小模型从大模型中间状态探索 |

## 目录结构

```
recipe/wyh_exp/
├── __init__.py
├── README.md
├── config/
│   └── base_config.yaml          # 基础配置
├── data/
│   ├── __init__.py
│   ├── turn_parser.py            # Turn 解析器
│   └── turn_dataset.py           # 数据集类
├── algos/
│   ├── __init__.py
│   └── core_algos.py             # 核心算法（优势估计）
├── scripts/
│   └── run_textcraft_turn_prefix.sh  # 训练脚本
├── ray_trainer.py                # Ray 训练器
├── fsdp_workers.py               # FSDP Workers
└── interaction_adapter.py        # AgentGym 环境适配器
```

## 数据格式

```json
{
    "prompt_id": "textcraft_001",
    "query": "Craft a wooden pickaxe",
    "messages": [
        {"role": "observation", "content": "You are in a forest..."},
        {"role": "think", "content": "I need to get wood first..."},
        {"role": "action", "content": "[[ get 3 logs ]]"},
        {"role": "observation", "content": "You got 3 logs."},
        ...
    ],
    "reward": 1.0
}
```

## 快速开始

### 1. 安装依赖

```bash
cd /Data/wyh/verl
pip install -e .
```

### 2. 准备数据

数据应包含多轮交互轨迹，格式如上所示。

### 3. 运行训练

**方式A: Full-Trajectory RL**
```bash
cd /Data/wyh/verl/recipe/wyh_exp/scripts

TRAINING_MODE=full_trajectory \
MODEL_PATH=/path/to/model \
TRAIN_DATA=/path/to/train.parquet \
bash run_textcraft_turn_prefix.sh
```

**方式B: Prefix-Guided RL**
```bash
TRAINING_MODE=prefix_guided \
PREFIX_STRATEGY=random \
MIN_PREFIX_TURNS=1 \
MAX_PREFIX_TURNS=5 \
MODEL_PATH=/path/to/model \
TRAIN_DATA=/path/to/train.parquet \
bash run_textcraft_turn_prefix.sh
```

## 配置说明

### 数据配置

```yaml
data:
  mode: full_trajectory  # full_trajectory 或 prefix_guided
  prefix_strategy: random  # fixed, random, all
  min_prefix_turns: 1
  max_prefix_turns: 5
  num_rollouts_per_prefix: 2
```

### 算法配置

```yaml
algorithm:
  # 优势估计器
  adv_estimator: turn_full_trajectory  # 或 turn_prefix_guided, turn_prefix_guided_dr
  
  # GRPO 配置
  norm_adv_by_std_in_grpo: true
  
  # Prefix 优势模式（仅 prefix_guided）
  prefix_advantage_mode: zero  # zero, sft, relative
```

## 注册的优势估计器

通过 verl 的注册机制，本模块注册了以下优势估计器：

| 名称 | 描述 |
|------|------|
| `turn_full_trajectory` | 方式A: 全轨迹 GRPO |
| `turn_full_trajectory_with_turn_bonus` | 方式A + Turn 级别奖励 |
| `turn_prefix_guided` | 方式B: Prefix 引导 GRPO |
| `turn_prefix_guided_dr` | 方式B: Dr.GRPO 变体 |

## 复用 verl 框架

本模块充分复用了 verl 框架的以下组件：

- `verl.trainer.ppo.ray_trainer.RayPPOTrainer`: 训练器基类
- `verl.trainer.ppo.core_algos`: 优势估计器注册机制
- `verl.workers.fsdp_workers`: FSDP 分布式训练
- `verl.interactions.*`: AgentGym 环境交互

## 开发计划

- [x] Phase 1: 目录结构和数据格式设计
- [x] Phase 2: Turn 解析器和数据集类
- [x] Phase 3: 两种训练模式的优势计算
- [x] Phase 4: FSDP Worker 和环境适配
- [ ] Phase 5: 完整测试和调试

## 参考

- [verl](https://github.com/volcengine/verl): 字节跳动的 LLM RL 训练框架
- [prefix-rft](https://arxiv.org/abs/2507.01679): Blending Supervised and Reinforcement Fine-Tuning with Prefix Sampling
- [ARPO](https://arxiv.org/abs/2507.19849): Agentic Reinforced Policy Optimization

