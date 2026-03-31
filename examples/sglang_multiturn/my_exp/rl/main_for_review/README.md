# Prefix Optimization 主实验代码修改

## 概述

本目录包含将 prefix token 纳入 GRPO 优化目标的核心代码修改。

**关键变化（新主实验定义）**:
- prefix old_logprob 来自**离线缓存的 SFT old_logprob**（不是当前 rollout actor）
- continuation old_logprob 来自 rollout 时旧 actor snapshot（标准 GRPO）
- **不使用 KL / reference 约束**

## 修改的文件

### 1. `algorithm.py` (verl/trainer/config/)
**位置**: 第 467-469 行

添加了新的配置选项：
```python
# Prefix optimization: whether to include prefix tokens in GRPO loss
optimize_prefix_tokens: bool = False
prefix_loss_weight: float = 1.0
```

### 2. `ray_trainer.py` (verl/trainer/ppo/)
**修改内容**:

- **compute_prefix_mask 函数**: 修复字段名问题，正确识别 `prompt`（而非 `prompts`）
- **处理 assistant_prefix_old_log_probs**: 从数据集读取预缓存的 SFT old_logprob

### 3. `dp_actor.py` (verl/workers/actor/)
**修改内容**:

- **分离 prefix 和 continuation 的 old_logprob 来源**:
  - continuation: 使用 rollout actor 的 old_logprob
  - prefix: 使用预缓存的 SFT old_logprob
- **关闭 KL 约束**: 当 optimize_prefix_tokens=True 时禁用 KL 损失

## 实验逻辑

### 两阶段设计

1. **阶段 A: SFT 预处理**
   - 从 Qwen3-1.7B base 出发
   - 在完整 teacher 成功轨迹上做 SFT
   - 得到 SFT model

2. **阶段 B: RL 主实验**
   - 重新拿一个全新的 Qwen3-1.7B base
   - 这个 RL actor **不是**从 SFT checkpoint 初始化
   - SFT model 只用于缓存 old_logprob

### old_logprob 来源不同

| 段落 | old_logprob 来源 |
|------|-----------------|
| prefix (assistant token) | 离线缓存的 SFT old_logprob |
| continuation | rollout 时旧 actor snapshot |

### 关键约束

- **不要启用 KL / reference 约束**
- 只优化 assistant token（不含 system/user）
- replay 仍然保留
- reward 仍然是 trajectory-level sparse reward

## 使用方法

### 启用主实验（Prefix 优化）

在训练脚本中设置：
```bash
OPTIMIZE_PREFIX_TOKENS=true
PREFIX_LOSS_WEIGHT=1.0
USE_KL_LOSS=false
```

或 YAML 配置：
```yaml
algorithm:
  adv_estimator: grpo
  optimize_prefix_tokens: True
  prefix_loss_weight: 1.0
  use_kl_loss: False  # 主实验不使用 KL
```

### Baseline vs 主实验

| 模式 | optimize_prefix_tokens | 行为 |
|------|----------------------|------|
| baseline | false | 只优化 continuation，不优化 prefix |
| 主实验 | true | 同时优化 prefix (cached SFT old_logprob) + continuation |

## 数据要求

### 必须字段

- `prompt`: ChatML 风格的多轮对话列表
- `extra_info.interaction_kwargs.prefix_actions`: prefix 动作列表（用于 replay）
- `extra_info.interaction_kwargs.task_id`: 任务 ID

### 新增字段（预缓存）

- `assistant_prefix_old_log_probs`: 预计算的 SFT model 对 assistant token 的 logprob

**注意**: 当前数据集没有 `assistant_prefix_old_log_probs` 字段。需要在数据预处理阶段生成：
1. 加载 SFT model
2. 对完整 teacher 轨迹做 teacher-forced forward
3. 提取所有 assistant token 的 logprob
4. 保存到数据集的 `assistant_prefix_old_log_probs` 字段

## 训练配置

### 新增环境变量

```bash
# 主实验开关
OPTIMIZE_PREFIX_TOKENS=${OPTIMIZE_PREFIX_TOKENS:-false}

# Prefix loss 权重
PREFIX_LOSS_WEIGHT=${PREFIX_LOSS_WEIGHT:-1.0}

# KL 约束（主实验关闭）
USE_KL_LOSS=${USE_KL_LOSS:-false}
```

## Metrics 记录

训练时会记录以下指标：
- `actor/prefix_loss`: prefix 策略损失
- `actor/continuation_loss`: continuation 策略损失
- `actor/prefix_loss_weight`: prefix loss 权重
- `actor/prefix_old_logprob_source`: prefix old_logprob 来源 ("cached_sft" 或 "rollout_actor")
- `actor/kl_disabled`: KL 损失是否被禁用
- `actor/use_cached_prefix_old_logprob`: 是否使用缓存的 prefix old_logprob
