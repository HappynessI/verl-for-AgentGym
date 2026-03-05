# 实验清单

> 环境：TextCraft（当前）/ BabyAI（待配置）
> 默认模型：Qwen3-1.7B，可通过 `MODEL_NAME` 切换

---

## 实验总览

| 实验 | 预计 GPU 时间 | 依赖 |
|------|-------------|------|
| GRPO 基线 | ~10h | — |
| Prefix-RL 主方法 | ~10h | — |
| 优势估计器消融（4 种）| ~24h | GRPO 基线完成 |
| Prefix 长度消融（7 种）| ~30h | Prefix-RL 完成 |
| 模型规模（0.6B/4B/8B）| ~45h | 以上完成 |
| 教师样本质量（正/负/混合）| ~20h | 需采集更多负样本 |

**总计**：~139 GPU-hours（4 卡约 35 小时）

---

## 第一批：基础实验

### 1. GRPO 基线

```bash
EXPERIMENT=grpo_baseline MODEL_NAME=Qwen3-1.7B AUTO_CONFIRM=1 \
  docker exec textcraft-rl bash /workspace/verl/docker/textcraft-rl/run_training.sh
```

### 2. Prefix-RL 主方法

```bash
EXPERIMENT=prefix_full MODEL_NAME=Qwen3-1.7B AUTO_CONFIRM=1 \
  docker exec textcraft-rl bash /workspace/verl/docker/textcraft-rl/run_training.sh
```

---

## 第二批：消融实验

### 优势估计器对比（固定 Qwen3-1.7B）

```bash
for EST in grpo turn_full_trajectory turn_prefix_guided turn_prefix_guided_dr; do
  EXPERIMENT=prefix_full ADV_ESTIMATOR=$EST AUTO_CONFIRM=1 \
    docker exec textcraft-rl bash /workspace/verl/docker/textcraft-rl/run_training.sh
done
```

### Prefix 长度消融（固定 turn_full_trajectory）

通过 `PREFIX_TURNS` 控制前缀轮数（`0` = GRPO 基线）：

```bash
for N in 0 1 2 3 5 random curriculum; do
  PREFIX_TURNS=$N EXPERIMENT=prefix_full AUTO_CONFIRM=1 \
    docker exec textcraft-rl bash /workspace/verl/docker/textcraft-rl/run_training.sh
done
```

### 模型规模（每个规模跑 GRPO + Prefix-RL）

```bash
for MODEL in Qwen3-0.6B Qwen3-4B Qwen3-8B; do
  for EXP in grpo_baseline prefix_full; do
    EXPERIMENT=$EXP MODEL_NAME=$MODEL AUTO_CONFIRM=1 \
      docker exec textcraft-rl bash /workspace/verl/docker/textcraft-rl/run_training.sh
  done
done
```

---

## 第三批：教师样本质量

### 正/负样本消融

> **前置条件**：用 Gemini-3-pro 重新采样 TextCraft 获取更多失败轨迹（当前仅 1 条 reward=0），BabyAI 已有 361 负 / 646 正可直接用。

```bash
for SPLIT in positive_only negative_only mixed reward_weighted; do
  TEACHER_SPLIT=$SPLIT EXPERIMENT=prefix_full AUTO_CONFIRM=1 \
    docker exec textcraft-rl bash /workspace/verl/docker/textcraft-rl/run_training.sh
done
```

---

## 评估指标

每个实验记录：

- `Success Rate (Avg@1)` — 主要指标
- `Pass@2 / Pass@4 / Pass@8` — 探索多样性
- `Avg Steps` — 交互效率

---

## 注意事项

1. **BabyAI** 环境尚未配置，待 TextCraft 实验完成后补充
2. **过度思考分析**（Think/Action Ratio）和**反向泛化**通过对 checkpoint 单独 eval 获取，不需要额外训练
3. 每个实验运行 3 次取 mean ± std
4. 输出目录：`OUTPUTS_DIR/<experiment_tag>/`，wandb 项目建议按 `textcraft_<exp>` 命名
