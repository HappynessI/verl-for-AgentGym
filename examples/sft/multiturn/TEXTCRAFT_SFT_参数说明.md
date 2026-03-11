# TextCraft SFT 训练参数完整说明

## 📍 参数位置总览

### 训练脚本中的参数
**位置**: `/Data/wyh/verl/examples/sft/multiturn/run_textcraft_qwen3_sft.sh`
这些是SFT训练的超参数，控制模型如何学习。

### 评估脚本中的参数  
**位置**: `/Data/wyh/verl/examples/sglang_multiturn/my_exp/eval/run_textcraft_eval.sh`
这些是推理参数，控制模型如何生成文本。

---

## 🔧 训练参数详解（在 run_textcraft_qwen3_sft.sh 中）

### 1. 数据参数
| 参数 | 值 | 说明 |
|------|----|----|
| `data.train_files` | `/Data/wyh/datasets/Parquet-Data/textcraft/train.parquet` | 训练数据路径 |
| `data.val_files` | `/Data/wyh/datasets/Parquet-Data/textcraft/train.parquet` | 验证数据路径 |
| `data.train_batch_size` | `256` | 全局batch size |
| `data.micro_batch_size` | `2` | 每张GPU的batch size |
| `data.max_length` | `4096` | 最大序列长度（tokens） |

**计算公式**:  
梯度累积步数 = `train_batch_size` / (`micro_batch_size` × GPU数量)  
= 256 / (2 × 2) = **64步累积一次梯度更新**

### 2. 优化器参数（核心训练超参数）
| 参数 | 值 | 说明 |
|------|----|----|
| `optim.lr` | `1e-5` | **学习率**（0.00001），控制参数更新步长 |
| `optim.betas` | `[0.9, 0.95]` | Adam优化器动量参数 |
| `optim.weight_decay` | `0.01` | **权重衰减**，防止过拟合 |
| `optim.lr_warmup_steps_ratio` | `0.1` | 学习率预热比例（前10%步数线性增长） |
| `optim.clip_grad` | `1.0` | **梯度裁剪**阈值，防止梯度爆炸 |
| `optim.lr_scheduler` | `cosine` | 学习率调度策略（余弦退火） |

### 3. 模型参数
| 参数 | 值 | 说明 |
|------|----|----|
| `model.partial_pretrain` | `/Data/public/Qwen3-1.7B` | 基础模型路径 |
| `model.enable_gradient_checkpointing` | `true` | 梯度检查点（用时间换显存） |

### 4. 并行与效率参数
| 参数 | 值 | 说明 |
|------|----|----|
| `ulysses_sequence_parallel_size` | `2` | 序列并行度（必须等于GPU数） |
| `use_remove_padding` | `true` | 移除padding优化 |

### 5. 训练流程参数
| 参数 | 值 | 说明 |
|------|----|----|
| `trainer.seed` | `42` | 随机种子（保证可复现） |
| `trainer.total_epochs` | `10` | 训练总轮数 |
| `trainer.save_freq` | `40` | 每40步保存checkpoint |
| `trainer.test_freq` | `40` | 每40步验证一次 |

### 6. 日志参数
| 参数 | 值 | 说明 |
|------|----|----|
| `trainer.logger` | `[console,wandb]` | 日志输出方式 |
| `trainer.project_name` | `textcraft-sft` | WandB项目名 |
| `trainer.experiment_name` | `textcraft-sft-qwen3-1.7b-gpt4o` | WandB实验名 |

### 7. GPU设置
| 参数 | 值 | 说明 |
|------|----|----|
| `CUDA_VISIBLE_DEVICES` | `0,1` | 使用的GPU编号（默认） |
| `nproc_per_node` | `2` | 使用的GPU数量 |

---

## 🎯 推理参数详解（在 run_textcraft_eval.sh 中）

这些参数控制模型**生成文本**时的行为，与训练无关！

| 参数 | 值 | 说明 |
|------|----|----|
| `MAX_NEW_TOKENS` | `150` | 每次生成的最大token数 |
| `TEMPERATURE` | `0.0` | **温度**（0=贪婪，越高越随机） |
| `TOP_P` | `1.0` | **核采样**阈值（1.0=不限制） |
| `DO_SAMPLE` | `""` | 是否启用采样（空=贪婪解码） |

---

## 📊 训练进度估算

基于当前配置：
- 数据量: 374条
- batch_size: 2 per GPU × 2 GPUs = 4
- 每个epoch步数: 374 / 4 = 94步
- 总步数: 94 × 10 epochs = **940步**
- checkpoint数量: 940 / 40 = **23个**

---

## 🔄 如何修改参数

### 调整学习率
```bash
bash run_textcraft_qwen3_sft.sh 2 /output optim.lr=5e-6
```

### 调整batch size
```bash
bash run_textcraft_qwen3_sft.sh 2 /output \
  data.train_batch_size=128 \
  data.micro_batch_size=1
```

### 修改保存频率
```bash
bash run_textcraft_qwen3_sft.sh 2 /output \
  trainer.save_freq=20 \
  trainer.test_freq=20
```

---

## ❓ 常见问题

### Q: temperature在哪里设置？
A: **不在训练脚本中**！temperature是推理参数，在评估脚本中设置。

### Q: 如何提高训练速度？
A: 增大 `data.micro_batch_size`（需要更多显存）

### Q: 如何减少显存占用？
A: 减小 `data.micro_batch_size` 或 `data.max_length`

### Q: 训练卡在某一步不动了？
A: 检查是否在做梯度累积（每64步才更新一次参数）

