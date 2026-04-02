# Prefix-GRPO 主实验技术说明（基于当前代码审查）

原始审查时间：2026-03-30  
本文在原始审查基础上，已同步 2026-04-02 的主实验运行口径与验证结论。

2026-03-31 更新：
- 仓库当前默认主实验数据已切换为 `data/textcraft/new_prefix_rl/stage7_audit_release/textcraft_prefix_main_train_step200.audited.parquet`
- 文中保留的 `step200_v2` / `cleaned_v2` 路径仅作为旧链路审计背景，不再是 repo 默认训练入口

2026-04-02 口径同步：
- 主实验默认训练数据应以 `new_prefix_rl/stage7_audit_release/textcraft_prefix_main_train_step200.audited.parquet` 为准
- 主实验默认 `OPTIMIZE_PREFIX_TOKENS=true`，只有显式设置 `false` 才是 continuation-only baseline
- 按当前实验要求，主实验不使用 KL/reference 约束，prefix 和 rollout 都不做 KL
- 经本轮验证，主实验入口默认应使用 `ENABLE_ACTIVATION_OFFLOAD=false`
- 下文中保留的旧 active 数据、`step200_v2` 路径与当时静态风险分析，主要用于解释历史问题来源；当前运行结论应以上述更新和 H200 维护说明为准

本说明主要对应以下对象：

- 主实验脚本：`/Data/wyh/verl/examples/sglang_multiturn/my_exp/rl/run_textcraft_grpo_validated.sh`
- 主训练配置：`/Data/wyh/verl/examples/sglang_multiturn/config/textcraft_grpo_train.yaml`
- 当前默认训练数据：`/Data/wyh/datasets/Verl-Data/train/textcraft/new_prefix_rl/stage7_audit_release/textcraft_prefix_main_train_step200.audited.parquet`
- 历史 smoke test 日志：`/Data/wyh/datasets/Verl-Data/outputs/textcraft_grpo_prefix_smoke_test_canonicalized/logs/train_canonicalized_prefix_smoke_test_20260322_231829.log`
- 2026-04-02 最新验证日志：
  - `/Data/wyh/h200_grpo_from_oss/outputs/prefix_grpo_validation_20260401/logs/preflight_tmux_36012.log`
  - `/Data/wyh/h200_grpo_from_oss/outputs/prefix_grpo_validation_20260401/logs/train_tmux_36012_no_actoff_fix2.log`
  - `/Data/wyh/h200_grpo_from_oss/outputs/prefix_grpo_smoke_20260402_joint_direct/logs/train_direct.log`
  - `/Data/wyh/h200_grpo_from_oss/outputs/prefix_grpo_smoke_20260402_cont_mean_abs_direct/logs/train_direct.log`
  - `/Data/wyh/h200_grpo_from_oss/outputs/prefix_grpo_smoke_20260402_constant_direct/logs/train_direct.log`
  - `/Data/wyh/h200_grpo_from_oss/outputs/prefix_grpo_short_nondebug_20260402/logs/train_direct.log`

---

## 0.1 2026-04-02 本轮测试补充

这部分对应主实验修改完成后的实际验证结果，不再是纯静态代码审查。

### 测试环境

- 代码落地与验证起点：`/Data/wyh/h200_grpo_from_oss`
- 验证环境：本地 `verl` Python 3.12 环境
- 模型：`/Data/public/Qwen3-1.7B`
- GPU：`2 x L20`
- TextCraft 服务：本地补丁服务 `http://127.0.0.1:36012`
- 验证数据：8 条 debug 子集

### 已执行测试

1. preflight smoke test
   - 日志：`/Data/wyh/h200_grpo_from_oss/outputs/prefix_grpo_validation_20260401/logs/preflight_tmux_36012.log`
   - 结果：通过
   - 结论：重建 prompt、`prefix_actions` replay、goal 绑定和基础环境创建链路正常

2. 2x L20 debug 训练，`enable_activation_offload=true`
   - 日志：`/Data/wyh/h200_grpo_from_oss/outputs/prefix_grpo_validation_20260401/logs/train_tmux_36012.log`
   - 结果：失败
   - 失败点：`activation_offload.py` 触发 `KeyError: 27`
   - 结论：当前主实验路径下不应默认开启 `activation_offload`

3. 2x L20 debug 训练，`enable_activation_offload=false`
   - 首次日志：`/Data/wyh/h200_grpo_from_oss/outputs/prefix_grpo_validation_20260401/logs/train_tmux_36012_no_actoff.log`
   - 结果：失败
   - 失败点：`prefix_token_count` 在 micro-batch 前未完全归一化为 tensor，导致 `dp_actor.py` 命中 `IndexError`

4. 修复后再次运行同一组 2x L20 debug 训练
   - 成功日志：`/Data/wyh/h200_grpo_from_oss/outputs/prefix_grpo_validation_20260401/logs/train_tmux_36012_no_actoff_fix2.log`
   - metrics：`/Data/wyh/h200_grpo_from_oss/outputs/prefix_grpo_validation_20260401/train_run_no_actoff_fix2/metrics/training_metrics.csv`
   - 结果：成功完成 `2/2` step，`EXIT_CODE=0`
   - 结论：当前 prefix 主路径已经跑通，不仅是预处理通过，而且 actor update 真实执行完成

5. 2x L20 最小 smoke train，`prefix_loss_mode=joint`
   - 日志：`/Data/wyh/h200_grpo_from_oss/outputs/prefix_grpo_smoke_20260402_joint_direct/logs/train_direct.log`
   - metrics：`/Data/wyh/h200_grpo_from_oss/outputs/prefix_grpo_smoke_20260402_joint_direct/metrics/training_metrics.csv`
   - 配置要点：`trainer.save_freq=0`、`trainer.test_freq=-1`、不保存 checkpoint、2 张 GPU
   - 结果：成功完成 `2/2` step
   - 结论：`joint` 模式已做最小端到端 smoke，日志中 `actor/prefix_loss_mode_joint=1.0`

6. 2x L20 最小 smoke train，`prefix_loss_mode=split` + `prefix_advantage_mode=cont_mean_abs`
   - 日志：`/Data/wyh/h200_grpo_from_oss/outputs/prefix_grpo_smoke_20260402_cont_mean_abs_direct/logs/train_direct.log`
   - metrics：`/Data/wyh/h200_grpo_from_oss/outputs/prefix_grpo_smoke_20260402_cont_mean_abs_direct/metrics/training_metrics.csv`
   - 配置要点：`trainer.save_freq=0`、`trainer.test_freq=-1`、不保存 checkpoint、2 张 GPU
   - 结果：成功完成 `2/2` step
   - 结论：`cont_mean_abs` 模式已做最小端到端 smoke，日志中配置项明确为 `prefix_advantage_mode='cont_mean_abs'`

7. 2x L20 最小 smoke train，`prefix_loss_mode=split` + `prefix_advantage_mode=constant`
   - 日志：`/Data/wyh/h200_grpo_from_oss/outputs/prefix_grpo_smoke_20260402_constant_direct/logs/train_direct.log`
   - metrics：`/Data/wyh/h200_grpo_from_oss/outputs/prefix_grpo_smoke_20260402_constant_direct/metrics/training_metrics.csv`
   - 配置要点：`trainer.save_freq=0`、`trainer.test_freq=-1`、不保存 checkpoint、2 张 GPU
   - 结果：成功完成 `2/2` step
   - 结论：`constant` 模式已做最小端到端 smoke，且 step 级日志里的 `actor/prefix_loss` 为非零

8. 2x L20 非 debug 短程 smoke train（正式 `stage7_audit_release` 数据）
   - 日志：`/Data/wyh/h200_grpo_from_oss/outputs/prefix_grpo_short_nondebug_20260402/logs/train_direct.log`
   - metrics：`/Data/wyh/h200_grpo_from_oss/outputs/prefix_grpo_short_nondebug_20260402/metrics/training_metrics.csv`
   - 配置要点：`data.train_max_samples=16`、`data.val_max_samples=8`、`trainer.save_freq=0`、`trainer.test_freq=-1`、不保存 checkpoint、2 张 GPU
   - 结果：成功完成 `2/2` step
   - 结论：在非 debug 路径下直接使用正式 `stage7` parquet 也已跑通，且出现了非零 reward / advantage / prefix loss

### 已验证到的关键证据

- `actor/use_cached_prefix_old_logprob=True`
- 训练 step 级日志中出现：
  `actor/prefix_loss`
  `actor/prefix_pg_clipfrac`
  `actor/prefix_ppo_kl`
  `actor/prefix_token_count`
- `joint` smoke log 中进一步出现：
  `actor/prefix_loss_mode_joint=1.0`
  `actor/joint_loss`
- `split + cont_mean_abs` smoke log 中配置和 step 指标同时证明：
  `prefix_advantage_mode='cont_mean_abs'`
  `actor/prefix_loss_mode_joint=0.0`
- `split + constant` smoke log 中进一步出现：
  `prefix_advantage_mode='constant'`
  `actor/prefix_loss` 为非零
- 非 debug 短程 `stage7` smoke log 中进一步出现：
  非零 `critic/rewards/mean`
  非零 `critic/advantages/mean`
  非零 `actor/prefix_loss`
- 这说明 prefix old logprob 的读取、prefix gather、prefix PPO loss 和训练指标回传都已真正发生

### 当前测试覆盖与未覆盖

- 已覆盖：
  - stage7 重建数据协议
  - goal / commands 从 prompt 到 env create/reset 的绑定
  - `split` 模式下的 prefix 训练主路径
  - `joint` 模式下的 prefix 训练主路径
  - `cont_mean` 模式下的 prefix advantage
  - `cont_mean_abs` 模式下的 prefix advantage
  - `constant` 模式下的 prefix advantage
  - 不启用 KL 的主实验默认路径
  - 最小 smoke 设置下的“不保存 checkpoint”路径
  - 非 debug 路径下直接使用正式 `stage7` parquet 的短程 smoke
  - 非零 reward / advantage 条件下的 prefix loss 更新信号

- 尚未做独立端到端验证：
  - 正式规模、非 debug 训练下的长期学习收益验证
  - 如需更严格对照，`joint` 模式下 prefix 独立 clip 的实现与验证

---

## 0.2 对本轮审查点的逐点回应

这里按本轮审查中的几个核心点，直接给出当前状态。

### 回应 1：新数据协议与训练实现是否一致

答：现在已经对齐。

- trainer 侧已按 `assistant_prefix_span + prefix_mask + prefix_token_count` 的 span-window 协议恢复 prefix sidecar
- 不再假设 `len(old_log_probs) == prefix_token_count == window_len`
- preflight 和 debug train 已经在 stage7 数据上跑通

因此，这一点不再是 blocker。

### 回应 2：prefix gather 是否仍然错误地假设在 prompt 尾部

答：已改为按绝对 span 位置 gather，不再依赖尾对齐假设。

- 被优化的 prefix token 可以位于 prompt 中部
- 当前成功训练日志已经证明 gather 后的 prefix loss 能真实参与 actor update

因此，这一点已经由实现和运行日志共同支撑。

### 回应 3：prefix advantage 是否需要显式可配

答：已经支持，且当前成功验证已经覆盖 `cont_mean`、`cont_mean_abs` 与 `constant`。

当前已实现：

- `cont_mean`
- `cont_mean_abs`
- `constant`

其中：

- `cont_mean` 已在本轮 debug train 中实际跑通
- `cont_mean_abs` 已在本轮 2x L20 最小 smoke train 中实际跑通
- `constant` 已在本轮 2x L20 最小 smoke train 中实际跑通

所以这项从“实现要求”到“最小实验覆盖”现在都已经完成。

### 回应 4：prefix clip 是否需要独立控制

答：已经支持独立配置。

当前已增加：

- `prefix_clip_ratio`
- `prefix_clip_ratio_low`
- `prefix_clip_ratio_high`
- `prefix_clip_ratio_c`

补充说明：

- 在 `split` 模式下，这些独立 clip 参数会单独作用于 prefix loss
- 在 `joint` 模式下，当前仍共用 actor 的主 clip 配置

所以如果后续要求“joint 也完全独立 clip”，那还需要继续改；如果要求只是“prefix 路径具备独立 clip 控制能力”，当前已经满足。

### 回应 5：split-loss 与 joint-loss 是否都支持

答：代码层面都已支持，而且运行验证现在已经同时覆盖 `split` 与 `joint`。

- `split`：已实现，并已在 2x L20 debug train 以及 `split + cont_mean_abs` 最小 smoke train 中跑通
- `joint`：已实现，并已在 2x L20 最小 smoke train 中跑通

因此，这一条现在可以直接表述为：

- “实现完成”
- “split 已验证”
- “joint 已验证”

### 回应 6：KL/reference 是否需要保留

答：按当前实验要求，不需要。

当前统一口径已经明确：

- `USE_KL_LOSS=false`
- rollout token 不做 KL
- prefix token 也不做 KL

这不是暂时关闭，而是当前主实验的正式运行设定。

### 回应 7：当前还剩下什么最值得做

如果只追求把这轮审查彻底收口，剩下最高优先级的是两件事：

1. 跑一轮更长程、更多样本的非 debug run，观察学习曲线是否稳定
2. 如果后续需要更严格的 joint-mode 对照，再补 `joint` 下 prefix 独立 clip 的实现与验证

这样这轮要求中的“实现支持”和“最基本验证”已经齐了，剩下主要是把长期实验覆盖和更严格对照补完整。

---

## 1. 主实验概览

当前这套主实验代码的目标，是在 TextCraft 多轮交互环境里，把“历史 assistant prefix”不仅作为 rollout 的上下文，还作为 actor policy update 的显式优化对象。具体做法不是把 prefix 当成监督微调数据单独训练，而是在 PPO/GRPO 的 actor update 阶段，同时计算两套 policy loss：

1. continuation loss：对 rollout 产生的 response token 做标准 GRPO/PPO-style 更新；
2. prefix loss：对 prompt 中被标记为 assistant prefix 的 token，使用离线缓存的 old logprob 作为旧策略锚点，再用当前 actor 的 prompt-side current logprob 计算第二个 PPO-style loss。

需要先强调当前已经统一的主实验运行口径：

- `OPTIMIZE_PREFIX_TOKENS=true` 是当前主实验默认值；只有显式设置 `OPTIMIZE_PREFIX_TOKENS=false` 时，才会退化成 continuation-only baseline
- `USE_KL_LOSS=false` 是当前主实验默认值；prefix 和 rollout 都不使用 KL/reference 约束
- `ENABLE_ACTIVATION_OFFLOAD=false` 是当前主实验默认值；这是基于本轮 debug 验证后同步的运行设置

因此，按当前默认值直接运行时，进入的是 Prefix-GRPO 主实验路径，而不是 baseline。

---

## 2. 数据集与输入数据说明

### 2.1 当前主实验默认使用的数据文件

主脚本默认数据路径是：

`/Data/wyh/datasets/Verl-Data/train/textcraft/new_prefix_rl/stage7_audit_release/textcraft_prefix_main_train_step200.audited.parquet`

对应脚本位置：

- `run_textcraft_grpo_validated.sh` 的模型和数据默认值定义段
- 训练和验证都覆盖为同一个 `DATA_PATH`

这里需要额外说明：

- 下文如果继续讨论 `textcraft_old_logits/active`、`step200_v2`、`cleaned_v2`，应视为旧链路审计背景
- 它们用于解释 2026-03-30 这次代码审查时的历史问题来源
- 但它们不再代表当前 repo 默认训练入口，也不再代表当前主实验的推荐运行口径

当前主线训练默认使用的是 `stage7_audit_release` 审核版 parquet。

### 2.2 当前 active 数据的实际列

我直接读取了当前 active parquet，实际列为：

| 字段 | 是否存在于当前 active parquet | 语义 |
| --- | --- | --- |
| `data_source` | 是 | 当前为 `textcraft`，reward 路由依赖该字段 |
| `prompt` | 是 | 多轮 chat message 列表，作为训练 prompt 的原始对话结构 |
| `ability` | 是 | 数据能力标签，当前训练主链路未见特殊分支依赖 |
| `reward_model` | 是 | 当前样本中为 `{'ground_truth': '', 'style': 'interaction'}` |
| `extra_info` | 是 | 附加元数据，含 `index` 与 `interaction_kwargs` |
| `assistant_prefix_old_log_probs` | 是 | 离线缓存的 prefix token old logprob |
| `prefix_token_count` | 是 | prefix token 数量 |
| `assistant_prefix_span` | 是 | prefix token span |
| `prefix_mask` | 是 | prefix token 的 0/1 mask |

当前 active parquet **不存在** 顶层列 `item_id`、`sample_idx`。这一点和旧说明不一致，当前代码审查应以实际 parquet schema 为准。

### 2.3 当前 active 数据中关键字段的语义

#### `prompt`

`prompt` 是原始 chat message 列表。`RLHFDataset.__getitem__()` 会把它喂给 `tokenizer.apply_chat_template(..., add_generation_prompt=True)`，再左填充/截断到 `data.max_prompt_length`。对应代码：

- `/Data/wyh/verl/verl/utils/dataset/rl_dataset.py` 第 305-392 行
- 第 451-452 行：当 `return_raw_chat=True` 时，保留 `raw_prompt`

当前主配置 `textcraft_grpo_train.yaml` 中 `data.return_raw_chat: True`（第 19 行），所以 rollout/interaction 端可以拿到原始 message 列表。

#### `extra_info`

`extra_info` 在 dataset 阶段继续拆成：

- `index`
- `tools_kwargs`
- `interaction_kwargs`

对应：

- `/Data/wyh/verl/verl/utils/dataset/rl_dataset.py` 第 458-469 行

当前 active parquet 里，`interaction_kwargs` 的实际 key 集合我做了全量统计，只有：

- `eval_mode`
- `name`
- `prefix_actions`
- `task_id`

也就是说，**当前 active 训练数据里并没有显式存 `goal`、`data_idx`、`item_id`、`session_id`**。这一点很重要，因为它决定了运行时的任务绑定部分仍然要靠 prompt 解析兜底，而不是完全由 parquet 显式提供。

#### `assistant_prefix_old_log_probs`

这是 prefix 分支使用的旧策略 logprob。当前 active parquet 中它是每样本一个变长数组，长度与 `prefix_token_count` 一致；我核查了 active parquet，`len(assistant_prefix_old_log_probs) == prefix_token_count` 对所有样本成立。

训练时它不会直接保持 ragged object array，而是先在 trainer 侧 densify 成 `[B, max_prefix_len]` 的张量，再传给 actor：

- `/Data/wyh/verl/verl/trainer/ppo/ray_trainer.py` 第 1584-1661 行

#### `prefix_mask`

这是 prefix token 的 0/1 mask。它的语义不是“整个 input_ids 上的 mask”，而是**prompt 紧凑坐标系下**的 mask，后续 actor 会再结合左 padding 偏移做 gather。

当前 active parquet 中，`sum(prefix_mask) == prefix_token_count` 对所有样本成立。trainer 侧会再次做一致性校验：

- `/Data/wyh/verl/verl/trainer/ppo/ray_trainer.py` 第 1627-1655 行

#### `prefix_token_count`

它是每个样本 prefix token 的数量，也是：

- `assistant_prefix_old_log_probs` 的长度基准
- `prefix_mask.sum()` 的一致性校验基准
- trainer densify 时的有效宽度依据

当前 active 数据统计：

- 样本数：1093
- `prefix_token_count` 均值：约 162.39
- 最小值：29
- 最大值：656

#### `assistant_prefix_span`

这是 prefix 的 token span。当前样本看起来是形如 `[start, end]` 的区间信息。它保存在 parquet 中并会被 restore 到 batch，但在当前 actor prefix loss 主路径里，**真正参与对齐的是 `prefix_mask` 和 `assistant_prefix_old_log_probs`，不是 `assistant_prefix_span` 本身**。因此它更像辅助审计/回溯字段，而非当前 loss 计算的核心对齐依据。

### 2.4 prefix 数据是如何构建出来的

当前代码里的 prefix old-logprob 数据管线是离线两段式：

#### 第一步：对完整 teacher 轨迹预计算 full-trajectory scalar old logprob

脚本：

`/Data/wyh/verl/examples/sglang_multiturn/my_exp/legacy/prefix_old_logits/precompute_sft_old_logprobs_from_trajectories.py`

关键点：

- 输入是原始完整 teacher 轨迹 jsonl，不是截断后的 prefix parquet（脚本第 1-18 行）
- 用 `AutoModelForCausalLM` teacher-forcing 计算每个真实 next token 的 scalar old logprob（第 236-257 行）
- 同时保留 assistant turn spans，便于后续任意 cut 策略重建

这一步输出 sidecar parquet，例如当前 active 目录里的：

- `textcraft_trajectories_old_logprobs_step200.parquet`
- `textcraft_trajectories_old_logprobs_step460.parquet`

#### 第二步：把 full-trajectory sidecar 对齐到截断 prompt，提取 prefix 部分

脚本：

`/Data/wyh/verl/examples/sglang_multiturn/my_exp/legacy/prefix_old_logits/build_prefix_old_logprob_dataset.py`

关键点：

- 读取训练 parquet、sidecar parquet、原始 jsonl（脚本第 33-67 行）
- 通过 `task_id -> item_id = textcraft_{task_id}` 和 task 内 `cumcount` 做 join（第 70-100 行）
- 按“截断后的 prompt 中前 K 个 assistant turns”重新对齐 full conversation tokenization（第 134-241 行）
- 输出：
  - `prefix_old_logprobs`
  - `prefix_mask`
  - `assistant_prefix_span`

这个脚本明确修的是“full trajectory 坐标系”和“训练 prompt 截断坐标系”不一致的问题，不是简单切片。

#### 第三步：字段适配

脚本：

`/Data/wyh/verl/examples/sglang_multiturn/my_exp/legacy/prefix_old_logits/adapt_data_for_training.py`

作用：

- 把 `prefix_old_logprobs` 重命名为训练代码期望的 `assistant_prefix_old_log_probs`（第 36-43 行）

不过当前 active parquet 最终并未保留该脚本中可选添加的顶层 `item_id`/`sample_idx` 字段，因此不能把这两个字段当成当前主线数据的既有 schema。

### 2.5 prefix old logprob 当前到底怎么来的

结论分三层：

1. **来源类型**：离线缓存，不是 online rollout 现场产生。
2. **生成逻辑**：来自 `precompute_sft_old_logprobs_from_trajectories.py` 对完整 teacher 轨迹做 teacher-forced scalar old logprob 计算，再由 `build_prefix_old_logprob_dataset.py` 对齐到截断 prompt。
3. **具体 checkpoint 版本**：从当前 active parquet 文件名和 sidecar 文件名看，它对应的是 `step200` 版本；但 **当前 parquet 内没有保存更强的 provenance 元数据**，我不能仅靠代码进一步确认当时使用的是哪个具体 checkpoint 绝对路径，只能确认“是离线 step200 版本缓存链路”，不能脑补更细。

### 2.6 active 主线数据与 cleaned 数据的角色

当前 active 主线数据：

- `/Data/wyh/datasets/Verl-Data/outputs/textcraft_old_logits/active/textcraft_validated_prefix_history_canonicalized_with_prefix_old_logprobs_step200_v2.parquet`
- 1093 条
- **主脚本默认直接使用它**

当前 cleaned_v2 数据：

- `/Data/wyh/datasets/Verl-Data/outputs/textcraft_old_logits/active/cleaned_v2/textcraft_validated_cleaned_v2_20260326_000658.parquet`
- 762 条
- 来自 `audit_data_quality_v2.py`
- 当前主脚本默认**不**切到 cleaned_v2

审计脚本：

`/Data/wyh/verl/examples/sglang_multiturn/my_exp/short_learning_validation/audit_data_quality_v2.py`

其 v2 规则中：

- placeholder pollution、`prefix_token_count`/old-logprob 不一致、missing_fields 是 hard delete
- prefix action mismatch、多 action 拼接是 suspicious flag
- task-level inconsistency 只删非众数版本

因此，cleaned_v2 的角色更像“数据质量保守子集”，不是当前主实验默认主线。

---

## 3. 训练流程的端到端逻辑

下面按当前代码实际执行顺序，串起从 parquet 到 loss 的链路。

### 3.1 样本如何从 parquet 读入

`RLHFDataset` 读取 parquet 后：

1. 保留原始行字段；
2. 把 `prompt` 走 chat template，得到 `input_ids/attention_mask/position_ids`；
3. 把 `raw_prompt` 保留下来；
4. 从 `extra_info` 拆出 `index`、`tools_kwargs`、`interaction_kwargs`。

对应代码：

- `/Data/wyh/verl/verl/utils/dataset/rl_dataset.py` 第 167-188 行：读 parquet
- 第 305-470 行：`__getitem__`
- 第 38-81 行：`collate_fn`，非 tensor 字段变成 `np.ndarray(dtype=object)`

### 3.2 prefix/history 如何进入 interaction 与 replay

rollout 之前，trainer 会调用 `_get_gen_batch()`，保留 `extra_info` 等 reward/interaction 所需非 tensor 字段，同时把 `input_ids/attention_mask/position_ids` pop 到 generation batch：

- `/Data/wyh/verl/verl/trainer/ppo/ray_trainer.py` 第 850-870 行

在 agent loop 里：

- `ToolAgentLoop.run()` 从 `kwargs["extra_info"]["interaction_kwargs"]` 取 `prefix_actions`
- 再把 `raw_prompt` 注入给 interaction
- 调用 `interaction.start_interaction(request_id, **interaction_kwargs_with_prompt)`

对应：

- `/Data/wyh/verl/verl/experimental/agent_loop/tool_agent_loop.py` 第 156-174 行

### 3.3 环境如何创建

当前 TextCraft interaction 配置：

- `/Data/wyh/verl/examples/sglang_multiturn/config/interaction_config/textcraft_interaction.yaml` 第 4-17 行

运行时的 `TextCraftInteraction.start_interaction()` 做了几件事：

1. 取出 `prefix_actions`（第 31-39 行）
2. 优先读 `goal`；若没有，则从 `prompt` 中解析 `Goal: craft ...`（第 40-57 行）
3. 可选读取 `data_idx`（第 59-67 行）
4. 调 `/create` 创建环境（第 69-84 行）
5. 从返回 observation 中再解析实际 goal，并与 `expected_goal` 做 fail-fast 校验（第 86-124 行）
6. 若有 `prefix_actions`，逐步 replay（第 153-206 行）

注意：**当前 active parquet 不含 `goal` 和 `data_idx`，所以训练主线至少当前默认数据下，goal 绑定是靠 prompt 解析兜底，`data_idx` 的确定性任务绑定没有实际用上。**

### 3.4 rollout 如何生成 response

agent loop 的 sampling params 明确来自 rollout config：

- `temperature`
- `top_p`
- `logprobs=config.calculate_log_probs`

对应：

- `/Data/wyh/verl/verl/experimental/agent_loop/agent_loop.py` 第 338-344 行

而主脚本把 `actor_rollout_ref.rollout.calculate_log_probs=$CALCULATE_LOG_PROBS`，默认值是 `true`（主脚本第 82 行和第 373 行）。因此，**当前主脚本不是 `calculate_log_probs=false`，而是默认会让 rollout 端产出 response-side `rollout_log_probs`。**

agent loop 后处理时：

- `raw_prompt` 被塞回 `extra_fields`（第 436-438 行）
- 若 rollout 端有 `response_logprobs`，则合并成 batch 里的 `rollout_log_probs`（第 619-621 行）

### 3.5 reward 如何计算

TextCraft rollout 的逐轮环境 reward 被记录在 `turn_scores` 中：

- `/Data/wyh/verl/verl/experimental/agent_loop/tool_agent_loop.py` 第 84 行
- 第 218 行：写入 `extra_fields`
- 第 245 行、第 519 行：逐轮 append reward

trainer 在 reward manager 里：

1. 从 `data.non_tensor_batch["turn_scores"]` 取出 rollout reward；
2. 规范化写回 `extra_info["turn_scores"]`；
3. `default_compute_score()` 对 `data_source=="textcraft"` 时返回 `sum(turn_scores)`；
4. 最终把这个标量 reward 填到 response 最后一个有效 token 位置。

对应代码：

- `/Data/wyh/verl/verl/workers/reward_manager/naive.py` 第 64-123 行
- `/Data/wyh/verl/verl/utils/reward_score/__init__.py` 第 115-126 行

### 3.6 advantages 如何计算

当前配置 `algorithm.adv_estimator=grpo`（主脚本第 344 行）。

trainer 侧在 reward 后调用 `compute_advantage(...)`，最终落到 `compute_grpo_outcome_advantage()`：

- `/Data/wyh/verl/verl/trainer/ppo/ray_trainer.py` 第 1813-1826 行
- `/Data/wyh/verl/verl/trainer/ppo/core_algos.py` 第 527-551 行

当前 GRPO advantage 的实现是：

1. 先把每条 trajectory 的 `token_level_rewards` 沿 response 维度求和，得到 scalar `score`；
2. 按同一 prompt 的多个 rollout（通过 `uid` 分组）做组内 baseline；
3. 再把这个 scalar 扩展回 response token 维度，并乘上 `response_mask`。

因此，continuation advantage 是 response-side 的 token mask 上广播的 trajectory-level 标量。

### 3.7 old logprobs / current logprobs 如何准备

#### continuation old logprob

当前主训练路径里，trainer 在 rollout 完成、reward 计算之后，会再次调用 actor worker 的 `compute_log_prob(batch)` 来重算 `old_log_probs`：

- `/Data/wyh/verl/verl/trainer/ppo/ray_trainer.py` 第 1717-1749 行

actor worker 内部又强制：

- `data.meta_info["logprob_temperature"] = 1.0`

对应：

- `/Data/wyh/verl/verl/workers/fsdp_workers.py` 第 990-1002 行

所以当前主实验里 continuation old logprob 的旧策略锚点，不是 offline 数据，也不是默认直接拿 rollout server 回传的 `rollout_log_probs`，而是**trainer 在 rollout 后用 actor worker 重新计算得到的 `old_log_probs`**。

#### continuation current logprob

actor update 阶段，`dp_actor.update_policy()` 用：

- `_forward_micro_batch(..., return_full_seq=False)`

计算 response token 的 current logprob：

- `/Data/wyh/verl/verl/workers/actor/dp_actor.py` 第 583-587 行

#### prefix old logprob

prefix old logprob 来自 parquet 的离线缓存 `assistant_prefix_old_log_probs`。trainer 会先把 ragged object array densify 成 tensor：

- `/Data/wyh/verl/verl/trainer/ppo/ray_trainer.py` 第 1584-1661 行

#### prefix current logprob

actor update 阶段，`dp_actor.update_policy()` 再做一次：

- `_forward_micro_batch(..., return_full_seq=True)`

得到 prompt 部分的 current logprob：

- `/Data/wyh/verl/verl/workers/actor/dp_actor.py` 第 598-603 行

### 3.8 actor 更新时 prefix 分支和 continuation 分支如何共同参与 loss

只有当 `optimize_prefix_tokens=true` 时，actor 才进入 prefix 路径。对应：

- `/Data/wyh/verl/verl/trainer/ppo/ray_trainer.py` 第 1845-1849 行
- `/Data/wyh/verl/verl/workers/actor/dp_actor.py` 第 576-580 行

此时 actor 会：

1. 算 continuation current logprob
2. 取 continuation old logprob
3. 算 prompt-side current logprob
4. 用 `prefix_mask + pad_offset` 从 prompt current logprob 里 gather 出 prefix current logprob
5. 用 cached `assistant_prefix_old_log_probs` 作为 prefix old logprob
6. 分别算 continuation loss 和 prefix loss
7. 按权重线性相加成总 policy loss

所以当前代码里：

- continuation 部分是 **online rollout + online old-logprob recompute**
- prefix 部分是 **offline old-logprob cache + online current-logprob recompute**

---

## 4. 损失函数设计（重点）

## 4.1 continuation 分支

### current logprob

continuation current logprob 由 actor 当前参数前向得到：

`log_prob_resp = _forward_micro_batch(..., return_full_seq=False)`

对应：

- `/Data/wyh/verl/verl/workers/actor/dp_actor.py` 第 583-587 行

### old logprob

当前主配置下，continuation old logprob 最终取：

`cont_old_log_prob = model_inputs["old_log_probs"]`

进入该分支的条件是：

- `use_rollout_log_probs` 未启用
- `on_policy` 为 `False`

对应：

- `/Data/wyh/verl/verl/workers/actor/dp_actor.py` 第 589-597 行

为什么当前主实验里 `on_policy=False`：

- rollout 展开后每步样本数是 `train_batch_size * rollout.n = 32 * 4 = 128`
- actor `ppo_mini_batch_size = 32`
- 因此 `mini_batches = data.split(32)` 后不是单个 mini-batch，而是 4 个
- 同时 actor 默认 `ppo_epochs=1`

对应：

- `/Data/wyh/verl/verl/workers/actor/dp_actor.py` 第 487-490 行
- `/Data/wyh/verl/verl/trainer/config/actor/actor.yaml` 第 116-117 行

### advantage

continuation advantage 来自 trainer 侧的 GRPO advantage：

- `/Data/wyh/verl/verl/trainer/ppo/ray_trainer.py` 第 1818-1826 行
- `/Data/wyh/verl/verl/trainer/ppo/core_algos.py` 第 527-551 行

其本质是：

```text
score_b = sum_t token_level_rewards[b, t]
adv_b = group_normalized(score_b within same prompt's rollout group)
A_cont[b, t] = adv_b * response_mask[b, t]
```

### continuation loss 公式

continuation loss 走标准 `compute_policy_loss_vanilla()`：

- `/Data/wyh/verl/verl/workers/actor/dp_actor.py` 第 826-833 行
- `/Data/wyh/verl/verl/trainer/ppo/core_algos.py` 第 950-1057 行

核心定义：

```text
ratio_cont = exp(log_prob_current_cont - log_prob_old_cont)
L_cont = PPOClip(ratio_cont, A_cont, response_mask)
```

代码里更具体是：

```text
negative_approx_kl = log_prob - old_log_prob
ratio = exp(negative_approx_kl)
pg_loss = clipped objective aggregated under response_mask
```

见：

- `/Data/wyh/verl/verl/trainer/ppo/core_algos.py` 第 1001-1057 行

## 4.2 prefix 分支

### current logprob

prefix current logprob 不是直接从 dataset 里来，而是 actor 对整条 `input_ids` 再做一次 prompt-side forward：

```text
log_prob_prefix = _forward_micro_batch(..., return_full_seq=True)
```

得到的是 prompt 区域上的 dense logprob，坐标系已经带左 padding：

- `/Data/wyh/verl/verl/workers/actor/dp_actor.py` 第 598-603 行

### old logprob

prefix old logprob 来自 offline cache：

- parquet 字段：`assistant_prefix_old_log_probs`
- trainer densify 后传给 actor

对应：

- `/Data/wyh/verl/verl/trainer/ppo/ray_trainer.py` 第 1584-1661 行

### prefix 的 mask / span / token count 如何参与对齐

当前真正参与 prefix 对齐的是：

- `prefix_mask`
- `prefix_token_count`
- `assistant_prefix_old_log_probs`

`assistant_prefix_span` 虽然在 batch 中被 restore，但当前 actor prefix 主路径并不直接用它做 gather。

actor 中的 gather 逻辑是：

1. `prefix_mask` 在紧凑 prompt 坐标系里标出 prefix token
2. `actual_prompt_len - compact_prompt_len` 算出左 padding 偏移 `pad_offset`
3. prefix token 的“被预测位置”使用 `dense_next_token_pos = compact_pos + pad_offset - 1`
4. 从 `log_prob_prefix[n]` 中 gather 出当前 prefix logprob

对应：

- `/Data/wyh/verl/verl/workers/actor/dp_actor.py` 第 613-717 行

### prefix advantage 怎么定义

当前实现里，prefix advantage 不是单独从 prefix reward 估计出来的，而是：

```text
trajectory_reward = advantages.mean(dim=1, keepdim=True)
prefix_advantages = trajectory_reward.expand(-1, max_num_prefix)
```

对应：

- `/Data/wyh/verl/verl/workers/actor/dp_actor.py` 第 773-778 行

这意味着当前 prefix advantage 的定义是：

```text
A_prefix[b, k] = mean_j A_cont[b, j]
```

并广播到该样本所有 prefix token。

需要注意：这里的 `advantages` 已经带了 response mask，因此无效 response 位置是 0；但 `.mean(dim=1)` 仍然是对整条 response 宽度求平均，而不是只对有效 response token 求 masked mean。这是当前实现中的一个明确设计选择，是否最合理，仍值得重点复核。

### prefix loss 公式

prefix loss 也直接复用 `compute_policy_loss_vanilla()`：

- `/Data/wyh/verl/verl/workers/actor/dp_actor.py` 第 779-788 行

因此：

```text
ratio_prefix = exp(log_prob_current_prefix - log_prob_old_prefix)
L_prefix = PPOClip(ratio_prefix, A_prefix, prefix_mask_gathered)
```

### prefix 分支和 continuation 分支的关系

当前代码不是“prefix 替代 continuation”，而是“双分支并行，再线性相加”：

- continuation：response token
- prefix：prompt 中被标为 assistant prefix 的 token

两者共享同一个当前 actor，但旧策略锚点不同：

- continuation old logprob：trainer 重新计算的 `old_log_probs`
- prefix old logprob：offline cached `assistant_prefix_old_log_probs`

## 4.3 总损失

当前代码对应的总损失就是：

```text
policy_loss = continuation_loss + prefix_loss_weight * prefix_loss
```

对应：

- `/Data/wyh/verl/verl/workers/actor/dp_actor.py` 第 835-837 行

`prefix_loss_weight` 的配置来源：

- shell 默认：`PREFIX_LOSS_WEIGHT=${PREFIX_LOSS_WEIGHT:-1.0}`  
  `/Data/wyh/verl/examples/sglang_multiturn/my_exp/rl/run_textcraft_grpo_validated.sh` 第 67-68 行
- Hydra override：脚本第 389-390 行
- 运行期写入 batch meta：`ray_trainer.py` 第 1846-1848 行

当前默认值：

```text
prefix_loss_weight = 1.0
```

这意味着一旦启用 prefix 分支，代码层面默认给予 continuation loss 和 prefix loss 同量纲线性相加的权重。当前代码里没有更复杂的 schedule、warmup 或 token-count normalization。

## 4.4 ratio 的定义和含义

当前 PPO/GRPO ratio 的定义在 continuation 和 prefix 两边都相同：

```text
ratio = exp(log_prob_current - log_prob_old)
```

对应：

- `/Data/wyh/verl/verl/trainer/ppo/core_algos.py` 第 1001-1005 行

于是：

- `ratio_cont`：当前 actor 对 response token 的概率，相对于 old continuation policy 的变化倍数
- `ratio_prefix`：当前 actor 对 prefix token 的概率，相对于 cached prefix old policy 的变化倍数

当前代码还单独记录了 prefix ratio 统计：

- `/Data/wyh/verl/verl/workers/actor/dp_actor.py` 第 859-876 行

smoke log 中也能看到这些指标实际被打出来，例如：

- `actor/prefix_ratio_mean`
- `actor/prefix_ratio_min`
- `actor/prefix_ratio_max`

## 4.5 是否启用 KL / reference

基于当前主脚本、主配置以及本轮统一后的实验要求，结论很明确：

1. 主实验脚本默认 `USE_KL_LOSS=false`  
   `run_textcraft_grpo_validated.sh` 默认参数段
2. 主配置 `algorithm.use_kl_in_reward: False`  
   `textcraft_grpo_train.yaml` 第 99-102 行
3. actor 配置 `use_kl_loss: False`  
   `textcraft_grpo_train.yaml` 第 37-41 行
4. 按当前实验要求，当前主实验对 rollout token 和 prefix token 都不施加 KL/reference 约束

所以当前主实验**不启用 KL / reference**，而且这个结论不是“暂时没开”，而是当前明确采用的实验口径。

在 trainer 里，若 `self.use_reference_policy=False`，会明确跳过 ref logprob 计算：

- `/Data/wyh/verl/verl/trainer/ppo/ray_trainer.py` 第 1751-1761 行

在 worker 初始化路径里，即使保留了 ref 兼容逻辑，只要 KL 关闭，就不会走主实验链路。并且当前 prefix 路径里，即使强行打开 `use_kl_loss`，actor 代码在 `optimize_prefix_tokens=True` 分支下也只是记 `actor/kl_disabled=1.0`，并不真正把 KL 加进 policy loss：

- `/Data/wyh/verl/verl/workers/actor/dp_actor.py` 第 889-899 行

因此应当把“存在兼容代码”和“当前主实验实际使用”严格区分：**兼容逻辑存在，但当前主实验不使用。**

---

## 5. prefix 的技术实现细节

### 5.1 `assistant_prefix_old_log_probs`

这是 prefix 分支的旧策略锚点。当前主链路中它：

1. 离线构建；
2. 进入 parquet；
3. 在 trainer 中从 ragged object array densify 成 dense tensor；
4. 传给 actor 参与 PPO-style prefix loss。

因此它不是分析信号，而是当前 prefix 分支的直接输入。

### 5.2 `prefix_mask`

这是 prefix token 的权威对齐信号。当前 trainer 的 `compute_prefix_mask()` 明确写着：

- 优先相信 parquet 里的预计算 `prefix_mask`
- 没有时才 fallback 到 `raw_prompt` 现算

对应：

- `/Data/wyh/verl/verl/trainer/ppo/ray_trainer.py` 第 248-405 行

但在当前主实验默认数据里，`prefix_mask` 是存在的，所以主路径应是“使用预计算 mask”，不是 runtime 重新推断。

### 5.3 `prefix_token_count`

`prefix_token_count` 是 densify 和一致性校验的核心字段。trainer 当前做了两层 fail-fast：

1. `len(cached_olp[b]) == prefix_token_count[b]`
2. `prefix_mask.sum() == prefix_token_count[b]`

对应：

- `/Data/wyh/verl/verl/trainer/ppo/ray_trainer.py` 第 1614-1623 行
- 第 1645-1655 行

### 5.4 `assistant_prefix_span`

当前 active parquet 里保留了 `assistant_prefix_span`，trainer 也会 restore 它：

- `/Data/wyh/verl/verl/trainer/ppo/ray_trainer.py` 第 1486-1494 行

但在 actor prefix loss 主路径里，它没有直接参与 gather。当前实现真正依赖的对齐信号仍是 `prefix_mask`。因此它更适合作为审计/回溯信息。

### 5.5 replay prefix actions

prefix 不只进入 loss，也进入 rollout 初始状态构造：

- `interaction_kwargs["prefix_actions"]` 被传给 `TextCraftInteraction.start_interaction()`
- 环境创建后 `_replay_prefix_actions()` 逐步回放这些 action
- 学生策略从 replay 后的环境状态继续 rollout

对应：

- `/Data/wyh/verl/verl/experimental/agent_loop/tool_agent_loop.py` 第 156-174 行
- `/Data/wyh/verl/verl/interactions/textcraft_interaction.py` 第 153-206 行

所以当前 Prefix-GRPO 不是“只在 loss 层利用 prefix”，而是：

1. rollout 初始状态由 prefix replay 决定；
2. policy loss 里额外再优化 prefix token 概率。

### 5.6 prefix 对齐 / gather 逻辑

当前 actor 的 prefix gather 逻辑是这套实现最关键也最容易出 bug 的部分：

1. prompt-side forward 得到 `log_prob_prefix[B, actual_prompt_len]`
2. `prefix_mask[B, compact_prompt_len]` 给出紧凑坐标
3. 用 `pad_offset = actual_prompt_len - compact_prompt_len`
4. 用 `dense_next_token_pos = compact_pos + pad_offset - 1`
5. 从 `log_prob_prefix` gather 出当前 prefix logprob

对应：

- `/Data/wyh/verl/verl/workers/actor/dp_actor.py` 第 613-717 行

这段代码已经明确写了 off-by-one 说明，意图是正确的：logprob 位置对应“预测下一个 token”的坐标。

### 5.7 trainer materialization 逻辑

这是当前 Prefix-GRPO 能真正跑通的关键工程点之一。

trainer 在 rollout 后、union 之前，先把 prefix 相关字段从 `batch.non_tensor_batch` restore 到 `batch.batch`：

- `assistant_prefix_old_log_probs`
- `prefix_token_count`
- `prefix_mask`
- `assistant_prefix_span`

然后再做 densify，写回 dense tensor，供 Ray/actor worker 序列化和前向使用。

对应：

- `/Data/wyh/verl/verl/trainer/ppo/ray_trainer.py` 第 1478-1688 行

### 5.8 actor 侧 prefix current logprob 的计算方式

actor 并不是对 prefix 单独构造一条新序列，而是对完整 `input_ids` 做一次 full-seq forward，然后截 prompt 部分：

- `/Data/wyh/verl/verl/workers/actor/dp_actor.py` 第 95-341 行：`_forward_micro_batch()`
- 第 271-277 行：`return_full_seq=True` 时返回 prompt 区域 logprob

因此，prefix current logprob 是当前 actor 在“当前 prompt 上”对这些 prefix token 的 next-token logprob。

### 5.9 当前代码如何确保 current / old 的温度语义一致

这是当前实现里的一个重要工程处理。

rollout sampling temperature 和 logprob 计算 temperature 被显式分开：

- rollout sampling 使用 `rollout.temperature`
- old/current logprob 计算统一使用 `logprob_temperature = 1.0`

对应：

- trainer 启动 batch 时：`ray_trainer.py` 第 1397-1399 行
- actor update 前：`ray_trainer.py` 第 1840-1843 行
- actor worker 重新算 old logprob 时：`fsdp_workers.py` 第 990-995 行
- actor 侧断言温度必须 `> 0`：`dp_actor.py` 第 103 行、第 437-438 行

因此，当前 continuation 和 prefix 的 old/current logprob，语义上都按 temperature 1.0 比较，而不是按采样温度比较。这样可以避免 `temperature=0` 导致 logprob 计算异常，也避免把 sampling 温度混进 PPO ratio。

### 5.10 当前 prefix 到底是“真实参与优化”还是“仅分析信号”

结论：**在 `optimize_prefix_tokens=true` 时，prefix 是真实参与优化的，不只是分析信号。**

证据有三层：

1. actor 代码明确计算 `prefix_loss` 并加到 `policy_loss`  
   `/Data/wyh/verl/verl/workers/actor/dp_actor.py` 第 779-837 行
2. trainer 只在启用 prefix 时把 `optimize_prefix_tokens` 和 `prefix_loss_weight` 传给 actor  
   `/Data/wyh/verl/verl/trainer/ppo/ray_trainer.py` 第 1845-1849 行
3. smoke log 中确实出现了：
   - `actor/use_cached_prefix_old_logprob: True`
   - `actor/prefix_loss`
   - `actor/prefix_ratio_mean`
   - `actor/prefix_ppo_kl`

但也要如实说明：当前给出的 smoke log 中 reward/advantage 全为 0，因此那次 smoke run 只能证明“prefix 分支进入了更新路径并产生了指标”，**不能证明在非零 reward 下 prefix 一定产生了有效非零梯度。**

### 5.11 当前实现中的关键修复点

与 prefix 直接相关、并且对“当前实现能否成立”有帮助的修复，主要有：

1. trainer 侧 ragged prefix sidecar materialization  
   当前通过 restore + densify，避免 rollout.n 重复扩展时 prefix object array 和 batch 尺寸错位。
2. logprob 温度固定为 1.0  
   避免 `temperature=0` 语义进入 logprob 路径。
3. prompt 截断坐标系下重新构造 prefix old logprob  
   避免 full trajectory tokenization 与训练 prompt tokenization 不一致。

---

## 6. 训练配置与关键超参数

以下先写“主脚本实际 override 后的默认值”，再解释它们的作用。

| 项目 | 当前主脚本默认值 | 说明 |
| --- | --- | --- |
| 模型路径 | `/Data/public/Qwen3-1.7B` | `MODEL_PATH` |
| 训练数据 | active `...step200_v2.parquet` | 主线默认数据 |
| 验证数据 | 同训练数据 | 脚本把 `data.val_files=$DATA_PATH` |
| `train_batch_size` | `32` | 每步 prompt 数 |
| `rollout.n` | `4` | 每个 prompt 采样 4 条 rollout |
| `ppo_mini_batch_size` | `32` | actor update 的 mini-batch 大小 |
| `ppo_micro_batch_size_per_gpu` | `16` | actor 侧显存切分 |
| `max_prompt_length` | `2048` | dataset 侧 prompt 上限 |
| `max_response_length` | `4096` | response 上限 |
| `rollout.prompt_length` | `4096` | rollout server 允许的 prompt 长度 |
| `max_model_len` | `10240` | vLLM 总上下文上限 |
| `ppo_max_token_len_per_gpu` | `12288` | PPO/update token 上限 |
| `temperature` | `1.0` | rollout 采样温度 |
| `top_p` | `0.95` | rollout nucleus sampling |
| `max_num_batched_tokens` | `10240` | vLLM batching 限制 |
| `max_num_seqs` | `64` | vLLM 同时序列数 |
| `learning_rate` | `5e-6` | actor lr |
| `total_epochs` | `10` | trainer 训练 epoch |
| `save_freq` | `5` | 每 5 step/末步存档 |
| `test_freq` | `5` | 每 5 step/末步验证 |
| `optimize_prefix_tokens` | `true` | 当前主实验默认启用 prefix |
| `prefix_loss_weight` | `1.0` | prefix loss 权重 |
| `use_kl_loss` | `false` | 当前主实验禁用 KL |
| `enable_activation_offload` | `false` | 当前主实验默认关闭 activation offload |

主要来源：

- `/Data/wyh/verl/examples/sglang_multiturn/my_exp/rl/run_textcraft_grpo_validated.sh` 第 54-99 行
- 第 341-392 行：最终 Hydra override

### 6.1 模型 / dtype / FSDP / offload / checkpointing / vLLM

当前主脚本显式设置：

- gradient checkpointing：开启  
  `run_textcraft_grpo_validated.sh` 第 353 行
- activation offload：关闭  
  当前主实验默认口径已同步为关闭，以避开本轮 debug 中触发的 `activation_offload.py` 运行时错误
- actor FSDP param offload：开启  
  第 355 行
- ref FSDP param offload：开启  
  第 356 行

主配置里：

- actor/ref `model_dtype: bfloat16`
- `hybrid_engine: True`
- rollout backend: `vllm`

对应：

- `/Data/wyh/verl/examples/sglang_multiturn/config/textcraft_grpo_train.yaml` 第 22-46 行
- 第 47-89 行

### 6.2 TextCraft interaction 相关配置

当前 interaction config：

- `class_name: verl.interactions.textcraft_interaction.TextCraftInteraction`
- `env_server_base: http://127.0.0.1:36001`
- `timeout: 600`
- `max_retries: 3`

对应：

- `/Data/wyh/verl/examples/sglang_multiturn/config/interaction_config/textcraft_interaction.yaml` 第 4-17 行

server 启动入口：

- `/Data/wyh/h200_grpo/envs/AgentGym/agentenv-textcraft/agentenv_textcraft/launch.py` 第 9-15 行

### 6.3 `train_batch_size × rollout.n` 之后，数据怎样进入 PPO 更新

这是当前配置里最容易混淆的部分，按当前代码应这样理解：

1. dataloader 每步先给 trainer `train_batch_size=32` 个 prompt
2. trainer 做 `batch.repeat(repeat_times=rollout.n, interleave=True)`，展开成 `32 * 4 = 128` 条 rollout trajectory  
   `/Data/wyh/verl/verl/trainer/ppo/ray_trainer.py` 第 1416-1418 行、第 1460-1464 行
3. actor update 阶段，`update_policy()` 按 `ppo_mini_batch_size=32` 把这 128 条数据切成 4 个 mini-batch  
   `/Data/wyh/verl/verl/workers/actor/dp_actor.py` 第 485-489 行
4. 每个 mini-batch 再按 `ppo_micro_batch_size_per_gpu=16` 继续切 micro-batch，用于梯度累积/显存控制  
   `/Data/wyh/verl/verl/workers/actor/dp_actor.py` 第 497-505 行

因此：

- **决定“每轮 rollout 数据如何被分批参与 PPO 更新”的参数**：`ppo_mini_batch_size`
- **主要决定显存切分和梯度累积的参数**：`ppo_micro_batch_size_per_gpu`

更直白地说：

```text
32 prompts
-> 每个 prompt rollout 4 次
-> 128 trajectories
-> 按 ppo_mini_batch_size=32 切成 4 个 mini-batches
-> 每个 mini-batch 再按 micro_batch_size_per_gpu=16 切小做反传
```

这也是为什么在当前主配置下 continuation old logprob 分支里 `on_policy=False`。

---

## 7. 当前代码中的关键修复与踩坑历史

下面只列对理解当前实现真正有帮助的关键点。

### 7.1 trainer 侧 prefix ragged materialization 问题

这是历史 bug，同时也是当前实现的一部分工程修复。

问题本质：

- parquet 中 prefix sidecar 是 ragged object array
- rollout.n 重复扩展后，若 restore/materialize 顺序不对，很容易 batch size 对不齐

当前修法：

- 先从 `batch.non_tensor_batch` restore 回 `batch.batch`
- 再 densify 成可序列化 tensor

对应：

- `/Data/wyh/verl/verl/trainer/ppo/ray_trainer.py` 第 1478-1688 行

### 7.2 actor 侧 `temperature=0` 导致 logprob NaN 的问题

这是历史风险，当前实现通过“sampling 温度和 logprob 温度分离”规避。

当前代码中：

- logprob 计算温度强制 `1.0`
- actor 对 logprob 温度做 `> 0` 断言

对应：

- `/Data/wyh/verl/verl/workers/actor/dp_actor.py` 第 103 行、第 392-393 行、第 437-438 行
- `/Data/wyh/verl/verl/workers/fsdp_workers.py` 第 993-994 行

### 7.3 TextCraft 任务绑定 mismatch（`item_id` vs `task_id`）问题

这类问题分成两个层面：

#### 离线 join 层

`build_prefix_old_logprob_dataset.py` 当前明确把训练样本里的 `task_id` 映射成 `item_id = textcraft_{task_id}`，再用 task 内 `cumcount` 做 join：

- `/Data/wyh/verl/examples/sglang_multiturn/my_exp/legacy/prefix_old_logits/build_prefix_old_logprob_dataset.py` 第 70-100 行

这是离线 sidecar 对齐层的关键修复。

#### 运行时 env 绑定层

当前 `TextCraftInteraction.start_interaction()` 支持显式 `goal` / `data_idx`，并做 goal fail-fast 校验：

- `/Data/wyh/verl/verl/interactions/textcraft_interaction.py` 第 40-67 行、第 86-124 行

但当前 active 主线 parquet **并没有显式提供 `goal` 和 `data_idx`**。所以就当前默认数据而言，运行时主要依赖：

- 从 prompt 中解析 goal
- 对 server 返回 observation 再次校验 goal

这部分不是“完全通过 parquet 显式绑定”，这一点需要在审查时说清楚。

验证脚本：

- `/Data/wyh/verl/examples/sglang_multiturn/my_exp/short_learning_validation/audit/verify_goal_binding_fix.py`

### 7.4 `turn_scores` 在 trainer 中被 pop 丢失导致 reward 为 0 的问题

这是历史 bug，当前修法是：

- `_get_gen_batch()` 把 `turn_scores` 保留在 `reward_model_keys` 中，不随 generation pop 掉
- reward manager 再把它注入 `extra_info["turn_scores"]`
- `default_compute_score()` 对 textcraft 返回 `sum(turn_scores)`

对应：

- `/Data/wyh/verl/verl/trainer/ppo/ray_trainer.py` 第 850-866 行
- `/Data/wyh/verl/verl/workers/reward_manager/naive.py` 第 96-113 行
- `/Data/wyh/verl/verl/utils/reward_score/__init__.py` 第 115-126 行

### 7.5 cleaned 数据与审计脚本的角色

这是工具链，不是主实验 loss 设计的一部分。

- `audit_data_quality_v2.py` 的角色是生成保守 cleaned 子集和 corruption report
- 当前主实验默认入口已切到 `new_prefix_rl/stage7_audit_release`，不会自动切到 cleaned_v2

因此不应把 cleaned_v2 包装成“当前主实验必要设计”；它更像主线之外的数据审计/筛查资产。

---

## 8. 当前实验的已知限制与剩余风险

### 8.1 已经解决的关键 blocker

当前代码层面已明确打通的关键点包括：

1. prefix old logprob 已经有离线缓存链路
2. trainer 能把 ragged prefix sidecar 正确 materialize 给 actor
3. actor 确实存在 prefix loss 分支并参与总 loss
4. TextCraft rollout 能 replay prefix actions
5. reward 链路里 `turn_scores` 已能保留下来并参与 textcraft reward
6. KL/reference 已被显式关闭，主实验路径更单纯

### 8.2 仍然存在的风险

#### 风险 1：文档中保留了旧 active 数据审计背景，阅读时需要和当前默认入口区分

本文保留了大量 2026-03-30 对 `textcraft_old_logits/active`、`step200_v2` 和旧实现的静态审查内容，用于解释历史问题来源；但当前默认训练入口已切到 `new_prefix_rl/stage7`。如果读者不区分“历史审计背景”和“当前运行口径”，很容易把旧结论误当成当前默认行为。

#### 风险 2：当前数据仍然缺少部分显式环境绑定字段

从旧数据审查看，样本里并不总是显式提供 `goal/data_idx` 等环境绑定字段。当前主实验运行路径已经通过 prompt 解析与 env reset 优先级修补，使这件事不再是 blocker；但从“可审查性”和“可追责性”看，后续如果能把这些字段显式落到 parquet，仍然更稳。

#### 风险 3：`joint` 模式下仍与 continuation 共用 clip 配置

当前统一口径里已经支持 `split` / `joint` 两种 prefix loss 模式，也支持多种 prefix advantage 模式；但在 `joint` 模式下，prefix 仍和 continuation 共用 actor 的 clip 配置。若后续要求更严格的 joint-mode 对照，这一点仍需单独注意。

#### 风险 4：activation offload 路径仍不建议直接重开

2026-04-02 的 debug 验证已经表明，`activation_offload.py` 在当前这套 TextCraft Prefix-GRPO 路径里会触发 `KeyError: 27`。因此当前主实验默认明确采用 `ENABLE_ACTIVATION_OFFLOAD=false`。如果未来有人重新开启它，需要先补专门验证，而不能直接沿用旧的大模型省显存经验。

旧文档里对 prefix gather / 长度校验的静态风险分析，已经在后续 H200 落地路径中得到针对性修复与 debug 验证；当前更现实的剩余风险反而是“有人忽略最新口径，重新把 activation offload 打开”。

#### 风险 5：短程 smoke 已证明非零 learning signal，但仍不能替代长期验证

新增的非 debug `stage7` 短程 smoke log 中已经出现：

- 非零 `critic/rewards/mean`
- 非零 `critic/advantages/mean`
- 非零 `actor/prefix_loss`

因此当前不再只是“路径活着”，而是已经证明 prefix 分支在非零 reward 条件下进入了更新；但它仍然不能替代更长程训练对学习收益和稳定性的验证。

### 8.3 工程已通但研究上仍未完全验证的点

1. prefix advantage 的定义是否足够合理
2. prefix loss 和 continuation loss 直接等权相加是否合适
3. offline prefix old policy 与 online continuation old policy 混用时，梯度行为是否稳定
4. 在真实非零 reward 训练中，prefix 分支是否真的带来策略改进，而不只是 ratio 指标变化

### 8.4 当前最需要重点复核的点

1. `joint` 模式下 prefix 与 continuation 共用 clip 配置，是否足够满足后续对照实验要求
2. 当前数据仍不总是显式提供 `goal/data_idx`，是否需要在 parquet 中进一步补齐以增强可审查性
3. prefix advantage 模式、`prefix_loss_weight` 和 `split/joint` ablation 是否已经覆盖当前想看的核心对照
4. `activation_offload` 路径是否值得单独修；在未修前应继续保持默认关闭
5. 是否还需要补一轮更长程的非 debug run，验证 learning signal 的持续性而不仅是短程打通

---

## 9. 对现有 Claude 版说明的修订点

我审查了现有说明文件：

`/Data/wyh/verl/prefix_grpo_main_experiment_overview.md`

其中至少有以下几处与当前代码/当前数据不一致，已在本说明里纠正：

1. 旧说明把当前 active parquet 写成含顶层 `item_id`、`sample_idx`；实际当前 active parquet 没有这两列。
2. 旧说明把 `reward_model.style` 写成固定 `"model"` 且 `ground_truth` 作为主 reward 参考；当前 active 样本实际是 `style="interaction"`，`ground_truth` 为空串。
3. 旧说明写 rollout 期间 `calculate_log_probs=false`；当前主脚本默认 `CALCULATE_LOG_PROBS=true`。
4. 旧说明把 `interaction_kwargs` 描述成含 `goal`、`data_idx`、`item_id` 等字段；当前 active parquet 全量统计只有 `eval_mode/name/prefix_actions/task_id`。

因此，后续对外使用的版本建议以本 reviewed 版为准。

---

## 10. 文件级索引

### 10.1 训练入口 / 配置

| 文件 | 作用 |
| --- | --- |
| `/Data/wyh/verl/examples/sglang_multiturn/my_exp/rl/run_textcraft_grpo_validated.sh` | 主实验启动脚本，定义默认数据、模型、batch、rollout、prefix 开关与 Hydra override |
| `/Data/wyh/verl/examples/sglang_multiturn/config/textcraft_grpo_train.yaml` | 主 Hydra 配置，定义 data / actor / rollout / algorithm / trainer 默认值 |
| `/Data/wyh/verl/examples/sglang_multiturn/config/interaction_config/textcraft_interaction.yaml` | TextCraft interaction 配置 |
| `/Data/wyh/verl/verl/trainer/config/actor/actor.yaml` | actor 通用配置源，包含 `ppo_epochs=1`、clip ratio、loss mode 等 |

### 10.2 trainer

| 文件 | 作用 |
| --- | --- |
| `/Data/wyh/verl/verl/trainer/ppo/ray_trainer.py` | 训练主循环；rollout、reward、old logprob、advantage、prefix sidecar materialization、actor update 调度都在这里 |
| `/Data/wyh/verl/verl/trainer/ppo/core_algos.py` | GRPO advantage 与 PPO-style policy loss 公式实现 |

### 10.3 actor

| 文件 | 作用 |
| --- | --- |
| `/Data/wyh/verl/verl/workers/actor/dp_actor.py` | actor current/old logprob 计算、continuation/prefix 双分支 loss、总损失拼接的核心文件 |
| `/Data/wyh/verl/verl/workers/fsdp_workers.py` | actor/ref worker；负责 old/ref logprob 计算时的 micro-batch、temperature、序列化等 |

### 10.4 reward manager

| 文件 | 作用 |
| --- | --- |
| `/Data/wyh/verl/verl/workers/reward_manager/naive.py` | 从 `turn_scores` 组装 textcraft reward，并把最终 reward 放到 response 最后一个有效 token |
| `/Data/wyh/verl/verl/utils/reward_score/__init__.py` | `data_source=="textcraft"` 时把 `sum(turn_scores)` 作为标量 reward |

### 10.5 interaction / rollout

| 文件 | 作用 |
| --- | --- |
| `/Data/wyh/verl/verl/utils/dataset/rl_dataset.py` | parquet 读取、chat template、raw prompt 保留、`interaction_kwargs` 拆出 |
| `/Data/wyh/verl/verl/experimental/agent_loop/agent_loop.py` | rollout 采样参数、`calculate_log_probs`、`rollout_log_probs` 回写 |
| `/Data/wyh/verl/verl/experimental/agent_loop/tool_agent_loop.py` | 多轮 tool/interaction agent loop；启动 interaction、记录 `turn_scores` |
| `/Data/wyh/verl/verl/interactions/textcraft_interaction.py` | TextCraft 环境创建、goal 校验、prefix replay、action 提取 |

### 10.6 TextCraft server / env

| 文件 | 作用 |
| --- | --- |
| `/Data/wyh/h200_grpo/envs/AgentGym/agentenv-textcraft/agentenv_textcraft/launch.py` | TextCraft server 启动入口 |
| `/Data/wyh/h200_grpo/envs/AgentGym/agentenv-textcraft/agentenv_textcraft/environment.py` | TextCraft 环境逻辑；goal 奖励和终止条件在这里定义 |

### 10.7 数据转换 / 数据审计

| 文件 | 作用 |
| --- | --- |
| `/Data/wyh/verl/examples/sglang_multiturn/my_exp/legacy/prefix_old_logits/precompute_sft_old_logprobs_from_trajectories.py` | 对完整 teacher 轨迹离线预计算 scalar old logprob |
| `/Data/wyh/verl/examples/sglang_multiturn/my_exp/legacy/prefix_old_logits/build_prefix_old_logprob_dataset.py` | 把 full trajectory sidecar 对齐到截断 prompt，生成 prefix old logprob / mask / span |
| `/Data/wyh/verl/examples/sglang_multiturn/my_exp/legacy/prefix_old_logits/adapt_data_for_training.py` | 字段名适配到训练代码期望格式 |
| `/Data/wyh/verl/examples/sglang_multiturn/my_exp/short_learning_validation/audit_data_quality_v2.py` | active 数据质量审计与 cleaned_v2 生成 |

### 10.8 验证 / 调试

| 文件 | 作用 |
| --- | --- |
| `/Data/wyh/verl/examples/sglang_multiturn/my_exp/short_learning_validation/audit/verify_goal_binding_fix.py` | goal binding 修复验证脚本 |
| `/Data/wyh/verl/examples/sglang_multiturn/my_exp/rl/debug/preflight_test.py` | preflight smoke test，验证数据、replay、continuation 基本链路 |
| `/Data/wyh/datasets/Verl-Data/outputs/textcraft_grpo_prefix_smoke_test_canonicalized/logs/train_canonicalized_prefix_smoke_test_20260322_231829.log` | prefix 分支已进入 actor update 的成功 smoke log |

---

## 11. 建议审查顺序

建议按下面顺序读，会最快抓到关键实现点：

1. `run_textcraft_grpo_validated.sh`  
   先确认默认配置、数据路径、`optimize_prefix_tokens` 开关和 Hydra override。
2. `ray_trainer.py`  
   看 rollout 后 prefix sidecar 的 restore/materialize、old logprob 计算、advantage 调度。
3. `dp_actor.py`  
   重点看 prefix gather、prefix advantage、`policy_loss = cont + w * prefix`。
4. `core_algos.py`  
   看 continuation/prefix 两边复用的 PPO/GRPO 公式。
5. `textcraft_interaction.py`  
   看 prefix replay、goal 解析与 fail-fast。
6. 数据构建脚本  
   `precompute_sft_old_logprobs_from_trajectories.py` + `build_prefix_old_logprob_dataset.py`，确认 offline prefix old logprob 的 provenance 与对齐逻辑。
