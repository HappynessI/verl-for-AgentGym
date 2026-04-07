# Entropy Analysis

这个目录用于存放 TextCraft teacher 轨迹的 entropy 可视化产物。

当前约定：

- 根目录：放全体 teacher 样本聚合后的 `user -> assistant -> user -> assistant ...` entropy 波动图。
- `representative_sample_traces/`：放若干代表性样本的原始轨迹图。
- 当前脚本会同时输出两套版本：
  - `full`：保留 assistant 原始完整文本
  - `no_think`：去掉 assistant 中整个 `<think> ... </think>` 内部推理块后再统计 entropy

当前数据的一个重要观察：

- `entropy_based_prefix/stage1_entropy/textcraft_teacher_entropy_step200.parquet` 是在 `enable_thinking=False` 下计算的。
- 对当前 Qwen chat template，这意味着 assistant 原始文本中的 `<think> ... </think>` 不会进入实际用于 entropy 计算的 token 序列。
- 因此，基于这份 stage1 sidecar 画出来的 `full` 和 `no_think` 图在当前数据上应当基本一致；这不是绘图错误，而是上游 entropy 计算时已经等价于“去掉内部推理”。

后续建议：

- 聚合图命名示例：`teacher_entropy_user_assistant_aggregate_u16_a32_turn9.png`
- 去推理聚合图命名示例：`teacher_entropy_user_assistant_aggregate_no_think_u16_a32_turn9.png`
- 代表样本图命名示例：`sample_textcraft_100__0_entropy_trace_u16_a32.png`
