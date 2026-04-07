# Legacy Entropy Prefix Data

这份归档保存的是 `2026-04-06` 修复前的 entropy prefix 旧产物。

## 归档原因

旧版 entropy 数据链在 canonicalize 阶段把首轮包含 `Crafting commands + Goal` 的 `user` 任务描述错误地当作 warmup user message 过滤掉了。

直接后果：

- 最终训练 prompt 变成 `system -> assistant -> user ...`
- 不再保留和 `fixed_ratio_0p4` 主数据一致的首轮任务描述格式
- TextCraft 训练时无法再从 prompt 中可靠解析 `commands + goal`
- 训练只能退化依赖 `task_id/data_idx` 重建环境
- 一旦本地 TextCraft 环境版本的 `task_id -> goal` 映射与数据生成时不一致，就会在 rollout 前触发 `Goal mismatch`

## 受影响范围

本目录只归档受该问题影响的下游结果：

- `stage4_canonicalized`
- `stage6_training_build`
- `stage7_audit_release`
- 对应 `stage4/stage6/stage7` manifests

上游结果未归档、仍作为当前有效输入继续复用：

- `stage1_entropy`
- `stage2_splits`
- `stage3_replay_validation`

## 修复目标

修复后的新数据链需要满足：

- prompt 保留首轮 `Crafting commands + Goal`
- prompt 结构与 `fixed_ratio_0p4` 主数据格式对齐
- TextCraft rollout 能从 prompt 中解析出正确的 `commands + goal`
- 不再依赖漂移的 `task_id -> goal` 环境映射来对齐任务
