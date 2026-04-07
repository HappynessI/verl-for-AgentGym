# Entropy Window 消融报告

## 1. 实验范围

- 数据根目录：`/Data/wyh/datasets/Verl-Data/train/textcraft/entropy_based_prefix`
- 关注策略：`entropy_change_topk_w{window}_interaction_assistant_k3`
- 固定设置：
  - scorer：`change_topk`
  - domain：`interaction_assistant`
  - `top_k = 3`
  - 仅改变 `change_window`
- 测试窗口：`[1, 3, 5, 7, 9, 11, 15, 21]`
- 比较层级：`stage2 candidate export + stage3 replay validation`

本次实验不重跑 teacher entropy 计算，只在已经固定的 entropy 结果上重新做不同窗口的平滑、候选导出与 replay 验证。

## 2. 实验目的

这轮消融要回答的问题只有一个：`change_topk_w11_interaction_assistant_k3` 里的 `window=11` 是否真的更优，还是只是一个经验超参。

这里的“更优”先只按数据构造质量来衡量，不直接看下游 RL 训练效果，因此主要比较：

- `validated_rate`：replay 后被判成 `validated` 的比例
- `validated_rows`：最终保留下来的合法候选条数
- `validated_samples_with_3_ranks`：同时拥有合法 `rank1/2/3` 的样本数
- `unverifiable_rate`：无法被当前 validator 严格证明的比例
- `rank1_same_cut_vs_w11`：相对 `w11` 的切点稳定性

## 3. 方法说明

对每个窗口，执行两步：

1. 重新运行 `05_export_entropy_prefix_candidates.py`
   - 固定 `domain=interaction_assistant`
   - 固定 `scorer=change_topk`
   - 固定 `top_k=3`
   - 只改变 `change_window`
2. 重新运行 `06_replay_validate_entropy_candidates.py`
   - 只验证该窗口对应的一条策略
   - 对比 `validated / mismatch / unverifiable / error`

额外说明：

- 当前实现要求窗口为奇数。
- 平滑实现本质上使用 `radius = window // 2`，所以 `w11` 的真实含义是“左右各看 5 个 token 做局部平均”。
- 偶数窗口在当前实现下会塌缩到和前一个奇数窗口相同的 `radius`，因此不单独测试。

## 4. 核心结果

| window | validated_rate | mismatch_rate | unverifiable_rate | validated_rows | 3-rank validated samples | mean cut delta vs fixed_ratio_0p4 | rank1 same cut vs w11 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 72.83% | 1.11% | 26.05% | 3204 | 785 | 0.193 | 78.14% |
| 3 | 73.31% | 1.07% | 25.62% | 3225 | 793 | 0.197 | 84.02% |
| 5 | 73.61% | 1.14% | 25.26% | 3238 | 795 | 0.216 | 84.16% |
| 7 | 73.49% | 1.14% | 25.37% | 3233 | 792 | 0.225 | 84.76% |
| 9 | 73.33% | 1.25% | 25.41% | 3226 | 794 | 0.258 | 84.09% |
| 11 | 72.68% | 1.14% | 26.19% | 3197 | 777 | 0.220 | 100.00% |
| 15 | 71.72% | 1.00% | 27.28% | 3155 | 754 | 0.216 | 79.48% |
| 21 | 71.06% | 1.14% | 27.80% | 3126 | 744 | 0.239 | 75.33% |

## 5. 结论

### 5.1 `w11` 不是最优窗口

- 从 `validated_rate` 看，最优窗口是 `w5 = 73.61%`。
- `w11 = 72.68%`，比 `w5` 低 `0.93` 个百分点。
- 从绝对条数看，`w5` 比 `w11` 多保留 `41` 条 `validated` 候选，`3238 vs 3197`。
- 从三 rank 完整覆盖看，`w5` 比 `w11` 多 `18` 个“同时拥有合法 `rank1/2/3`”的样本，`795 vs 777`。

因此，`w11` 不能被表述为“在当前数据构造指标下表现最好”的窗口。

### 5.2 最稳定的高位区间是 `w3-w9`

- `w3 / w5 / w7 / w9` 的 `validated_rate` 都在 `73.3%+`，差距很小。
- 这个区间里，`w5` 略优，`w7` 次之。
- 相对 `w11` 的 `rank1` 切点一致率约为 `84%`，说明中等窗口之间的切点虽然会变化，但整体仍处在同一族局部最优附近。

这说明当前任务上存在一个比较平稳的“中等平滑窗口”区间，而不是只有 `11` 这一个特殊答案。

### 5.3 大窗口 `w15 / w21` 明显过平滑

- `w15` 和 `w21` 的 `validated_rate` 明显下降到 `71.72%` 和 `71.06%`。
- 同时 `unverifiable_rate` 上升到 `27.28%` 和 `27.80%`。
- `rank1_same_cut_vs_w11` 也进一步下降到 `79.48%` 和 `75.33%`。

这说明窗口过大后，平滑会把局部 entropy 变化抹得过于平坦，导致切点质量下降。

### 5.4 小窗口 `w1` 也不是最优

- `w1` 已经比 `w11` 略好，但仍弱于 `w3-w9` 这段高位区间。
- 这说明完全不平滑或几乎不平滑时，局部 token 级噪声仍然会影响切点排序。

因此当前结果更像是：

- 太小：去噪不够
- 太大：过平滑
- 中等窗口：效果最好

## 6. 对 `unverifiable` 的观察

窗口变化并不会让 `unverifiable` 消失，它始终维持在 `25% - 28%` 左右。

更重要的是，很多 `unverifiable` 并不是明显的坏样本，而是“当前 validator 证据不够”。本次统计里：

- `w11` 的 `1152` 条 `unverifiable` 中，有 `727` 条 cut observation 完全一致，`498` 条 cut 和 next 都完全一致。
- `w21` 的 `1223` 条 `unverifiable` 中，有 `803` 条 cut observation 完全一致，`569` 条 cut 和 next 都完全一致。

所以不能把 `unverifiable` 简单理解成“环境一定没对齐”；它更像“现有校验器无法严格证明”。

## 7. 对 `w11` 的解释

基于这轮消融，可以更准确地描述 `w11`：

- `w11` 不是最优窗口。
- `w11` 属于一个可用的中等窗口配置，但并不是当前实验里 replay 指标最好的选择。
- 如果只按本轮 stage3 质量指标看，`w5` 或 `w7` 更值得优先考虑。

换句话说，现有证据更支持把 `w11` 视为“经验上可用的默认值”，而不是“经过严格消融后证明最优的窗口”。

## 8. 建议

### 8.1 如果目标是继续构造更高质量数据集

优先考虑：

- 首选：`w5`
- 次选：`w7`
- 兼容旧版本：继续保留 `w11`

### 8.2 如果目标是决定最终训练用哪个窗口

本轮结论还不够，需要继续做下游训练对比。建议下一步至少做三组：

- `w5`
- `w7`
- `w11`

然后比较：

- RL 训练稳定性
- 成功率 / reward
- prefix 相关指标
- 最终主实验效果

只有把“数据构造质量”与“下游训练效果”合起来看，才能决定最终是否替换 `w11`。

## 9. 产物路径

- 中文报告：`/Data/wyh/datasets/Verl-Data/train/textcraft/entropy_based_prefix/window_ablation/notes/window_ablation_report_zh.md`
- 英文汇总：`/Data/wyh/datasets/Verl-Data/train/textcraft/entropy_based_prefix/window_ablation/notes/window_ablation_report.md`
- JSON 摘要：`/Data/wyh/datasets/Verl-Data/train/textcraft/entropy_based_prefix/window_ablation/notes/window_ablation_report.json`
- CSV 摘要：`/Data/wyh/datasets/Verl-Data/train/textcraft/entropy_based_prefix/window_ablation/notes/window_ablation_report.csv`
- 可视化图：`/Data/wyh/datasets/Verl-Data/train/textcraft/entropy_based_prefix/window_ablation/outputs/window_ablation_summary.png`
