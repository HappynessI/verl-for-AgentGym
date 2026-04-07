# Window Ablation

这个目录用于放置 `entropy_based_prefix` 的窗口大小消融相关内容。

建议后续按以下结构补充：
- `configs/`: 不同 `change_window` 配置
- `manifests/`: 运行摘要与统计
- `outputs/`: 中间结果、对比表与可视化
- `notes/`: 结论记录与观察

当前状态：
- 已完成首轮窗口消融
- 当前消融范围：`change_topk + interaction_assistant + top_k=3`
- 已测试窗口：`[1, 3, 5, 7, 9, 11, 15, 21]`

主要产物：
- 中文报告：`notes/window_ablation_report_zh.md`
- 英文汇总：`notes/window_ablation_report.md`
- 结构化摘要：`notes/window_ablation_report.json`
- CSV 摘要：`notes/window_ablation_report.csv`
- 可视化图：`outputs/window_ablation_summary.png`
