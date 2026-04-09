# TextCraft Main Prefix Runtime Parser Aligned

## Contents
- complete_split: `/Data/wyh/datasets/Verl-Data/train/textcraft/main_prefix/runtime_parser_aligned/complete_split`
- replay_validated: `/Data/wyh/datasets/Verl-Data/train/textcraft/main_prefix/runtime_parser_aligned/replay_validated`
- audit_report.md: `/Data/wyh/datasets/Verl-Data/train/textcraft/main_prefix/runtime_parser_aligned/audit_report.md`

## Meaning
- This branch rebuilds main-prefix datasets under a parser that matches the runtime TextCraft ReAct rule exactly.
- Training-side interaction logic is intentionally left unchanged; this branch is for offline data audit and alternate experiments only.
- Prefix rows are dropped if any non-warmup assistant turn violates the runtime single-`Action:` protocol, or if no runtime-valid action remains after filtering.
- Dropped rows are stored as audit artifacts with per-reason counts.
