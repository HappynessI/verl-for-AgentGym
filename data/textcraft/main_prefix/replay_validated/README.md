# TextCraft Main Prefix Replay Validated

## Meaning
- This directory contains replay-filtered supplemental datasets that explicitly follow the legacy `new_prefix_rl` / `entropy_based_prefix` full-flow semantics.
- They do not replace the cutpoint-complete datasets in `main_prefix/complete_split/`.
- The datasets in `complete_split/` keep all cutpoints and rebuild prompt-space sidecars.
- The `full_flow` datasets below additionally inherit replay validation / refinement filtering semantics.

## Inputs
- teacher_normalized: `/Data/wyh/datasets/Verl-Data/train/textcraft/new_prefix_rl/stage0_teacher/teacher_normalized.parquet`
- official raw train parquet: `/Data/wyh/datasets/Verl-Data/train/textcraft/train.parquet`
- prompt-space old-logprob model: `/Data/wyh/datasets/Verl-Data/outputs/textcraft_sft/qwen3-1.7b-sft/global_step_200/huggingface`
- replay validation server: `http://127.0.0.1:36001`

## Output Datasets
### main_change_top3_w11_fullflow
- rows: `4693`
- unique_sample_uid: `1496`
- raw_rows: `1496`
- prefix_rows: `3197`
- zero_prefix_rows: `1496`
- positive_prefix_rows: `3197`
- output_path: `/Data/wyh/datasets/Verl-Data/train/textcraft/main_prefix/replay_validated/main_change_top3_w11_fullflow.parquet`
- strategy_counts: `{'entropy_change_topk_w11_interaction_assistant_k3': 3197, 'raw': 1496}`
- candidate_rank_counts: `{'1': 1083, '2': 1086, '3': 1028}`

### main_fixed_gp1_fullflow
- rows: `4970`
- unique_sample_uid: `1496`
- raw_rows: `1496`
- prefix_rows: `3474`
- zero_prefix_rows: `1496`
- positive_prefix_rows: `3474`
- output_path: `/Data/wyh/datasets/Verl-Data/train/textcraft/main_prefix/replay_validated/main_fixed_gp1_fullflow.parquet`
- strategy_counts: `{'fixed_ratio_0p1': 1115, 'fixed_ratio_0p3': 1120, 'fixed_ratio_0p5': 1239, 'raw': 1496}`

### main_fixed_gp2_fullflow
- rows: `5114`
- unique_sample_uid: `1496`
- raw_rows: `1496`
- prefix_rows: `3618`
- zero_prefix_rows: `1496`
- positive_prefix_rows: `3618`
- output_path: `/Data/wyh/datasets/Verl-Data/train/textcraft/main_prefix/replay_validated/main_fixed_gp2_fullflow.parquet`
- strategy_counts: `{'fixed_ratio_0p25': 1120, 'fixed_ratio_0p5': 1239, 'fixed_ratio_0p7': 1259, 'raw': 1496}`

### main_raw_top3_fullflow
- rows: `4726`
- unique_sample_uid: `1496`
- raw_rows: `1496`
- prefix_rows: `3230`
- zero_prefix_rows: `1496`
- positive_prefix_rows: `3230`
- output_path: `/Data/wyh/datasets/Verl-Data/train/textcraft/main_prefix/replay_validated/main_raw_top3_fullflow.parquet`
- strategy_counts: `{'entropy_raw_topk_interaction_assistant_k3': 3230, 'raw': 1496}`
- candidate_rank_counts: `{'1': 1105, '2': 1075, '3': 1050}`

## Notes
- Dataset-specific counts and protocol summaries are also stored beside each parquet as `*.manifest.json`.
- `main_fixed_*_fullflow` keeps all `1496` raw rows, but prefix rows are replay-filtered (`validated + usable_state_feedback`).
- `main_*_fullflow` is the pipeline-aligned replay-validated version; `complete_split/` remains the cutpoint-complete prompt-space version.
