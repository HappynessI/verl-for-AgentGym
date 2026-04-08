# TextCraft Main Prefix Complete Split

## Inputs
- teacher_normalized: `/Data/wyh/datasets/Verl-Data/train/textcraft/new_prefix_rl/stage0_teacher/teacher_normalized.parquet`
- entropy stage2 candidates: `/Data/wyh/datasets/Verl-Data/train/textcraft/entropy_based_prefix/stage2_splits/prefix_candidates_entropy_topk.parquet`
- official raw train parquet: `/Data/wyh/datasets/Verl-Data/train/textcraft/train.parquet`
- prompt-space old-logprob model: `/Data/wyh/datasets/Verl-Data/outputs/textcraft_sft/qwen3-1.7b-sft/global_step_200/huggingface`

## Output Datasets
### main_change_top3_w11
- rows: `5895`
- unique_sample_uid: `1496`
- raw_rows: `1496`
- prefix_rows: `4399`
- zero_prefix_rows: `1496`
- positive_prefix_rows: `4399`
- output_path: `/Data/wyh/datasets/Verl-Data/train/textcraft/main_prefix/complete_split/main_change_top3_w11.parquet`
- strategy_counts: `{'entropy_change_topk_w11_interaction_assistant_k3': 4399, 'raw': 1496}`
- candidate_rank_counts: `{'1': 1496, '2': 1496, '3': 1407}`

### main_fixed_gp1
- rows: `5984`
- unique_sample_uid: `1496`
- raw_rows: `1496`
- prefix_rows: `4488`
- zero_prefix_rows: `1496`
- positive_prefix_rows: `4488`
- output_path: `/Data/wyh/datasets/Verl-Data/train/textcraft/main_prefix/complete_split/main_fixed_gp1.parquet`
- strategy_counts: `{'fixed_ratio_0p1': 1496, 'fixed_ratio_0p3': 1496, 'fixed_ratio_0p5': 1496, 'raw': 1496}`

### main_fixed_gp2
- rows: `5984`
- unique_sample_uid: `1496`
- raw_rows: `1496`
- prefix_rows: `4488`
- zero_prefix_rows: `1496`
- positive_prefix_rows: `4488`
- output_path: `/Data/wyh/datasets/Verl-Data/train/textcraft/main_prefix/complete_split/main_fixed_gp2.parquet`
- strategy_counts: `{'fixed_ratio_0p25': 1496, 'fixed_ratio_0p5': 1496, 'fixed_ratio_0p7': 1496, 'raw': 1496}`

### main_raw_top3
- rows: `5895`
- unique_sample_uid: `1496`
- raw_rows: `1496`
- prefix_rows: `4399`
- zero_prefix_rows: `1496`
- positive_prefix_rows: `4399`
- output_path: `/Data/wyh/datasets/Verl-Data/train/textcraft/main_prefix/complete_split/main_raw_top3.parquet`
- strategy_counts: `{'entropy_raw_topk_interaction_assistant_k3': 4399, 'raw': 1496}`
- candidate_rank_counts: `{'1': 1496, '2': 1496, '3': 1407}`

## Notes
- `main_fixed_gp1` and `main_fixed_gp2` each duplicate every sampled trajectory into `3 prefix variants + 1 raw variant`.
- `main_raw_top3` and `main_change_top3_w11` use the full stage2 top-k split points to preserve all `1496` sampled trajectories.
- `raw` variants keep the official train parquet prompt and attach an empty prefix sidecar (`prefix_token_count=0`).
- Prefix sidecars are rebuilt in prompt-space on the canonicalized training prompt.
- The legacy `new_prefix_rl` fixed-ratio release sidecars are not reused here; `main_prefix` rebuilds all prefix sidecars under one prompt-space protocol.
- Replay-filtered datasets are stored in:
  `/Data/wyh/datasets/Verl-Data/train/textcraft/main_prefix/replay_validated/`
