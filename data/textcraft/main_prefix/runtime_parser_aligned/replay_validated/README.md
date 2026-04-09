# TextCraft Main Prefix Replay Validated (Runtime Parser Aligned)

## Meaning
- This directory rebuilds the replay-filtered datasets under a parser that matches the runtime TextCraft ReAct rule exactly.
- Fixed-ratio datasets keep `validated + usable_state_feedback`; entropy datasets keep `validated` only.
- Replay validation is rerun from parser-aligned candidate rows instead of reusing legacy stage4/stage7 outputs.
- Candidate rows are dropped if any non-warmup assistant turn violates the runtime single-`Action:` protocol, or if no runtime-valid action remains after filtering.
- Dropped rows are recorded under `../audit/replay_validated/` with per-reason counts.

## Inputs
- teacher_normalized: `/Data/wyh/datasets/Verl-Data/train/textcraft/new_prefix_rl/stage0_teacher/teacher_normalized.parquet`
- entropy stage2 candidates: `/Data/wyh/datasets/Verl-Data/train/textcraft/entropy_based_prefix/stage2_splits/prefix_candidates_entropy_topk.parquet`
- official raw train parquet: `/Data/wyh/datasets/Verl-Data/train/textcraft/train.parquet`
- prompt-space old-logprob model: `/Data/wyh/datasets/Verl-Data/outputs/textcraft_sft/qwen3-1.7b-sft/global_step_200/huggingface`
- replay validation server: `http://127.0.0.1:36001`

## Output Datasets
### main_fixed_gp1_fullflow
- rows: `5473`
- unique_sample_uid: `1496`
- raw_rows: `1496`
- prefix_rows: `3977`
- dropped_runtime_invalid_prefix_rows: `115`
- output_path: `/Data/wyh/datasets/Verl-Data/train/textcraft/main_prefix/runtime_parser_aligned/replay_validated/main_fixed_gp1_fullflow.parquet`

### main_fixed_gp2_fullflow
- rows: `5564`
- unique_sample_uid: `1496`
- raw_rows: `1496`
- prefix_rows: `4068`
- dropped_runtime_invalid_prefix_rows: `150`
- output_path: `/Data/wyh/datasets/Verl-Data/train/textcraft/main_prefix/runtime_parser_aligned/replay_validated/main_fixed_gp2_fullflow.parquet`

### main_raw_top3_fullflow
- rows: `5180`
- unique_sample_uid: `1496`
- raw_rows: `1496`
- prefix_rows: `3684`
- dropped_runtime_invalid_prefix_rows: `150`
- output_path: `/Data/wyh/datasets/Verl-Data/train/textcraft/main_prefix/runtime_parser_aligned/replay_validated/main_raw_top3_fullflow.parquet`
- candidate_rank_counts: `{1: 1264, 2: 1230, 3: 1190}`

### main_change_top3_w11_fullflow
- rows: `5132`
- unique_sample_uid: `1496`
- raw_rows: `1496`
- prefix_rows: `3636`
- dropped_runtime_invalid_prefix_rows: `153`
- output_path: `/Data/wyh/datasets/Verl-Data/train/textcraft/main_prefix/runtime_parser_aligned/replay_validated/main_change_top3_w11_fullflow.parquet`
- candidate_rank_counts: `{1: 1240, 2: 1236, 3: 1160}`

## Notes
- This family is replay-filtered and parser-aligned, but it still uses the legacy replay evidence policy (`validated` / `usable_state_feedback`).
- It should be interpreted as a new audit branch; it does not overwrite the current `main_prefix/replay_validated/` release.
