# TextCraft Main Prefix Replay Validated (Sample Faithful)

## Meaning
- This directory reruns replay from sample-faithful candidate rows built with the same loose action parser used online.
- Fixed-ratio datasets keep `validated + usable_state_feedback`; entropy datasets keep `validated` only, matching the legacy replay evidence policy.
- Candidate rows are not dropped just because a prefix message contains multiple `Action:` tags.
- Parser-risk candidates are audited under `../audit/replay_validated/` for later inspection.

## Inputs
- teacher_normalized: `/Data/wyh/datasets/Verl-Data/train/textcraft/new_prefix_rl/stage0_teacher/teacher_normalized.parquet`
- entropy stage2 candidates: `/Data/wyh/datasets/Verl-Data/train/textcraft/entropy_based_prefix/stage2_splits/prefix_candidates_entropy_topk.parquet`
- official raw train parquet: `/Data/wyh/datasets/Verl-Data/train/textcraft/train.parquet`
- prompt-space old-logprob model: `/Data/wyh/datasets/Verl-Data/outputs/textcraft_sft/qwen3-1.7b-sft/global_step_200/huggingface`
- replay validation server: `http://127.0.0.1:36001`

## Output Datasets
### main_fixed_gp1_fullflow
- rows: `5513`
- unique_sample_uid: `1496`
- raw_rows: `1496`
- prefix_rows: `4017`
- zero_prefix_rows: `1496`
- output_path: `/Data/wyh/datasets/Verl-Data/train/textcraft/main_prefix/new_main_prefix/replay_validated/main_fixed_gp1_fullflow.parquet`
- rows_with_empty_sampling_prefix_actions: `3`

### main_fixed_gp2_fullflow
- rows: `5627`
- unique_sample_uid: `1496`
- raw_rows: `1496`
- prefix_rows: `4131`
- zero_prefix_rows: `1496`
- output_path: `/Data/wyh/datasets/Verl-Data/train/textcraft/main_prefix/new_main_prefix/replay_validated/main_fixed_gp2_fullflow.parquet`
- rows_with_empty_sampling_prefix_actions: `3`

### main_raw_top3_fullflow
- rows: `5239`
- unique_sample_uid: `1496`
- raw_rows: `1496`
- prefix_rows: `3743`
- zero_prefix_rows: `1496`
- output_path: `/Data/wyh/datasets/Verl-Data/train/textcraft/main_prefix/new_main_prefix/replay_validated/main_raw_top3_fullflow.parquet`
- rows_with_empty_sampling_prefix_actions: `3`

### main_change_top3_w11_fullflow
- rows: `5190`
- unique_sample_uid: `1496`
- raw_rows: `1496`
- prefix_rows: `3694`
- zero_prefix_rows: `1496`
- output_path: `/Data/wyh/datasets/Verl-Data/train/textcraft/main_prefix/new_main_prefix/replay_validated/main_change_top3_w11_fullflow.parquet`
- rows_with_empty_sampling_prefix_actions: `3`

## Notes
- This branch keeps the legacy replay evidence policy but switches candidate reconstruction to the online loose parser.
- Replay, not the parser prefilter, is the first hard filter here.
