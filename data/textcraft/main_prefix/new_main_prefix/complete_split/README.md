# TextCraft Main Prefix Complete Split (Sample Faithful)

## Meaning
- This directory keeps all sampled cutpoints and rebuilds them with the same loose action parser used by online TextCraft interaction.
- Prefix rows are not pre-dropped for `multiple_action_tags` or `missing_action_tag`.
- If the loose online parser still extracts zero prefix actions, the row is kept and receives a zero-length prefix sidecar.
- Parser-risk rows are audited under `../audit/complete_split/` instead of being filtered out here.

## Inputs
- teacher_normalized: `/Data/wyh/datasets/Verl-Data/train/textcraft/new_prefix_rl/stage0_teacher/teacher_normalized.parquet`
- entropy stage2 candidates: `/Data/wyh/datasets/Verl-Data/train/textcraft/entropy_based_prefix/stage2_splits/prefix_candidates_entropy_topk.parquet`
- official raw train parquet: `/Data/wyh/datasets/Verl-Data/train/textcraft/train.parquet`
- prompt-space old-logprob model: `/Data/wyh/datasets/Verl-Data/outputs/textcraft_sft/qwen3-1.7b-sft/global_step_200/huggingface`

## Output Datasets
### main_fixed_gp1
- rows: `5984`
- unique_sample_uid: `1496`
- raw_rows: `1496`
- prefix_rows: `4488`
- zero_prefix_rows: `1499`
- output_path: `/Data/wyh/datasets/Verl-Data/train/textcraft/main_prefix/new_main_prefix/complete_split/main_fixed_gp1.parquet`
- rows_with_empty_sampling_prefix_actions: `3`

### main_fixed_gp2
- rows: `5984`
- unique_sample_uid: `1496`
- raw_rows: `1496`
- prefix_rows: `4488`
- zero_prefix_rows: `1499`
- output_path: `/Data/wyh/datasets/Verl-Data/train/textcraft/main_prefix/new_main_prefix/complete_split/main_fixed_gp2.parquet`
- rows_with_empty_sampling_prefix_actions: `3`

### main_raw_top3
- rows: `5895`
- unique_sample_uid: `1496`
- raw_rows: `1496`
- prefix_rows: `4399`
- zero_prefix_rows: `1499`
- output_path: `/Data/wyh/datasets/Verl-Data/train/textcraft/main_prefix/new_main_prefix/complete_split/main_raw_top3.parquet`
- rows_with_empty_sampling_prefix_actions: `3`

### main_change_top3_w11
- rows: `5895`
- unique_sample_uid: `1496`
- raw_rows: `1496`
- prefix_rows: `4399`
- zero_prefix_rows: `1499`
- output_path: `/Data/wyh/datasets/Verl-Data/train/textcraft/main_prefix/new_main_prefix/complete_split/main_change_top3_w11.parquet`
- rows_with_empty_sampling_prefix_actions: `3`

## Notes
- This branch is sample-faithful rather than runtime-audit-strict.
- `rows_with_multi_action_tag_in_prefix_messages` can remain non-zero here by design.
- Zero-prefix rows are expected when the online parser would also fail to recover any prefix action from the stored assistant text.
