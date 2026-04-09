# TextCraft Main Prefix Complete Split (Runtime Parser Aligned)

## Meaning
- This directory rebuilds the complete-split datasets under a parser that matches the runtime TextCraft ReAct rule exactly.
- A prefix assistant message contributes one action iff it contains exactly one `Action:` tag after chat-template marker stripping.
- Prefix rows are dropped if any non-warmup assistant turn violates the runtime single-`Action:` protocol, or if no runtime-valid action remains after filtering.
- Dropped rows are recorded under `../audit/complete_split/` with per-reason counts.

## Inputs
- teacher_normalized: `/Data/wyh/datasets/Verl-Data/train/textcraft/new_prefix_rl/stage0_teacher/teacher_normalized.parquet`
- entropy stage2 candidates: `/Data/wyh/datasets/Verl-Data/train/textcraft/entropy_based_prefix/stage2_splits/prefix_candidates_entropy_topk.parquet`
- official raw train parquet: `/Data/wyh/datasets/Verl-Data/train/textcraft/train.parquet`
- prompt-space old-logprob model: `/Data/wyh/datasets/Verl-Data/outputs/textcraft_sft/qwen3-1.7b-sft/global_step_200/huggingface`

## Output Datasets
### main_fixed_gp1
- rows: `5869`
- unique_sample_uid: `1496`
- raw_rows: `1496`
- prefix_rows: `4373`
- dropped_runtime_invalid_prefix_rows: `115`
- output_path: `/Data/wyh/datasets/Verl-Data/train/textcraft/main_prefix/runtime_parser_aligned/complete_split/main_fixed_gp1.parquet`

### main_fixed_gp2
- rows: `5834`
- unique_sample_uid: `1496`
- raw_rows: `1496`
- prefix_rows: `4338`
- dropped_runtime_invalid_prefix_rows: `150`
- output_path: `/Data/wyh/datasets/Verl-Data/train/textcraft/main_prefix/runtime_parser_aligned/complete_split/main_fixed_gp2.parquet`

### main_raw_top3
- rows: `5745`
- unique_sample_uid: `1496`
- raw_rows: `1496`
- prefix_rows: `4249`
- dropped_runtime_invalid_prefix_rows: `150`
- output_path: `/Data/wyh/datasets/Verl-Data/train/textcraft/main_prefix/runtime_parser_aligned/complete_split/main_raw_top3.parquet`
- candidate_rank_counts: `{1: 1449, 2: 1445, 3: 1355}`

### main_change_top3_w11
- rows: `5742`
- unique_sample_uid: `1496`
- raw_rows: `1496`
- prefix_rows: `4246`
- dropped_runtime_invalid_prefix_rows: `153`
- output_path: `/Data/wyh/datasets/Verl-Data/train/textcraft/main_prefix/runtime_parser_aligned/complete_split/main_change_top3_w11.parquet`
- candidate_rank_counts: `{1: 1448, 2: 1439, 3: 1359}`

## Notes
- This is still the cutpoint-complete family: it preserves all remaining parser-valid cutpoints and then rebuilds prompt-space sidecars.
- `concat_like_prefix_action_rows` may remain because the runtime env accepts a single `Action:` line whose payload still contains multiple commands.
- `placeholder_like_prefix_action_rows` and `rows_with_multi_action_tag_in_prefix_messages` are the main parser-alignment audit fields to watch.
