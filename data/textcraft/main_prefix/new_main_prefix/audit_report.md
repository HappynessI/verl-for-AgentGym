# Sample-Faithful Main Prefix Audit

- Output root: `/Data/wyh/datasets/Verl-Data/train/textcraft/main_prefix/new_main_prefix`
- Comparison baseline: existing `main_prefix/complete_split/` and `main_prefix/replay_validated/` datasets.
- Parser rule: use the same loose extraction order as online TextCraft interaction (`[[...]]` last, then `Action:\n`, then inline `Action:`).

## Dataset Diffs
### main_fixed_gp1
- legacy_rows: `5984`
- rebuilt_rows: `5984`
- row_delta: `0`
- legacy_empty_prefix_action_rows: `0`
- rebuilt_empty_prefix_action_rows: `3`
- legacy_rows_with_multi_action_tag_in_prefix_messages: `112`
- rebuilt_rows_with_multi_action_tag_in_prefix_messages: `112`

### main_fixed_gp1_fullflow
- legacy_rows: `4970`
- rebuilt_rows: `5513`
- row_delta: `543`
- legacy_empty_prefix_action_rows: `0`
- rebuilt_empty_prefix_action_rows: `0`
- legacy_rows_with_multi_action_tag_in_prefix_messages: `43`
- rebuilt_rows_with_multi_action_tag_in_prefix_messages: `40`

### main_fixed_gp2
- legacy_rows: `5984`
- rebuilt_rows: `5984`
- row_delta: `0`
- legacy_empty_prefix_action_rows: `0`
- rebuilt_empty_prefix_action_rows: `3`
- legacy_rows_with_multi_action_tag_in_prefix_messages: `147`
- rebuilt_rows_with_multi_action_tag_in_prefix_messages: `147`

### main_fixed_gp2_fullflow
- legacy_rows: `5114`
- rebuilt_rows: `5627`
- row_delta: `513`
- legacy_empty_prefix_action_rows: `0`
- rebuilt_empty_prefix_action_rows: `0`
- legacy_rows_with_multi_action_tag_in_prefix_messages: `73`
- rebuilt_rows_with_multi_action_tag_in_prefix_messages: `63`

### main_raw_top3
- legacy_rows: `5895`
- rebuilt_rows: `5895`
- row_delta: `0`
- legacy_empty_prefix_action_rows: `0`
- rebuilt_empty_prefix_action_rows: `3`
- legacy_rows_with_multi_action_tag_in_prefix_messages: `147`
- rebuilt_rows_with_multi_action_tag_in_prefix_messages: `147`

### main_raw_top3_fullflow
- legacy_rows: `4726`
- rebuilt_rows: `5239`
- row_delta: `513`
- legacy_empty_prefix_action_rows: `0`
- rebuilt_empty_prefix_action_rows: `0`
- legacy_rows_with_multi_action_tag_in_prefix_messages: `81`
- rebuilt_rows_with_multi_action_tag_in_prefix_messages: `59`

### main_change_top3_w11
- legacy_rows: `5895`
- rebuilt_rows: `5895`
- row_delta: `0`
- legacy_empty_prefix_action_rows: `0`
- rebuilt_empty_prefix_action_rows: `3`
- legacy_rows_with_multi_action_tag_in_prefix_messages: `150`
- rebuilt_rows_with_multi_action_tag_in_prefix_messages: `150`

### main_change_top3_w11_fullflow
- legacy_rows: `4693`
- rebuilt_rows: `5190`
- row_delta: `497`
- legacy_empty_prefix_action_rows: `0`
- rebuilt_empty_prefix_action_rows: `0`
- legacy_rows_with_multi_action_tag_in_prefix_messages: `79`
- rebuilt_rows_with_multi_action_tag_in_prefix_messages: `58`

