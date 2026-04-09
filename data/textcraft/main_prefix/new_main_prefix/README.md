# TextCraft Main Prefix New Main Prefix

## Contents
- complete_split: `/Data/wyh/datasets/Verl-Data/train/textcraft/main_prefix/new_main_prefix/complete_split`
- replay_validated: `/Data/wyh/datasets/Verl-Data/train/textcraft/main_prefix/new_main_prefix/replay_validated`
- audit: `/Data/wyh/datasets/Verl-Data/train/textcraft/main_prefix/new_main_prefix/audit`
- audit_report.md: `/Data/wyh/datasets/Verl-Data/train/textcraft/main_prefix/new_main_prefix/audit_report.md`

## Meaning
- This branch keeps sampled cutpoints first, then lets replay decide which prefixes remain usable.
- Action extraction reuses the same loose parser as online TextCraft interaction instead of the runtime-audit strict single-`Action:` rule.
- Parser-risk rows are audited instead of being pre-dropped from `complete_split/`.

## Task Coverage Notes
- The coverage notes below count prefix variants only (`is_raw_variant=false`). `raw` rows always keep all `374` training tasks.
- Each replay-validated dataset family still reaches `374/374` task coverage after unioning all of its prefix variants together.
- No replay-validated task falls back to `raw-only`. For every dataset family, each task keeps at least `2/3` prefix variants:
  - `main_fixed_gp1_fullflow`: `366` tasks keep `3/3` variants, `8` tasks keep `2/3`, `0` tasks keep `1/3` or `0/3`
  - `main_fixed_gp2_fullflow`: `371` tasks keep `3/3` variants, `3` tasks keep `2/3`, `0` tasks keep `1/3` or `0/3`
  - `main_raw_top3_fullflow`: `368` tasks keep `3/3` variants, `6` tasks keep `2/3`, `0` tasks keep `1/3` or `0/3`
  - `main_change_top3_w11_fullflow`: `362` tasks keep `3/3` variants, `12` tasks keep `2/3`, `0` tasks keep `1/3` or `0/3`
- TextCraft goal depth is derived from the environment's `sorted_item_depth_list[task_id]`.
- The `374` training tasks have depth distribution:
  - `depth1=94`
  - `depth2=232`
  - `depth3=47`
  - `depth4=1`
- The only `depth4` task is `task_id=537` (`lectern`), and it is not missing from any replay-validated variant.

### main_fixed_gp1_fullflow
- `fixed_ratio_0p1`: missing `4` tasks: `259 birch sign (d2)`, `312 leather boots (d2)`, `355 bucket (d2)`, `416 rabbit stew (d3)`
- `fixed_ratio_0p3`: missing `4` tasks: `251 item frame (d2)`, `383 jungle sign (d3)`, `449 orange banner (d3)`, `528 light blue banner (d2)`
- `fixed_ratio_0p5`: missing `0` tasks

### main_fixed_gp2_fullflow
- `fixed_ratio_0p25`: missing `2` tasks: `324 diamond hoe (d2)`, `383 jungle sign (d2)`
- `fixed_ratio_0p5`: missing `0` tasks
- `fixed_ratio_0p7`: missing `1` task: `531 red banner (d3)`

### main_raw_top3_fullflow
- `rank1`: missing `2` tasks: `209 pink wool (d2)`, `324 diamond hoe (d2)`
- `rank2`: missing `1` task: `388 spruce fence gate (d2)`
- `rank3`: missing `3` tasks: `76 oak wood (d1)`, `409 warped trapdoor (d2)`, `449 orange banner (d3)`
- `rank3` note: `task_id=76` is already absent in `complete_split/rank3`, because this sample-faithful branch never had a third candidate cutpoint for that task. The other missing `rank3` tasks are replay-filtered out.

### main_change_top3_w11_fullflow
- `rank1`: missing `4` tasks: `136 brown stained glass pane (d2)`, `209 pink wool (d2)`, `266 blue stained glass pane (d2)`, `324 diamond hoe (d2)`
- `rank2`: missing `2` tasks: `240 white stained glass (d2)`, `273 orange terracotta (d2)`
- `rank3`: missing `6` tasks: `76 oak wood (d1)`, `343 birch trapdoor (d2)`, `397 iron helmet (d3)`, `409 warped trapdoor (d2)`, `449 orange banner (d3)`, `528 light blue banner (d2)`
- `rank3` note: `task_id=76` is also already absent in `complete_split/rank3`; the other `rank3` gaps come from replay filtering.
