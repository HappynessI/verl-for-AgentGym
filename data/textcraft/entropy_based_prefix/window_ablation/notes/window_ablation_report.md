# Entropy Change-Window Ablation Report

## Scope

- Pipeline root: `/Data/wyh/datasets/Verl-Data/train/textcraft/entropy_based_prefix`
- Strategy family: `change_topk`
- Domain: `interaction_assistant`
- `top_k = 3`
- Windows tested: `[1, 3, 5, 7, 9, 11, 15, 21]`
- Comparison level: stage2 candidate export + stage3 replay validation

## Method

- For each `change_window`, rerun `05_export_entropy_prefix_candidates.py` with `--domains interaction_assistant --scorers change_topk`.
- Then rerun `06_replay_validate_entropy_candidates.py` only for the corresponding strategy.
- Use the resulting replay categories to compare validated / mismatch / unverifiable behavior.
- Compare each window to `w11` by checking whether the `rank1` cut turn stays the same for the same `sample_uid`.

Important implementation note:
- The exporter requires odd windows. In the current implementation, smoothing depends on `radius = window // 2`, so even windows would collapse to the same radius as the previous odd window and are not separately informative.

## Main Findings

- Best validated rate: `w5` with `73.61%`.
- Best triple-rank validated coverage: `w5` with `795` samples having all `rank1/2/3` validated.
- Baseline `w11` validated rate: `72.68%`, triple-rank validated samples: `777`.
- `unverifiable` does not disappear as the window changes; much of it still comes from conservative validator limits rather than obvious replay failure.

## Summary Table

| window | validated_rate | mismatch_rate | unverifiable_rate | validated_rows | validated_samples_with_3_ranks | cut_exact_in_unverifiable | cut+next_exact_in_unverifiable | mean_cut_delta_vs_fixed_ratio | rank1_same_cut_vs_w11 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 72.83% | 1.11% | 26.05% | 3204 | 785 | 739 | 490 | 0.193 | 78.14% |
| 3 | 73.31% | 1.07% | 25.62% | 3225 | 793 | 712 | 474 | 0.197 | 84.02% |
| 5 | 73.61% | 1.14% | 25.26% | 3238 | 795 | 693 | 458 | 0.216 | 84.16% |
| 7 | 73.49% | 1.14% | 25.37% | 3233 | 792 | 707 | 468 | 0.225 | 84.76% |
| 9 | 73.33% | 1.25% | 25.41% | 3226 | 794 | 711 | 476 | 0.258 | 84.09% |
| 11 | 72.68% | 1.14% | 26.19% | 3197 | 777 | 727 | 498 | 0.220 | 100.00% |
| 15 | 71.72% | 1.00% | 27.28% | 3155 | 754 | 772 | 542 | 0.216 | 79.48% |
| 21 | 71.06% | 1.14% | 27.80% | 3126 | 744 | 803 | 569 | 0.239 | 75.33% |

## Per-Window Notes

### w1

- Stage2 rows: `4399`; rank counts: `{'1': 1496, '2': 1496, '3': 1407}`.
- Replay counts: `{'mismatch': 49, 'unverifiable': 1146, 'validated': 3204}`.
- Validated unique samples: `1299`.
- Samples with all three validated ranks: `785`.
- Mean cut delta vs `fixed_ratio_0p4`: `0.193` turns.
- Rank1 same cut as `w11`: `78.14%`.
- In `unverifiable`, cut observation exact matches: `739`; cut+next exact matches: `490`.

### w3

- Stage2 rows: `4399`; rank counts: `{'1': 1496, '2': 1496, '3': 1407}`.
- Replay counts: `{'mismatch': 47, 'unverifiable': 1127, 'validated': 3225}`.
- Validated unique samples: `1300`.
- Samples with all three validated ranks: `793`.
- Mean cut delta vs `fixed_ratio_0p4`: `0.197` turns.
- Rank1 same cut as `w11`: `84.02%`.
- In `unverifiable`, cut observation exact matches: `712`; cut+next exact matches: `474`.

### w5

- Stage2 rows: `4399`; rank counts: `{'1': 1496, '2': 1496, '3': 1407}`.
- Replay counts: `{'mismatch': 50, 'unverifiable': 1111, 'validated': 3238}`.
- Validated unique samples: `1301`.
- Samples with all three validated ranks: `795`.
- Mean cut delta vs `fixed_ratio_0p4`: `0.216` turns.
- Rank1 same cut as `w11`: `84.16%`.
- In `unverifiable`, cut observation exact matches: `693`; cut+next exact matches: `458`.

### w7

- Stage2 rows: `4399`; rank counts: `{'1': 1496, '2': 1496, '3': 1407}`.
- Replay counts: `{'mismatch': 50, 'unverifiable': 1116, 'validated': 3233}`.
- Validated unique samples: `1303`.
- Samples with all three validated ranks: `792`.
- Mean cut delta vs `fixed_ratio_0p4`: `0.225` turns.
- Rank1 same cut as `w11`: `84.76%`.
- In `unverifiable`, cut observation exact matches: `707`; cut+next exact matches: `468`.

### w9

- Stage2 rows: `4399`; rank counts: `{'1': 1496, '2': 1496, '3': 1407}`.
- Replay counts: `{'mismatch': 55, 'unverifiable': 1118, 'validated': 3226}`.
- Validated unique samples: `1302`.
- Samples with all three validated ranks: `794`.
- Mean cut delta vs `fixed_ratio_0p4`: `0.258` turns.
- Rank1 same cut as `w11`: `84.09%`.
- In `unverifiable`, cut observation exact matches: `711`; cut+next exact matches: `476`.

### w11

- Stage2 rows: `4399`; rank counts: `{'1': 1496, '2': 1496, '3': 1407}`.
- Replay counts: `{'mismatch': 50, 'unverifiable': 1152, 'validated': 3197}`.
- Validated unique samples: `1295`.
- Samples with all three validated ranks: `777`.
- Mean cut delta vs `fixed_ratio_0p4`: `0.220` turns.
- Rank1 same cut as `w11`: `100.00%`.
- In `unverifiable`, cut observation exact matches: `727`; cut+next exact matches: `498`.

### w15

- Stage2 rows: `4399`; rank counts: `{'1': 1496, '2': 1496, '3': 1407}`.
- Replay counts: `{'mismatch': 44, 'unverifiable': 1200, 'validated': 3155}`.
- Validated unique samples: `1294`.
- Samples with all three validated ranks: `754`.
- Mean cut delta vs `fixed_ratio_0p4`: `0.216` turns.
- Rank1 same cut as `w11`: `79.48%`.
- In `unverifiable`, cut observation exact matches: `772`; cut+next exact matches: `542`.

### w21

- Stage2 rows: `4399`; rank counts: `{'1': 1496, '2': 1496, '3': 1407}`.
- Replay counts: `{'mismatch': 50, 'unverifiable': 1223, 'validated': 3126}`.
- Validated unique samples: `1289`.
- Samples with all three validated ranks: `744`.
- Mean cut delta vs `fixed_ratio_0p4`: `0.239` turns.
- Rank1 same cut as `w11`: `75.33%`.
- In `unverifiable`, cut observation exact matches: `803`; cut+next exact matches: `569`.

## Artifacts

- Plot: `/Data/wyh/datasets/Verl-Data/train/textcraft/entropy_based_prefix/window_ablation/outputs/window_ablation_summary.png`
- Structured summary: `/Data/wyh/datasets/Verl-Data/train/textcraft/entropy_based_prefix/window_ablation/notes/window_ablation_report.json`
- CSV summary: `/Data/wyh/datasets/Verl-Data/train/textcraft/entropy_based_prefix/window_ablation/notes/window_ablation_report.csv`

