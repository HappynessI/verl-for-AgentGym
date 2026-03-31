# Stage7 Audit Report

- Dataset: `/Data/wyh/datasets/Verl-Data/train/textcraft/new_prefix_rl/stage6_training_build/textcraft_prefix_main_train_step200.parquet`
- Rows: `1189`
- Passed: `True`

## Core Checks

- Missing columns: `[]`
- Duplicate sample_uid: `0`
- Duplicate (item_id, sample_idx): `0`
- Empty prefix old logprobs: `0`
- old_logprobs vs prefix_mask length mismatch: `0`
- prefix_token_count mismatch: `0`
- Drop-set overlap: `0`

## Composition

- Replay category counts: `{'validated': 1020, 'unverifiable': 169}`
- Empty prefix_actions: `0`
- Multi-action-like rows: `57`

## Identity Stability

- task_id -> multiple goals: `0`
- goal -> multiple task_ids: `0`

