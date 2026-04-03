# Stage7 Audit Report

- Dataset: `/Data/wyh/verl/data/textcraft/new_prefix_rl/stage6_training_build/recompute_oldlogprob_20260402/textcraft_prefix_main_train_step200.prompt_space_recomputed.full.parquet`
- Rows: `1189`
- Passed: `True`

## Core Checks

- Missing columns: `[]`
- Duplicate sample_uid: `0`
- Duplicate (item_id, sample_idx): `0`
- Empty prefix old logprobs: `0`
- old_logprobs vs prefix_mask length mismatch: `0`
- prefix_token_count mismatch: `0`
- Prefix semantic span mismatch: `0`
- Prefix semantic mask mismatch: `0`
- Prefix semantic block mismatch: `0`
- Prefix semantic audit errors: `0`
- Drop-set overlap: `0`

## Composition

- Replay category counts: `{'validated': 1020, 'unverifiable': 169}`
- Empty prefix_actions: `0`
- Multi-action-like rows: `57`

## Identity Stability

- task_id -> multiple goals: `0`
- goal -> multiple task_ids: `0`

