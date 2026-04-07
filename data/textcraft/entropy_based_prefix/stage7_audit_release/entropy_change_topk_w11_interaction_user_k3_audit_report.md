# Entropy Stage7 Audit Report

- Dataset: `/Data/wyh/datasets/Verl-Data/train/textcraft/entropy_based_prefix/stage6_training_build/textcraft_prefix_entropy_change_topk_w11_interaction_user_k3_step200.prompt_space_recomputed.full.parquet`
- Rows: `3071`
- Passed: `True`

## Core Checks

- Missing columns: `[]`
- Strategy values: `['entropy_change_topk_w11_interaction_user_k3']`
- Duplicate candidate_uid: `0`
- Duplicate (item_id, sample_idx, strategy, cut_turn_idx): `0`
- Empty prefix old logprobs: `0`
- old_logprobs vs prefix_mask length mismatch: `0`
- prefix_token_count mismatch: `0`
- Prefix semantic span mismatch: `0`
- Prefix semantic mask mismatch: `0`
- Prefix semantic block mismatch: `0`
- Prefix semantic audit errors: `0`
- Drop-set overlap: `0`

## Composition

- Replay category counts: `{'validated': 3071}`
- Candidate rank counts: `{1: 1050, 2: 1006, 3: 1015}`
- Unique sample_uid: `1290`
- Duplicate sample_uid: `1781`
- Empty prefix_actions: `0`
- Multi-action-like rows: `80`

## Identity Stability

- task_id -> multiple goals: `0`
- goal -> multiple task_ids: `0`
- prefix_coordinate_system values: `['canonicalized_prompt']`

