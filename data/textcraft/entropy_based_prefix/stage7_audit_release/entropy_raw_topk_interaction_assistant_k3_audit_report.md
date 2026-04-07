# Entropy Stage7 Audit Report

- Dataset: `/Data/wyh/datasets/Verl-Data/train/textcraft/entropy_based_prefix/stage6_training_build/textcraft_prefix_entropy_raw_topk_interaction_assistant_k3_step200.prompt_space_recomputed.full.parquet`
- Rows: `3230`
- Passed: `True`

## Core Checks

- Missing columns: `[]`
- Strategy values: `['entropy_raw_topk_interaction_assistant_k3']`
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

- Replay category counts: `{'validated': 3230}`
- Candidate rank counts: `{1: 1105, 2: 1075, 3: 1050}`
- Unique sample_uid: `1301`
- Duplicate sample_uid: `1929`
- Empty prefix_actions: `0`
- Multi-action-like rows: `87`

## Identity Stability

- task_id -> multiple goals: `0`
- goal -> multiple task_ids: `0`
- prefix_coordinate_system values: `['canonicalized_prompt']`

