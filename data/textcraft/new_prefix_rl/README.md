# TextCraft Rebuilt Prefix-Main Data

This directory stores the rebuilt TextCraft prefix-main data pipeline artifacts kept in the repository.

What is kept here:
- the raw teacher trajectory used to rebuild the chain
- normalized teacher data with stable `item_id`, `sample_idx`, and `sample_uid`
- the rebuilt prefix split / replay-validation / canonicalization outputs
- the `step200` full-trajectory teacher-forcing old-logprob sidecar
- the final audited main-experiment parquet

What was intentionally removed:
- the old incorrect `step200_v2` active parquet
- the old cleaned-v2 comparison parquet
- old entropy-analysis artifacts and offline-minimax artifacts that belonged to the broken chain

Canonical data layout:
- `stage0_teacher/`
  - `textcraft_trajectories.raw.jsonl`
  - `teacher_normalized.jsonl`
  - `teacher_normalized.parquet`
- `stage2_splits/`
  - `prefix_candidates_fixed_ratio_0p4.parquet`
- `stage3_replay_validation/`
  - validated / mismatch / unverifiable splits
  - refined `usable_state_feedback` and `drop_nonusable`
- `stage4_canonicalized/`
  - canonicalized prompt-format datasets
- `stage5_old_logits/`
  - `teacher_old_logprobs_step200.parquet`
- `stage6_training_build/`
  - exact-join training parquet
- `stage7_audit_release/`
  - audited release parquet
  - audit reports
  - smoke-test reports

Main experiment dataset:
- `stage7_audit_release/textcraft_prefix_main_train_step200.audited.parquet`

Training smoke subset:
- `stage7_audit_release/textcraft_prefix_main_train_step200.smoke_train.parquet`

Pipeline scripts live in:
- `examples/sglang_multiturn/my_exp/short_learning_validation/new_prefix_rl/scripts/`

Current default main-experiment script:
- `examples/sglang_multiturn/my_exp/rl/run_textcraft_grpo_validated.sh`
- its default `DATA_PATH` now points to the audited parquet above

Notes:
- the rebuilt chain does not use position-based alignment
- all joins are keyed by exact sample identity
- old-logprob computation is intended to run with at most 2 GPUs
