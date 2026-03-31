# Prefix-GRPO Review Assets

This directory snapshots the main review materials for the current Prefix-GRPO TextCraft experiment.

Contents:

- `datasets/active/`: current default training parquet used by `run_textcraft_grpo_validated.sh`
- `datasets/cleaned_v2/`: cleaned audit subset kept for comparison
- `logs/`: successful prefix smoke-test log

Authoritative source paths on the server:

- active parquet:
  `/Data/wyh/datasets/Verl-Data/outputs/textcraft_old_logits/active/textcraft_validated_prefix_history_canonicalized_with_prefix_old_logprobs_step200_v2.parquet`
- cleaned parquet:
  `/Data/wyh/datasets/Verl-Data/outputs/textcraft_old_logits/active/cleaned_v2/textcraft_validated_cleaned_v2_20260326_000658.parquet`
- smoke log:
  `/Data/wyh/datasets/Verl-Data/outputs/textcraft_grpo_prefix_smoke_test_canonicalized/logs/train_canonicalized_prefix_smoke_test_20260322_231829.log`

Related technical report:

- `/Data/wyh/verl/prefix_grpo_main_experiment_reviewed.md`

Notes:

- These files are copied into the repository for reviewer convenience.
- The active parquet is the default dataset used by the current main training script.
- The cleaned parquet is not the script default; it is an audited comparison subset.
