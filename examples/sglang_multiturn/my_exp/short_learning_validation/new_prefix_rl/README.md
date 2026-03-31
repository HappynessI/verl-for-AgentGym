# TextCraft New Prefix RL Scripts

This directory contains the rebuilt TextCraft prefix-main data pipeline scripts.

Repository data root:
- `data/textcraft/new_prefix_rl/`

Key entrypoints:
- `scripts/01_normalize_teacher.py`
- `scripts/02_export_prefix_fixed_ratio.py`
- `scripts/05_replay_validate_prefix.py`
- `scripts/03_canonicalize_validated_prefix.py`
- `scripts/10_run_oldlogprob_step200_parallel.sh`
- `scripts/06_build_training_dataset_exact.py`
- `scripts/11_audit_release.py`
- `scripts/12_smoke_test_single_task.py`
- `scripts/run_prefix_main_train_smoke_test.sh`

The default script paths have been adjusted to use repo-relative data under `data/textcraft/new_prefix_rl/`.
