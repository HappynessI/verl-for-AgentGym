# Data

This directory contains the replay-validated Prefix-GRPO parquet datasets used
by the review artifact.

Use `DATASETS.csv` as the authoritative inventory. It includes environment,
split, artifact id, relative path, row count, file size, SHA256 hash, and the
presence of the prompt-space prefix sidecar fields.

The main datasets are:

```text
textcraft/main/main_change_top3_w11_fullflow.parquet
babyai/main/main_change_top3_w11_fullflow.parquet
alfworld/main/main_change_top3_w11_fullflow.parquet
```

TextCraft and BabyAI ablation datasets are under each environment's
`ablations/` directory. TextCraft includes the w5 window ablation; the w7
window ablation is intentionally not included.
