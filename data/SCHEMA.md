# Dataset Schema

The complete per-dataset column inventory is in:

```text
data/SCHEMA_COLUMNS.csv
```

All included parquet files are training datasets for online continuation from a
replayed cut state. Each row is either a raw no-prefix training query or a
prefix query produced by one of the cut-selection strategies.

## Core Fields

| Field | Meaning |
| --- | --- |
| `data_source` | Environment or dataset source label used by the trainer. |
| `prompt` | Canonicalized chat prompt at the training state. This includes the system prompt, replayed prefix history, and the cut-state user observation. |
| `ability` | Task family label consumed by the trainer and reward manager. |
| `reward_model` | Reward-model metadata used by the `verl` data protocol. |
| `extra_info` | Per-row metadata, including task identifiers and replay/cut-selection annotations. |

## Prefix-GRPO Sidecars

| Field | Meaning |
| --- | --- |
| `assistant_prefix_span` | Prompt-token span for historical assistant tokens selected for prefix optimization. Coordinates are in the current canonicalized prompt, not in the original teacher trajectory. |
| `prefix_mask` | Prompt-space token mask selecting prefix tokens that enter the prefix policy loss. |
| `assistant_prefix_old_log_probs` | Old-policy log probabilities for the selected historical assistant prefix tokens. These are computed by teacher forcing under the SFT old-policy checkpoint. |
| `prefix_token_count` | Number of prompt tokens selected by `prefix_mask`. |

## Important Conventions

- Prefix annotations are in prompt-token coordinates on the canonicalized prompt.
- Prefix tokens can occur in the middle of the prompt; they are not assumed to be
  at the tail.
- Raw no-prefix rows are included alongside replayed prefix rows. They have
  zero selected prefix tokens.
- Continuation advantage is a per-sample sequence scalar broadcast to valid
  response tokens.
- Prefix advantage is a per-sample sequence scalar broadcast to selected prefix
  tokens.

## Dataset Inventory

Use `data/DATASETS.csv` for row counts, SHA256 hashes, file sizes, and
sidecar-field presence checks.
