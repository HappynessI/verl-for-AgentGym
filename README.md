<div align="center">

# Prefix-GRPO

**Replayed prefix optimization for small-model agents**

[![Method](https://img.shields.io/badge/method-Prefix--GRPO-2f6fed)](#method)
[![Environments](https://img.shields.io/badge/envs-TextCraft%20%7C%20BabyAI%20%7C%20ALFWorld-0f8b8d)](#environments)
[![Code](https://img.shields.io/badge/code-verl%20%2B%20AgentGym-6b5cff)](#repository-layout)
[![Data](https://img.shields.io/badge/data-not%20tracked-lightgrey)](#data)

Code for **From Trajectories to Prefixes: Reusing Teacher Trajectories via
Replayed Prefixes and Online Continuation**.

</div>

## Overview

Prefix-GRPO reuses teacher trajectories as more than one-shot imitation targets.
It splits a teacher rollout into replayable prefix queries, restores each
intermediate environment state by replay, and then trains the student on online
continuations from those cut states.

Unlike response-only GRPO, the implementation can also optimize historical
assistant tokens inside the replayed prefix. Prefix old log-probabilities are
computed under a policy-distilled SFT checkpoint and stored as prompt-space
sidecars, which lets prefix tokens enter the same clipped policy optimization
family as continuation tokens.

## Method

```text
teacher trajectory
      |
      |  entropy-change cut selection
      v
replayable prefix queries
      |
      |  environment replay validation
      v
canonical prompt at cut state
      |
      |  online rollout
      v
Prefix-GRPO objective
  - continuation token update
  - historical assistant prefix token update
```

The main TextCraft pipeline uses assistant-token **Entropy-Change Top-3**
selection with a centered window of `w=11`, then validates candidate cuts by
environment replay before admitting them to training.

## Highlights

| Component | What this repository contains |
| --- | --- |
| Prefix objective | Split prefix/continuation GRPO loss with prompt-space prefix sidecars |
| Old-policy anchor | SFT checkpoint teacher-forcing log-probabilities for prefix tokens |
| Data pipeline | Entropy scoring, cut selection, replay validation, canonicalization |
| Agent stack | Modified `verl` trainer plus AgentGym-style environment interactions |
| Environments | TextCraft, BabyAI, ALFWorld, SciWorld, WebShop wrappers |

## Current Main Setting

The current TextCraft main experiment is:

```text
Dataset: Entropy-Change Top-3, replay validated
Objective: split Prefix-GRPO
Prefix advantage: cont_mean_abs
KL loss: disabled
Entropy coefficient: 0
```

The launch script sets the active defaults:

```bash
scripts/train/run_textcraft_grpo_validated.sh
```

Core overrides:

```text
optimize_prefix_tokens=true
prefix_loss_mode=split
prefix_advantage_mode=cont_mean_abs
use_kl_loss=false
enable_activation_offload=false
entropy_coeff=0
```

`constant` prefix advantage is still supported for ablations, but it is not the
main setting.

## Repository Layout

```text
Prefix_GRPO/
├── config/                  # GRPO and environment interaction configs
├── envs/AgentGym/           # Environment wrappers and local servers
├── legacy_eval/             # Legacy vLLM-server evaluation utilities
├── scripts/
│   ├── build_data/          # Prefix data construction and validation
│   ├── train/               # Training and SFT launch scripts
│   └── utils/               # Small helper utilities
├── third_party/             # Runtime requirement manifests only
└── verl/                    # Modified verl training stack
```

Generated datasets, checkpoints, logs, offline wheels, and local run outputs are
not tracked.

## Data

Training data is intentionally excluded from Git. For the default TextCraft run,
place the replay-validated parquet at:

```text
data/textcraft/replay_validated/main_change_top3_w11_fullflow.parquet
```

or pass it explicitly:

```bash
DATA_PATH=/path/to/main_change_top3_w11_fullflow.parquet \
  bash scripts/train/run_textcraft_grpo_validated.sh
```

The official internal path used for the main experiment was:

```text
/Data/wyh/datasets/Verl-Data/train/textcraft/main_prefix/new_main_prefix/replay_validated/main_change_top3_w11_fullflow.parquet
```

## Build Prefix Data

Data construction lives under `scripts/build_data/`, separate from training
launchers.

Key TextCraft scripts:

```text
compute_textcraft_teacher_entropy.py
build_textcraft_main_prefix_new_main_prefix.py
build_textcraft_teacher_demo_rows.py
textcraft_entropy_utils.py
```

BabyAI and ALFWorld builders:

```text
build_babyai_prefix_rl_change_top3.py
build_alfworld_prefix_rl_change_top3.py
```

The data pipeline stores these prompt-space fields for prefix optimization:

```text
assistant_prefix_span
prefix_mask
assistant_prefix_old_log_probs
prefix_token_count
```

## Train

Main TextCraft Prefix-GRPO:

```bash
MODEL_ROOT=/path/to/models \
OUTPUT_ROOT=/path/to/outputs \
DATA_PATH=/path/to/main_change_top3_w11_fullflow.parquet \
NUM_GPUS=2 \
bash scripts/train/run_textcraft_grpo_validated.sh
```

Small smoke run:

```bash
MODEL_ROOT=/path/to/models \
OUTPUT_ROOT=/tmp/prefix_grpo_outputs \
DATA_PATH=/path/to/main_change_top3_w11_fullflow.parquet \
SAVE_FREQ=-1 \
TEST_FREQ=-1 \
DEBUG_MODE=1 \
DEBUG_MAX_SAMPLES=2 \
NUM_EPOCHS=1 \
bash scripts/train/run_textcraft_grpo_validated.sh
```

For smoke tests, keep checkpoint saving disabled unless a checkpoint artifact is
part of the test.

## Environments

| Environment | Status in this codebase |
| --- | --- |
| TextCraft | Primary Prefix-GRPO path |
| BabyAI | Prefix data builder and training scripts included |
| ALFWorld | Prefix data builder and training scripts included |
| SciWorld | GRPO/SFT environment wrapper included |
| WebShop | Wrapper included; Python/runtime constraints may require extra setup |

## Notes

- Prefix annotations are in prompt-token coordinates on the canonicalized
  prompt. Prefix tokens are often in the middle of the prompt, not at the tail.
- The current `joint` objective variant shares clipping with continuation tokens
  and should not be described as strict full-trajectory PPO.
- Entropy is used for offline cut selection; `entropy_coeff=0`, so entropy is
  not an active reward term or training regularizer in the main experiment.
