#!/usr/bin/env python3
"""Convert replay-validated prefix splits into final training prompt format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from common import (
    DEFAULT_TRAIN_PARQUET_PATH,
    NEW_PREFIX_ROOT,
    canonicalize_assistant_content,
    is_warmup_assistant_message,
    is_warmup_user_message,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-path",
        type=Path,
        default=NEW_PREFIX_ROOT / "stage3_replay_validation" / "fixed_ratio_0p4_validated.parquet",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=NEW_PREFIX_ROOT / "stage4_canonicalized" / "fixed_ratio_0p4_validated_canonicalized.parquet",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=NEW_PREFIX_ROOT / "manifests" / "stage4_canonicalized_manifest.json",
    )
    parser.add_argument("--train-parquet-path", type=Path, default=DEFAULT_TRAIN_PARQUET_PATH)
    return parser.parse_args()


def load_reference_system_prompt(train_parquet_path: Path) -> str:
    df = pd.read_parquet(train_parquet_path)
    if df.empty:
        raise RuntimeError(f"Reference train parquet is empty: {train_parquet_path}")

    row = df.iloc[0]
    for column in ("messages", "prompt"):
        if column not in row:
            continue
        messages = row[column]
        if hasattr(messages, "tolist"):
            messages = messages.tolist()
        for msg in messages:
            if msg.get("role") == "system":
                return msg.get("content", "")
    raise RuntimeError(f"Could not find a system prompt in {train_parquet_path}")


def build_training_prompt(
    prefix_messages: List[Dict[str, Any]],
    continuation_messages: List[Dict[str, Any]],
    system_prompt: str,
) -> List[Dict[str, str]]:
    prompt: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    for msg in prefix_messages:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "user":
            if is_warmup_user_message(content):
                continue
            prompt.append({"role": "user", "content": content})
            continue

        if role == "assistant":
            if is_warmup_assistant_message(content):
                continue
            prompt.append({"role": "assistant", "content": canonicalize_assistant_content(content)})

    cut_observation: Optional[str] = None
    for msg in continuation_messages:
        if msg.get("role") == "user":
            cut_observation = msg.get("content", "")
            break
    if cut_observation:
        prompt.append({"role": "user", "content": cut_observation})

    return prompt


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.input_path)
    if df.empty:
        raise RuntimeError(f"Input parquet is empty: {args.input_path}")

    system_prompt = load_reference_system_prompt(args.train_parquet_path)
    records: List[Dict[str, Any]] = []

    for row in df.to_dict(orient="records"):
        prefix_messages = row["prefix_messages"]
        continuation_messages = row["continuation_messages"]
        prompt = build_training_prompt(prefix_messages, continuation_messages, system_prompt)

        record = {
            "sample_uid": row["sample_uid"],
            "item_id": row["item_id"],
            "sample_idx": int(row["sample_idx"]),
            "task_id": int(row["task_id"]),
            "goal": row.get("goal"),
            "strategy": row.get("strategy"),
            "cut_turn_idx": int(row.get("cut_turn_idx", 0)),
            "replay_category": row.get("replay_category", "validated"),
            "data_source": "textcraft",
            "ability": "crafting",
            "prompt": prompt,
            "reward_model": {"ground_truth": "", "style": "interaction"},
            "extra_info": {
                "index": int(row["sample_idx"]),
                "sample_uid": row["sample_uid"],
                "interaction_kwargs": {
                    "name": "textcraft",
                    "task_id": int(row["task_id"]),
                    "eval_mode": False,
                    "prefix_actions": list(row.get("prefix_actions", [])),
                    "goal": row.get("goal"),
                },
                "reward_model": {"ground_truth": "", "style": "interaction"},
            },
            "prefix_messages": prefix_messages,
            "continuation_messages": continuation_messages,
            "prefix_actions": list(row.get("prefix_actions", [])),
        }
        records.append(record)

    out_df = pd.DataFrame(records)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.output_path, index=False)

    manifest = {
        "input_path": str(args.input_path),
        "output_path": str(args.output_path),
        "rows": len(out_df),
        "unique_sample_uid": int(out_df["sample_uid"].nunique()),
        "reference_train_parquet": str(args.train_parquet_path),
    }
    args.manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

