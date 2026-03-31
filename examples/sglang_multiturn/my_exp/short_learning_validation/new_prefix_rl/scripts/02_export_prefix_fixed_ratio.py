#!/usr/bin/env python3
"""Export fixed-ratio prefix split candidates with exact sample identity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from common import (
    NEW_PREFIX_ROOT,
    assistant_message_indices,
    extract_actions_from_messages,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-parquet",
        type=Path,
        default=NEW_PREFIX_ROOT / "stage0_teacher" / "teacher_normalized.parquet",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=NEW_PREFIX_ROOT / "stage2_splits" / "prefix_candidates_fixed_ratio_0p4.parquet",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=NEW_PREFIX_ROOT / "manifests" / "stage2_fixed_ratio_0p4_manifest.json",
    )
    parser.add_argument("--target-ratio", type=float, default=0.4)
    return parser.parse_args()


def choose_cut_turn_idx(num_assistant_messages: int, target_ratio: float) -> Tuple[int, float]:
    if num_assistant_messages <= 0:
        return 0, 0.0
    if num_assistant_messages == 1:
        return 0, 0.0

    for turn_idx in range(num_assistant_messages):
        q = turn_idx / (num_assistant_messages - 1)
        if q >= target_ratio:
            return turn_idx, q
    return num_assistant_messages - 1, 1.0


def split_messages(messages: List[Dict[str, Any]], cut_turn_idx: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    assistant_indices = assistant_message_indices(messages)
    if not assistant_indices:
        return [], list(messages)

    last_prefix_message_index = assistant_indices[min(cut_turn_idx, len(assistant_indices) - 1)]
    cut_position = last_prefix_message_index + 1
    return list(messages[:cut_position]), list(messages[cut_position:])


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.input_parquet)
    if df.empty:
        raise RuntimeError(f"Input parquet is empty: {args.input_parquet}")

    records: List[Dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        messages = row["conversations"]
        num_assistant_messages = int(row["num_assistant_messages"])
        cut_turn_idx, cut_q = choose_cut_turn_idx(num_assistant_messages, args.target_ratio)
        prefix_messages, continuation_messages = split_messages(messages, cut_turn_idx)

        record = {
            "sample_uid": row["sample_uid"],
            "item_id": row["item_id"],
            "sample_idx": int(row["sample_idx"]),
            "task_id": int(row["task_id"]),
            "goal": row.get("goal"),
            "success": int(row.get("success", 0)),
            "reward": row.get("reward", 0),
            "strategy": f"fixed_ratio_{str(args.target_ratio).replace('.', 'p')}",
            "cut_turn_idx": int(cut_turn_idx),
            "cut_relative_position_q": float(cut_q),
            "num_assistant_messages_total": num_assistant_messages,
            "num_prefix_messages": len(prefix_messages),
            "num_continuation_messages": len(continuation_messages),
            "num_prefix_assistant_messages": sum(msg.get("role") == "assistant" for msg in prefix_messages),
            "num_continuation_assistant_messages": sum(
                msg.get("role") == "assistant" for msg in continuation_messages
            ),
            "prefix_messages": prefix_messages,
            "continuation_messages": continuation_messages,
            "prefix_actions": extract_actions_from_messages(prefix_messages),
        }
        records.append(record)

    out_df = pd.DataFrame(records)
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.output_parquet, index=False)

    manifest = {
        "input_parquet": str(args.input_parquet),
        "output_parquet": str(args.output_parquet),
        "target_ratio": args.target_ratio,
        "rows": len(out_df),
        "unique_sample_uid": int(out_df["sample_uid"].nunique()),
        "empty_prefix_actions": int((out_df["prefix_actions"].apply(len) == 0).sum()),
        "with_continuation": int((out_df["num_continuation_messages"] > 0).sum()),
    }
    args.manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

