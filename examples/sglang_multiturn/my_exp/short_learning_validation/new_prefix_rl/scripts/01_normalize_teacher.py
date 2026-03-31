#!/usr/bin/env python3
"""Normalize raw TextCraft teacher trajectories into a stable keyed schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from common import (
    DEFAULT_RAW_TEACHER_PATH,
    NEW_PREFIX_ROOT,
    count_assistant_messages,
    extract_goal_from_messages,
    iter_jsonl,
    make_sample_uid,
    parse_task_id,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=DEFAULT_RAW_TEACHER_PATH)
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=NEW_PREFIX_ROOT / "stage0_teacher" / "teacher_normalized.parquet",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=NEW_PREFIX_ROOT / "stage0_teacher" / "teacher_normalized.jsonl",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=NEW_PREFIX_ROOT / "manifests" / "stage0_teacher_manifest.json",
    )
    return parser.parse_args()


def normalize_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    item_id = raw["item_id"]
    sample_idx = int(raw["sample_idx"])
    conversations: List[Dict[str, Any]] = raw.get("conversations", [])
    goal = extract_goal_from_messages(conversations)

    return {
        "sample_uid": make_sample_uid(item_id, sample_idx),
        "item_id": item_id,
        "sample_idx": sample_idx,
        "task_id": parse_task_id(item_id),
        "goal": goal,
        "success": int(raw.get("success", 0)),
        "reward": raw.get("reward", 0),
        "task_name": raw.get("task_name"),
        "model": raw.get("model"),
        "num_messages": len(conversations),
        "num_assistant_messages": count_assistant_messages(conversations),
        "conversations": conversations,
    }


def main() -> None:
    args = parse_args()

    rows = [normalize_record(raw) for raw in iter_jsonl(args.input_path)]
    if not rows:
        raise RuntimeError(f"No rows found in {args.input_path}")

    df = pd.DataFrame(rows)
    duplicated = df["sample_uid"].duplicated().sum()
    if duplicated:
        raise RuntimeError(f"Found {duplicated} duplicate sample_uid values")

    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(args.output_parquet, index=False)
    write_jsonl(args.output_jsonl, rows)

    manifest = {
        "input_path": str(args.input_path),
        "output_parquet": str(args.output_parquet),
        "output_jsonl": str(args.output_jsonl),
        "total_rows": len(df),
        "unique_sample_uid": int(df["sample_uid"].nunique()),
        "unique_item_id": int(df["item_id"].nunique()),
        "missing_goal_rows": int(df["goal"].isna().sum()),
        "success_rows": int((df["success"] == 1).sum()),
        "failed_rows": int((df["success"] != 1).sum()),
    }
    args.manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

