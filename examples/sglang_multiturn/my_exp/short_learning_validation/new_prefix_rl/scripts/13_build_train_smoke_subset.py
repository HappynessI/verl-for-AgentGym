#!/usr/bin/env python3
"""Build a tiny deterministic subset for prefix-main training smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from common import NEW_PREFIX_ROOT, extract_action


DEFAULT_INPUT = (
    NEW_PREFIX_ROOT / "stage7_audit_release" / "textcraft_prefix_main_train_step200.audited.parquet"
)
DEFAULT_OUTPUT = (
    NEW_PREFIX_ROOT / "stage7_audit_release" / "textcraft_prefix_main_train_step200.smoke_train.parquet"
)
DEFAULT_MANIFEST = NEW_PREFIX_ROOT / "manifests" / "stage7_smoke_train_subset_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--max-samples", type=int, default=16)
    return parser.parse_args()


def to_list(value: Any) -> List[Any]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value)


def count_continuation_actions(messages: Any) -> int:
    count = 0
    for msg in to_list(messages):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        if extract_action(msg.get("content", "")):
            count += 1
    return count


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.input_path)

    ranked_rows: List[Dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        if row.get("replay_category") != "validated":
            continue
        continuation_action_count = count_continuation_actions(row["continuation_messages"])
        if continuation_action_count != 1:
            continue
        prefix_actions = to_list(row["prefix_actions"])
        ranked_rows.append(
            {
                **row,
                "_prefix_len": len(prefix_actions),
                "_continuation_action_count": continuation_action_count,
            }
        )

    ranked_rows.sort(
        key=lambda row: (
            row["_prefix_len"],
            int(row["prefix_token_count"]),
            int(row["task_id"]),
            int(row["sample_idx"]),
        )
    )
    selected = ranked_rows[: args.max_samples]
    if not selected:
        raise RuntimeError("No suitable smoke-train samples found")

    output_rows = []
    for row in selected:
        clean_row = dict(row)
        clean_row.pop("_prefix_len", None)
        clean_row.pop("_continuation_action_count", None)
        output_rows.append(clean_row)

    out_df = pd.DataFrame(output_rows)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.output_path, index=False)

    manifest = {
        "input_path": str(args.input_path),
        "output_path": str(args.output_path),
        "max_samples": args.max_samples,
        "selected_rows": len(out_df),
        "sample_uids": out_df["sample_uid"].tolist(),
        "task_ids": sorted(out_df["task_id"].astype(int).unique().tolist()),
        "mean_prefix_token_count": float(out_df["prefix_token_count"].mean()),
        "max_prefix_token_count": int(out_df["prefix_token_count"].max()),
        "mean_prefix_actions": float(
            np.mean([len(to_list(v)) for v in out_df["prefix_actions"].tolist()])
        ),
    }
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
