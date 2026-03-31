#!/usr/bin/env python3
"""
Build an official-aligned TextCraft eval parquet.

This script keeps the existing prompt/template rows, but rewrites task-binding
metadata so the parquet matches the official sparse TextCraft item ids instead
of relying on row order.
"""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-parquet",
        default="/Data/wyh/datasets/Verl-Data/eval/textcraft/test.parquet",
        help="Existing TextCraft eval parquet used as the row template.",
    )
    parser.add_argument(
        "--official-json",
        default="/Data/wyh/datasets/AgentGym-RL-Data/eval/textcraft_test.json",
        help="Official TextCraft eval json containing item_id values.",
    )
    parser.add_argument(
        "--output-parquet",
        default="/Data/wyh/datasets/Verl-Data/eval/textcraft/test_official_aligned.parquet",
        help="Path to write the aligned parquet.",
    )
    return parser.parse_args()


def extract_session_id(row: dict) -> int | None:
    item_id = row.get("item_id")
    if isinstance(item_id, str) and item_id.startswith("textcraft_"):
        return int(item_id.split("_")[-1])

    extra_info = row.get("extra_info", {}) or {}
    interaction_kwargs = extra_info.get("interaction_kwargs", {}) or {}
    session_id = interaction_kwargs.get("session_id")
    if session_id is not None:
        return int(session_id)
    return None


def main() -> None:
    args = parse_args()

    source_path = Path(args.source_parquet)
    official_path = Path(args.official_json)
    output_path = Path(args.output_parquet)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(source_path)
    with official_path.open() as f:
        official_rows = json.load(f)

    official_item_ids = [row["item_id"] for row in official_rows]
    official_session_ids = [int(item_id.split("_")[-1]) for item_id in official_item_ids]

    source_rows = df.to_dict(orient="records")
    source_by_session_id = {}
    for row in source_rows:
        session_id = extract_session_id(row)
        if session_id is not None:
            source_by_session_id[session_id] = row

    aligned_rows = []
    for item_id, session_id in zip(official_item_ids, official_session_ids):
        template = source_by_session_id.get(session_id)
        if template is None:
            raise ValueError(
                f"Could not find source parquet row for official item_id={item_id}. "
                "This means the current parquet does not preserve the official task ids."
            )

        row = deepcopy(template)
        row["item_id"] = item_id
        row["original_index"] = session_id

        extra_info = deepcopy(row.get("extra_info", {}) or {})
        interaction_kwargs = deepcopy(extra_info.get("interaction_kwargs", {}) or {})
        extra_info["index"] = session_id
        interaction_kwargs["name"] = "textcraft"
        interaction_kwargs["session_id"] = session_id
        interaction_kwargs["data_idx"] = session_id
        interaction_kwargs["item_id"] = item_id
        extra_info["interaction_kwargs"] = interaction_kwargs
        row["extra_info"] = extra_info

        aligned_rows.append(row)

    aligned_df = pd.DataFrame(aligned_rows)
    aligned_df.to_parquet(output_path, index=False)

    print(f"Wrote {len(aligned_df)} aligned rows to {output_path}")
    print(
        "Aligned official item_id bands:",
        {
            "0_30": sum(0 <= x <= 30 for x in official_session_ids),
            "140_180": sum(140 <= x <= 180 for x in official_session_ids),
            "420_444": sum(420 <= x <= 444 for x in official_session_ids),
            "533_535": sum(533 <= x <= 535 for x in official_session_ids),
        },
    )


if __name__ == "__main__":
    main()
