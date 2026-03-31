#!/usr/bin/env python3
"""Merge old-logprob shard parquets into a single exact-key sidecar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd

from common import NEW_PREFIX_ROOT, make_sample_uid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=NEW_PREFIX_ROOT / "stage5_old_logits" / "shards",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=NEW_PREFIX_ROOT / "stage5_old_logits" / "teacher_old_logprobs_step200.parquet",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=NEW_PREFIX_ROOT / "manifests" / "stage5_oldlogprob_merge_manifest.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shard_paths = sorted(args.input_dir.glob("teacher_old_logprobs_step200.shard*.parquet"))
    if not shard_paths:
        raise RuntimeError(f"No shard parquets found in {args.input_dir}")

    dfs: List[pd.DataFrame] = []
    for path in shard_paths:
        df = pd.read_parquet(path)
        if "sample_uid" not in df.columns:
            df = df.copy()
            df["sample_uid"] = df.apply(lambda row: make_sample_uid(row["item_id"], int(row["sample_idx"])), axis=1)
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    dup = int(merged["sample_uid"].duplicated().sum())
    if dup:
        raise RuntimeError(f"Found {dup} duplicate sample_uid rows while merging")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.output_path, index=False)

    manifest = {
        "input_dir": str(args.input_dir),
        "shards": [str(path) for path in shard_paths],
        "output_path": str(args.output_path),
        "rows": len(merged),
        "unique_sample_uid": int(merged["sample_uid"].nunique()),
    }
    args.manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

