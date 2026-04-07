#!/usr/bin/env python3
"""Merge entropy shard parquets into a single exact-key sidecar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd

from common import ENTROPY_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ENTROPY_ROOT / "stage1_entropy" / "shards")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ENTROPY_ROOT / "stage1_entropy" / "textcraft_teacher_entropy_step200.parquet",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=ENTROPY_ROOT / "manifests" / "stage1_entropy_merge_manifest.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shard_paths = sorted(args.input_dir.glob("textcraft_teacher_entropy_step200.shard*.parquet"))
    if not shard_paths:
        raise RuntimeError(f"No entropy shard parquets found in {args.input_dir}")

    dfs: List[pd.DataFrame] = [pd.read_parquet(path) for path in shard_paths]
    merged = pd.concat(dfs, ignore_index=True)
    duplicate_rows = int(merged["sample_uid"].duplicated().sum())
    if duplicate_rows:
        raise RuntimeError(f"Found {duplicate_rows} duplicate sample_uid rows while merging entropy shards")

    merged = merged.sort_values(["sample_uid"]).reset_index(drop=True)
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
