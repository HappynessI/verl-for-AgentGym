#!/usr/bin/env python3
"""Split normalized teacher JSONL into deterministic shards for parallel forward-only precompute."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from common import NEW_PREFIX_ROOT, iter_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-path",
        type=Path,
        default=NEW_PREFIX_ROOT / "stage0_teacher" / "teacher_normalized.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=NEW_PREFIX_ROOT / "stage5_old_logits" / "shards",
    )
    parser.add_argument("--num-shards", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(iter_jsonl(args.input_path))
    if not rows:
        raise RuntimeError(f"No rows found in {args.input_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_buffers: List[List[dict]] = [[] for _ in range(args.num_shards)]

    for idx, row in enumerate(rows):
        shard_buffers[idx % args.num_shards].append(row)

    for shard_idx, shard_rows in enumerate(shard_buffers):
        out_path = args.output_dir / f"teacher_normalized.shard{shard_idx}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for row in shard_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"shard{shard_idx}: {len(shard_rows)} -> {out_path}")


if __name__ == "__main__":
    main()

