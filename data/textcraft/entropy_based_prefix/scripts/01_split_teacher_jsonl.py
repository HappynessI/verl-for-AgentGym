#!/usr/bin/env python3
"""Split normalized teacher trajectories into balanced shards for entropy forward."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from common import DEFAULT_INPUT_JSONL, ENTROPY_ROOT, ensure_parent, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--output-dir", type=Path, default=ENTROPY_ROOT / "stage1_entropy" / "shards")
    parser.add_argument("--manifest-path", type=Path, default=ENTROPY_ROOT / "manifests" / "stage1_split_manifest.json")
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--balance-by",
        type=str,
        choices=("char_length", "num_messages", "round_robin"),
        default="char_length",
    )
    return parser.parse_args()


def row_weight(row: Dict[str, Any], balance_by: str) -> int:
    if balance_by == "round_robin":
        return 1
    if balance_by == "num_messages":
        return int(row.get("num_messages", len(row.get("conversations", []))))
    return sum(len(msg.get("content", "")) for msg in row.get("conversations", []))


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input_path)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    if not rows:
        raise RuntimeError(f"No rows found in {args.input_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ensure_parent(args.manifest_path)

    shard_buffers: List[List[Dict[str, Any]]] = [[] for _ in range(args.num_shards)]
    shard_loads = [0 for _ in range(args.num_shards)]

    if args.balance_by == "round_robin":
        for idx, row in enumerate(rows):
            shard_idx = idx % args.num_shards
            shard_buffers[shard_idx].append(row)
            shard_loads[shard_idx] += 1
    else:
        weighted_rows = sorted(
            ((row_weight(row, args.balance_by), row) for row in rows),
            key=lambda item: item[0],
            reverse=True,
        )
        for weight, row in weighted_rows:
            shard_idx = min(range(args.num_shards), key=lambda i: shard_loads[i])
            shard_buffers[shard_idx].append(row)
            shard_loads[shard_idx] += weight

    shard_records = []
    for shard_idx, shard_rows in enumerate(shard_buffers):
        out_path = args.output_dir / f"teacher_normalized.shard{shard_idx}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for row in shard_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        shard_records.append(
            {
                "shard_idx": shard_idx,
                "rows": len(shard_rows),
                "load_proxy": shard_loads[shard_idx],
                "output_path": str(out_path),
            }
        )
        print(f"shard{shard_idx}: rows={len(shard_rows)} load_proxy={shard_loads[shard_idx]} -> {out_path}")

    manifest = {
        "input_path": str(args.input_path),
        "output_dir": str(args.output_dir),
        "num_shards": args.num_shards,
        "max_samples": args.max_samples,
        "balance_by": args.balance_by,
        "rows": len(rows),
        "shards": shard_records,
    }
    args.manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
