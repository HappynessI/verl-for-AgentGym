#!/usr/bin/env python3
"""Merge sharded entropy stage6 training parquet outputs."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from common import ENTROPY_ROOT


SHARD_RE = re.compile(r"^textcraft_prefix_(.+?)_step200\.prompt_space_recomputed\.shard(\d+)\.parquet$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-paths", nargs="*", type=Path, default=None)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ENTROPY_ROOT / "stage6_training_build",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ENTROPY_ROOT / "stage6_training_build",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=ENTROPY_ROOT / "manifests" / "stage6_entropy_training_merge_manifest.json",
    )
    parser.add_argument("--strategies", nargs="*", default=None)
    return parser.parse_args()


def discover_input_paths(input_paths: Optional[List[Path]], input_dir: Path) -> List[Path]:
    if input_paths:
        return list(input_paths)
    return sorted(input_dir.glob("textcraft_prefix_*_step200.prompt_space_recomputed.shard*.parquet"))


def parse_strategy_and_shard(path: Path) -> tuple[str, int]:
    match = SHARD_RE.match(path.name)
    if not match:
        raise RuntimeError(f"Unexpected shard filename: {path.name}")
    return match.group(1), int(match.group(2))


def main() -> None:
    args = parse_args()
    input_paths = discover_input_paths(args.input_paths, args.input_dir)
    if not input_paths:
        raise RuntimeError("No entropy stage6 shard parquet files found to merge")

    grouped: Dict[str, List[Path]] = defaultdict(list)
    for path in input_paths:
        strategy, _ = parse_strategy_and_shard(path)
        if args.strategies and strategy not in set(args.strategies):
            continue
        grouped[strategy].append(path)

    if not grouped:
        raise RuntimeError("No shard files remain after strategy filtering")

    summaries = []
    for strategy, paths in sorted(grouped.items()):
        sorted_paths = sorted(paths, key=lambda p: parse_strategy_and_shard(p)[1])
        frames = [pd.read_parquet(path) for path in sorted_paths]
        out_df = pd.concat(frames, ignore_index=True)
        if "candidate_uid" in out_df.columns:
            out_df = out_df.sort_values(by=["candidate_uid"]).reset_index(drop=True)
        else:
            out_df = out_df.sort_values(by=["item_id", "sample_idx", "cut_turn_idx"]).reset_index(drop=True)

        output_path = args.output_dir / f"textcraft_prefix_{strategy}_step200.prompt_space_recomputed.full.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_parquet(output_path, index=False)

        summaries.append(
            {
                "strategy": strategy,
                "input_paths": [str(path) for path in sorted_paths],
                "output_path": str(output_path),
                "rows": int(len(out_df)),
                "unique_candidate_uid": int(out_df["candidate_uid"].nunique()),
                "unique_sample_uid": int(out_df["sample_uid"].nunique()),
                "duplicate_candidate_uid": int(out_df["candidate_uid"].duplicated().sum()),
                "duplicate_sample_uid": int(out_df["sample_uid"].duplicated().sum()),
                "empty_prefix_old_logprobs": int((out_df["assistant_prefix_old_log_probs"].apply(len) == 0).sum()),
                "olp_vs_mask_len_mismatch": int(
                    (~out_df.apply(lambda r: len(r["assistant_prefix_old_log_probs"]) == len(r["prefix_mask"]), axis=1)).sum()
                ),
                "prefix_token_count_mismatch": int(
                    (~out_df.apply(lambda r: int(sum(r["prefix_mask"])) == int(r["prefix_token_count"]), axis=1)).sum()
                ),
            }
        )

    manifest = {
        "build_mode": "prompt_space_recompute_sharded_merge",
        "output_dir": str(args.output_dir),
        "files": summaries,
        "total_rows": int(sum(item["rows"] for item in summaries)),
    }
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
