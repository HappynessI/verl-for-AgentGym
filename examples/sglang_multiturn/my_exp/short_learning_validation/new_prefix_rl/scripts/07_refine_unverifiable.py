#!/usr/bin/env python3
"""Split stage3 unverifiable rows into usable state-feedback vs drop buckets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from common import NEW_PREFIX_ROOT


STATE_FEEDBACK_PREFIXES = (
    "Could not find",
    "Could not find enough items",
    "Could not find a valid recipe",
)

FORMAT_ERROR_PATTERNS = (
    "Only one 'Action' is allowed",
    "Error:",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validated-path",
        type=Path,
        default=NEW_PREFIX_ROOT / "stage3_replay_validation" / "fixed_ratio_0p4_validated.parquet",
    )
    parser.add_argument(
        "--unverifiable-path",
        type=Path,
        default=NEW_PREFIX_ROOT / "stage3_replay_validation" / "fixed_ratio_0p4_unverifiable.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=NEW_PREFIX_ROOT / "stage3_replay_validation",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=NEW_PREFIX_ROOT / "manifests" / "stage3_unverifiable_refine_manifest.json",
    )
    return parser.parse_args()


def normalize_text(text: Optional[str]) -> str:
    return " ".join((text or "").split()).strip()


def is_state_feedback(text: str) -> bool:
    return text.startswith(STATE_FEEDBACK_PREFIXES)


def is_format_error(text: str) -> bool:
    return any(pattern in text for pattern in FORMAT_ERROR_PATTERNS)


def classify_row(row: pd.Series) -> tuple[str, str]:
    expected_cut = normalize_text(row.get("expected_cut_observation"))
    replay_cut = normalize_text(row.get("replay_cut_observation"))
    expected_next = normalize_text(row.get("expected_next_observation"))
    replay_next = normalize_text(row.get("replay_next_observation"))

    if (
        expected_cut
        and expected_cut == replay_cut
        and is_state_feedback(expected_cut)
        and not is_format_error(expected_cut)
    ):
        return "usable_state_feedback", "cut_state_feedback_exact_match"

    if any(is_format_error(text) for text in (expected_cut, replay_cut, expected_next, replay_next) if text):
        return "drop_nonusable", "format_error"

    return "drop_nonusable", "other_unverifiable"


def main() -> None:
    args = parse_args()
    validated_df = pd.read_parquet(args.validated_path)
    unverifiable_df = pd.read_parquet(args.unverifiable_path)

    if unverifiable_df.empty:
        raise RuntimeError(f"Unverifiable parquet is empty: {args.unverifiable_path}")

    refined = unverifiable_df.copy()
    refined[["unverifiable_bucket", "unverifiable_subreason"]] = refined.apply(
        lambda row: pd.Series(classify_row(row)),
        axis=1,
    )

    usable_df = refined[refined["unverifiable_bucket"] == "usable_state_feedback"].copy()
    drop_df = refined[refined["unverifiable_bucket"] == "drop_nonusable"].copy()

    trainable_df = pd.concat([validated_df, usable_df], ignore_index=True)
    trainable_df = trainable_df.sort_values(["task_id", "sample_idx"]).reset_index(drop=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    usable_path = args.output_dir / "fixed_ratio_0p4_unverifiable_state_feedback_usable.parquet"
    drop_path = args.output_dir / "fixed_ratio_0p4_unverifiable_drop_nonusable.parquet"
    trainable_path = args.output_dir / "fixed_ratio_0p4_stage4_trainable.parquet"

    usable_df.to_parquet(usable_path, index=False)
    drop_df.to_parquet(drop_path, index=False)
    trainable_df.to_parquet(trainable_path, index=False)

    manifest = {
        "validated_path": str(args.validated_path),
        "unverifiable_path": str(args.unverifiable_path),
        "usable_state_feedback_rows": int(len(usable_df)),
        "drop_nonusable_rows": int(len(drop_df)),
        "trainable_rows": int(len(trainable_df)),
        "drop_subreasons": drop_df["unverifiable_subreason"].value_counts().to_dict(),
        "usable_subreasons": usable_df["unverifiable_subreason"].value_counts().to_dict(),
        "usable_output_path": str(usable_path),
        "drop_output_path": str(drop_path),
        "trainable_output_path": str(trainable_path),
    }
    args.manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
