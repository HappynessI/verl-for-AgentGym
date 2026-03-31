#!/usr/bin/env python3
"""Audit the final stage6 training parquet and write a stage7 release report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from common import NEW_PREFIX_ROOT


REQUIRED_COLUMNS = [
    "sample_uid",
    "item_id",
    "sample_idx",
    "task_id",
    "goal",
    "prompt",
    "prefix_actions",
    "assistant_prefix_old_log_probs",
    "prefix_mask",
    "prefix_token_count",
    "assistant_prefix_span",
    "replay_category",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-path",
        type=Path,
        default=NEW_PREFIX_ROOT / "stage6_training_build" / "textcraft_prefix_main_train_step200.parquet",
    )
    parser.add_argument(
        "--drop-path",
        type=Path,
        default=NEW_PREFIX_ROOT / "stage3_replay_validation" / "fixed_ratio_0p4_unverifiable_drop_nonusable.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=NEW_PREFIX_ROOT / "stage7_audit_release",
    )
    return parser.parse_args()


def is_multi_action_like(action: str) -> bool:
    text = " ".join(str(action).split()).strip().lower()
    get_count = text.count("get ")
    craft_count = text.count("craft ")
    return get_count > 1 or craft_count > 1 or (get_count >= 1 and craft_count >= 1)


def build_report(df: pd.DataFrame, drop_df: pd.DataFrame) -> Dict[str, Any]:
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    report: Dict[str, Any] = {
        "rows": len(df),
        "missing_columns": missing_columns,
        "unique_sample_uid": int(df["sample_uid"].nunique()),
        "duplicate_sample_uid": int(df["sample_uid"].duplicated().sum()),
        "duplicate_item_id_sample_idx": int(df.duplicated(subset=["item_id", "sample_idx"]).sum()),
        "replay_category_counts": df["replay_category"].value_counts().to_dict(),
        "empty_prefix_old_logprobs": int((df["assistant_prefix_old_log_probs"].apply(len) == 0).sum()),
        "olp_vs_mask_len_mismatch": int(
            (~df.apply(lambda r: len(r["assistant_prefix_old_log_probs"]) == len(r["prefix_mask"]), axis=1)).sum()
        ),
        "prefix_token_count_mismatch": int(
            (~df.apply(lambda r: int(sum(r["prefix_mask"])) == int(r["prefix_token_count"]), axis=1)).sum()
        ),
        "empty_prefix_actions": int((df["prefix_actions"].apply(len) == 0).sum()),
        "multi_action_like_rows": int(
            df["prefix_actions"].apply(lambda xs: any(is_multi_action_like(x) for x in xs)).sum()
        ),
        "task_id_to_multiple_goals": int(
            (df.groupby("task_id")["goal"].nunique() > 1).sum()
        ),
        "goal_to_multiple_task_ids": int(
            (df.groupby("goal")["task_id"].nunique() > 1).sum()
        ),
        "drop_set_overlap": 0,
    }

    if not drop_df.empty and "sample_uid" in drop_df.columns:
        report["drop_set_overlap"] = int(df["sample_uid"].isin(set(drop_df["sample_uid"])).sum())

    report["passed"] = all(
        [
            not missing_columns,
            report["duplicate_sample_uid"] == 0,
            report["duplicate_item_id_sample_idx"] == 0,
            report["empty_prefix_old_logprobs"] == 0,
            report["olp_vs_mask_len_mismatch"] == 0,
            report["prefix_token_count_mismatch"] == 0,
            report["drop_set_overlap"] == 0,
        ]
    )
    return report


def write_markdown(report: Dict[str, Any], md_path: Path, dataset_path: Path) -> None:
    lines = [
        "# Stage7 Audit Report",
        "",
        f"- Dataset: `{dataset_path}`",
        f"- Rows: `{report['rows']}`",
        f"- Passed: `{report['passed']}`",
        "",
        "## Core Checks",
        "",
        f"- Missing columns: `{report['missing_columns']}`",
        f"- Duplicate sample_uid: `{report['duplicate_sample_uid']}`",
        f"- Duplicate (item_id, sample_idx): `{report['duplicate_item_id_sample_idx']}`",
        f"- Empty prefix old logprobs: `{report['empty_prefix_old_logprobs']}`",
        f"- old_logprobs vs prefix_mask length mismatch: `{report['olp_vs_mask_len_mismatch']}`",
        f"- prefix_token_count mismatch: `{report['prefix_token_count_mismatch']}`",
        f"- Drop-set overlap: `{report['drop_set_overlap']}`",
        "",
        "## Composition",
        "",
        f"- Replay category counts: `{report['replay_category_counts']}`",
        f"- Empty prefix_actions: `{report['empty_prefix_actions']}`",
        f"- Multi-action-like rows: `{report['multi_action_like_rows']}`",
        "",
        "## Identity Stability",
        "",
        f"- task_id -> multiple goals: `{report['task_id_to_multiple_goals']}`",
        f"- goal -> multiple task_ids: `{report['goal_to_multiple_task_ids']}`",
        "",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input_path)
    drop_df = pd.read_parquet(args.drop_path) if args.drop_path.exists() else pd.DataFrame()

    report = build_report(df, drop_df)

    json_path = args.output_dir / "audit_report.json"
    md_path = args.output_dir / "audit_report.md"
    report["dataset_path"] = str(args.input_path)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(report, md_path, args.input_path)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
