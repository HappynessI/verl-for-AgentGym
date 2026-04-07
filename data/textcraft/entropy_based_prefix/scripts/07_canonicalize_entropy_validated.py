#!/usr/bin/env python3
"""Canonicalize validated entropy-based prefix candidates into training prompt format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from common import (
    ENTROPY_ROOT,
    canonicalize_assistant_content,
    is_warmup_assistant_message,
    is_warmup_user_message,
)


DEFAULT_TRAIN_PARQUET_PATH = Path("/Data/wyh/datasets/Verl-Data/train/textcraft/train.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-paths",
        nargs="*",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ENTROPY_ROOT / "stage3_replay_validation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ENTROPY_ROOT / "stage4_canonicalized",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=ENTROPY_ROOT / "manifests" / "stage4_entropy_canonicalized_manifest.json",
    )
    parser.add_argument("--train-parquet-path", type=Path, default=DEFAULT_TRAIN_PARQUET_PATH)
    parser.add_argument("--strategies", nargs="*", default=None)
    parser.add_argument("--max-files", type=int, default=None)
    return parser.parse_args()


def load_reference_system_prompt(train_parquet_path: Path) -> str:
    df = pd.read_parquet(train_parquet_path)
    if df.empty:
        raise RuntimeError(f"Reference train parquet is empty: {train_parquet_path}")

    row = df.iloc[0]
    for column in ("messages", "prompt"):
        if column not in row:
            continue
        messages = row[column]
        if hasattr(messages, "tolist"):
            messages = messages.tolist()
        for msg in messages:
            if msg.get("role") == "system":
                return msg.get("content", "")
    raise RuntimeError(f"Could not find a system prompt in {train_parquet_path}")


def build_training_prompt(
    prefix_messages: List[Dict[str, Any]],
    continuation_messages: List[Dict[str, Any]],
    system_prompt: str,
) -> List[Dict[str, str]]:
    prompt: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    for msg in prefix_messages:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "user":
            if is_warmup_user_message(content):
                continue
            prompt.append({"role": "user", "content": content})
            continue

        if role == "assistant":
            if is_warmup_assistant_message(content):
                continue
            prompt.append({"role": "assistant", "content": canonicalize_assistant_content(content)})

    cut_observation: Optional[str] = None
    for msg in continuation_messages:
        if msg.get("role") == "user":
            cut_observation = msg.get("content", "")
            break
    if cut_observation:
        prompt.append({"role": "user", "content": cut_observation})

    return prompt


def discover_input_paths(
    input_paths: Optional[List[Path]],
    input_dir: Path,
    strategies: Optional[List[str]],
    max_files: Optional[int],
) -> List[Path]:
    if input_paths:
        paths = list(input_paths)
    else:
        paths = sorted(input_dir.glob("*_validated.parquet"))

    if strategies:
        allowed = set(strategies)
        filtered: List[Path] = []
        for path in paths:
            strategy = path.stem.removesuffix("_validated")
            if strategy in allowed:
                filtered.append(path)
        paths = filtered

    if max_files is not None:
        paths = paths[:max_files]
    return paths


def canonicalize_one_file(
    input_path: Path,
    output_dir: Path,
    system_prompt: str,
) -> Dict[str, Any]:
    df = pd.read_parquet(input_path)
    if df.empty:
        raise RuntimeError(f"Input parquet is empty: {input_path}")

    strategies = sorted(str(strategy) for strategy in df["strategy"].unique())
    if len(strategies) != 1:
        raise RuntimeError(f"Expected exactly one strategy in {input_path}, got {strategies}")
    strategy = strategies[0]

    records: List[Dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        prompt = build_training_prompt(
            prefix_messages=list(row["prefix_messages"]),
            continuation_messages=list(row["continuation_messages"]),
            system_prompt=system_prompt,
        )

        output = dict(row)
        output.update(
            {
                "data_source": "textcraft",
                "ability": "crafting",
                "prompt": prompt,
                "reward_model": {"ground_truth": "", "style": "interaction"},
                "extra_info": {
                    "index": int(row["sample_idx"]),
                    "sample_uid": row["sample_uid"],
                    "candidate_uid": row.get("candidate_uid"),
                    "strategy": row.get("strategy"),
                    "interaction_kwargs": {
                        "name": "textcraft",
                        "task_id": int(row["task_id"]),
                        "eval_mode": False,
                        "prefix_actions": list(row.get("prefix_actions", [])),
                        "goal": row.get("goal"),
                    },
                    "reward_model": {"ground_truth": "", "style": "interaction"},
                },
            }
        )
        records.append(output)

    out_df = pd.DataFrame(records)
    output_path = output_dir / f"{strategy}_validated_canonicalized.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)

    return {
        "strategy": strategy,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "rows": int(len(out_df)),
        "unique_candidate_uid": int(out_df["candidate_uid"].nunique()),
        "unique_sample_uid": int(out_df["sample_uid"].nunique()),
    }


def main() -> None:
    args = parse_args()
    input_paths = discover_input_paths(
        input_paths=args.input_paths,
        input_dir=args.input_dir,
        strategies=args.strategies,
        max_files=args.max_files,
    )
    if not input_paths:
        raise RuntimeError("No validated parquet files found for stage4 canonicalization")

    system_prompt = load_reference_system_prompt(args.train_parquet_path)
    summaries = []
    candidate_uid_union = set()
    sample_uid_union = set()
    for path in input_paths:
        summary = canonicalize_one_file(
            input_path=path,
            output_dir=args.output_dir,
            system_prompt=system_prompt,
        )
        summaries.append(summary)

        out_df = pd.read_parquet(summary["output_path"], columns=["candidate_uid", "sample_uid"])
        candidate_uid_union.update(str(value) for value in out_df["candidate_uid"].tolist())
        sample_uid_union.update(str(value) for value in out_df["sample_uid"].tolist())

    manifest = {
        "input_paths": [str(path) for path in input_paths],
        "output_dir": str(args.output_dir),
        "reference_train_parquet": str(args.train_parquet_path),
        "strategies": [summary["strategy"] for summary in summaries],
        "files": summaries,
        "total_rows": int(sum(summary["rows"] for summary in summaries)),
        "total_unique_candidate_uid": int(len(candidate_uid_union)),
        "total_unique_sample_uid": int(len(sample_uid_union)),
    }
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
