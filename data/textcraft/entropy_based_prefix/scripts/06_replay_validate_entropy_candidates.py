#!/usr/bin/env python3
"""Replay-validate entropy-based prefix candidates against the local TextCraft server."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm

from common import ENTROPY_ROOT, extract_action


DEFAULT_SERVER = "http://127.0.0.1:36001"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-path",
        type=Path,
        default=ENTROPY_ROOT / "stage2_splits" / "prefix_candidates_entropy_topk.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ENTROPY_ROOT / "stage3_replay_validation",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=ENTROPY_ROOT / "manifests" / "stage3_entropy_replay_validation_manifest.json",
    )
    parser.add_argument("--server", type=str, default=DEFAULT_SERVER)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--strategies", nargs="*", default=None)
    return parser.parse_args()


def first_user_message(messages: List[Dict[str, Any]]) -> Optional[str]:
    for msg in messages:
        if msg.get("role") == "user":
            return msg.get("content", "")
    return None


def first_assistant_action(messages: List[Dict[str, Any]]) -> Optional[str]:
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        action = extract_action(msg.get("content", ""))
        if action:
            return action
    return None


def second_user_after_first_assistant(messages: List[Dict[str, Any]]) -> Optional[str]:
    saw_assistant = False
    for msg in messages:
        role = msg.get("role")
        if role == "assistant":
            action = extract_action(msg.get("content", ""))
            if action:
                saw_assistant = True
                continue
        if saw_assistant and role == "user":
            return msg.get("content", "")
    return None


def extract_observation_fields(text: str) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    if not text:
        return fields

    for prefix, key in (
        ("Inventory:", "inventory"),
        ("Got ", "got"),
        ("Crafted ", "crafted"),
    ):
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(prefix):
                fields[key] = stripped[len(prefix) :].strip()
                break
    return fields


def compare_observations(lhs: str, rhs: str) -> Dict[str, Any]:
    lhs_fields = extract_observation_fields(lhs)
    rhs_fields = extract_observation_fields(rhs)

    shared = sorted(set(lhs_fields) & set(rhs_fields))
    matched = [key for key in shared if lhs_fields[key] == rhs_fields[key]]
    mismatched = [key for key in shared if lhs_fields[key] != rhs_fields[key]]

    return {
        "shared_fields": shared,
        "matched_fields": matched,
        "mismatched_fields": mismatched,
        "strict_match": bool(shared) and not mismatched,
        "lhs_fields": lhs_fields,
        "rhs_fields": rhs_fields,
    }


def create_env(server: str, task_id: int, goal: Optional[str], timeout: float) -> Tuple[str, str]:
    payload: Dict[str, Any] = {"task_id": task_id, "eval_mode": False}
    if goal:
        payload["goal"] = goal

    response = requests.post(f"{server}/create", json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data["id"], data.get("observation", "")


def step_env(server: str, env_id: str, action: str, timeout: float) -> Dict[str, Any]:
    response = requests.post(
        f"{server}/step",
        json={"id": env_id, "action": action, "return_raw_obs": False},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def close_env(server: str, env_id: str, timeout: float) -> None:
    try:
        requests.post(f"{server}/close", json={"id": env_id}, timeout=timeout)
    except Exception:
        pass


def classify_row(row: Dict[str, Any], server: str, timeout: float) -> Dict[str, Any]:
    task_id = int(row["task_id"])
    goal = row.get("goal")
    prefix_actions = [str(action) for action in list(row.get("prefix_actions", []))]
    continuation_messages = list(row["continuation_messages"])

    expected_cut_obs = first_user_message(continuation_messages)
    next_action = first_assistant_action(continuation_messages)
    expected_next_obs = second_user_after_first_assistant(continuation_messages)

    replay_cut_obs = ""
    replay_next_obs = None
    initial_obs = ""
    env_id = None
    error = None

    try:
        env_id, initial_obs = create_env(server, task_id=task_id, goal=goal, timeout=timeout)
        current_obs = initial_obs
        for action in prefix_actions:
            result = step_env(server, env_id, action, timeout=timeout)
            current_obs = result.get("observation", "")
        replay_cut_obs = current_obs

        if next_action:
            next_result = step_env(server, env_id, next_action, timeout=timeout)
            replay_next_obs = next_result.get("observation", "")
    except Exception as exc:
        error = str(exc)
    finally:
        if env_id is not None:
            close_env(server, env_id, timeout=timeout)

    cut_cmp = compare_observations(replay_cut_obs, expected_cut_obs or "")
    next_cmp = compare_observations(replay_next_obs or "", expected_next_obs or "")

    if error:
        replay_category = "error"
    elif not cut_cmp["shared_fields"]:
        replay_category = "unverifiable"
    elif not cut_cmp["strict_match"]:
        replay_category = "mismatch"
    elif next_action and expected_next_obs and next_cmp["shared_fields"] and not next_cmp["strict_match"]:
        replay_category = "mismatch"
    elif next_action and expected_next_obs and not next_cmp["shared_fields"]:
        replay_category = "unverifiable"
    else:
        replay_category = "validated"

    output = dict(row)
    output.update(
        {
            "replay_category": replay_category,
            "initial_observation": initial_obs,
            "replay_cut_observation": replay_cut_obs,
            "expected_cut_observation": expected_cut_obs,
            "cut_comparison_json": json.dumps(cut_cmp, ensure_ascii=False),
            "continuation_first_action": next_action,
            "replay_next_observation": replay_next_obs,
            "expected_next_observation": expected_next_obs,
            "next_comparison_json": json.dumps(next_cmp, ensure_ascii=False),
            "replay_error": error,
        }
    )
    return output


def strategy_summary(out_df: pd.DataFrame) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    if out_df.empty:
        return summary

    for strategy, subset in out_df.groupby("strategy", sort=True):
        category_counts = {
            str(category): int(count)
            for category, count in subset["replay_category"].value_counts(dropna=False).sort_index().items()
        }
        summary[str(strategy)] = {
            "rows": int(len(subset)),
            "unique_candidate_uid": int(subset["candidate_uid"].nunique()),
            "unique_sample_uid": int(subset["sample_uid"].nunique()),
            "category_counts": category_counts,
            "validated_rate": float((subset["replay_category"] == "validated").mean()),
            "mismatch_rate": float((subset["replay_category"] == "mismatch").mean()),
            "unverifiable_rate": float((subset["replay_category"] == "unverifiable").mean()),
            "error_rate": float((subset["replay_category"] == "error").mean()),
        }
    return summary


def main() -> None:
    args = parse_args()
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive")

    df = pd.read_parquet(args.input_path)
    if args.strategies:
        df = df[df["strategy"].isin(args.strategies)]
    if args.max_samples is not None:
        df = df.head(args.max_samples)
    if df.empty:
        raise RuntimeError(f"Input parquet is empty after filtering: {args.input_path}")

    rows_in = df.to_dict(orient="records")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows_out: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        iterator = executor.map(
            lambda row: classify_row(row, server=args.server, timeout=args.request_timeout),
            rows_in,
        )
        for row in tqdm(iterator, total=len(rows_in), desc="replay-validate"):
            rows_out.append(row)

    out_df = pd.DataFrame(rows_out)

    for strategy, strategy_df in out_df.groupby("strategy", sort=True):
        all_path = args.output_dir / f"{strategy}_all.parquet"
        strategy_df.to_parquet(all_path, index=False)
        for category in ("validated", "mismatch", "unverifiable", "error"):
            subset = strategy_df[strategy_df["replay_category"] == category]
            subset.to_parquet(args.output_dir / f"{strategy}_{category}.parquet", index=False)

    manifest = {
        "input_path": str(args.input_path),
        "output_dir": str(args.output_dir),
        "server": args.server,
        "rows": int(len(out_df)),
        "unique_candidate_uid": int(out_df["candidate_uid"].nunique()),
        "unique_sample_uid": int(out_df["sample_uid"].nunique()),
        "strategies": sorted(str(strategy) for strategy in out_df["strategy"].unique()),
        "concurrency": int(args.concurrency),
        "request_timeout": float(args.request_timeout),
        "max_samples": args.max_samples,
        "strategy_summary": strategy_summary(out_df),
    }
    args.manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
