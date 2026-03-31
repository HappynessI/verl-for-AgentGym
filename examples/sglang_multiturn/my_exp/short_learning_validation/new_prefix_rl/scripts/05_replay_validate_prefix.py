#!/usr/bin/env python3
"""Replay-validate prefix candidates while preserving exact sample identity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

from common import NEW_PREFIX_ROOT, extract_action


DEFAULT_SERVER = "http://127.0.0.1:36001"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-path",
        type=Path,
        default=NEW_PREFIX_ROOT / "stage2_splits" / "prefix_candidates_fixed_ratio_0p4.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=NEW_PREFIX_ROOT / "stage3_replay_validation",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=NEW_PREFIX_ROOT / "manifests" / "stage3_replay_validation_manifest.json",
    )
    parser.add_argument("--server", type=str, default=DEFAULT_SERVER)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--request-timeout", type=float, default=30.0)
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
    sample_uid = row["sample_uid"]
    task_id = int(row["task_id"])
    goal = row.get("goal")
    prefix_actions = list(row.get("prefix_actions", []))
    continuation_messages = row["continuation_messages"]

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


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.input_path)
    if args.max_samples is not None:
        df = df.head(args.max_samples)
    if df.empty:
        raise RuntimeError(f"Input parquet is empty: {args.input_path}")

    rows = [classify_row(row, server=args.server, timeout=args.request_timeout) for row in df.to_dict(orient="records")]
    out_df = pd.DataFrame(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    outputs = {
        "validated": args.output_dir / "fixed_ratio_0p4_validated.parquet",
        "mismatch": args.output_dir / "fixed_ratio_0p4_mismatch.parquet",
        "unverifiable": args.output_dir / "fixed_ratio_0p4_unverifiable.parquet",
        "error": args.output_dir / "fixed_ratio_0p4_error.parquet",
    }

    for category, path in outputs.items():
        subset = out_df[out_df["replay_category"] == category]
        subset.to_parquet(path, index=False)

    manifest = {
        "input_path": str(args.input_path),
        "output_dir": str(args.output_dir),
        "server": args.server,
        "rows": len(out_df),
        "validated": int((out_df["replay_category"] == "validated").sum()),
        "mismatch": int((out_df["replay_category"] == "mismatch").sum()),
        "unverifiable": int((out_df["replay_category"] == "unverifiable").sum()),
        "error": int((out_df["replay_category"] == "error").sum()),
        "max_samples": args.max_samples,
    }
    args.manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
