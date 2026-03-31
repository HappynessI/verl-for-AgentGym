#!/usr/bin/env python3
"""
Summarize official-aligned TextCraft eval results by task depth.

This is a pure post-processing script. It reads eval_results_*.jsonl generated
by eval_textcraft_vllm_server.py and groups tasks using the official sparse
session_id bands:
  depth 1:   0..30
  depth 2: 140..180
  depth 3: 420..444
  depth 4: 533..535
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


OFFICIAL_DEPTH_BANDS = {
    1: range(0, 31),
    2: range(140, 181),
    3: range(420, 445),
    4: range(533, 536),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-file", required=True, help="Path to eval_results_*.jsonl")
    parser.add_argument(
        "--num-samples-per-task",
        type=int,
        default=8,
        help="Expected number of samples per task.",
    )
    return parser.parse_args()


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float | None:
    if num_samples < k:
        return None
    if num_correct == 0:
        return 0.0
    if num_samples - num_correct < k:
        return 1.0

    prob_all_failure = 1.0
    for i in range(k):
        prob_all_failure *= (num_samples - num_correct - i) / (num_samples - i)
    return 1.0 - prob_all_failure


def session_id_to_depth(session_id: int) -> int:
    for depth, band in OFFICIAL_DEPTH_BANDS.items():
        if session_id in band:
            return depth
    raise ValueError(f"session_id={session_id} is not in the official aligned 100-task bands")


def load_results(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_no}: {e}") from e
    return rows


def summarize_depth(rows: list[dict], depth: int, expected_samples_per_task: int) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["session_id"]].append(row)

    pass_k_values = {k: [] for k in (1, 2, 4, 8)}
    task_sample_counts = Counter(len(task_rows) for task_rows in grouped.values())
    missing_tasks = []
    avg_reward_sum = 0.0
    avg_success_sum = 0.0

    for session_id, task_rows in sorted(grouped.items()):
        n = len(task_rows)
        c = sum(1 for row in task_rows if row.get("success", False))
        rewards = [row.get("reward", 0.0) for row in task_rows]
        avg_reward_sum += sum(rewards) / n
        avg_success_sum += c / n

        if n != expected_samples_per_task:
            present = sorted(row.get("sample_idx") for row in task_rows)
            missing = [idx for idx in range(expected_samples_per_task) if idx not in present]
            missing_tasks.append((session_id, n, missing))

        for k in pass_k_values:
            pass_k = estimate_pass_at_k(n, c, k)
            if pass_k is not None:
                pass_k_values[k].append(pass_k)

    num_tasks = len(grouped)
    return {
        "depth": depth,
        "num_tasks": num_tasks,
        "expected_samples": num_tasks * expected_samples_per_task,
        "finished_samples": len(rows),
        "avg_reward": avg_reward_sum / num_tasks if num_tasks else 0.0,
        "avg_success": avg_success_sum / num_tasks if num_tasks else 0.0,
        "task_sample_counts": dict(sorted(task_sample_counts.items())),
        "missing_tasks": missing_tasks,
        "pass_at_k": {
            k: (sum(values) / len(values) if values else None, len(values))
            for k, values in pass_k_values.items()
        },
    }


def main() -> None:
    args = parse_args()
    path = Path(args.results_file)
    rows = load_results(path)

    by_depth = defaultdict(list)
    for row in rows:
        session_id = int(row["session_id"])
        depth = session_id_to_depth(session_id)
        by_depth[depth].append(row)

    print("=" * 72)
    print(f"TextCraft Depth Summary: {path}")
    print("=" * 72)

    total_sessions = sorted({int(row["session_id"]) for row in rows})
    print("Observed task counts by official band:")
    for depth in sorted(OFFICIAL_DEPTH_BANDS):
        band = OFFICIAL_DEPTH_BANDS[depth]
        count = sum(session_id in band for session_id in total_sessions)
        print(f"  depth {depth}: {count}")
    print("-" * 72)

    for depth in sorted(by_depth):
        summary = summarize_depth(by_depth[depth], depth, args.num_samples_per_task)
        print(f"Depth {depth}")
        print(f"  Tasks: {summary['num_tasks']}")
        print(
            f"  Samples: {summary['finished_samples']}/{summary['expected_samples']} "
            f"(hist={summary['task_sample_counts']})"
        )
        print(f"  Average Reward: {summary['avg_reward']:.4f}")
        print(f"  Average Success (Avg@1): {summary['avg_success']:.4f}")
        for k in (1, 2, 4, 8):
            value, covered = summary["pass_at_k"][k]
            if value is None:
                print(f"  Pass@{k}: N/A")
            else:
                print(f"  Pass@{k}: {value:.4f} (tasks: {covered}/{summary['num_tasks']})")
        if summary["missing_tasks"]:
            print("  Incomplete tasks:")
            for session_id, count, missing in summary["missing_tasks"]:
                print(f"    session {session_id}: {count} samples, missing {missing}")
        print("-" * 72)


if __name__ == "__main__":
    main()
