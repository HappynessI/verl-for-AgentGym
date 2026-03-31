#!/usr/bin/env python3
"""Shared helpers for the rebuilt TextCraft prefix data pipeline."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional



def _find_repo_root(start: Path) -> Path:
    for candidate in [start] + list(start.parents):
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError(f"Could not locate repo root from: {start}")


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
NEW_PREFIX_ROOT = REPO_ROOT / "data" / "textcraft" / "new_prefix_rl"
DEFAULT_RAW_TEACHER_PATH = NEW_PREFIX_ROOT / "stage0_teacher" / "textcraft_trajectories.raw.jsonl"
DEFAULT_TRAIN_PARQUET_PATH = REPO_ROOT / "data" / "textcraft" / "train.parquet"

GOAL_RE = re.compile(r"Goal:\s*craft\s+(.+?)\.?\s*$", re.IGNORECASE | re.MULTILINE)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_sample_uid(item_id: str, sample_idx: int) -> str:
    return f"{item_id}__{sample_idx}"


def parse_task_id(item_id: str) -> int:
    if not item_id.startswith("textcraft_"):
        raise ValueError(f"Unexpected item_id format: {item_id}")
    return int(item_id.split("_", 1)[1])


def normalize_ws(text: str) -> str:
    return " ".join(text.split()).strip()


def extract_goal_from_messages(messages: List[Dict[str, Any]]) -> Optional[str]:
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        match = GOAL_RE.search(content)
        if match:
            return match.group(1).strip()
    return None


def extract_action(text: str) -> Optional[str]:
    if not text:
        return None

    box_matches = re.findall(r"\[\[\s*(.*?)\s*\]\]", text, re.DOTALL)
    if box_matches:
        action = normalize_ws(box_matches[-1])
        return action or None

    action_match = re.search(r"Action:\s*(.+?)(?:$|\n\n|\n[A-Z][a-z]+:)", text, re.DOTALL)
    if action_match:
        action = normalize_ws(action_match.group(1))
        return action or None

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("> inventory"):
            return "inventory"
        if stripped.startswith("> get "):
            return normalize_ws(stripped[2:])
        if stripped.startswith("> craft "):
            return normalize_ws(stripped[2:])
        if stripped == "inventory":
            return "inventory"
        if stripped.startswith("get "):
            return normalize_ws(stripped)
        if stripped.startswith("craft "):
            return normalize_ws(stripped)

    bare_match = re.search(r"\b(inventory|get\s+.+?|craft\s+.+?)\s*$", text, re.IGNORECASE | re.MULTILINE)
    if bare_match:
        action = normalize_ws(bare_match.group(1))
        lowered = action.lower()
        if lowered == "inventory" or lowered.startswith("get ") or lowered.startswith("craft "):
            return action

    return None


def extract_actions_from_messages(messages: List[Dict[str, Any]]) -> List[str]:
    actions: List[str] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        action = extract_action(msg.get("content", ""))
        if action:
            actions.append(action)
    return actions


def assistant_message_indices(messages: List[Dict[str, Any]]) -> List[int]:
    return [idx for idx, msg in enumerate(messages) if msg.get("role") == "assistant"]


def count_assistant_messages(messages: List[Dict[str, Any]]) -> int:
    return len(assistant_message_indices(messages))


def is_warmup_user_message(content: str) -> bool:
    return (
        "You are given few useful crafting recipes" in content
        or content.startswith("Every round I will give you")
    )


def is_warmup_assistant_message(content: str) -> bool:
    return "OK. I'll follow your instructions" in content


def extract_think_text(content: str) -> Optional[str]:
    think_match = re.search(r"<think>\s*(.*?)\s*</think>", content, re.DOTALL)
    if think_match:
        value = think_match.group(1).strip()
        return value or None

    thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", content, re.DOTALL)
    if thought_match:
        value = thought_match.group(1).strip()
        return value or None

    return None


def canonicalize_assistant_content(content: str) -> str:
    action = extract_action(content)
    think_text = extract_think_text(content)

    if action and think_text:
        return f"Think: {think_text}\nAction: [[ {action} ]]"
    if action:
        return f"Action: [[ {action} ]]"
    return content
