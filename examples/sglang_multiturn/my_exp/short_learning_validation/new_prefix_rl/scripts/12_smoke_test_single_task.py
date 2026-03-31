#!/usr/bin/env python3
"""Minimal smoke test: replay prefix, let Qwen3-1.7B finish one task, and check reward."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import NEW_PREFIX_ROOT, extract_action


DEFAULT_DATASET = (
    NEW_PREFIX_ROOT / "stage7_audit_release" / "textcraft_prefix_main_train_step200.audited.parquet"
)
DEFAULT_MODEL = "/Data/public/Qwen3-1.7B"
DEFAULT_SERVER = "http://127.0.0.1:36001"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--textcraft-server", type=str, default=DEFAULT_SERVER)
    parser.add_argument("--sample-uid", type=str, default=None)
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--report-path",
        type=Path,
        default=NEW_PREFIX_ROOT / "stage7_audit_release" / "smoke_test_report.json",
    )
    return parser.parse_args()


def to_list(value: Any) -> List[Any]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value)


def extract_continuation_actions(messages: List[Dict[str, Any]]) -> List[str]:
    actions: List[str] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        action = extract_action(msg.get("content", ""))
        if action:
            actions.append(action)
    return actions


def choose_candidates(df: pd.DataFrame, max_candidates: int) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        continuation_messages = to_list(row["continuation_messages"])
        continuation_actions = extract_continuation_actions(continuation_messages)
        records.append(
            {
                **row,
                "_prefix_actions_len": len(to_list(row["prefix_actions"])),
                "_continuation_actions": continuation_actions,
                "_continuation_actions_len": len(continuation_actions),
            }
        )

    ranked = [
        row
        for row in records
        if row["replay_category"] == "validated" and row["_continuation_actions_len"] == 1
    ]
    ranked.sort(key=lambda row: (row["_prefix_actions_len"], row["task_id"], row["sample_idx"]))
    return ranked[:max_candidates]


def load_model(model_path: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()
    return tokenizer, model


def generate_assistant_text(
    tokenizer,
    model,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    device: str,
) -> str:
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to(device)
    attention_mask = torch.ones_like(input_ids, device=device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][input_ids.shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def create_env(server: str, task_id: int, goal: Optional[str]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"task_id": task_id, "eval_mode": False}
    if goal:
        payload["goal"] = goal
    response = requests.post(f"{server}/create", json=payload, timeout=20)
    response.raise_for_status()
    return response.json()


def step_env(server: str, env_id: str, action: str) -> Dict[str, Any]:
    response = requests.post(
        f"{server}/step",
        json={"id": env_id, "action": action, "return_raw_obs": False},
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def close_env(server: str, env_id: str) -> None:
    try:
        requests.post(f"{server}/close", json={"id": env_id}, timeout=5)
    except Exception:
        pass


def run_one_candidate(
    candidate: Dict[str, Any],
    tokenizer,
    model,
    server: str,
    max_new_tokens: int,
    device: str,
) -> Dict[str, Any]:
    env = create_env(server, task_id=int(candidate["task_id"]), goal=candidate.get("goal"))
    env_id = env["id"]
    trace: List[Dict[str, Any]] = []

    try:
        for action in to_list(candidate["prefix_actions"]):
            result = step_env(server, env_id, action)
            trace.append(
                {
                    "phase": "prefix_replay",
                    "action": action,
                    "observation": result.get("observation", ""),
                    "reward": result.get("reward", 0.0),
                    "done": result.get("done", False),
                }
            )

        prompt_messages = to_list(candidate["prompt"])
        generated_text = generate_assistant_text(
            tokenizer=tokenizer,
            model=model,
            messages=prompt_messages,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        action = extract_action(generated_text)

        outcome: Dict[str, Any] = {
            "sample_uid": candidate["sample_uid"],
            "task_id": int(candidate["task_id"]),
            "goal": candidate.get("goal"),
            "expected_continuation_actions": candidate["_continuation_actions"],
            "generated_text": generated_text,
            "parsed_action": action,
            "success": False,
            "reward": 0.0,
            "done": False,
            "trace": trace,
        }

        if not action:
            outcome["error"] = "Could not parse action from generated text"
            return outcome

        result = step_env(server, env_id, action)
        outcome["step_observation"] = result.get("observation", "")
        outcome["reward"] = float(result.get("reward", 0.0))
        outcome["done"] = bool(result.get("done", False))
        outcome["success"] = outcome["reward"] > 0.0
        return outcome
    finally:
        close_env(server, env_id)


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.dataset_path)
    if args.sample_uid:
        df = df[df["sample_uid"] == args.sample_uid]
        if df.empty:
            raise RuntimeError(f"sample_uid not found: {args.sample_uid}")
        candidates = choose_candidates(df, max_candidates=1)
        if not candidates:
            candidates = [df.iloc[0].to_dict()]
    else:
        candidates = choose_candidates(df, max_candidates=args.max_candidates)
    if not candidates:
        raise RuntimeError("No suitable smoke-test candidates found")

    tokenizer, model = load_model(args.model_path, args.device)

    attempts: List[Dict[str, Any]] = []
    success: Optional[Dict[str, Any]] = None

    for candidate in candidates:
        attempt = run_one_candidate(
            candidate=candidate,
            tokenizer=tokenizer,
            model=model,
            server=args.textcraft_server,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )
        attempts.append(attempt)
        if attempt.get("success"):
            success = attempt
            break

    report = {
        "dataset_path": str(args.dataset_path),
        "model_path": args.model_path,
        "textcraft_server": args.textcraft_server,
        "attempt_count": len(attempts),
        "success": success is not None,
        "successful_sample_uid": success.get("sample_uid") if success else None,
        "attempts": attempts,
    }

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
