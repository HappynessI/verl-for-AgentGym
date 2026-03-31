#!/usr/bin/env python3
"""
调试脚本：测试带长上下文的请求
"""
import os
import json
import requests

TRAJ_DIR = "/Data/wyh/datasets/Sampling-Data/babyai_MiniMax-M2.1_20260307_150356"
VLLM_URL = "http://localhost:8000"
MODEL_NAME = "qwen3"


def load_one_trajectory(traj_dir: str):
    traj_path = os.path.join(traj_dir, "babyai_trajectories.jsonl")
    with open(traj_path, "r") as f:
        traj = json.loads(f.readline())
    return traj


def parse_assistant_turns(conversations):
    return [m["content"] for m in conversations if m.get("role") == "assistant"]


def build_prefix_for_turn(conversations, turn_idx, role="assistant"):
    prefix_messages = []
    role_count = 0
    for msg in conversations:
        msg_role = msg.get("role")
        if msg_role == role:
            if role_count == turn_idx:
                break
            role_count += 1
        prefix_messages.append(msg)
    return prefix_messages


def test_request(prefix_messages, max_tokens=8192):
    clean_messages = []
    for msg in prefix_messages:
        clean_msg = {"role": msg.get("role"), "content": msg.get("content", "")}
        clean_messages.append(clean_msg)

    total_chars = sum(len(m.get("content", "")) for m in clean_messages)
    total_tokens = total_chars // 4

    payload = {
        "model": MODEL_NAME,
        "messages": clean_messages,
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_p": 1.0,
        "stream": False,
        "logprobs": True,
        "top_logprobs": 100,
    }

    try:
        resp = requests.post(
            f"{VLLM_URL}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60,
        )
        if resp.status_code != 200:
            print(f"  HTTP {resp.status_code}: {resp.text[:200]}")
            return False
        result = resp.json()
        usage = result.get("usage", {})
        print(f"  OK! prompt_tokens={usage.get('prompt_tokens')}, completion_tokens={usage.get('completion_tokens')}")
        return True
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        return False


traj = load_one_trajectory(TRAJ_DIR)
conversations = traj.get("conversations", [])
item_id = traj.get("item_id", "unknown")
assistant_turns = parse_assistant_turns(conversations)

print(f"Item: {item_id}")
print(f"Assistant turns: {len(assistant_turns)}")

for turn_idx in range(min(3, len(assistant_turns))):
    print(f"\nTurn {turn_idx}:")
    prefix = build_prefix_for_turn(conversations, turn_idx, "assistant")
    test_request(prefix)

print(f"\n=== Done ===")
