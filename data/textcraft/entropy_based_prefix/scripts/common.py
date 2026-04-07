#!/usr/bin/env python3
"""Common helpers for the TextCraft entropy-based prefix pipeline."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch


ENTROPY_ROOT = Path("/Data/wyh/datasets/Verl-Data/train/textcraft/entropy_based_prefix")
DEFAULT_INPUT_JSONL = Path(
    "/Data/wyh/datasets/Verl-Data/train/textcraft/new_prefix_rl/stage0_teacher/teacher_normalized.jsonl"
)
DEFAULT_MODEL_PATH = "/Data/wyh/datasets/Verl-Data/outputs/textcraft_sft/qwen3-1.7b-sft/global_step_200/huggingface"

START_TAG_RE = re.compile(r"<\|im_start\|>(user|assistant|tool|system)")
END_TAG_RE = re.compile(r"<\|im_end\|>")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return list(iter_jsonl(path))


def make_sample_uid(item_id: str, sample_idx: int) -> str:
    return f"{item_id}__{sample_idx}"


def parse_task_id(item_id: str) -> int:
    if not item_id.startswith("textcraft_"):
        raise ValueError(f"Unexpected item_id format: {item_id}")
    return int(item_id.split("_", 1)[1])


def normalize_ws(text: str) -> str:
    return " ".join(text.split()).strip()


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


def extract_actions_from_messages(messages: Sequence[Dict[str, Any]]) -> List[str]:
    actions: List[str] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        action = extract_action(msg.get("content", ""))
        if action:
            actions.append(action)
    return actions


def assistant_message_indices(messages: Sequence[Dict[str, Any]]) -> List[int]:
    return [idx for idx, msg in enumerate(messages) if msg.get("role") == "assistant"]


def split_messages_at_cut_turn(
    messages: Sequence[Dict[str, Any]],
    cut_turn_idx: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    assistant_indices = assistant_message_indices(messages)
    if not assistant_indices:
        return [], list(messages)

    if cut_turn_idx < 0:
        return [], list(messages)

    last_prefix_message_index = assistant_indices[min(cut_turn_idx, len(assistant_indices) - 1)]
    cut_position = last_prefix_message_index + 1
    return list(messages[:cut_position]), list(messages[cut_position:])


def choose_fixed_ratio_cut_turn_idx(num_assistant_messages: int, target_ratio: float) -> Tuple[int, float]:
    if num_assistant_messages <= 0:
        return 0, 0.0
    if num_assistant_messages == 1:
        return 0, 0.0

    for turn_idx in range(num_assistant_messages):
        q = turn_idx / (num_assistant_messages - 1)
        if q >= target_ratio:
            return turn_idx, q
    return num_assistant_messages - 1, 1.0


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


def render_conversations_text(tokenizer, conversations: Sequence[Dict[str, Any]], enable_thinking: bool = False) -> str:
    if enable_thinking:
        text = ""
        for msg in conversations:
            role = msg["role"]
            content = msg.get("content", "")
            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        return text

    return tokenizer.apply_chat_template(
        list(conversations),
        add_generation_prompt=False,
        tokenize=False,
    )


def tokenize_conversations_with_offsets(
    tokenizer,
    conversations: Sequence[Dict[str, Any]],
    enable_thinking: bool = False,
) -> Tuple[str, torch.Tensor, Sequence[Tuple[int | None, int | None]]]:
    text = render_conversations_text(tokenizer, conversations, enable_thinking=enable_thinking)
    result = tokenizer(
        text,
        add_special_tokens=True,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    input_ids = result.input_ids[0].cpu()
    offset_mapping = [tuple(item) for item in result.offset_mapping[0].tolist()]
    return text, input_ids, offset_mapping


def compute_message_spans_from_offsets(
    full_text: str,
    conversations: Sequence[Dict[str, Any]],
    offset_mapping: Sequence[Tuple[int | None, int | None]],
) -> List[Dict[str, Any]]:
    start_matches = list(START_TAG_RE.finditer(full_text))
    end_matches = list(END_TAG_RE.finditer(full_text))
    if len(start_matches) != len(conversations) or len(end_matches) != len(conversations):
        raise ValueError(
            "Conversation tag count does not match message count: "
            f"{len(start_matches)} starts, {len(end_matches)} ends, {len(conversations)} messages"
        )

    role_counters: Dict[str, int] = {}
    message_spans: List[Dict[str, Any]] = []
    for message_index, msg in enumerate(conversations):
        role = msg.get("role", "")
        role_counters[role] = role_counters.get(role, 0) + 1
        role_turn_idx = role_counters[role]

        start_char = start_matches[message_index].end()
        end_char = end_matches[message_index].end()

        token_start = None
        token_end = None
        for token_index, (char_start, char_end) in enumerate(offset_mapping):
            if char_start is None:
                continue
            if char_start < end_char and char_end > start_char:
                if token_start is None:
                    token_start = token_index
                token_end = token_index + 1

        if token_start is None or token_end is None:
            raise ValueError(f"Could not map message {message_index} ({role}) to token span")

        content = msg.get("content", "")
        is_warmup = False
        if role == "user":
            is_warmup = is_warmup_user_message(content)
        elif role == "assistant":
            is_warmup = is_warmup_assistant_message(content)

        message_spans.append(
            {
                "message_index": message_index,
                "role": role,
                "role_turn_idx": role_turn_idx,
                "token_start": token_start,
                "token_end": token_end,
                "entropy_start": max(0, token_start - 1),
                "entropy_end": max(0, token_end - 1),
                "is_warmup": bool(is_warmup),
            }
        )

    return message_spans


def prepare_sample(
    tokenizer,
    sample: Dict[str, Any],
    enable_thinking: bool = False,
) -> Dict[str, Any]:
    conversations = sample["conversations"]
    sample_uid = sample.get("sample_uid") or make_sample_uid(sample["item_id"], int(sample["sample_idx"]))
    full_text, input_ids, offset_mapping = tokenize_conversations_with_offsets(
        tokenizer,
        conversations,
        enable_thinking=enable_thinking,
    )
    message_spans = compute_message_spans_from_offsets(full_text, conversations, offset_mapping)

    return {
        "sample_uid": sample_uid,
        "item_id": sample["item_id"],
        "sample_idx": int(sample["sample_idx"]),
        "task_id": int(sample.get("task_id", parse_task_id(sample["item_id"]))),
        "goal": sample.get("goal"),
        "success": int(sample.get("success", 0)),
        "reward": sample.get("reward", 0),
        "conversations": conversations,
        "input_ids": input_ids,
        "token_length": int(input_ids.shape[0]),
        "message_spans": message_spans,
    }


def compute_token_entropy_batch(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = outputs.logits[:, :-1, :].to(torch.float32)
        probs = logits.softmax(dim=-1)
        entropies = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)
        return entropies


def _message_entropy_values(sequence_entropies: Sequence[float], span: Dict[str, Any]) -> Tuple[List[int], List[float]]:
    start_position = max(1, int(span["token_start"]))
    end_position = int(span["token_end"])
    entropy_values = list(sequence_entropies[start_position - 1 : end_position - 1])
    token_positions = list(range(start_position, end_position))
    if len(token_positions) != len(entropy_values):
        raise ValueError(
            "Token positions and entropy values length mismatch: "
            f"{len(token_positions)} vs {len(entropy_values)} for span={span}"
        )
    return token_positions, entropy_values


def build_message_stats(
    conversations: Sequence[Dict[str, Any]],
    message_spans: Sequence[Dict[str, Any]],
    sequence_entropies: Sequence[float],
) -> List[Dict[str, Any]]:
    stats: List[Dict[str, Any]] = []
    for span in message_spans:
        msg = conversations[int(span["message_index"])]
        token_positions, entropy_values = _message_entropy_values(sequence_entropies, span)
        content = msg.get("content", "")
        stats.append(
            {
                "message_index": int(span["message_index"]),
                "role": span["role"],
                "role_turn_idx": int(span["role_turn_idx"]),
                "token_start": int(span["token_start"]),
                "token_end": int(span["token_end"]),
                "entropy_start": int(span["entropy_start"]),
                "entropy_end": int(span["entropy_end"]),
                "token_count": int(span["token_end"] - span["token_start"]),
                "entropy_count": len(entropy_values),
                "entropy_sum": float(sum(entropy_values)) if entropy_values else 0.0,
                "entropy_mean": float(sum(entropy_values) / len(entropy_values)) if entropy_values else 0.0,
                "entropy_max": float(max(entropy_values)) if entropy_values else 0.0,
                "entropy_min": float(min(entropy_values)) if entropy_values else 0.0,
                "token_positions": token_positions,
                "entropy_values": entropy_values,
                "is_warmup": bool(span.get("is_warmup", False)),
                "content_preview": content[:160],
            }
        )
    return stats


def filter_message_stats(
    message_stats: Sequence[Dict[str, Any]],
    role: str | None = None,
    include_warmup: bool = True,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for stat in message_stats:
        if role is not None and stat["role"] != role:
            continue
        if not include_warmup and stat.get("is_warmup", False):
            continue
        out.append(dict(stat))
    return out


def flatten_role_entropy(message_stats: Sequence[Dict[str, Any]]) -> Tuple[List[int], List[float]]:
    token_positions: List[int] = []
    entropy_values: List[float] = []
    for stat in message_stats:
        token_positions.extend(int(pos) for pos in stat["token_positions"])
        entropy_values.extend(float(v) for v in stat["entropy_values"])
    return token_positions, entropy_values
