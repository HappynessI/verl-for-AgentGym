#!/usr/bin/env python3
"""Build the final prefix training parquet via exact sample_uid / item_id+sample_idx joins."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from transformers import AutoTokenizer

from common import NEW_PREFIX_ROOT, extract_action, make_sample_uid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix-data-path",
        type=Path,
        default=NEW_PREFIX_ROOT / "stage4_canonicalized" / "fixed_ratio_0p4_validated_canonicalized.parquet",
    )
    parser.add_argument(
        "--sidecar-path",
        type=Path,
        default=NEW_PREFIX_ROOT / "stage5_old_logits" / "teacher_old_logprobs_step200.parquet",
    )
    parser.add_argument(
        "--teacher-parquet-path",
        type=Path,
        default=NEW_PREFIX_ROOT / "stage0_teacher" / "teacher_normalized.parquet",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=NEW_PREFIX_ROOT / "stage6_training_build" / "textcraft_prefix_main_train_step200.parquet",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=NEW_PREFIX_ROOT / "manifests" / "stage6_training_build_manifest.json",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/Data/wyh/datasets/Verl-Data/outputs/textcraft_sft/qwen3-1.7b-sft/global_step_200/huggingface",
    )
    return parser.parse_args()


def compute_full_assistant_token_spans(conversations: List[Dict[str, Any]], tokenizer) -> List[Dict[str, Any]]:
    full_text = tokenizer.apply_chat_template(
        conversations,
        add_generation_prompt=False,
        tokenize=False,
    )
    result = tokenizer(
        full_text,
        add_special_tokens=True,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    offset_mapping = result.offset_mapping[0]

    start_pattern = re.compile(r"<\|im_start\|>(user|assistant|tool|system)")
    end_pattern = re.compile(r"<\|im_end\|>")

    start_matches = list(start_pattern.finditer(full_text))
    end_matches = list(end_pattern.finditer(full_text))
    if len(start_matches) != len(conversations) or len(end_matches) != len(conversations):
        raise ValueError("Conversation tag count does not match message count")

    token_spans: List[Dict[str, Any]] = []
    for idx, msg in enumerate(conversations):
        role = msg.get("role", "")
        if role != "assistant":
            continue

        start_char = start_matches[idx].end()
        end_char = end_matches[idx].end()

        start_token = None
        end_token = None
        for token_idx, (char_start, char_end) in enumerate(offset_mapping):
            if char_start is None:
                continue
            if char_start < end_char and char_end > start_char:
                if start_token is None:
                    start_token = token_idx
                end_token = token_idx + 1

        if start_token is None or end_token is None:
            raise ValueError(f"Could not map assistant message to token span: idx={idx}")

        token_spans.append(
            {
                "token_start": start_token,
                "token_end": end_token,
                "action": extract_action(msg.get("content", "")),
            }
        )

    return token_spans


def tokenize_conversations(tokenizer, conversations: List[Dict[str, Any]]) -> Tuple[Any, int]:
    text = tokenizer.apply_chat_template(
        conversations,
        add_generation_prompt=False,
        tokenize=False,
    )
    tokens = tokenizer(
        text,
        add_special_tokens=True,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids[0]
    return input_ids, len(input_ids)


def match_prompt_actions_to_full_spans(prompt: List[Dict[str, Any]], full_token_spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prompt_actions = []
    for msg in prompt:
        if msg.get("role") != "assistant":
            continue
        action = extract_action(msg.get("content", ""))
        if action:
            prompt_actions.append(action)

    matched: List[Dict[str, Any]] = []
    next_full_idx = 0
    for action in prompt_actions:
        found_idx = None
        for full_idx in range(next_full_idx, len(full_token_spans)):
            if full_token_spans[full_idx]["action"] == action:
                found_idx = full_idx
                break
        if found_idx is None:
            raise ValueError(f"Could not match prompt action to full trajectory action: {action}")
        matched.append(full_token_spans[found_idx])
        next_full_idx = found_idx + 1
    return matched


def extract_prefix_old_logprobs(
    prompt: List[Dict[str, Any]],
    full_conversations: List[Dict[str, Any]],
    sequence_old_logprobs: List[float],
    tokenizer,
) -> Tuple[List[float], List[int], Dict[str, int], int]:
    full_token_spans = compute_full_assistant_token_spans(full_conversations, tokenizer)
    matched_spans = match_prompt_actions_to_full_spans(prompt, full_token_spans)

    if not matched_spans:
        return [], [], {"start": 0, "end": 0}, 0

    first_token = matched_spans[0]["token_start"]
    last_token = matched_spans[-1]["token_end"]

    lp_start = max(0, first_token - 1)
    lp_end = min(len(sequence_old_logprobs), last_token - 1)
    prefix_old_logprobs = list(sequence_old_logprobs[lp_start:lp_end])

    prefix_mask = [0] * (last_token - first_token)
    for span in matched_spans:
        for pos in range(span["token_start"], span["token_end"]):
            prefix_mask[pos - first_token] = 1

    prefix_token_count = int(sum(prefix_mask))
    return prefix_old_logprobs, prefix_mask, {"start": first_token, "end": last_token}, prefix_token_count


def main() -> None:
    args = parse_args()
    prefix_df = pd.read_parquet(args.prefix_data_path)
    sidecar_df = pd.read_parquet(args.sidecar_path)
    teacher_df = pd.read_parquet(args.teacher_parquet_path)

    if "sample_uid" not in sidecar_df.columns:
        sidecar_df = sidecar_df.copy()
        if not {"item_id", "sample_idx"} <= set(sidecar_df.columns):
            raise RuntimeError("Sidecar parquet must contain either sample_uid or item_id+sample_idx")
        sidecar_df["sample_uid"] = sidecar_df.apply(
            lambda row: make_sample_uid(row["item_id"], int(row["sample_idx"])),
            axis=1,
        )

    teacher_lookup = teacher_df.set_index("sample_uid").to_dict(orient="index")
    sidecar_lookup = sidecar_df.set_index("sample_uid").to_dict(orient="index")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    records: List[Dict[str, Any]] = []
    for row in prefix_df.to_dict(orient="records"):
        sample_uid = row["sample_uid"]
        if sample_uid not in teacher_lookup:
            raise RuntimeError(f"Missing teacher row for sample_uid={sample_uid}")
        if sample_uid not in sidecar_lookup:
            raise RuntimeError(f"Missing sidecar row for sample_uid={sample_uid}")

        teacher_row = teacher_lookup[sample_uid]
        sidecar_row = sidecar_lookup[sample_uid]

        prompt = row["prompt"]
        full_conversations = teacher_row["conversations"]
        sequence_old_logprobs = sidecar_row["sequence_old_logprobs"]

        prefix_old_logprobs, prefix_mask, prefix_span, prefix_token_count = extract_prefix_old_logprobs(
            prompt=prompt,
            full_conversations=full_conversations,
            sequence_old_logprobs=sequence_old_logprobs,
            tokenizer=tokenizer,
        )

        if len(prefix_old_logprobs) != len(prefix_mask):
            raise RuntimeError(f"Length mismatch for sample_uid={sample_uid}")

        output = dict(row)
        output.update(
            {
                "assistant_prefix_old_log_probs": prefix_old_logprobs,
                "prefix_mask": prefix_mask,
                "prefix_token_count": prefix_token_count,
                "assistant_prefix_span": prefix_span,
                "source_oldlogprob_model_path": args.model_path,
            }
        )
        records.append(output)

    out_df = pd.DataFrame(records)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.output_path, index=False)

    manifest = {
        "prefix_data_path": str(args.prefix_data_path),
        "sidecar_path": str(args.sidecar_path),
        "teacher_parquet_path": str(args.teacher_parquet_path),
        "output_path": str(args.output_path),
        "rows": len(out_df),
        "unique_sample_uid": int(out_df["sample_uid"].nunique()),
        "empty_prefix_old_logprobs": int((out_df["assistant_prefix_old_log_probs"].apply(len) == 0).sum()),
    }
    args.manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
