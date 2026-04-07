#!/usr/bin/env python3
"""Compute role-aware teacher-forced entropy sidecars for TextCraft trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (
    DEFAULT_INPUT_JSONL,
    DEFAULT_MODEL_PATH,
    ENTROPY_ROOT,
    build_message_stats,
    compute_token_entropy_batch,
    ensure_parent,
    filter_message_stats,
    flatten_role_entropy,
    prepare_sample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_JSONL)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ENTROPY_ROOT / "stage1_entropy" / "textcraft_teacher_entropy_step200.parquet",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=ENTROPY_ROOT / "manifests" / "stage1_entropy_manifest.json",
    )
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--max-batch-samples", type=int, default=8)
    parser.add_argument("--max-batch-tokens", type=int, default=12288)
    parser.add_argument("--sort-by-length", action="store_true")
    return parser.parse_args()


def build_batches(
    prepared_samples: List[Dict[str, Any]],
    max_batch_samples: int,
    max_batch_tokens: int,
) -> List[List[Dict[str, Any]]]:
    if not prepared_samples:
        return []

    batches: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_max_len = 0

    for sample in prepared_samples:
        token_len = int(sample["token_length"])
        if not current:
            current = [sample]
            current_max_len = token_len
            continue

        projected_max_len = max(current_max_len, token_len)
        projected_tokens = projected_max_len * (len(current) + 1)
        if len(current) >= max_batch_samples or projected_tokens > max_batch_tokens:
            batches.append(current)
            current = [sample]
            current_max_len = token_len
        else:
            current.append(sample)
            current_max_len = projected_max_len

    if current:
        batches.append(current)
    return batches


def run_entropy_forward_batch(
    model,
    batch: List[Dict[str, Any]],
    pad_token_id: int,
    device: torch.device,
) -> List[torch.Tensor]:
    max_len = max(int(sample["token_length"]) for sample in batch)
    batch_size = len(batch)

    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)

    for row_idx, sample in enumerate(batch):
        sample_ids = sample["input_ids"]
        sample_len = int(sample_ids.shape[0])
        input_ids[row_idx, :sample_len] = sample_ids.to(device)
        attention_mask[row_idx, :sample_len] = 1

    batch_entropies = compute_token_entropy_batch(model, input_ids, attention_mask)

    result: List[torch.Tensor] = []
    for row_idx, sample in enumerate(batch):
        sample_len = int(sample["token_length"])
        result.append(batch_entropies[row_idx, : sample_len - 1].detach().cpu())
    return result


def role_entropy_fields(
    message_stats: List[Dict[str, Any]],
    role: str,
    include_warmup: bool,
) -> Dict[str, Any]:
    scoped_stats = filter_message_stats(message_stats, role=role, include_warmup=include_warmup)
    token_positions, entropy_values = flatten_role_entropy(scoped_stats)
    return {
        "message_stats": scoped_stats,
        "token_positions": token_positions,
        "entropy_values": entropy_values,
        "entropy_mean": float(sum(entropy_values) / len(entropy_values)) if entropy_values else 0.0,
        "entropy_max": float(max(entropy_values)) if entropy_values else 0.0,
        "token_count": len(token_positions),
    }


def build_record(sample: Dict[str, Any], sequence_entropies: List[float], model_path: str) -> Dict[str, Any]:
    token_length = int(sample["token_length"])
    entropy_length = len(sequence_entropies)
    if entropy_length != max(0, token_length - 1):
        raise ValueError(
            f"Entropy length mismatch for {sample['sample_uid']}: {entropy_length} vs token_length-1={token_length - 1}"
        )

    message_stats = build_message_stats(
        conversations=sample["conversations"],
        message_spans=sample["message_spans"],
        sequence_entropies=sequence_entropies,
    )
    assistant_all = role_entropy_fields(message_stats, role="assistant", include_warmup=True)
    assistant_interaction = role_entropy_fields(message_stats, role="assistant", include_warmup=False)
    user_all = role_entropy_fields(message_stats, role="user", include_warmup=True)
    user_interaction = role_entropy_fields(message_stats, role="user", include_warmup=False)

    return {
        "sample_uid": sample["sample_uid"],
        "item_id": sample["item_id"],
        "sample_idx": sample["sample_idx"],
        "task_id": sample["task_id"],
        "goal": sample["goal"],
        "success": sample["success"],
        "reward": sample["reward"],
        "num_messages": len(sample["conversations"]),
        "num_assistant_messages": len([stat for stat in message_stats if stat["role"] == "assistant"]),
        "num_user_messages": len([stat for stat in message_stats if stat["role"] == "user"]),
        "token_length": token_length,
        "entropy_length": entropy_length,
        "entropy_coordinate_note": "sequence_entropies[i] aligns to token_position=i+1 in the full trajectory",
        "sequence_entropies": sequence_entropies,
        "message_stats": message_stats,
        "assistant_message_stats": assistant_all["message_stats"],
        "interaction_assistant_message_stats": assistant_interaction["message_stats"],
        "user_message_stats": user_all["message_stats"],
        "interaction_user_message_stats": user_interaction["message_stats"],
        "assistant_entropy_token_positions": assistant_all["token_positions"],
        "assistant_entropy_values": assistant_all["entropy_values"],
        "interaction_assistant_entropy_token_positions": assistant_interaction["token_positions"],
        "interaction_assistant_entropy_values": assistant_interaction["entropy_values"],
        "user_entropy_token_positions": user_all["token_positions"],
        "user_entropy_values": user_all["entropy_values"],
        "interaction_user_entropy_token_positions": user_interaction["token_positions"],
        "interaction_user_entropy_values": user_interaction["entropy_values"],
        "assistant_entropy_mean": assistant_all["entropy_mean"],
        "interaction_assistant_entropy_mean": assistant_interaction["entropy_mean"],
        "user_entropy_mean": user_all["entropy_mean"],
        "interaction_user_entropy_mean": user_interaction["entropy_mean"],
        "assistant_entropy_max": assistant_all["entropy_max"],
        "interaction_assistant_entropy_max": assistant_interaction["entropy_max"],
        "user_entropy_max": user_all["entropy_max"],
        "interaction_user_entropy_max": user_interaction["entropy_max"],
        "assistant_entropy_token_count": assistant_all["token_count"],
        "interaction_assistant_entropy_token_count": assistant_interaction["token_count"],
        "user_entropy_token_count": user_all["token_count"],
        "interaction_user_entropy_token_count": user_interaction["token_count"],
        "entropy_model_path": model_path,
    }


def summarize_batches(batches: Iterable[List[Dict[str, Any]]]) -> Dict[str, float]:
    batches = list(batches)
    if not batches:
        return {"num_batches": 0, "mean_batch_size": 0.0, "mean_padded_tokens": 0.0, "max_padded_tokens": 0.0}
    padded_tokens = [max(sample["token_length"] for sample in batch) * len(batch) for batch in batches]
    return {
        "num_batches": len(batches),
        "mean_batch_size": float(sum(len(batch) for batch in batches) / len(batches)),
        "mean_padded_tokens": float(sum(padded_tokens) / len(padded_tokens)),
        "max_padded_tokens": float(max(padded_tokens)),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        # Initialize the CUDA runtime before tokenizer/message preprocessing.
        # In this environment, delaying the first CUDA call until after
        # prepare-samples can leave torch.cuda in a bad state.
        torch.zeros(1, device=device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        pad_token="<|endoftext|>",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = int(tokenizer.pad_token_id)

    with args.input_path.open("r", encoding="utf-8") as f:
        raw_samples = [json.loads(line) for line in f if line.strip()]
    if args.max_samples is not None:
        raw_samples = raw_samples[: args.max_samples]
    if not raw_samples:
        raise RuntimeError(f"No rows found in {args.input_path}")

    prepared_samples = [
        prepare_sample(tokenizer=tokenizer, sample=sample, enable_thinking=args.enable_thinking)
        for sample in tqdm(raw_samples, desc="prepare-samples")
    ]
    if args.sort_by_length:
        prepared_samples = sorted(prepared_samples, key=lambda sample: sample["token_length"], reverse=True)

    batches = build_batches(
        prepared_samples,
        max_batch_samples=args.max_batch_samples,
        max_batch_tokens=args.max_batch_tokens,
    )
    batch_summary = summarize_batches(batches)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()

    records: List[Dict[str, Any]] = []
    for batch in tqdm(batches, desc="entropy-forward"):
        batch_entropies = run_entropy_forward_batch(
            model=model,
            batch=batch,
            pad_token_id=pad_token_id,
            device=device,
        )
        for sample, entropies in zip(batch, batch_entropies, strict=True):
            records.append(
                build_record(
                    sample=sample,
                    sequence_entropies=entropies.tolist(),
                    model_path=args.model_path,
                )
            )

    out_df = pd.DataFrame(records)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_parent(args.manifest_path)
    out_df.to_parquet(args.output_path, index=False)

    manifest = {
        "input_path": str(args.input_path),
        "output_path": str(args.output_path),
        "model_path": args.model_path,
        "rows": len(out_df),
        "unique_sample_uid": int(out_df["sample_uid"].nunique()),
        "max_samples": args.max_samples,
        "enable_thinking": args.enable_thinking,
        "max_batch_samples": args.max_batch_samples,
        "max_batch_tokens": args.max_batch_tokens,
        "sort_by_length": args.sort_by_length,
        **batch_summary,
        "token_length_mean": float(out_df["token_length"].mean()),
        "token_length_max": int(out_df["token_length"].max()),
        "interaction_user_entropy_mean_mean": float(out_df["interaction_user_entropy_mean"].mean()),
        "interaction_assistant_entropy_mean_mean": float(out_df["interaction_assistant_entropy_mean"].mean()),
    }
    args.manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
