#!/usr/bin/env python3
"""Build entropy-based stage6 training parquet in canonicalized prompt-space coordinates."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import DEFAULT_MODEL_PATH, ENTROPY_ROOT, extract_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-paths", nargs="*", type=Path, default=None)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ENTROPY_ROOT / "stage4_canonicalized",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ENTROPY_ROOT / "stage6_training_build",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=ENTROPY_ROOT / "manifests" / "stage6_entropy_training_build_manifest.json",
    )
    parser.add_argument("--strategies", nargs="*", default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--candidate-ranks", nargs="*", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    return parser.parse_args()


def resolve_torch_dtype(name: str):
    if name == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def discover_input_paths(
    input_paths: Optional[List[Path]],
    input_dir: Path,
    strategies: Optional[List[str]],
    max_files: Optional[int],
) -> List[Path]:
    if input_paths:
        paths = list(input_paths)
    else:
        paths = sorted(input_dir.glob("*_validated_canonicalized.parquet"))

    if strategies:
        allowed = set(strategies)
        filtered: List[Path] = []
        for path in paths:
            strategy = path.stem.removesuffix("_validated_canonicalized")
            if strategy in allowed:
                filtered.append(path)
        paths = filtered

    if max_files is not None:
        paths = paths[:max_files]
    return paths


def normalize_prompt(prompt: Any) -> List[Dict[str, Any]]:
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if not isinstance(prompt, list):
        raise ValueError(f"Prompt must be a list of messages, got {type(prompt)}")
    normalized: List[Dict[str, Any]] = []
    for msg in prompt:
        if not isinstance(msg, dict):
            raise ValueError(f"Prompt message must be dict, got {type(msg)}")
        normalized.append(msg)
    return normalized


def build_prompt_text(prompt: List[Dict[str, Any]], tokenizer) -> str:
    return tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=False,
        tokenize=False,
    )


def tokenize_prompt(tokenizer, prompt: Sequence[Dict[str, Any]]) -> Tuple[torch.Tensor, int]:
    text = build_prompt_text(list(prompt), tokenizer)
    tokens = tokenizer(
        text,
        add_special_tokens=True,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids[0]
    return input_ids, len(input_ids)


def compute_prompt_assistant_token_spans(prompt: List[Dict[str, Any]], tokenizer) -> Tuple[str, List[Dict[str, Any]]]:
    full_text = build_prompt_text(prompt, tokenizer)
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
    if len(start_matches) != len(prompt) or len(end_matches) != len(prompt):
        raise ValueError("Conversation tag count does not match prompt message count")

    token_spans: List[Dict[str, Any]] = []
    for idx, msg in enumerate(prompt):
        if msg.get("role") != "assistant":
            continue

        action = extract_action(msg.get("content", ""))
        if action is None:
            continue

        start_char = start_matches[idx].end()
        end_char = end_matches[idx].end()

        start_token = None
        end_token = None
        for token_idx, (char_start, char_end) in enumerate(offset_mapping.tolist()):
            if char_start is None:
                continue
            if char_start < end_char and char_end > start_char:
                if start_token is None:
                    start_token = token_idx
                end_token = token_idx + 1

        if start_token is None or end_token is None:
            raise ValueError(f"Could not map assistant prompt message to token span: idx={idx}")

        token_spans.append(
            {
                "token_start": start_token,
                "token_end": end_token,
                "action": action,
            }
        )

    return full_text, token_spans


def compute_batch_old_logprobs(model, tokenizer, texts: List[str], device: str) -> List[List[float]]:
    if not texts:
        return []

    try:
        batch = tokenizer(
            texts,
            add_special_tokens=True,
            return_tensors="pt",
            padding=True,
        )
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            logits = outputs.logits[:, :-1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            target_tokens = input_ids[:, 1:].unsqueeze(-1)
            scalar_logprobs = log_probs.gather(dim=-1, index=target_tokens).squeeze(-1)

        lengths = attention_mask.sum(dim=1).tolist()
        result: List[List[float]] = []
        for row_idx, seq_len in enumerate(lengths):
            valid_len = int(seq_len) - 1
            result.append(scalar_logprobs[row_idx, :valid_len].detach().cpu().float().tolist())

        del outputs, logits, log_probs, scalar_logprobs, input_ids, attention_mask, batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result
    except torch.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if len(texts) == 1:
            raise
        mid = len(texts) // 2
        return (
            compute_batch_old_logprobs(model, tokenizer, texts[:mid], device)
            + compute_batch_old_logprobs(model, tokenizer, texts[mid:], device)
        )


def build_prefix_window_from_prompt_spans(
    prompt_spans: List[Dict[str, Any]],
    sequence_old_logprobs: List[float],
) -> Tuple[List[float], List[int], Dict[str, int], int]:
    if not prompt_spans:
        return [], [], {"start": 0, "end": 0}, 0

    first_token = prompt_spans[0]["token_start"]
    last_token = prompt_spans[-1]["token_end"]

    lp_start = max(0, first_token - 1)
    lp_end = min(len(sequence_old_logprobs), last_token - 1)
    prefix_old_logprobs = list(sequence_old_logprobs[lp_start:lp_end])

    prefix_mask = [0] * (last_token - first_token)
    for span in prompt_spans:
        rel_start = span["token_start"] - first_token
        rel_end = span["token_end"] - first_token
        for pos in range(rel_start, rel_end):
            prefix_mask[pos] = 1

    prefix_token_count = int(sum(prefix_mask))
    return prefix_old_logprobs, prefix_mask, {"start": first_token, "end": last_token}, prefix_token_count


def validate_rebuilt_sample(
    sample_uid: str,
    candidate_uid: str,
    prompt_token_length: int,
    prompt_spans: List[Dict[str, Any]],
    sequence_old_logprobs: List[float],
    prefix_old_logprobs: List[float],
    prefix_mask: List[int],
    prefix_span: Dict[str, int],
    prefix_token_count: int,
) -> None:
    if not prompt_spans:
        raise RuntimeError(
            f"sample_uid={sample_uid} candidate_uid={candidate_uid}: no assistant action spans found in prompt"
        )

    if len(sequence_old_logprobs) != prompt_token_length - 1:
        raise RuntimeError(
            f"sample_uid={sample_uid} candidate_uid={candidate_uid}: "
            f"len(sequence_old_logprobs)={len(sequence_old_logprobs)} != prompt_token_length-1={prompt_token_length - 1}"
        )

    span_len = int(prefix_span["end"]) - int(prefix_span["start"])
    if span_len <= 0:
        raise RuntimeError(f"sample_uid={sample_uid} candidate_uid={candidate_uid}: invalid prefix span {prefix_span}")

    if len(prefix_old_logprobs) != span_len:
        raise RuntimeError(
            f"sample_uid={sample_uid} candidate_uid={candidate_uid}: "
            f"len(prefix_old_logprobs)={len(prefix_old_logprobs)} != span_len={span_len}"
        )

    if len(prefix_mask) != span_len:
        raise RuntimeError(
            f"sample_uid={sample_uid} candidate_uid={candidate_uid}: len(prefix_mask)={len(prefix_mask)} != span_len={span_len}"
        )

    if prefix_token_count != int(sum(prefix_mask)):
        raise RuntimeError(
            f"sample_uid={sample_uid} candidate_uid={candidate_uid}: "
            f"prefix_token_count={prefix_token_count} != sum(prefix_mask)={int(sum(prefix_mask))}"
        )

    if prefix_token_count <= 0:
        raise RuntimeError(
            f"sample_uid={sample_uid} candidate_uid={candidate_uid}: prefix_token_count must be positive"
        )


def build_output_path(output_dir: Path, strategy: str, shard_index: int, num_shards: int) -> Path:
    stem = f"textcraft_prefix_{strategy}_step200.prompt_space_recomputed"
    if num_shards > 1:
        stem += f".shard{shard_index}"
    return output_dir / f"{stem}.parquet"


def filter_dataframe(
    df: pd.DataFrame,
    candidate_ranks: Optional[List[int]],
    max_samples: Optional[int],
    shard_index: int,
    num_shards: int,
) -> pd.DataFrame:
    out_df = df
    if candidate_ranks:
        allowed = set(int(value) for value in candidate_ranks)
        out_df = out_df[out_df["candidate_rank"].isin(allowed)].copy()
    if max_samples is not None:
        out_df = out_df.head(max_samples).copy()
    if num_shards > 1:
        if shard_index < 0 or shard_index >= num_shards:
            raise RuntimeError(f"Invalid shard_index={shard_index} for num_shards={num_shards}")
        out_df = out_df.iloc[shard_index::num_shards].copy()
    return out_df.reset_index(drop=True)


def process_one_file(
    input_path: Path,
    output_dir: Path,
    tokenizer,
    model,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    df = pd.read_parquet(input_path)
    if df.empty:
        raise RuntimeError(f"Input parquet is empty: {input_path}")

    strategies = sorted(str(value) for value in df["strategy"].dropna().unique().tolist())
    if len(strategies) != 1:
        raise RuntimeError(f"Expected exactly one strategy in {input_path}, got {strategies}")
    strategy = strategies[0]

    original_rows = int(len(df))
    df = filter_dataframe(
        df=df,
        candidate_ranks=args.candidate_ranks,
        max_samples=args.max_samples,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    filtered_rows = int(len(df))
    if df.empty:
        raise RuntimeError(f"No rows remain after filtering/sharding for {input_path}")

    prepared_rows: List[Dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        prompt = normalize_prompt(row["prompt"])
        prompt_text, prompt_spans = compute_prompt_assistant_token_spans(prompt, tokenizer)
        _, prompt_token_length = tokenize_prompt(tokenizer, prompt)
        prepared_rows.append(
            {
                "row": row,
                "prompt_text": prompt_text,
                "prompt_spans": prompt_spans,
                "prompt_token_length": prompt_token_length,
            }
        )

    batch_size = max(1, int(args.batch_size))
    records: List[Dict[str, Any]] = []
    for start_idx in range(0, len(prepared_rows), batch_size):
        batch_rows = prepared_rows[start_idx : start_idx + batch_size]
        batch_texts = [item["prompt_text"] for item in batch_rows]
        batch_old_logprobs = compute_batch_old_logprobs(model, tokenizer, batch_texts, args.device)

        for item, sequence_old_logprobs in zip(batch_rows, batch_old_logprobs):
            row = item["row"]
            prompt_spans = item["prompt_spans"]
            prefix_old_logprobs, prefix_mask, prefix_span, prefix_token_count = build_prefix_window_from_prompt_spans(
                prompt_spans=prompt_spans,
                sequence_old_logprobs=sequence_old_logprobs,
            )

            validate_rebuilt_sample(
                sample_uid=str(row["sample_uid"]),
                candidate_uid=str(row["candidate_uid"]),
                prompt_token_length=item["prompt_token_length"],
                prompt_spans=prompt_spans,
                sequence_old_logprobs=sequence_old_logprobs,
                prefix_old_logprobs=prefix_old_logprobs,
                prefix_mask=prefix_mask,
                prefix_span=prefix_span,
                prefix_token_count=prefix_token_count,
            )

            output = dict(row)
            output.update(
                {
                    "assistant_prefix_old_log_probs": prefix_old_logprobs,
                    "prefix_mask": prefix_mask,
                    "prefix_token_count": prefix_token_count,
                    "assistant_prefix_span": prefix_span,
                    "source_oldlogprob_model_path": args.model_path,
                    "prefix_coordinate_system": "canonicalized_prompt",
                }
            )
            records.append(output)

    out_df = pd.DataFrame(records)
    output_path = build_output_path(
        output_dir=output_dir,
        strategy=strategy,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)

    return {
        "strategy": strategy,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "original_rows": original_rows,
        "filtered_rows": filtered_rows,
        "rows": int(len(out_df)),
        "unique_candidate_uid": int(out_df["candidate_uid"].nunique()),
        "unique_sample_uid": int(out_df["sample_uid"].nunique()),
        "duplicate_candidate_uid": int(out_df["candidate_uid"].duplicated().sum()),
        "duplicate_sample_uid": int(out_df["sample_uid"].duplicated().sum()),
        "empty_prefix_old_logprobs": int((out_df["assistant_prefix_old_log_probs"].apply(len) == 0).sum()),
        "olp_vs_mask_len_mismatch": int(
            (~out_df.apply(lambda r: len(r["assistant_prefix_old_log_probs"]) == len(r["prefix_mask"]), axis=1)).sum()
        ),
        "prefix_token_count_mismatch": int(
            (~out_df.apply(lambda r: int(sum(r["prefix_mask"])) == int(r["prefix_token_count"]), axis=1)).sum()
        ),
        "candidate_ranks": sorted(int(value) for value in out_df["candidate_rank"].unique().tolist()),
        "shard_index": int(args.shard_index),
        "num_shards": int(args.num_shards),
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
        raise RuntimeError("No stage4 canonicalized entropy files found for stage6 build")

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    torch_dtype = resolve_torch_dtype(args.torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    model.to(args.device)
    model.eval()

    summaries = []
    for input_path in input_paths:
        summaries.append(
            process_one_file(
                input_path=input_path,
                output_dir=args.output_dir,
                tokenizer=tokenizer,
                model=model,
                args=args,
            )
        )

    manifest = {
        "build_mode": "prompt_space_recompute",
        "input_paths": [str(path) for path in input_paths],
        "output_dir": str(args.output_dir),
        "model_path": args.model_path,
        "device": args.device,
        "batch_size": int(max(1, args.batch_size)),
        "candidate_ranks": args.candidate_ranks,
        "max_samples": args.max_samples,
        "shard_index": int(args.shard_index),
        "num_shards": int(args.num_shards),
        "files": summaries,
        "total_rows": int(sum(item["rows"] for item in summaries)),
        "total_unique_candidate_uid": int(sum(item["unique_candidate_uid"] for item in summaries)),
        "total_unique_sample_uid_sum": int(sum(item["unique_sample_uid"] for item in summaries)),
    }
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
