#!/usr/bin/env python3
"""Audit entropy stage6 datasets and publish strategy-specific stage7 audited copies."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from transformers import AutoTokenizer

from common import DEFAULT_MODEL_PATH, ENTROPY_ROOT, extract_action


REQUIRED_COLUMNS = [
    "candidate_uid",
    "sample_uid",
    "item_id",
    "sample_idx",
    "task_id",
    "goal",
    "strategy",
    "cut_turn_idx",
    "candidate_rank",
    "prompt",
    "prefix_actions",
    "assistant_prefix_old_log_probs",
    "prefix_mask",
    "prefix_token_count",
    "assistant_prefix_span",
    "replay_category",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-paths", nargs="*", type=Path, default=None)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ENTROPY_ROOT / "stage6_training_build",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ENTROPY_ROOT / "stage7_audit_release",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=ENTROPY_ROOT / "manifests" / "stage7_entropy_audit_manifest.json",
    )
    parser.add_argument("--strategies", nargs="*", default=None)
    parser.add_argument("--model-path", type=str, default=None)
    return parser.parse_args()


def discover_input_paths(
    input_paths: Optional[List[Path]],
    input_dir: Path,
    strategies: Optional[List[str]],
) -> List[Path]:
    if input_paths:
        paths = list(input_paths)
    else:
        paths = sorted(input_dir.glob("textcraft_prefix_*_step200.prompt_space_recomputed.full.parquet"))

    if strategies:
        allowed = set(strategies)
        filtered: List[Path] = []
        for path in paths:
            strategy = path.name.removeprefix("textcraft_prefix_").split("_step200.", 1)[0]
            if strategy in allowed:
                filtered.append(path)
        paths = filtered
    return paths


def is_multi_action_like(action: str) -> bool:
    text = " ".join(str(action).split()).strip().lower()
    get_count = text.count("get ")
    craft_count = text.count("craft ")
    return get_count > 1 or craft_count > 1 or (get_count >= 1 and craft_count >= 1)


def normalize_prompt_messages(prompt: Any) -> List[Dict[str, Any]]:
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if isinstance(prompt, list):
        return [msg for msg in prompt if isinstance(msg, dict)]
    return []


def parse_prefix_span(prefix_span: Any) -> Tuple[int, int]:
    if isinstance(prefix_span, dict):
        start = prefix_span.get("start")
        end = prefix_span.get("end")
    elif isinstance(prefix_span, (list, tuple)) and len(prefix_span) == 2:
        start, end = prefix_span
    else:
        raise ValueError(f"Unsupported assistant_prefix_span format: {type(prefix_span)}")
    start = int(start)
    end = int(end)
    if start < 0 or end < start:
        raise ValueError(f"Invalid assistant_prefix_span [{start}, {end})")
    return start, end


def contiguous_blocks(mask: Iterable[Any]) -> List[Tuple[int, int]]:
    mask_values = [int(value) for value in mask]
    blocks: List[Tuple[int, int]] = []
    start = None
    for idx, value in enumerate(mask_values):
        is_active = value > 0
        if is_active and start is None:
            start = idx
        elif not is_active and start is not None:
            blocks.append((start, idx))
            start = None
    if start is not None:
        blocks.append((start, len(mask_values)))
    return blocks


def build_expected_prefix_from_prompt(prompt: Any, tokenizer) -> Dict[str, Any]:
    messages = normalize_prompt_messages(prompt)
    if not messages:
        return {"span": (0, 0), "mask": [], "blocks": [], "assistant_action_count": 0}

    full_text = tokenizer.apply_chat_template(
        messages,
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
    if len(start_matches) != len(messages) or len(end_matches) != len(messages):
        raise ValueError("Conversation tag count does not match prompt message count")

    selected_spans: List[Tuple[int, int]] = []
    for idx, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        if extract_action(msg.get("content", "")) is None:
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
        selected_spans.append((start_token, end_token))

    if not selected_spans:
        return {"span": (0, 0), "mask": [], "blocks": [], "assistant_action_count": 0}

    first_token = selected_spans[0][0]
    last_token = selected_spans[-1][1]
    expected_mask = [0] * (last_token - first_token)
    expected_blocks: List[Tuple[int, int]] = []
    for token_start, token_end in selected_spans:
        rel_start = token_start - first_token
        rel_end = token_end - first_token
        expected_blocks.append((rel_start, rel_end))
        for pos in range(rel_start, rel_end):
            expected_mask[pos] = 1

    return {
        "span": (first_token, last_token),
        "mask": expected_mask,
        "blocks": expected_blocks,
        "assistant_action_count": len(selected_spans),
    }


def resolve_model_path(df: pd.DataFrame, override: Optional[str]) -> str:
    if override:
        return override
    if "source_oldlogprob_model_path" in df.columns:
        values = [str(v).strip() for v in df["source_oldlogprob_model_path"].dropna().unique().tolist() if str(v).strip()]
        if len(values) == 1:
            return values[0]
        if len(values) > 1:
            raise RuntimeError(f"Multiple source_oldlogprob_model_path values found: {values}")
    return DEFAULT_MODEL_PATH


def run_prefix_message_boundary_audit(df: pd.DataFrame, tokenizer) -> Dict[str, Any]:
    span_mismatch = 0
    mask_mismatch = 0
    block_mismatch = 0
    audit_errors = 0
    examples: List[Dict[str, Any]] = []

    for row in df.to_dict(orient="records"):
        candidate_uid = row.get("candidate_uid", "<missing>")
        try:
            expected = build_expected_prefix_from_prompt(row.get("prompt", []), tokenizer)
            actual_span = parse_prefix_span(row.get("assistant_prefix_span"))
            actual_mask = list(row.get("prefix_mask", []))
            actual_blocks = contiguous_blocks(actual_mask)

            row_span_mismatch = actual_span != expected["span"]
            row_mask_mismatch = actual_mask != expected["mask"]
            row_block_mismatch = actual_blocks != expected["blocks"]

            if row_span_mismatch:
                span_mismatch += 1
            if row_mask_mismatch:
                mask_mismatch += 1
            if row_block_mismatch:
                block_mismatch += 1

            if (row_span_mismatch or row_mask_mismatch or row_block_mismatch) and len(examples) < 10:
                examples.append(
                    {
                        "candidate_uid": candidate_uid,
                        "sample_uid": row.get("sample_uid"),
                        "actual_span": list(actual_span),
                        "expected_span": list(expected["span"]),
                        "actual_mask_len": len(actual_mask),
                        "expected_mask_len": len(expected["mask"]),
                        "actual_blocks": [list(block) for block in actual_blocks],
                        "expected_blocks": [list(block) for block in expected["blocks"]],
                    }
                )
        except Exception as exc:
            audit_errors += 1
            if len(examples) < 10:
                examples.append({"candidate_uid": candidate_uid, "sample_uid": row.get("sample_uid"), "error": str(exc)})

    return {
        "span_mismatch": int(span_mismatch),
        "mask_mismatch": int(mask_mismatch),
        "block_mismatch": int(block_mismatch),
        "audit_errors": int(audit_errors),
        "examples": examples,
    }


def build_row_keys(df: pd.DataFrame) -> set[str]:
    if "candidate_uid" in df.columns:
        return set(str(value) for value in df["candidate_uid"].dropna().tolist())
    return set(
        df.apply(
            lambda row: f"{row['item_id']}__{int(row['sample_idx'])}__{row['strategy']}__{int(row['cut_turn_idx'])}",
            axis=1,
        ).tolist()
    )


def load_drop_frames(input_path: Path, strategy: str) -> pd.DataFrame:
    stage3_dir = ENTROPY_ROOT / "stage3_replay_validation"
    frames: List[pd.DataFrame] = []
    for suffix in ("mismatch", "unverifiable", "error"):
        path = stage3_dir / f"{strategy}_{suffix}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_report(df: pd.DataFrame, drop_df: pd.DataFrame, model_path: str) -> Dict[str, Any]:
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    semantic_audit = run_prefix_message_boundary_audit(df, tokenizer)

    duplicate_strategy_cut = int(df.duplicated(subset=["item_id", "sample_idx", "strategy", "cut_turn_idx"]).sum())
    duplicate_candidate_uid = int(df["candidate_uid"].duplicated().sum()) if "candidate_uid" in df.columns else 0
    strategy_values = sorted(str(value) for value in df["strategy"].dropna().unique().tolist()) if "strategy" in df.columns else []

    report: Dict[str, Any] = {
        "rows": int(len(df)),
        "tokenizer_model_path": model_path,
        "missing_columns": missing_columns,
        "strategy_values": strategy_values,
        "unique_candidate_uid": int(df["candidate_uid"].nunique()) if "candidate_uid" in df.columns else None,
        "duplicate_candidate_uid": duplicate_candidate_uid,
        "unique_sample_uid": int(df["sample_uid"].nunique()),
        "duplicate_sample_uid": int(df["sample_uid"].duplicated().sum()),
        "duplicate_item_id_sample_idx_strategy_cut": duplicate_strategy_cut,
        "replay_category_counts": df["replay_category"].value_counts().to_dict(),
        "candidate_rank_counts": df["candidate_rank"].value_counts().sort_index().to_dict(),
        "empty_prefix_old_logprobs": int((df["assistant_prefix_old_log_probs"].apply(len) == 0).sum()),
        "olp_vs_mask_len_mismatch": int(
            (~df.apply(lambda r: len(r["assistant_prefix_old_log_probs"]) == len(r["prefix_mask"]), axis=1)).sum()
        ),
        "prefix_token_count_mismatch": int(
            (~df.apply(lambda r: int(sum(r["prefix_mask"])) == int(r["prefix_token_count"]), axis=1)).sum()
        ),
        "empty_prefix_actions": int((df["prefix_actions"].apply(len) == 0).sum()),
        "multi_action_like_rows": int(
            df["prefix_actions"].apply(lambda xs: any(is_multi_action_like(x) for x in xs)).sum()
        ),
        "task_id_to_multiple_goals": int((df.groupby("task_id")["goal"].nunique() > 1).sum()),
        "goal_to_multiple_task_ids": int((df.groupby("goal")["task_id"].nunique() > 1).sum()),
        "prefix_coordinate_system_values": (
            sorted(str(v) for v in df["prefix_coordinate_system"].dropna().unique().tolist())
            if "prefix_coordinate_system" in df.columns
            else []
        ),
        "drop_set_overlap": 0,
        "prefix_message_boundary_semantics": semantic_audit,
    }

    if not drop_df.empty:
        report["drop_set_overlap"] = int(len(build_row_keys(df) & build_row_keys(drop_df)))

    report["passed"] = all(
        [
            not missing_columns,
            len(strategy_values) == 1,
            duplicate_candidate_uid == 0,
            duplicate_strategy_cut == 0,
            report["empty_prefix_old_logprobs"] == 0,
            report["olp_vs_mask_len_mismatch"] == 0,
            report["prefix_token_count_mismatch"] == 0,
            report["drop_set_overlap"] == 0,
            semantic_audit["span_mismatch"] == 0,
            semantic_audit["mask_mismatch"] == 0,
            semantic_audit["block_mismatch"] == 0,
            semantic_audit["audit_errors"] == 0,
        ]
    )
    return report


def write_markdown(report: Dict[str, Any], md_path: Path, dataset_path: Path) -> None:
    lines = [
        "# Entropy Stage7 Audit Report",
        "",
        f"- Dataset: `{dataset_path}`",
        f"- Rows: `{report['rows']}`",
        f"- Passed: `{report['passed']}`",
        "",
        "## Core Checks",
        "",
        f"- Missing columns: `{report['missing_columns']}`",
        f"- Strategy values: `{report['strategy_values']}`",
        f"- Duplicate candidate_uid: `{report['duplicate_candidate_uid']}`",
        f"- Duplicate (item_id, sample_idx, strategy, cut_turn_idx): `{report['duplicate_item_id_sample_idx_strategy_cut']}`",
        f"- Empty prefix old logprobs: `{report['empty_prefix_old_logprobs']}`",
        f"- old_logprobs vs prefix_mask length mismatch: `{report['olp_vs_mask_len_mismatch']}`",
        f"- prefix_token_count mismatch: `{report['prefix_token_count_mismatch']}`",
        f"- Prefix semantic span mismatch: `{report['prefix_message_boundary_semantics']['span_mismatch']}`",
        f"- Prefix semantic mask mismatch: `{report['prefix_message_boundary_semantics']['mask_mismatch']}`",
        f"- Prefix semantic block mismatch: `{report['prefix_message_boundary_semantics']['block_mismatch']}`",
        f"- Prefix semantic audit errors: `{report['prefix_message_boundary_semantics']['audit_errors']}`",
        f"- Drop-set overlap: `{report['drop_set_overlap']}`",
        "",
        "## Composition",
        "",
        f"- Replay category counts: `{report['replay_category_counts']}`",
        f"- Candidate rank counts: `{report['candidate_rank_counts']}`",
        f"- Unique sample_uid: `{report['unique_sample_uid']}`",
        f"- Duplicate sample_uid: `{report['duplicate_sample_uid']}`",
        f"- Empty prefix_actions: `{report['empty_prefix_actions']}`",
        f"- Multi-action-like rows: `{report['multi_action_like_rows']}`",
        "",
        "## Identity Stability",
        "",
        f"- task_id -> multiple goals: `{report['task_id_to_multiple_goals']}`",
        f"- goal -> multiple task_ids: `{report['goal_to_multiple_task_ids']}`",
        f"- prefix_coordinate_system values: `{report['prefix_coordinate_system_values']}`",
        "",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit_one_file(input_path: Path, output_dir: Path, model_path_override: Optional[str]) -> Dict[str, Any]:
    df = pd.read_parquet(input_path)
    if df.empty:
        raise RuntimeError(f"Input parquet is empty: {input_path}")
    strategy_values = sorted(str(value) for value in df["strategy"].dropna().unique().tolist())
    if len(strategy_values) != 1:
        raise RuntimeError(f"Expected exactly one strategy in {input_path}, got {strategy_values}")
    strategy = strategy_values[0]

    drop_df = load_drop_frames(input_path=input_path, strategy=strategy)
    model_path = resolve_model_path(df, model_path_override)
    report = build_report(df, drop_df, model_path)
    report["dataset_path"] = str(input_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{strategy}_audit_report.json"
    md_path = output_dir / f"{strategy}_audit_report.md"
    audited_path = output_dir / f"textcraft_prefix_{strategy}_step200.audited.parquet"
    report["audited_dataset_path"] = str(audited_path) if report["passed"] else None

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(report, md_path, input_path)
    if report["passed"]:
        shutil.copy2(input_path, audited_path)

    return {
        "strategy": strategy,
        "input_path": str(input_path),
        "report_json": str(json_path),
        "report_md": str(md_path),
        "audited_dataset_path": report["audited_dataset_path"],
        "passed": bool(report["passed"]),
        "rows": int(report["rows"]),
        "duplicate_candidate_uid": int(report["duplicate_candidate_uid"]),
        "duplicate_item_id_sample_idx_strategy_cut": int(report["duplicate_item_id_sample_idx_strategy_cut"]),
        "drop_set_overlap": int(report["drop_set_overlap"]),
        "semantic_span_mismatch": int(report["prefix_message_boundary_semantics"]["span_mismatch"]),
        "semantic_mask_mismatch": int(report["prefix_message_boundary_semantics"]["mask_mismatch"]),
        "semantic_block_mismatch": int(report["prefix_message_boundary_semantics"]["block_mismatch"]),
        "semantic_audit_errors": int(report["prefix_message_boundary_semantics"]["audit_errors"]),
    }


def main() -> None:
    args = parse_args()
    input_paths = discover_input_paths(
        input_paths=args.input_paths,
        input_dir=args.input_dir,
        strategies=args.strategies,
    )
    if not input_paths:
        raise RuntimeError("No entropy stage6 full parquet files found for stage7 audit")

    summaries = [audit_one_file(path, args.output_dir, args.model_path) for path in input_paths]
    manifest = {
        "files": summaries,
        "output_dir": str(args.output_dir),
        "all_passed": all(item["passed"] for item in summaries),
        "total_rows": int(sum(item["rows"] for item in summaries)),
    }
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
