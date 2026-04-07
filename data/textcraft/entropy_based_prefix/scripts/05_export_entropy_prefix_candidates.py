#!/usr/bin/env python3
"""Export entropy-based prefix split candidates from the stage1 teacher entropy sidecar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from common import (
    ENTROPY_ROOT,
    choose_fixed_ratio_cut_turn_idx,
    extract_actions_from_messages,
    split_messages_at_cut_turn,
)


DOMAIN_SPECS = {
    "assistant": {"roles": {"assistant"}, "include_warmup": True},
    "interaction_assistant": {"roles": {"assistant"}, "include_warmup": False},
    "user": {"roles": {"user"}, "include_warmup": True},
    "interaction_user": {"roles": {"user"}, "include_warmup": False},
    "all": {"roles": {"user", "assistant"}, "include_warmup": True},
    "interaction_all": {"roles": {"user", "assistant"}, "include_warmup": False},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entropy-parquet",
        type=Path,
        default=ENTROPY_ROOT / "stage1_entropy" / "textcraft_teacher_entropy_step200.parquet",
    )
    parser.add_argument(
        "--teacher-parquet",
        type=Path,
        default=ENTROPY_ROOT.parent / "new_prefix_rl" / "stage0_teacher" / "teacher_normalized.parquet",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=ENTROPY_ROOT / "stage2_splits" / "prefix_candidates_entropy_topk.parquet",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=ENTROPY_ROOT / "manifests" / "stage2_entropy_topk_manifest.json",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=sorted(DOMAIN_SPECS),
        default=["interaction_user", "interaction_assistant"],
    )
    parser.add_argument(
        "--scorers",
        nargs="+",
        choices=("raw_topk", "change_topk"),
        default=["raw_topk", "change_topk"],
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--change-window", type=int, default=11)
    parser.add_argument("--min-domain-gap", type=int, default=0)
    parser.add_argument("--fixed-ratio", type=float, default=0.4)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def strategy_name(scorer: str, domain: str, top_k: int, change_window: int) -> str:
    if scorer == "raw_topk":
        return f"entropy_raw_topk_{domain}_k{top_k}"
    return f"entropy_change_topk_w{change_window}_{domain}_k{top_k}"


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size <= 1:
        return values.astype(float, copy=True)

    radius = window // 2
    prefix = np.concatenate(([0.0], np.cumsum(values, dtype=float)))
    out = np.empty(values.shape[0], dtype=float)
    for idx in range(values.shape[0]):
        left = max(0, idx - radius)
        right = min(values.shape[0], idx + radius + 1)
        out[idx] = (prefix[right] - prefix[left]) / float(right - left)
    return out


def build_domain_tokens(
    message_stats: Sequence[Dict[str, Any]],
    domain: str,
    num_assistant_messages_total: int,
) -> List[Dict[str, Any]]:
    spec = DOMAIN_SPECS[domain]
    sorted_stats = sorted((dict(stat) for stat in message_stats), key=lambda stat: int(stat["message_index"]))

    tokens: List[Dict[str, Any]] = []
    previous_assistant_turn_idx = None
    previous_assistant_message_index = None

    for stat in sorted_stats:
        role = stat["role"]
        message_index = int(stat["message_index"])
        role_turn_idx = int(stat["role_turn_idx"])
        is_warmup = bool(stat.get("is_warmup", False))
        token_positions = [int(pos) for pos in list(stat["token_positions"])]
        entropy_values = [float(v) for v in list(stat["entropy_values"])]

        current_assistant_turn_idx = role_turn_idx - 1 if role == "assistant" else None
        mapped_cut_turn_idx = current_assistant_turn_idx if role == "assistant" else previous_assistant_turn_idx
        mapped_assistant_message_index = (
            message_index if role == "assistant" else previous_assistant_message_index
        )

        include_role = role in spec["roles"]
        include_warmup = spec["include_warmup"] or not is_warmup
        if include_role and include_warmup and mapped_cut_turn_idx is not None:
            if num_assistant_messages_total > 1:
                cut_relative_q = mapped_cut_turn_idx / float(num_assistant_messages_total - 1)
            else:
                cut_relative_q = 0.0

            for token_position, entropy_value in zip(token_positions, entropy_values, strict=True):
                tokens.append(
                    {
                        "source_role": role,
                        "source_message_index": message_index,
                        "source_role_turn_idx": role_turn_idx,
                        "source_message_is_warmup": is_warmup,
                        "source_message_content_preview": stat.get("content_preview", ""),
                        "token_position": token_position,
                        "raw_entropy": entropy_value,
                        "mapped_cut_turn_idx": int(mapped_cut_turn_idx),
                        "mapped_cut_relative_position_q": float(cut_relative_q),
                        "mapped_assistant_message_index": int(mapped_assistant_message_index),
                    }
                )

        if role == "assistant":
            previous_assistant_turn_idx = current_assistant_turn_idx
            previous_assistant_message_index = message_index

    tokens.sort(key=lambda token: int(token["token_position"]))
    for domain_rank, token in enumerate(tokens):
        token["domain_rank"] = domain_rank
        token["num_domain_tokens"] = len(tokens)
    return tokens


def score_domain_tokens(
    domain_tokens: Sequence[Dict[str, Any]],
    scorer: str,
    change_window: int,
) -> List[Dict[str, Any]]:
    if not domain_tokens:
        return []

    entropies = np.asarray([float(token["raw_entropy"]) for token in domain_tokens], dtype=float)
    smoothed = moving_average(entropies, change_window)
    cumulative = np.cumsum(smoothed, dtype=float)
    change_score = np.zeros(entropies.shape[0], dtype=float)
    if entropies.shape[0] > 1:
        change_score[1:] = np.abs(np.diff(smoothed))

    scored_tokens: List[Dict[str, Any]] = []
    for idx, token in enumerate(domain_tokens):
        scored = dict(token)
        scored["smoothed_entropy"] = float(smoothed[idx])
        scored["cumulative_entropy"] = float(cumulative[idx])
        scored["change_score"] = float(change_score[idx])
        scored["selection_score"] = float(entropies[idx] if scorer == "raw_topk" else change_score[idx])
        scored_tokens.append(scored)
    return scored_tokens


def select_candidate_tokens(
    scored_tokens: Sequence[Dict[str, Any]],
    top_k: int,
    min_domain_gap: int,
) -> List[Dict[str, Any]]:
    ranked = sorted(
        (dict(token) for token in scored_tokens),
        key=lambda token: (-float(token["selection_score"]), int(token["token_position"])),
    )

    selected: List[Dict[str, Any]] = []
    used_turns = set()
    used_domain_ranks: List[int] = []
    for token in ranked:
        turn_idx = int(token["mapped_cut_turn_idx"])
        if turn_idx in used_turns:
            continue
        if min_domain_gap > 0 and any(
            abs(int(token["domain_rank"]) - prev) < min_domain_gap for prev in used_domain_ranks
        ):
            continue
        used_turns.add(turn_idx)
        used_domain_ranks.append(int(token["domain_rank"]))
        selected.append(token)
        if len(selected) >= top_k:
            break

    for candidate_rank, token in enumerate(selected, start=1):
        token["candidate_rank"] = candidate_rank
    return selected


def build_candidate_row(
    entropy_row: Dict[str, Any],
    teacher_row: Dict[str, Any],
    scorer: str,
    domain: str,
    top_k: int,
    change_window: int,
    min_domain_gap: int,
    selected_token: Dict[str, Any],
    fixed_ratio_cut_turn_idx: int,
    fixed_ratio_cut_q: float,
) -> Dict[str, Any]:
    cut_turn_idx = int(selected_token["mapped_cut_turn_idx"])
    messages = list(teacher_row["conversations"])
    prefix_messages, continuation_messages = split_messages_at_cut_turn(messages, cut_turn_idx)
    prefix_actions = extract_actions_from_messages(prefix_messages)
    strategy = strategy_name(scorer=scorer, domain=domain, top_k=top_k, change_window=change_window)

    sample_uid = teacher_row["sample_uid"]
    candidate_rank = int(selected_token["candidate_rank"])
    candidate_uid = f"{sample_uid}__{strategy}__r{candidate_rank}"

    return {
        "candidate_uid": candidate_uid,
        "sample_uid": sample_uid,
        "item_id": teacher_row["item_id"],
        "sample_idx": int(teacher_row["sample_idx"]),
        "task_id": int(teacher_row["task_id"]),
        "goal": teacher_row.get("goal"),
        "success": int(teacher_row.get("success", 0)),
        "reward": teacher_row.get("reward", 0),
        "strategy": strategy,
        "strategy_family": "entropy_topk",
        "scorer": scorer,
        "domain": domain,
        "candidate_rank": candidate_rank,
        "top_k": top_k,
        "change_window": change_window,
        "min_domain_gap": min_domain_gap,
        "mapping_mode": "current_or_previous_assistant",
        "cut_turn_idx": cut_turn_idx,
        "cut_assistant_turn_idx_one_based": cut_turn_idx + 1,
        "cut_relative_position_q": float(selected_token["mapped_cut_relative_position_q"]),
        "num_assistant_messages_total": int(entropy_row["num_assistant_messages"]),
        "fixed_ratio_0p4_cut_turn_idx": int(fixed_ratio_cut_turn_idx),
        "fixed_ratio_0p4_cut_relative_position_q": float(fixed_ratio_cut_q),
        "cut_turn_idx_delta_vs_fixed_ratio_0p4": int(cut_turn_idx - fixed_ratio_cut_turn_idx),
        "source_role": selected_token["source_role"],
        "source_message_index": int(selected_token["source_message_index"]),
        "source_role_turn_idx": int(selected_token["source_role_turn_idx"]),
        "source_message_is_warmup": bool(selected_token["source_message_is_warmup"]),
        "source_message_content_preview": selected_token["source_message_content_preview"],
        "source_token_position": int(selected_token["token_position"]),
        "source_domain_rank": int(selected_token["domain_rank"]),
        "num_domain_tokens": int(selected_token["num_domain_tokens"]),
        "mapped_assistant_message_index": int(selected_token["mapped_assistant_message_index"]),
        "source_token_entropy": float(selected_token["raw_entropy"]),
        "smoothed_entropy": float(selected_token["smoothed_entropy"]),
        "cumulative_entropy": float(selected_token["cumulative_entropy"]),
        "change_score": float(selected_token["change_score"]),
        "selection_score": float(selected_token["selection_score"]),
        "num_prefix_messages": len(prefix_messages),
        "num_continuation_messages": len(continuation_messages),
        "num_prefix_assistant_messages": sum(msg.get("role") == "assistant" for msg in prefix_messages),
        "num_continuation_assistant_messages": sum(
            msg.get("role") == "assistant" for msg in continuation_messages
        ),
        "prefix_messages": prefix_messages,
        "continuation_messages": continuation_messages,
        "prefix_actions": prefix_actions,
    }


def summarize_strategies(out_df: pd.DataFrame) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    if out_df.empty:
        return summary

    for strategy, subset in out_df.groupby("strategy", sort=True):
        role_counts = {
            str(role): int(count)
            for role, count in subset["source_role"].value_counts(dropna=False).sort_index().items()
        }
        summary[str(strategy)] = {
            "rows": int(len(subset)),
            "unique_sample_uid": int(subset["sample_uid"].nunique()),
            "mean_candidate_rank": float(subset["candidate_rank"].mean()),
            "mean_cut_turn_idx": float(subset["cut_turn_idx"].mean()),
            "mean_cut_relative_position_q": float(subset["cut_relative_position_q"].mean()),
            "mean_cut_turn_idx_delta_vs_fixed_ratio_0p4": float(
                subset["cut_turn_idx_delta_vs_fixed_ratio_0p4"].mean()
            ),
            "mean_selection_score": float(subset["selection_score"].mean()),
            "source_role_counts": role_counts,
        }
    return summary


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.change_window <= 0:
        raise ValueError("--change-window must be positive")
    if args.change_window % 2 == 0:
        raise ValueError("--change-window must be odd for symmetric smoothing")
    if args.min_domain_gap < 0:
        raise ValueError("--min-domain-gap must be non-negative")

    entropy_df = pd.read_parquet(args.entropy_parquet)
    teacher_df = pd.read_parquet(args.teacher_parquet)
    if args.max_samples is not None:
        entropy_df = entropy_df.head(args.max_samples)

    if entropy_df.empty:
        raise RuntimeError(f"Entropy parquet is empty: {args.entropy_parquet}")
    if teacher_df.empty:
        raise RuntimeError(f"Teacher parquet is empty: {args.teacher_parquet}")

    teacher_rows = {row["sample_uid"]: row for row in teacher_df.to_dict(orient="records")}

    records: List[Dict[str, Any]] = []
    skipped_samples = 0
    for entropy_row in entropy_df.to_dict(orient="records"):
        sample_uid = entropy_row["sample_uid"]
        teacher_row = teacher_rows.get(sample_uid)
        if teacher_row is None:
            raise KeyError(f"sample_uid={sample_uid} exists in entropy parquet but not in teacher parquet")

        fixed_ratio_cut_turn_idx, fixed_ratio_cut_q = choose_fixed_ratio_cut_turn_idx(
            int(entropy_row["num_assistant_messages"]),
            target_ratio=args.fixed_ratio,
        )

        per_sample_added = 0
        for domain in args.domains:
            domain_tokens = build_domain_tokens(
                message_stats=entropy_row["message_stats"],
                domain=domain,
                num_assistant_messages_total=int(entropy_row["num_assistant_messages"]),
            )
            if not domain_tokens:
                continue

            for scorer in args.scorers:
                scored_tokens = score_domain_tokens(
                    domain_tokens=domain_tokens,
                    scorer=scorer,
                    change_window=args.change_window,
                )
                selected_tokens = select_candidate_tokens(
                    scored_tokens=scored_tokens,
                    top_k=args.top_k,
                    min_domain_gap=args.min_domain_gap,
                )
                for selected_token in selected_tokens:
                    records.append(
                        build_candidate_row(
                            entropy_row=entropy_row,
                            teacher_row=teacher_row,
                            scorer=scorer,
                            domain=domain,
                            top_k=args.top_k,
                            change_window=args.change_window,
                            min_domain_gap=args.min_domain_gap,
                            selected_token=selected_token,
                            fixed_ratio_cut_turn_idx=fixed_ratio_cut_turn_idx,
                            fixed_ratio_cut_q=fixed_ratio_cut_q,
                        )
                    )
                    per_sample_added += 1

        if per_sample_added == 0:
            skipped_samples += 1

    out_df = pd.DataFrame(records)
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.output_parquet, index=False)

    manifest = {
        "entropy_parquet": str(args.entropy_parquet),
        "teacher_parquet": str(args.teacher_parquet),
        "output_parquet": str(args.output_parquet),
        "rows": int(len(out_df)),
        "unique_candidate_uid": int(out_df["candidate_uid"].nunique()) if not out_df.empty else 0,
        "unique_sample_uid": int(out_df["sample_uid"].nunique()) if not out_df.empty else 0,
        "domains": list(args.domains),
        "scorers": list(args.scorers),
        "top_k": int(args.top_k),
        "change_window": int(args.change_window),
        "min_domain_gap": int(args.min_domain_gap),
        "fixed_ratio": float(args.fixed_ratio),
        "max_samples": args.max_samples,
        "skipped_samples": int(skipped_samples),
        "strategy_summary": summarize_strategies(out_df),
    }
    args.manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
