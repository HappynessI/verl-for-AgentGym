#!/usr/bin/env python3
"""Plot aggregate and representative teacher entropy traces for TextCraft."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer


DEFAULT_STAGE1_PATH = Path(
    "/Data/wyh/datasets/Verl-Data/train/textcraft/entropy_based_prefix/stage1_entropy/textcraft_teacher_entropy_step200.parquet"
)
DEFAULT_TEACHER_JSONL = Path(
    "/Data/wyh/datasets/Verl-Data/train/textcraft/new_prefix_rl/stage0_teacher/teacher_normalized.jsonl"
)
DEFAULT_TOKENIZER_PATH = (
    "/Data/wyh/datasets/Verl-Data/outputs/textcraft_sft/qwen3-1.7b-sft/global_step_200/huggingface"
)
DEFAULT_OUTPUT_DIR = Path("/Data/wyh/datasets/Verl-Data/train/textcraft/entropy_analysis")
DEFAULT_REP_DIR = DEFAULT_OUTPUT_DIR / "representative_sample_traces"
START_TAG_RE = re.compile(r"<\|im_start\|>(user|assistant|tool|system)")
END_TAG_RE = re.compile(r"<\|im_end\|>")
THINK_BLOCK_RE = re.compile(r"<think>\s*.*?\s*</think>\s*", re.DOTALL)

USER_COLOR = "#d55e00"
ASSISTANT_COLOR = "#0072b2"
SEPARATOR_COLOR = "#c9c9c9"
TEXT_COLOR = "#1f1f1f"


@dataclass
class MessageTrace:
    role: str
    role_turn_idx: int
    message_index: int
    entropy_values: List[float]


@dataclass
class SampleTrace:
    sample_uid: str
    users: List[MessageTrace]
    assistants: List[MessageTrace]
    complete_turns: int
    mean_user_entropy: float
    mean_assistant_entropy: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-path", type=Path, default=DEFAULT_STAGE1_PATH)
    parser.add_argument("--teacher-jsonl", type=Path, default=DEFAULT_TEACHER_JSONL)
    parser.add_argument("--tokenizer-path", type=str, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--representative-dir", type=Path, default=DEFAULT_REP_DIR)
    parser.add_argument("--user-tokens", type=int, default=16)
    parser.add_argument("--assistant-tokens", type=int, default=32)
    parser.add_argument("--min-coverage-ratio", type=float, default=0.2)
    parser.add_argument("--num-representatives", type=int, default=6)
    return parser.parse_args()


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def mean_or_zero(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def load_teacher_rows(path: Path) -> Dict[str, Dict[str, Any]]:
    teacher_rows: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_uid = str(row.get("sample_uid") or f"{row['item_id']}__{int(row['sample_idx'])}")
            teacher_rows[sample_uid] = row
    return teacher_rows


def render_conversations_text(tokenizer, conversations: Sequence[Dict[str, Any]]) -> str:
    return tokenizer.apply_chat_template(
        list(conversations),
        add_generation_prompt=False,
        tokenize=False,
    )


def tokenize_conversations_with_offsets(
    tokenizer,
    conversations: Sequence[Dict[str, Any]],
) -> Tuple[str, Sequence[Tuple[int | None, int | None]]]:
    text = render_conversations_text(tokenizer, conversations)
    result = tokenizer(
        text,
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    offset_mapping = [tuple(item) for item in result["offset_mapping"]]
    return text, offset_mapping


def compute_message_token_spans(
    full_text: str,
    conversations: Sequence[Dict[str, Any]],
    offset_mapping: Sequence[Tuple[int | None, int | None]],
) -> List[Tuple[int, int]]:
    start_matches = list(START_TAG_RE.finditer(full_text))
    end_matches = list(END_TAG_RE.finditer(full_text))
    if len(start_matches) != len(conversations) or len(end_matches) != len(conversations):
        raise ValueError(
            "Conversation tag count mismatch: "
            f"{len(start_matches)} starts, {len(end_matches)} ends, {len(conversations)} messages"
        )

    spans: List[Tuple[int, int]] = []
    for message_index, _ in enumerate(conversations):
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
            raise ValueError(f"Could not map message {message_index} to token span")
        spans.append((int(token_start), int(token_end)))
    return spans


def find_message_content_start(full_text: str, content: str, message_index: int) -> int:
    if not content:
        return -1
    start_matches = list(START_TAG_RE.finditer(full_text))
    end_matches = list(END_TAG_RE.finditer(full_text))
    search_start = start_matches[message_index].end()
    search_end = end_matches[message_index].start()
    return full_text.find(content, search_start, search_end)


def build_no_think_index_map(
    teacher_rows: Dict[str, Dict[str, Any]],
    tokenizer,
) -> Dict[str, Dict[int, List[int]]]:
    index_map: Dict[str, Dict[int, List[int]]] = {}
    for sample_uid, row in teacher_rows.items():
        conversations = list(row["conversations"])
        full_text, offset_mapping = tokenize_conversations_with_offsets(tokenizer, conversations)
        message_spans = compute_message_token_spans(
            full_text=full_text,
            conversations=conversations,
            offset_mapping=offset_mapping,
        )

        per_message: Dict[int, List[int]] = {}
        for message_index, msg in enumerate(conversations):
            if msg.get("role") != "assistant":
                continue
            content = str(msg.get("content", ""))
            think_match = THINK_BLOCK_RE.search(content)
            if not think_match:
                continue

            content_start = find_message_content_start(
                full_text=full_text,
                content=content,
                message_index=message_index,
            )
            if content_start < 0:
                continue

            think_start_char = content_start + think_match.start()
            think_end_char = content_start + think_match.end()
            token_start, token_end = message_spans[message_index]
            keep_local_indices: List[int] = []
            for token_index in range(token_start, token_end):
                char_start, char_end = offset_mapping[token_index]
                if char_start is None:
                    keep_local_indices.append(token_index - token_start)
                    continue
                overlaps_think = char_start < think_end_char and char_end > think_start_char
                if not overlaps_think:
                    keep_local_indices.append(token_index - token_start)
            per_message[message_index] = keep_local_indices
        index_map[sample_uid] = per_message
    return index_map


def build_sample_traces(
    rows: Sequence[Dict[str, Any]],
    no_think_index_map: Optional[Dict[str, Dict[int, List[int]]]] = None,
) -> List[SampleTrace]:
    traces: List[SampleTrace] = []
    for row in rows:
        stats = list(row["message_stats"])
        interaction = [stat for stat in stats if not stat["is_warmup"] and stat["role"] in ("user", "assistant")]
        sample_uid = str(row["sample_uid"])
        sample_no_think = no_think_index_map.get(sample_uid, {}) if no_think_index_map else {}
        users = [
            MessageTrace(
                role="user",
                role_turn_idx=int(stat["role_turn_idx"]),
                message_index=int(stat["message_index"]),
                entropy_values=[float(v) for v in stat["entropy_values"]],
            )
            for stat in interaction
            if stat["role"] == "user"
        ]
        assistants = [
            MessageTrace(
                role="assistant",
                role_turn_idx=int(stat["role_turn_idx"]),
                message_index=int(stat["message_index"]),
                entropy_values=(
                    [float(v) for v in stat["entropy_values"]]
                    if int(stat["message_index"]) not in sample_no_think
                    else [
                        float(stat["entropy_values"][idx])
                        for idx in sample_no_think[int(stat["message_index"])]
                        if idx < len(stat["entropy_values"])
                    ]
                ),
            )
            for stat in interaction
            if stat["role"] == "assistant"
        ]
        complete_turns = min(len(users), len(assistants))
        traces.append(
            SampleTrace(
                sample_uid=str(row["sample_uid"]),
                users=users,
                assistants=assistants,
                complete_turns=complete_turns,
                mean_user_entropy=mean_or_zero([v for msg in users for v in msg.entropy_values]),
                mean_assistant_entropy=mean_or_zero([v for msg in assistants for v in msg.entropy_values]),
            )
        )
    return traces


def compute_max_turn_to_plot(sample_traces: Sequence[SampleTrace], min_coverage_ratio: float) -> Tuple[int, List[Dict[str, Any]]]:
    total = len(sample_traces)
    max_turn = 0
    coverage_rows: List[Dict[str, Any]] = []
    for turn_idx in range(1, max(trace.complete_turns for trace in sample_traces) + 1):
        count = sum(1 for trace in sample_traces if trace.complete_turns >= turn_idx)
        ratio = count / float(total)
        coverage_rows.append(
            {"turn_idx": turn_idx, "sample_count": count, "coverage_ratio": ratio}
        )
        if ratio >= min_coverage_ratio:
            max_turn = turn_idx
    return max_turn, coverage_rows


def allocate_block_arrays(max_turn: int, user_tokens: int, assistant_tokens: int) -> Dict[Tuple[str, int], List[List[float]]]:
    blocks: Dict[Tuple[str, int], List[List[float]]] = {}
    for turn_idx in range(1, max_turn + 1):
        blocks[("user", turn_idx)] = [[] for _ in range(user_tokens)]
        blocks[("assistant", turn_idx)] = [[] for _ in range(assistant_tokens)]
    return blocks


def collect_aggregate_stats(
    sample_traces: Sequence[SampleTrace],
    max_turn: int,
    user_tokens: int,
    assistant_tokens: int,
) -> Dict[Tuple[str, int], Dict[str, List[float]]]:
    blocks = allocate_block_arrays(max_turn=max_turn, user_tokens=user_tokens, assistant_tokens=assistant_tokens)
    for trace in sample_traces:
        for turn_idx in range(1, min(trace.complete_turns, max_turn) + 1):
            for role, messages, limit in (
                ("user", trace.users, user_tokens),
                ("assistant", trace.assistants, assistant_tokens),
            ):
                msg = messages[turn_idx - 1]
                for pos_idx, value in enumerate(msg.entropy_values[:limit]):
                    blocks[(role, turn_idx)][pos_idx].append(float(value))

    stats: Dict[Tuple[str, int], Dict[str, List[float]]] = {}
    for key, position_lists in blocks.items():
        means: List[float] = []
        lows: List[float] = []
        highs: List[float] = []
        counts: List[float] = []
        for values in position_lists:
            if values:
                arr = np.asarray(values, dtype=float)
                means.append(float(arr.mean()))
                lows.append(float(np.percentile(arr, 25)))
                highs.append(float(np.percentile(arr, 75)))
                counts.append(float(arr.shape[0]))
            else:
                means.append(np.nan)
                lows.append(np.nan)
                highs.append(np.nan)
                counts.append(0.0)
        stats[key] = {"mean": means, "low": lows, "high": highs, "count": counts}
    return stats


def block_sequence(max_turn: int, user_tokens: int, assistant_tokens: int) -> List[Tuple[str, int, int]]:
    sequence: List[Tuple[str, int, int]] = []
    for turn_idx in range(1, max_turn + 1):
        sequence.append(("user", turn_idx, user_tokens))
        sequence.append(("assistant", turn_idx, assistant_tokens))
    return sequence


def plot_aggregate(
    stats: Dict[Tuple[str, int], Dict[str, List[float]]],
    coverage_rows: Sequence[Dict[str, Any]],
    output_path: Path,
    user_tokens: int,
    assistant_tokens: int,
    max_turn: int,
    mode_name: str,
) -> None:
    fig, (ax_main, ax_count) = plt.subplots(
        2,
        1,
        figsize=(18, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [4.2, 1.2], "hspace": 0.08},
    )

    xticks: List[float] = []
    xlabels: List[str] = []
    cursor = 0
    for role, turn_idx, block_len in block_sequence(max_turn=max_turn, user_tokens=user_tokens, assistant_tokens=assistant_tokens):
        x = np.arange(cursor, cursor + block_len)
        block = stats[(role, turn_idx)]
        mean = np.asarray(block["mean"], dtype=float)
        low = np.asarray(block["low"], dtype=float)
        high = np.asarray(block["high"], dtype=float)
        count = np.asarray(block["count"], dtype=float)
        mask = ~np.isnan(mean)
        color = USER_COLOR if role == "user" else ASSISTANT_COLOR

        ax_main.plot(x[mask], mean[mask], color=color, linewidth=2.2)
        ax_main.fill_between(x[mask], low[mask], high[mask], color=color, alpha=0.18)
        ax_count.plot(x[mask], count[mask], color=color, linewidth=1.8)

        ax_main.axvline(cursor - 0.5, color=SEPARATOR_COLOR, linewidth=0.8, alpha=0.9)
        ax_count.axvline(cursor - 0.5, color=SEPARATOR_COLOR, linewidth=0.8, alpha=0.9)
        xticks.append(cursor + block_len / 2.0 - 0.5)
        xlabels.append(f"{'U' if role == 'user' else 'A'}{turn_idx}")
        cursor += block_len
    ax_main.axvline(cursor - 0.5, color=SEPARATOR_COLOR, linewidth=0.8, alpha=0.9)
    ax_count.axvline(cursor - 0.5, color=SEPARATOR_COLOR, linewidth=0.8, alpha=0.9)

    total_samples = int(coverage_rows[0]["sample_count"]) if coverage_rows else 0
    ax_main.set_title(
        f"Teacher Entropy Trajectory ({mode_name}): interaction-only user/assistant aggregate\n"
        f"all {total_samples} teacher samples, warmup removed, showing U1-A1-U2-A2..., user first {user_tokens} tokens, assistant first {assistant_tokens} tokens",
        fontsize=13,
        color=TEXT_COLOR,
    )
    ax_main.set_ylabel("Entropy")
    ax_main.grid(axis="y", alpha=0.22)
    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)

    ax_count.set_ylabel("Count")
    ax_count.set_xlabel("Interaction message blocks")
    ax_count.grid(axis="y", alpha=0.18)
    ax_count.spines["top"].set_visible(False)
    ax_count.spines["right"].set_visible(False)
    ax_count.set_xticks(xticks)
    ax_count.set_xticklabels(xlabels, rotation=0)

    legend_lines = [
        plt.Line2D([0], [0], color=USER_COLOR, lw=2.2, label="user"),
        plt.Line2D([0], [0], color=ASSISTANT_COLOR, lw=2.2, label="assistant"),
    ]
    ax_main.legend(handles=legend_lines, loc="upper right", frameon=False)

    coverage_note = ", ".join(
        f"T{row['turn_idx']}={int(row['sample_count'])}"
        for row in coverage_rows[:max_turn]
    )
    ax_count.text(
        0.0,
        -0.62,
        f"turn coverage counts: {coverage_note}",
        transform=ax_count.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#555555",
    )

    fig.subplots_adjust(left=0.06, right=0.995, top=0.9, bottom=0.17, hspace=0.08)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def choose_representative_samples(
    sample_traces: Sequence[SampleTrace],
    num_representatives: int,
) -> List[SampleTrace]:
    traces = [trace for trace in sample_traces if trace.complete_turns > 0]
    turn_counts = np.asarray(sorted(trace.complete_turns for trace in traces), dtype=float)
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.9, 1.0]
    target_counts = [
        int(np.quantile(turn_counts, q=q, method="nearest"))  # type: ignore[arg-type]
        for q in quantiles[:num_representatives]
    ]

    selected: List[SampleTrace] = []
    used_ids = set()
    for target_count in target_counts:
        same_turn = [trace for trace in traces if trace.complete_turns == target_count and trace.sample_uid not in used_ids]
        if not same_turn:
            continue
        group_median = float(np.median([trace.mean_assistant_entropy for trace in same_turn]))
        same_turn.sort(
            key=lambda trace: (
                abs(trace.mean_assistant_entropy - group_median),
                trace.sample_uid,
            )
        )
        chosen = same_turn[0]
        selected.append(chosen)
        used_ids.add(chosen.sample_uid)

    if len(selected) < num_representatives:
        remaining = [trace for trace in traces if trace.sample_uid not in used_ids]
        remaining.sort(key=lambda trace: (trace.complete_turns, trace.sample_uid))
        stride = max(1, math.floor(len(remaining) / max(1, num_representatives - len(selected))))
        for idx in range(0, len(remaining), stride):
            if len(selected) >= num_representatives:
                break
            trace = remaining[idx]
            selected.append(trace)
            used_ids.add(trace.sample_uid)

    selected.sort(key=lambda trace: (trace.complete_turns, trace.sample_uid))
    return selected[:num_representatives]


def plot_representative_trace(
    sample_trace: SampleTrace,
    output_path: Path,
    user_tokens: int,
    assistant_tokens: int,
    mode_name: str,
) -> None:
    max_turn = sample_trace.complete_turns
    num_blocks = max_turn * 2
    fig_width = min(24, max(12, 0.55 * num_blocks))
    fig, ax = plt.subplots(figsize=(fig_width, 5.2))

    xticks: List[float] = []
    xlabels: List[str] = []
    cursor = 0
    for turn_idx in range(1, max_turn + 1):
        for role, messages, block_len in (
            ("user", sample_trace.users, user_tokens),
            ("assistant", sample_trace.assistants, assistant_tokens),
        ):
            msg = messages[turn_idx - 1]
            values = np.asarray(msg.entropy_values[:block_len], dtype=float)
            x = np.arange(cursor, cursor + values.shape[0])
            color = USER_COLOR if role == "user" else ASSISTANT_COLOR
            ax.plot(x, values, color=color, linewidth=2.0)
            ax.axvline(cursor - 0.5, color=SEPARATOR_COLOR, linewidth=0.8, alpha=0.9)
            xticks.append(cursor + block_len / 2.0 - 0.5)
            xlabels.append(f"{'U' if role == 'user' else 'A'}{turn_idx}")
            cursor += block_len
    ax.axvline(cursor - 0.5, color=SEPARATOR_COLOR, linewidth=0.8, alpha=0.9)

    ax.set_title(
        f"{sample_trace.sample_uid} | mode={mode_name} | complete turns={sample_trace.complete_turns} | "
        f"mean user entropy={sample_trace.mean_user_entropy:.3f} | "
        f"mean assistant entropy={sample_trace.mean_assistant_entropy:.3f}",
        fontsize=12,
        color=TEXT_COLOR,
    )
    ax.set_ylabel("Entropy")
    ax.set_xlabel("Interaction message blocks")
    ax.grid(axis="y", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if len(xlabels) > 20:
        filtered_ticks = [tick for idx, tick in enumerate(xticks) if idx % 2 == 0]
        filtered_labels = [label for idx, label in enumerate(xlabels) if idx % 2 == 0]
        ax.set_xticks(filtered_ticks)
        ax.set_xticklabels(filtered_labels, rotation=0)
    else:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, rotation=0)

    legend_lines = [
        plt.Line2D([0], [0], color=USER_COLOR, lw=2.0, label="user"),
        plt.Line2D([0], [0], color=ASSISTANT_COLOR, lw=2.0, label="assistant"),
    ]
    ax.legend(handles=legend_lines, loc="upper right", frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.representative_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(args.stage1_path, columns=["sample_uid", "message_stats"])
    rows = table.to_pylist()
    teacher_rows = load_teacher_rows(args.teacher_jsonl)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    no_think_index_map = build_no_think_index_map(teacher_rows, tokenizer)

    mode_outputs: Dict[str, Dict[str, Any]] = {}
    for mode_name, no_think in (("full", False), ("no_think", True)):
        sample_traces = build_sample_traces(rows, no_think_index_map=no_think_index_map if no_think else None)
        max_turn, coverage_rows = compute_max_turn_to_plot(
            sample_traces=sample_traces,
            min_coverage_ratio=args.min_coverage_ratio,
        )
        aggregate_stats = collect_aggregate_stats(
            sample_traces=sample_traces,
            max_turn=max_turn,
            user_tokens=args.user_tokens,
            assistant_tokens=args.assistant_tokens,
        )

        if mode_name == "full":
            aggregate_name = (
                f"teacher_entropy_user_assistant_aggregate_u{args.user_tokens}_a{args.assistant_tokens}_"
                f"turn{max_turn}.png"
            )
        else:
            aggregate_name = (
                f"teacher_entropy_user_assistant_aggregate_{mode_name}_u{args.user_tokens}_a{args.assistant_tokens}_"
                f"turn{max_turn}.png"
            )
        aggregate_path = args.output_dir / aggregate_name
        plot_aggregate(
            stats=aggregate_stats,
            coverage_rows=coverage_rows,
            output_path=aggregate_path,
            user_tokens=args.user_tokens,
            assistant_tokens=args.assistant_tokens,
            max_turn=max_turn,
            mode_name=mode_name,
        )

        representatives = choose_representative_samples(
            sample_traces=sample_traces,
            num_representatives=args.num_representatives,
        )
        representative_outputs: List[Dict[str, Any]] = []
        for trace in representatives:
            if mode_name == "full":
                filename = f"sample_{safe_name(trace.sample_uid)}_entropy_trace_u{args.user_tokens}_a{args.assistant_tokens}.png"
            else:
                filename = (
                    f"sample_{safe_name(trace.sample_uid)}_entropy_trace_{mode_name}_"
                    f"u{args.user_tokens}_a{args.assistant_tokens}.png"
                )
            output_path = args.representative_dir / filename
            plot_representative_trace(
                sample_trace=trace,
                output_path=output_path,
                user_tokens=args.user_tokens,
                assistant_tokens=args.assistant_tokens,
                mode_name=mode_name,
            )
            representative_outputs.append(
                {
                    "sample_uid": trace.sample_uid,
                    "complete_turns": trace.complete_turns,
                    "mean_user_entropy": trace.mean_user_entropy,
                    "mean_assistant_entropy": trace.mean_assistant_entropy,
                    "output_path": str(output_path),
                }
            )

        mode_outputs[mode_name] = {
            "num_samples": len(sample_traces),
            "max_turn_plotted": max_turn,
            "coverage_rows": coverage_rows,
            "aggregate_output_path": str(aggregate_path),
            "representative_outputs": representative_outputs,
        }

    manifest = {
        "stage1_path": str(args.stage1_path),
        "teacher_jsonl": str(args.teacher_jsonl),
        "tokenizer_path": args.tokenizer_path,
        "user_tokens": args.user_tokens,
        "assistant_tokens": args.assistant_tokens,
        "min_coverage_ratio": args.min_coverage_ratio,
        "modes": mode_outputs,
    }
    manifest_path = args.output_dir / "entropy_analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for mode_name, output in mode_outputs.items():
        print(f"AGGREGATE_{mode_name.upper()}={output['aggregate_output_path']}")
    print(f"MANIFEST={manifest_path}")
    for mode_name, output in mode_outputs.items():
        for item in output["representative_outputs"]:
            print(f"REP_{mode_name.upper()}={item['output_path']}")


if __name__ == "__main__":
    main()
