#!/usr/bin/env python3
"""Run change-window ablation for entropy change_topk interaction_assistant k=3."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/Data/wyh/datasets/Verl-Data/train/textcraft/entropy_based_prefix")
ABLATION_ROOT = ROOT / "window_ablation"
SCRIPTS_ROOT = ROOT / "scripts"
EXPORT_SCRIPT = SCRIPTS_ROOT / "05_export_entropy_prefix_candidates.py"
REPLAY_SCRIPT = SCRIPTS_ROOT / "06_replay_validate_entropy_candidates.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--server", type=str, default="http://127.0.0.1:36001")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=[1, 3, 5, 7, 9, 11, 15, 21],
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--fixed-ratio", type=float, default=0.4)
    parser.add_argument("--min-domain-gap", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def ensure_dirs() -> Dict[str, Path]:
    paths = {
        "configs": ABLATION_ROOT / "configs",
        "manifests": ABLATION_ROOT / "manifests",
        "notes": ABLATION_ROOT / "notes",
        "outputs": ABLATION_ROOT / "outputs",
        "stage2": ABLATION_ROOT / "outputs" / "stage2_candidates",
        "stage3": ABLATION_ROOT / "outputs" / "stage3_replay_validation",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def run_cmd(cmd: List[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def normalize_text(text: Any) -> str:
    return " ".join(str(text or "").split()).strip()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rank_counts(df: pd.DataFrame) -> Dict[str, int]:
    if df.empty or "candidate_rank" not in df.columns:
        return {}
    return {str(int(k)): int(v) for k, v in df["candidate_rank"].value_counts().sort_index().items()}


def compute_unverifiable_exact_stats(df: pd.DataFrame) -> Dict[str, int]:
    if df.empty:
        return {
            "rows": 0,
            "cut_exact": 0,
            "cut_and_next_exact": 0,
            "cut_exact_only": 0,
        }

    expected_cut = df["expected_cut_observation"].map(normalize_text)
    replay_cut = df["replay_cut_observation"].map(normalize_text)
    expected_next = df["expected_next_observation"].map(normalize_text)
    replay_next = df["replay_next_observation"].map(normalize_text)

    cut_exact = expected_cut == replay_cut
    next_exact = expected_next == replay_next
    cut_exact_count = int(cut_exact.sum())
    cut_and_next_exact = int((cut_exact & next_exact).sum())
    return {
        "rows": int(len(df)),
        "cut_exact": cut_exact_count,
        "cut_and_next_exact": cut_and_next_exact,
        "cut_exact_only": int(cut_exact_count - cut_and_next_exact),
    }


def sample_cut_map(df: pd.DataFrame, rank: int) -> Dict[str, int]:
    subset = df[df["candidate_rank"] == rank][["sample_uid", "cut_turn_idx"]].drop_duplicates("sample_uid")
    return {str(row.sample_uid): int(row.cut_turn_idx) for row in subset.itertuples(index=False)}


def overlap_rate(lhs: Dict[str, int], rhs: Dict[str, int]) -> float:
    if not lhs or not rhs:
        return float("nan")
    shared = sorted(set(lhs) & set(rhs))
    if not shared:
        return float("nan")
    same = sum(lhs[key] == rhs[key] for key in shared)
    return same / float(len(shared))


def run_window(
    python_bin: str,
    window: int,
    top_k: int,
    fixed_ratio: float,
    min_domain_gap: int,
    server: str,
    concurrency: int,
    request_timeout: float,
    paths: Dict[str, Path],
    skip_existing: bool,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    strategy = f"entropy_change_topk_w{window}_interaction_assistant_k3"
    stage2_parquet = paths["stage2"] / f"{strategy}.parquet"
    stage2_manifest = paths["manifests"] / f"{strategy}.stage2_manifest.json"
    stage3_dir = paths["stage3"] / f"w{window}"
    stage3_manifest = paths["manifests"] / f"{strategy}.stage3_manifest.json"
    stage3_validated = stage3_dir / f"{strategy}_validated.parquet"
    stage3_unverifiable = stage3_dir / f"{strategy}_unverifiable.parquet"
    stage3_mismatch = stage3_dir / f"{strategy}_mismatch.parquet"

    if not (skip_existing and stage2_parquet.exists() and stage2_manifest.exists()):
        run_cmd(
            [
                python_bin,
                str(EXPORT_SCRIPT),
                "--domains",
                "interaction_assistant",
                "--scorers",
                "change_topk",
                "--top-k",
                str(top_k),
                "--change-window",
                str(window),
                "--min-domain-gap",
                str(min_domain_gap),
                "--fixed-ratio",
                str(fixed_ratio),
                "--output-parquet",
                str(stage2_parquet),
                "--manifest-path",
                str(stage2_manifest),
            ]
        )

    if not (skip_existing and stage3_manifest.exists() and stage3_validated.exists() and stage3_unverifiable.exists()):
        run_cmd(
            [
                python_bin,
                str(REPLAY_SCRIPT),
                "--input-path",
                str(stage2_parquet),
                "--output-dir",
                str(stage3_dir),
                "--manifest-path",
                str(stage3_manifest),
                "--server",
                server,
                "--request-timeout",
                str(request_timeout),
                "--concurrency",
                str(concurrency),
                "--strategies",
                strategy,
            ]
        )

    stage2_manifest_data = load_json(stage2_manifest)
    stage3_manifest_data = load_json(stage3_manifest)
    stage2_df = pd.read_parquet(stage2_parquet)
    validated_df = pd.read_parquet(stage3_validated)
    unverifiable_df = pd.read_parquet(stage3_unverifiable)
    mismatch_df = pd.read_parquet(stage3_mismatch) if stage3_mismatch.exists() else pd.DataFrame()

    unique_samples_with_3_candidates = int((stage2_df.groupby("sample_uid")["candidate_rank"].nunique() == 3).sum())
    unique_samples_with_3_validated_ranks = int(
        (validated_df.groupby("sample_uid")["candidate_rank"].nunique() == 3).sum()
    )
    zero_delta_ratio = float((stage2_df["cut_turn_idx_delta_vs_fixed_ratio_0p4"] == 0).mean())

    summary = {
        "window": int(window),
        "strategy": strategy,
        "stage2_rows": int(len(stage2_df)),
        "stage2_unique_sample_uid": int(stage2_df["sample_uid"].nunique()),
        "stage2_rank_counts": rank_counts(stage2_df),
        "stage2_mean_cut_turn_idx": float(stage2_df["cut_turn_idx"].mean()),
        "stage2_mean_cut_turn_idx_delta_vs_fixed_ratio_0p4": float(
            stage2_df["cut_turn_idx_delta_vs_fixed_ratio_0p4"].mean()
        ),
        "stage2_zero_delta_vs_fixed_ratio_rate": zero_delta_ratio,
        "stage2_mean_selection_score": float(stage2_df["selection_score"].mean()),
        "stage2_samples_with_3_candidates": unique_samples_with_3_candidates,
        "stage3_category_counts": stage3_manifest_data["strategy_summary"][strategy]["category_counts"],
        "stage3_validated_rate": float(stage3_manifest_data["strategy_summary"][strategy]["validated_rate"]),
        "stage3_mismatch_rate": float(stage3_manifest_data["strategy_summary"][strategy]["mismatch_rate"]),
        "stage3_unverifiable_rate": float(stage3_manifest_data["strategy_summary"][strategy]["unverifiable_rate"]),
        "stage3_error_rate": float(stage3_manifest_data["strategy_summary"][strategy]["error_rate"]),
        "validated_rank_counts": rank_counts(validated_df),
        "validated_unique_sample_uid": int(validated_df["sample_uid"].nunique()),
        "validated_samples_with_3_ranks": unique_samples_with_3_validated_ranks,
        "unverifiable_exact_stats": compute_unverifiable_exact_stats(unverifiable_df),
        "mismatch_rows": int(len(mismatch_df)),
        "stage2_manifest_path": str(stage2_manifest),
        "stage3_manifest_path": str(stage3_manifest),
    }
    return summary, stage2_df, validated_df


def to_ratio_text(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{100.0 * value:.2f}%"


def to_float_text(value: float, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{value:.{digits}f}"


def make_summary_table(rows: List[Dict[str, Any]]) -> str:
    header = [
        "window",
        "validated_rate",
        "mismatch_rate",
        "unverifiable_rate",
        "validated_rows",
        "validated_samples_with_3_ranks",
        "cut_exact_in_unverifiable",
        "cut+next_exact_in_unverifiable",
        "mean_cut_delta_vs_fixed_ratio",
        "rank1_same_cut_vs_w11",
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows:
        counts = row["stage3_category_counts"]
        unverifiable_exact = row["unverifiable_exact_stats"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["window"]),
                    to_ratio_text(row["stage3_validated_rate"]),
                    to_ratio_text(row["stage3_mismatch_rate"]),
                    to_ratio_text(row["stage3_unverifiable_rate"]),
                    str(counts.get("validated", 0)),
                    str(row["validated_samples_with_3_ranks"]),
                    str(unverifiable_exact["cut_exact"]),
                    str(unverifiable_exact["cut_and_next_exact"]),
                    to_float_text(row["stage2_mean_cut_turn_idx_delta_vs_fixed_ratio_0p4"]),
                    to_ratio_text(row.get("rank1_same_cut_vs_w11")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_report(
    rows: List[Dict[str, Any]],
    report_path: Path,
    plot_path: Path,
    windows: Iterable[int],
) -> None:
    best_validated = max(rows, key=lambda row: row["stage3_validated_rate"])
    best_triple = max(rows, key=lambda row: row["validated_samples_with_3_ranks"])
    baseline = next(row for row in rows if row["window"] == 11)

    lines = [
        "# Entropy Change-Window Ablation Report",
        "",
        "## Scope",
        "",
        "- Pipeline root: `/Data/wyh/datasets/Verl-Data/train/textcraft/entropy_based_prefix`",
        "- Strategy family: `change_topk`",
        "- Domain: `interaction_assistant`",
        "- `top_k = 3`",
        f"- Windows tested: `{list(windows)}`",
        "- Comparison level: stage2 candidate export + stage3 replay validation",
        "",
        "## Method",
        "",
        "- For each `change_window`, rerun `05_export_entropy_prefix_candidates.py` with `--domains interaction_assistant --scorers change_topk`.",
        "- Then rerun `06_replay_validate_entropy_candidates.py` only for the corresponding strategy.",
        "- Use the resulting replay categories to compare validated / mismatch / unverifiable behavior.",
        "- Compare each window to `w11` by checking whether the `rank1` cut turn stays the same for the same `sample_uid`.",
        "",
        "Important implementation note:",
        "- The exporter requires odd windows. In the current implementation, smoothing depends on `radius = window // 2`, so even windows would collapse to the same radius as the previous odd window and are not separately informative.",
        "",
        "## Main Findings",
        "",
        f"- Best validated rate: `w{best_validated['window']}` with `{to_ratio_text(best_validated['stage3_validated_rate'])}`.",
        f"- Best triple-rank validated coverage: `w{best_triple['window']}` with `{best_triple['validated_samples_with_3_ranks']}` samples having all `rank1/2/3` validated.",
        f"- Baseline `w11` validated rate: `{to_ratio_text(baseline['stage3_validated_rate'])}`, triple-rank validated samples: `{baseline['validated_samples_with_3_ranks']}`.",
        "- `unverifiable` does not disappear as the window changes; much of it still comes from conservative validator limits rather than obvious replay failure.",
        "",
        "## Summary Table",
        "",
        make_summary_table(rows),
        "",
        "## Per-Window Notes",
        "",
    ]

    for row in rows:
        counts = row["stage3_category_counts"]
        unverifiable_exact = row["unverifiable_exact_stats"]
        lines.extend(
            [
                f"### w{row['window']}",
                "",
                f"- Stage2 rows: `{row['stage2_rows']}`; rank counts: `{row['stage2_rank_counts']}`.",
                f"- Replay counts: `{counts}`.",
                f"- Validated unique samples: `{row['validated_unique_sample_uid']}`.",
                f"- Samples with all three validated ranks: `{row['validated_samples_with_3_ranks']}`.",
                f"- Mean cut delta vs `fixed_ratio_0p4`: `{to_float_text(row['stage2_mean_cut_turn_idx_delta_vs_fixed_ratio_0p4'])}` turns.",
                f"- Rank1 same cut as `w11`: `{to_ratio_text(row.get('rank1_same_cut_vs_w11'))}`.",
                f"- In `unverifiable`, cut observation exact matches: `{unverifiable_exact['cut_exact']}`; cut+next exact matches: `{unverifiable_exact['cut_and_next_exact']}`.",
                "",
            ]
        )

    lines.extend(
        [
            "## Artifacts",
            "",
            f"- Plot: `{plot_path}`",
            f"- Structured summary: `{report_path.with_suffix('.json')}`",
            f"- CSV summary: `{report_path.with_suffix('.csv')}`",
            "",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(rows: List[Dict[str, Any]], output_path: Path) -> None:
    windows = [row["window"] for row in rows]
    validated_rate = [100.0 * row["stage3_validated_rate"] for row in rows]
    unverifiable_rate = [100.0 * row["stage3_unverifiable_rate"] for row in rows]
    triple_rank = [row["validated_samples_with_3_ranks"] for row in rows]
    cut_delta = [row["stage2_mean_cut_turn_idx_delta_vs_fixed_ratio_0p4"] for row in rows]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(windows, validated_rate, marker="o", color="#0f766e", linewidth=2.2)
    ax.set_title("Validated Rate")
    ax.set_xlabel("change_window")
    ax.set_ylabel("Percent")

    ax = axes[0, 1]
    ax.plot(windows, unverifiable_rate, marker="o", color="#b45309", linewidth=2.2)
    ax.set_title("Unverifiable Rate")
    ax.set_xlabel("change_window")
    ax.set_ylabel("Percent")

    ax = axes[1, 0]
    ax.plot(windows, triple_rank, marker="o", color="#1d4ed8", linewidth=2.2)
    ax.set_title("Samples With 3 Validated Ranks")
    ax.set_xlabel("change_window")
    ax.set_ylabel("Sample Count")

    ax = axes[1, 1]
    ax.plot(windows, cut_delta, marker="o", color="#7c3aed", linewidth=2.2)
    ax.axhline(0.0, color="#475569", linewidth=1.0, linestyle="--")
    ax.set_title("Mean Cut Delta vs fixed_ratio_0p4")
    ax.set_xlabel("change_window")
    ax.set_ylabel("Turns")

    fig.suptitle("Entropy change_window Ablation: interaction_assistant k=3", fontsize=16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    for window in args.windows:
        if window <= 0:
            raise ValueError("All --windows must be positive")
        if window % 2 == 0:
            raise ValueError("All --windows must be odd")

    paths = ensure_dirs()
    rows: List[Dict[str, Any]] = []
    stage2_cache: Dict[int, pd.DataFrame] = {}
    validated_cache: Dict[int, pd.DataFrame] = {}

    for window in args.windows:
        summary, stage2_df, validated_df = run_window(
            python_bin=args.python_bin,
            window=window,
            top_k=args.top_k,
            fixed_ratio=args.fixed_ratio,
            min_domain_gap=args.min_domain_gap,
            server=args.server,
            concurrency=args.concurrency,
            request_timeout=args.request_timeout,
            paths=paths,
            skip_existing=args.skip_existing,
        )
        rows.append(summary)
        stage2_cache[window] = stage2_df
        validated_cache[window] = validated_df

    rows.sort(key=lambda row: row["window"])
    baseline_rank1 = sample_cut_map(stage2_cache[11], rank=1) if 11 in stage2_cache else {}
    for row in rows:
        row["rank1_same_cut_vs_w11"] = overlap_rate(sample_cut_map(stage2_cache[row["window"]], rank=1), baseline_rank1)

    summary_df = pd.DataFrame(
        [
            {
                "window": row["window"],
                "validated_rate": row["stage3_validated_rate"],
                "mismatch_rate": row["stage3_mismatch_rate"],
                "unverifiable_rate": row["stage3_unverifiable_rate"],
                "validated_rows": row["stage3_category_counts"].get("validated", 0),
                "mismatch_rows": row["stage3_category_counts"].get("mismatch", 0),
                "unverifiable_rows": row["stage3_category_counts"].get("unverifiable", 0),
                "validated_unique_sample_uid": row["validated_unique_sample_uid"],
                "validated_samples_with_3_ranks": row["validated_samples_with_3_ranks"],
                "mean_cut_turn_idx_delta_vs_fixed_ratio_0p4": row["stage2_mean_cut_turn_idx_delta_vs_fixed_ratio_0p4"],
                "rank1_same_cut_vs_w11": row["rank1_same_cut_vs_w11"],
                "unverifiable_cut_exact": row["unverifiable_exact_stats"]["cut_exact"],
                "unverifiable_cut_and_next_exact": row["unverifiable_exact_stats"]["cut_and_next_exact"],
            }
            for row in rows
        ]
    )

    report_path = paths["notes"] / "window_ablation_report.md"
    json_path = report_path.with_suffix(".json")
    csv_path = report_path.with_suffix(".csv")
    plot_path = paths["outputs"] / "window_ablation_summary.png"

    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_df.to_csv(csv_path, index=False)
    write_plot(rows, plot_path)
    write_report(rows, report_path, plot_path, args.windows)

    print(f"REPORT={report_path}")
    print(f"SUMMARY_JSON={json_path}")
    print(f"SUMMARY_CSV={csv_path}")
    print(f"PLOT={plot_path}")


if __name__ == "__main__":
    main()
