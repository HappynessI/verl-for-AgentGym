#!/usr/bin/env python3
"""Plot paper-main training reward curves from included metric CSVs."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_METRICS = REPO_ROOT / "results/paper_metrics.csv"
OUTPUT_PATH = REPO_ROOT / "results/figures/main_reward_curves.png"
ROLLING_WINDOW = 25

COLORS = {
    "TextCraft": "#2f6fed",
    "BabyAI": "#0f8b8d",
    "ALFWorld": "#d55e00",
}

ENV_ORDER = {
    "TextCraft": 0,
    "BabyAI": 1,
    "ALFWorld": 2,
}


def load_main_runs() -> list[dict[str, str]]:
    with PAPER_METRICS.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return [
        row
        for row in rows
        if row["paper_use"] == "paper-main" and row["status"] == "available"
    ]


def main() -> None:
    runs = load_main_runs()
    if not runs:
        raise RuntimeError("No available paper-main runs found in results/paper_metrics.csv")

    runs = sorted(runs, key=lambda row: ENV_ORDER.get(row["environment"], 99))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, len(runs), figsize=(13.2, 3.9), dpi=180, sharey=True)
    if len(runs) == 1:
        axes = [axes]

    for ax, run in zip(axes, runs):
        env = run["environment"]
        metrics_path = REPO_ROOT / run["metrics_path"]
        df = pd.read_csv(metrics_path)
        if "step" not in df.columns or "critic_rewards_mean" not in df.columns:
            raise RuntimeError(f"Missing required columns in {run['metrics_path']}")

        df = df[["step", "critic_rewards_mean"]].dropna()
        df = df.sort_values("step")
        smoothed = df["critic_rewards_mean"].rolling(
            ROLLING_WINDOW, min_periods=1
        ).mean()
        color = COLORS.get(env)
        ax.plot(
            df["step"],
            df["critic_rewards_mean"],
            color=color,
            alpha=0.18,
            linewidth=0.8,
        )
        ax.plot(
            df["step"],
            smoothed,
            color=color,
            linewidth=2.1,
        )
        ax.set_title(env)
        ax.set_xlabel("Training step")
        ax.set_ylim(-0.02, 1.02)
        ax.text(
            0.98,
            0.06,
            f"final={df['critic_rewards_mean'].iloc[-1]:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#334155",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.0},
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Mean reward")
    fig.suptitle(f"Main Experiment Training Reward ({ROLLING_WINDOW}-step rolling mean)", y=1.02)
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
