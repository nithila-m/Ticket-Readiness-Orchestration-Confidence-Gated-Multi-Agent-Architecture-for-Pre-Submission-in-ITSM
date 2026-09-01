"""
Confidence calibration analysis (Metric 6).

Reads results/confidence_calibration_raw.csv (produced by
run_confidence_calibration_eval.py) and checks whether Agent 1's
self-reported category confidence is actually calibrated - i.e.
whether tickets the model calls "0.85 confidence" are correct
roughly 85% of the time.

No API calls here - pure offline analysis, safe to re-run as many
times as needed while iterating on the prompt.

Run from the project root: python scripts/analyze_calibration.py

Outputs:
    results/calibration_reliability_diagram.png
    results/calibration_summary.md

Requires: pip install matplotlib pandas numpy
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
INPUT_PATH = RESULTS_DIR / "confidence_calibration_raw.csv"
PNG_PATH = RESULTS_DIR / "calibration_reliability_diagram.png"
SUMMARY_PATH = RESULTS_DIR / "calibration_summary.md"

N_BINS = 5  # coarse bins - 150 tickets doesn't reliably support finer bins


def load_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    before = len(df)
    df = df.dropna(subset=["category_confidence", "category_correct"])
    df["category_correct"] = df["category_correct"].astype(bool)
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with errors/missing confidence.")
    return df


def compute_calibration(df: pd.DataFrame, n_bins: int = N_BINS) -> pd.DataFrame:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    df = df.copy()
    df["bin"] = pd.cut(df["category_confidence"], bins=bins, include_lowest=True)

    grouped = (
        df.groupby("bin", observed=True)
        .agg(
            count=("category_correct", "size"),
            mean_confidence=("category_confidence", "mean"),
            accuracy=("category_correct", "mean"),
        )
        .reset_index()
    )
    return grouped


def expected_calibration_error(grouped: pd.DataFrame, total_n: int) -> float:
    ece = 0.0
    for _, row in grouped.iterrows():
        if pd.isna(row["mean_confidence"]):
            continue
        weight = row["count"] / total_n
        ece += weight * abs(row["accuracy"] - row["mean_confidence"])
    return ece


def brier_score(df: pd.DataFrame) -> float:
    return float(np.mean((df["category_confidence"] - df["category_correct"].astype(float)) ** 2))


def plot_reliability_diagram(grouped: pd.DataFrame, ece: float, brier: float, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")

    valid = grouped.dropna(subset=["mean_confidence", "accuracy"])
    ax.plot(
        valid["mean_confidence"], valid["accuracy"],
        marker="o", color="#3C3489", label="Agent 1 (observed)",
    )
    for _, row in valid.iterrows():
        ax.annotate(
            f"n={int(row['count'])}",
            (row["mean_confidence"], row["accuracy"]),
            textcoords="offset points", xytext=(6, -6), fontsize=8,
        )

    ax.set_xlabel("Self-reported category confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_title(f"Agent 1 confidence calibration\nECE = {ece:.3f}, Brier score = {brier:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Reliability diagram saved to {out_path}")


def write_summary(grouped: pd.DataFrame, ece: float, brier: float, n: int, out_path: Path):
    lines = [
        "# Agent 1 confidence calibration — Metric 6",
        f"n = {n} tickets (rows with a returned category and confidence)",
        f"Expected Calibration Error (ECE): **{ece:.3f}**",
        f"Brier score: **{brier:.3f}**",
        "",
        "Lower is better for both. This is a single run on 150 synthetic "
        "tickets - report it as a directional first check, not a final "
        "calibration claim, consistent with how the rest of the evaluation "
        "suite is reported.",
        "",
        "| Confidence bin | n | Mean confidence | Observed accuracy | Gap |",
        "|---|---|---|---|---|",
    ]
    for _, row in grouped.iterrows():
        if pd.isna(row["mean_confidence"]):
            continue
        gap = row["accuracy"] - row["mean_confidence"]
        lines.append(
            f"| {row['bin']} | {int(row['count'])} | {row['mean_confidence']:.3f} "
            f"| {row['accuracy']:.3f} | {gap:+.3f} |"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Summary written to {out_path}")


def main():
    if not INPUT_PATH.exists():
        print(f"ERROR: {INPUT_PATH} not found.")
        print("Run scripts/run_confidence_calibration_eval.py first.")
        sys.exit(1)

    df = load_clean(INPUT_PATH)
    if df.empty:
        print("No usable rows after cleaning - nothing to analyze.")
        sys.exit(1)

    grouped = compute_calibration(df)
    ece = expected_calibration_error(grouped, len(df))
    brier = brier_score(df)

    print(grouped.to_string(index=False))
    print(f"\nECE = {ece:.3f}")
    print(f"Brier score = {brier:.3f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    plot_reliability_diagram(grouped, ece, brier, PNG_PATH)
    write_summary(grouped, ece, brier, len(df), SUMMARY_PATH)


if __name__ == "__main__":
    main()
