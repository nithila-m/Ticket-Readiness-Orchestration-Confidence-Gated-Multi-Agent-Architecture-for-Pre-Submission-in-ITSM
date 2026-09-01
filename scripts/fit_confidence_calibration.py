"""
Post-hoc confidence recalibration (Claim B, iteration 2).

analyze_calibration.py measures whether Agent 1's raw confidence is
calibrated. This script goes one step further: it FITS a monotonic
correction curve (isotonic regression) mapping raw confidence ->
calibrated confidence, using k-fold cross-validation so the reported
improvement isn't just overfitting to the same 150 tickets it was fit
on - fit on some folds, evaluate only on the held-out fold, rotate.

The fitted mapping is saved as a small, auditable JSON list of
(raw, calibrated) points - not a pickled black-box object - so it can
be inspected and explained in five minutes, consistent with the
project's own auditability requirement.

Run after run_confidence_calibration_eval.py:
    python scripts/fit_confidence_calibration.py

Requires: pip install scikit-learn pandas numpy matplotlib
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
INPUT_PATH = RESULTS_DIR / "confidence_calibration_raw.csv"
MAPPING_PATH = RESULTS_DIR / "calibration_mapping.json"
PNG_PATH = RESULTS_DIR / "calibration_before_after.png"
SUMMARY_PATH = RESULTS_DIR / "calibration_recalibration_summary.md"

N_FOLDS = 5
N_BINS = 5


def ece(confidence: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(confidence, bins[1:-1])
    total = len(confidence)
    err = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        err += (mask.sum() / total) * abs(correct[mask].mean() - confidence[mask].mean())
    return err


def reliability_points(values: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(values, bins[1:-1])
    xs, ys, ns = [], [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        xs.append(values[mask].mean())
        ys.append(correct[mask].mean())
        ns.append(int(mask.sum()))
    return xs, ys, ns


def main():
    if not INPUT_PATH.exists():
        print(f"ERROR: {INPUT_PATH} not found. Run run_confidence_calibration_eval.py first.")
        return

    df = pd.read_csv(INPUT_PATH).dropna(subset=["category_confidence", "category_correct"])
    df["category_correct"] = df["category_correct"].astype(bool).astype(float)

    conf = df["category_confidence"].to_numpy()
    correct = df["category_correct"].to_numpy()
    n = len(df)
    print(f"n = {n} tickets")

    raw_ece = ece(conf, correct)

    # Cross-validated calibrated confidence: for each fold, fit isotonic
    # regression on the OTHER folds and predict on the held-out fold.
    # This is what makes the reported improvement honest rather than
    # circular - fitting and evaluating on the same 150 points would
    # trivially "improve" ECE without proving anything generalizes.
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    calibrated = np.zeros_like(conf)
    for train_idx, test_idx in kf.split(conf):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(conf[train_idx], correct[train_idx])
        calibrated[test_idx] = iso.predict(conf[test_idx])

    calibrated_ece = ece(calibrated, correct)

    print(f"Raw ECE:        {raw_ece:.3f}")
    print(f"Calibrated ECE: {calibrated_ece:.3f}  (cross-validated, {N_FOLDS}-fold)")

    # Fit the FINAL mapping on all 150 points for actual downstream use -
    # that's separate from the ECE claim above, which only ever used
    # held-out folds.
    final_iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    final_iso.fit(conf, correct)
    breakpoints = sorted(set(np.round(conf, 4)))
    mapping = [
        {"raw_confidence": float(x), "calibrated_confidence": float(final_iso.predict([x])[0])}
        for x in breakpoints
    ]
    MAPPING_PATH.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    print(f"Calibration mapping saved to {MAPPING_PATH}")

    # Before/after reliability diagram, side by side.
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, values, title in [
        (axes[0], conf, "Before (raw confidence)"),
        (axes[1], calibrated, "After (cross-validated calibration)"),
    ]:
        xs, ys, ns = reliability_points(values, correct)
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.plot(xs, ys, marker="o", color="#3C3489")
        for x, y, count in zip(xs, ys, ns):
            ax.annotate(f"n={count}", (x, y), textcoords="offset points", xytext=(6, -6), fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("Confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Observed accuracy")
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=150)
    print(f"Before/after diagram saved to {PNG_PATH}")

    SUMMARY_PATH.write_text(
        "\n".join([
            "# Confidence recalibration (isotonic regression, cross-validated)",
            f"n = {n} tickets, {N_FOLDS}-fold cross-validation",
            f"Raw ECE: **{raw_ece:.3f}**",
            f"Calibrated ECE (held-out folds): **{calibrated_ece:.3f}**",
            "",
            "The calibrated ECE above is computed only on held-out folds - "
            "never the same points the correction was fit on - so this "
            "number reflects genuine generalization, not overfitting to "
            "150 points. The final mapping saved to calibration_mapping.json "
            "is fit on the full set for downstream use and is a plain, "
            "auditable lookup table, not a black-box model.",
        ]),
        encoding="utf-8",
    )
    print(f"Summary saved to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
