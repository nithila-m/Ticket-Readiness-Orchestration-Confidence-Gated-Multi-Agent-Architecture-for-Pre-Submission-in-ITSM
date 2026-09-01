"""
Lightweight trained fallback classifier - proof-of-concept for the
"XLNet/HGB phased fallback" item on the research roadmap (Sravani et
al. precedent), brought forward from "post-graduation future work" to
something evaluated tonight.

NOT the same thing as fine-tuning Gemini or the Groq-hosted model -
neither supports customer fine-tuning on the free tier (Gemini's free
tier explicitly excludes fine-tuning; Groq only serves inference on
open-weight models, no customer fine-tuning). This trains a small,
separate, classically-trained classifier (TF-IDF features +
Histogram Gradient Boosting) on the existing 150-ticket synthetic set
instead - a legitimate "training technique" answer, honestly scoped as
a proof-of-concept on a small dataset, not a production model.

Runs entirely on CPU in seconds - no GPU, no large model download.

Run from the project root:
    python scripts/train_category_classifier.py

Requires:
    pip install scikit-learn pandas numpy
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "eval"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

DATASET_FILES = [
    "wifi_internet.json",
    "ms_teams.json",
    "vit_email.json",
    "ad_account_creation.json",
    "printer_support.json",
]

# Known dataset labeling-drift fixes.
LABEL_FIX = {
    "AD Account Support": "AD Account Creation",
    "VIT Email": "VIT Email Support",
}

CATEGORY_DISPLAY_TO_KEY = {
    "Wifi/Internet Support": "wifi_internet",
    "Microsoft Teams Support": "ms_teams",
    "VIT Email Support": "vit_email",
    "AD Account Creation": "ad_account_creation",
    "Printer Support": "printer_support",
}


def load_all_tickets() -> list[dict]:
    """Load tickets from all configured evaluation datasets."""
    tickets = []

    for filename in DATASET_FILES:
        path = DATA_DIR / filename

        if not path.exists():
            print(f"WARNING: {path} not found, skipping.")
            continue

        with open(path, encoding="utf-8") as f:
            tickets.extend(json.load(f))

    return tickets


def build_input_text(ticket: dict) -> str:
    """Combine ticket fields into the text used by the classifier."""
    parts = [
        ticket["subject"],
        ticket["message"],
    ]

    if ticket.get("ocr_text"):
        parts.append(ticket["ocr_text"])

    return " ".join(parts)


def normalize_label(raw: str) -> str | None:
    """Normalize dataset labels into the classifier's category keys."""
    fixed = LABEL_FIX.get(raw, raw)
    return CATEGORY_DISPLAY_TO_KEY.get(fixed)


def main():
    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    tickets = load_all_tickets()

    if not tickets:
        print("No tickets loaded - check data/eval/ files exist.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Prepare training rows
    # ------------------------------------------------------------------
    rows = []
    skipped = 0

    for ticket in tickets:
        label = normalize_label(ticket["true_category"])

        if label is None:
            skipped += 1
            continue

        rows.append(
            {
                "text": build_input_text(ticket),
                "label": label,
                "ticket_id": ticket["ticket_id"],
            }
        )

    if skipped:
        print(
            f"Skipped {skipped} tickets with unrecognized "
            f"true_category labels."
        )

    df = pd.DataFrame(rows)

    print(
        f"Training set: {len(df)} tickets across "
        f"{df['label'].nunique()} categories"
    )

    print(df["label"].value_counts().to_string())

    # ------------------------------------------------------------------
    # 3. Build TF-IDF + HGB pipeline
    #
    # TF-IDF produces a sparse matrix.
    # HistGradientBoostingClassifier requires dense input.
    #
    # Therefore:
    #
    # TF-IDF -> sparse matrix -> dense conversion -> HGB
    # ------------------------------------------------------------------
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    min_df=1,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
            (
                "to_dense",
                FunctionTransformer(
                    lambda x: x.toarray(),
                    accept_sparse=True,
                ),
            ),
            (
                "clf",
                HistGradientBoostingClassifier(
                    random_state=42,
                ),
            ),
        ]
    )

    # ------------------------------------------------------------------
    # 4. Stratified 5-fold cross-validation
    # ------------------------------------------------------------------
    n_splits = 5

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    print("\nRunning cross-validation...")

    y_pred = cross_val_predict(
        pipeline,
        df["text"],
        df["label"],
        cv=skf,
    )

    # ------------------------------------------------------------------
    # 5. Accuracy
    # ------------------------------------------------------------------
    accuracy = float(np.mean(y_pred == df["label"]))

    print(
        f"\n{n_splits}-fold cross-validated accuracy: "
        f"{accuracy:.3f}"
    )

    # ------------------------------------------------------------------
    # 6. Classification report
    # ------------------------------------------------------------------
    print("\nClassification report:")

    report = classification_report(
        df["label"],
        y_pred,
        zero_division=0,
    )

    print(report)

    # ------------------------------------------------------------------
    # 7. Confusion matrix
    # ------------------------------------------------------------------
    labels_sorted = sorted(df["label"].unique())

    cm = confusion_matrix(
        df["label"],
        y_pred,
        labels=labels_sorted,
    )

    cm_df = pd.DataFrame(
        cm,
        index=labels_sorted,
        columns=labels_sorted,
    )

    print("Confusion matrix (rows=true, cols=predicted):")
    print(cm_df.to_string())

    # ------------------------------------------------------------------
    # 8. Save evaluation report
    # ------------------------------------------------------------------
    RESULTS_DIR.mkdir(exist_ok=True)

    out_path = RESULTS_DIR / "category_classifier_cv_report.md"

    lines = [
        "# Lightweight trained fallback classifier — proof of concept",
        "",
        (
            f"TF-IDF (1-2 grams) + HistGradientBoostingClassifier, "
            f"{n_splits}-fold stratified CV"
        ),
        "",
        (
            f"n = {len(df)} tickets, "
            f"{df['label'].nunique()} categories"
        ),
        "",
        f"Cross-validated accuracy: **{accuracy:.3f}**",
        "",
        (
            "The pipeline uses TF-IDF text features followed by a "
            "sparse-to-dense conversion because "
            "HistGradientBoostingClassifier requires dense input."
        ),
        "",
        (
            "This is a separate, classically-trained model — not a "
            "fine-tuned version of the Gemini or Groq LLMs used "
            "elsewhere in the pipeline. It is a first step toward "
            "the XLNet/HGB fallback item on the research roadmap, "
            "evaluated honestly on a small (150-example) synthetic "
            "set — a proof-of-concept, not a production-ready model."
        ),
        "",
        "## Classification Report",
        "",
        "```",
        report,
        "```",
        "",
        "## Confusion Matrix",
        "",
        "Rows = true labels, columns = predicted labels.",
        "",
        "```",
        cm_df.to_string(),
        "```",
    ]

    out_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()