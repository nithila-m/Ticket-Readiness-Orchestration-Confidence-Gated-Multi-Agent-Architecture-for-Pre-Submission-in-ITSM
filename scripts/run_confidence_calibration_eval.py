"""
Real-API evaluation run that captures raw confidence values (not just
completeness score) needed for confidence calibration (Metric 6).

This is a companion to run_agent1_eval.py, not a replacement - that
script measures category/completeness accuracy; this one captures the
confidence numbers themselves so analyze_calibration.py can check
whether "confidence: 0.85" actually means "correct 85% of the time."

Costs real Gemini API calls. Run manually from the project root:
    python scripts/run_confidence_calibration_eval.py

Rate-limited the same way as run_agent1_eval.py (free-tier RPM safety).
~150 tickets * 4.5s = ~11-12 minutes total.
"""

import asyncio
import csv
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.information_extractor import InformationExtractor
from app.config.category_profiles import normalize_category_label
from app.config.settings import settings
from app.providers.exceptions import LLMProviderError
from app.providers.gemini_provider import GeminiProvider

logging.getLogger("google_genai").setLevel(logging.ERROR)
logging.getLogger("google_genai.types").setLevel(logging.ERROR)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "eval"
RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "confidence_calibration_raw.csv"

DATASET_FILES = [
    "wifi_internet.json",
    "ms_teams.json",
    "vit_email.json",
    "ad_account_creation.json",
    "printer_support.json",
]

SECONDS_BETWEEN_CALLS = 4.5


def load_all_tickets() -> list[dict]:
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
    parts = [f"Subject: {ticket['subject']}", ticket["message"]]
    if ticket.get("ocr_text"):
        parts.append(f"[Screenshot text: {ticket['ocr_text']}]")
    return "\n".join(parts)


async def evaluate_ticket(extractor: InformationExtractor, ticket: dict, max_retries: int = 2) -> dict:
    input_text = build_input_text(ticket)
    expected_category = normalize_category_label(ticket["true_category"])

    error = None
    result = None
    for attempt in range(max_retries + 1):
        try:
            result = await extractor.extract(input_text)
            error = None
            break
        except LLMProviderError as exc:
            error = str(exc)
            if "429" in error and attempt < max_retries:
                wait = 10 * (attempt + 1)
                print(f"    -> 429 hit, retrying in {wait}s...")
                await asyncio.sleep(wait)
                continue
            break

    if result is not None:
        predicted_category = result.category.value
        category_confidence = result.category.confidence
        category_correct = predicted_category == expected_category

        field_confidences = [
            f.confidence for f in result.extracted_fields.values() if f.value is not None
        ]
        mean_field_confidence = (
            sum(field_confidences) / len(field_confidences) if field_confidences else None
        )
    else:
        predicted_category = None
        category_confidence = None
        category_correct = None
        mean_field_confidence = None

    return {
        "ticket_id": ticket["ticket_id"],
        "expected_category": expected_category,
        "predicted_category": predicted_category,
        "category_confidence": category_confidence,
        "category_correct": category_correct,
        "mean_field_confidence": mean_field_confidence,
        "num_fields_extracted": len(result.extracted_fields) if result else 0,
        "error": error,
    }


async def main():
    tickets = load_all_tickets()
    if not tickets:
        print("No tickets loaded - check data/eval/ files exist.")
        return

    print(f"Loaded {len(tickets)} tickets. Capturing raw confidence for calibration analysis...")
    est_minutes = len(tickets) * SECONDS_BETWEEN_CALLS / 60
    print(f"Rate-limited to 1 call per {SECONDS_BETWEEN_CALLS}s (~{est_minutes:.1f} min total).\n")

    provider = GeminiProvider(api_key=settings.gemini_api_key or "", model=settings.gemini_model)
    extractor = InformationExtractor(provider)

    results = []
    for i, ticket in enumerate(tickets, start=1):
        row = await evaluate_ticket(extractor, ticket)
        results.append(row)
        status = "ERROR" if row["error"] else ("OK" if row["category_correct"] else "MISS")
        conf = f"{row['category_confidence']:.2f}" if row["category_confidence"] is not None else "n/a"
        print(f"[{i}/{len(tickets)}] {row['ticket_id']} - {status} (conf={conf})")
        if i < len(tickets):
            await asyncio.sleep(SECONDS_BETWEEN_CALLS)

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nRaw confidence data saved to {RESULTS_PATH}")
    print("Next: python scripts/analyze_calibration.py")


if __name__ == "__main__":
    asyncio.run(main())
