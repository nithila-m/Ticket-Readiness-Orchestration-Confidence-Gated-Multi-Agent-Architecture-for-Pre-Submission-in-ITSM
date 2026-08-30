"""
Real-API evaluation of Agent 1 (InformationExtractor + completeness scoring)
against the team's synthetic ticket datasets.

Costs real Gemini API calls - not part of the pytest suite. Run manually:
    python scripts/run_agent1_eval.py

LIMITATION (state this honestly in the paper too): the ground-truth
`missing_fields` in these datasets were set by the data GENERATOR, not
measured from independent human labeling of the message text. This
script checks whether Agent 1's extraction roughly agrees with what the
generator intended to omit - a proxy for extractor quality, not a
certified gold-standard comparison.
"""

import asyncio
import csv
import json
import sys
from pathlib import Path

# Ensure project root is importable regardless of where this is run from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.information_extractor import InformationExtractor
from app.config.category_profiles import normalize_category_label
from app.core.settings import settings
from app.providers.exceptions import LLMProviderError
from app.providers.gemini_provider import GeminiProvider
from app.services.ticket_analysis_service import TicketAnalysisService

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "eval"
RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "agent1_eval_results.csv"

DATASET_FILES = [
    "wifi_internet.json",
    "ms_teams.json",
    "vit_email.json",
    "ad_account_creation.json",
    "printer_support.json",
]

# Rough score buckets for comparing against the dataset's qualitative
# completeness_level label. These are judgment calls, not derived from
# any labeled threshold-fitting exercise - stated as such in the paper.
COMPLETENESS_BUCKETS = {
    "vague": (0.0, 0.35),
    "partial": (0.35, 0.75),
    "complete": (0.75, 1.01),
}

# Gemini free-tier RPM protection. Adjust if you're on a paid tier.
SECONDS_BETWEEN_CALLS = 2.0


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
    """
    Mimics what a real submission would actually contain: subject + message
    + OCR text if a screenshot was attached. Agent 1's real API only takes
    one text field, so this is the realistic combined input, not just
    the bare `message` in isolation.
    """
    parts = [f"Subject: {ticket['subject']}", ticket["message"]]
    if ticket.get("ocr_text"):
        parts.append(f"[Screenshot text: {ticket['ocr_text']}]")
    return "\n".join(parts)


def completeness_bucket_matches(predicted_score: float, expected_level: str) -> bool:
    low, high = COMPLETENESS_BUCKETS.get(expected_level, (0.0, 1.01))
    return low <= predicted_score < high


async def evaluate_ticket(service: TicketAnalysisService, ticket: dict) -> dict:
    input_text = build_input_text(ticket)
    expected_category = normalize_category_label(ticket["true_category"])

    try:
        result = await service.analyze(input_text)
        error = None
        predicted_category = result.category.value
        completeness_score = result.completeness_score
        missing_fields_predicted = result.missing_or_uncertain_fields
    except LLMProviderError as exc:
        error = str(exc)
        predicted_category = None
        completeness_score = None
        missing_fields_predicted = []

    category_correct = (
        predicted_category == expected_category if error is None else None
    )
    bucket_match = (
        completeness_bucket_matches(completeness_score, ticket["completeness_level"])
        if error is None
        else None
    )

    return {
        "ticket_id": ticket["ticket_id"],
        "expected_category": expected_category,
        "predicted_category": predicted_category,
        "category_correct": category_correct,
        "completeness_level_expected": ticket["completeness_level"],
        "completeness_score_predicted": completeness_score,
        "completeness_bucket_match": bucket_match,
        "missing_fields_expected": ";".join(ticket.get("missing_fields", [])),
        "missing_fields_predicted": ";".join(missing_fields_predicted),
        "error": error,
    }


async def main():
    tickets = load_all_tickets()
    if not tickets:
        print("No tickets loaded - check data/eval/ files exist.")
        return

    print(f"Loaded {len(tickets)} tickets. Starting evaluation against real Gemini API...")
    print(f"Rate-limited to 1 call per {SECONDS_BETWEEN_CALLS}s.\n")

    provider = GeminiProvider(api_key=settings.gemini_api_key or "", model=settings.gemini_model)
    extractor = InformationExtractor(provider)
    service = TicketAnalysisService(extractor)

    results = []
    for i, ticket in enumerate(tickets, start=1):
        row = await evaluate_ticket(service, ticket)
        results.append(row)

        status = "ERROR" if row["error"] else ("OK" if row["category_correct"] else "MISS")
        print(f"[{i}/{len(tickets)}] {row['ticket_id']} - {status}")

        if i < len(tickets):
            await asyncio.sleep(SECONDS_BETWEEN_CALLS)

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nFull results saved to {RESULTS_PATH}\n")
    print_summary(results)


def print_summary(results: list[dict]):
    completed = [r for r in results if r["error"] is None]
    errored = [r for r in results if r["error"] is not None]

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total tickets: {len(results)}")
    print(f"Successful API calls: {len(completed)}")
    print(f"Failed API calls: {len(errored)}")

    if completed:
        correct = sum(1 for r in completed if r["category_correct"])
        print(f"\nOverall category accuracy: {correct}/{len(completed)} ({correct/len(completed):.1%})")

        by_category: dict[str, list[dict]] = {}
        for r in completed:
            by_category.setdefault(r["expected_category"], []).append(r)

        print("\nPer-category accuracy:")
        for cat, rows in sorted(by_category.items()):
            cat_correct = sum(1 for r in rows if r["category_correct"])
            print(f"  {cat}: {cat_correct}/{len(rows)} ({cat_correct/len(rows):.1%})")

        bucket_matches = sum(1 for r in completed if r["completeness_bucket_match"])
        print(
            f"\nCompleteness bucket agreement: {bucket_matches}/{len(completed)} "
            f"({bucket_matches/len(completed):.1%})"
        )

        misses = [r for r in completed if not r["category_correct"]]
        if misses:
            print(f"\nMisclassified tickets ({len(misses)}):")
            for r in misses[:15]:
                print(f"  {r['ticket_id']}: expected={r['expected_category']}, got={r['predicted_category']}")
            if len(misses) > 15:
                print(f"  ... and {len(misses) - 15} more (see CSV)")

    if errored:
        print(f"\nErrored tickets ({len(errored)}):")
        for r in errored[:5]:
            print(f"  {r['ticket_id']}: {r['error']}")


if __name__ == "__main__":
    asyncio.run(main())