"""
Standalone evaluation of Agent 3 (KB Retrieval / Deflection) against the
team's synthetic ticket datasets — no app/, no ConversationState, no API
calls. Pure library-level test of TRO_Codes/agents/kb_retrieval_agent.py.

Run manually from the repo root:
    python scripts/run_agent3_eval.py

LIMITATION (state this honestly in the paper too, same caveat as
run_agent1_eval.py): `is_deflectable` in these datasets was set by the
data GENERATOR's judgment of whether the issue is generically
self-resolvable — not by checking it against this specific kb_articles
collection. This script measures whether Agent 3's embedding-similarity
retrieval agrees with that generator judgment, which is a proxy for
retrieval quality, not a certified gold-standard comparison.
"""

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRO_CODES_DIR = REPO_ROOT / "TRO_Codes"

# TRO_Codes has no __init__.py (not a formal package) — this mirrors the
# same sys.path trick kb_retrieval_agent.py already uses internally, so
# `from agents.kb_retrieval_agent import deflect` resolves the same way
# it would if you ran the script directly from inside TRO_Codes/.
sys.path.insert(0, str(TRO_CODES_DIR))

from agents.kb_retrieval_agent import deflect  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "eval"
RESULTS_PATH = REPO_ROOT / "results" / "agent3_eval_results.csv"

DATASET_FILES = [
    "wifi_internet.json",
    "ms_teams.json",
    "vit_email.json",
    "ad_account_creation.json",
    "printer_support.json",
]


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


def build_ticket_text(ticket: dict) -> str:
    """Matches how tickets were embedded at ingestion time (see
    TRO_Codes/kb_insertion.py::prepare_ticket_records)."""
    parts = [ticket.get("subject", ""), ticket.get("message", "")]
    if ticket.get("ocr_text"):
        parts.append(ticket["ocr_text"])
    return ". ".join(p for p in parts if p)



def evaluate_ticket(ticket: dict) -> dict:
    ticket_text = build_ticket_text(ticket)

    # Run Agent 3 on this ticket
    result = deflect(ticket_text)

    # Agent 3's actual output contract uses "deflect"
    predicted_deflectable = bool(result.get("deflect", False))
    expected_deflectable = bool(ticket.get("is_deflectable", False))

    return {
        "ticket_id": ticket["ticket_id"],
        "category": ticket.get("true_category", ""),
        "expected_deflectable": expected_deflectable,
        "predicted_deflectable": predicted_deflectable,
        "correct": predicted_deflectable == expected_deflectable,
        "resolution_confidence": float(
            result.get("resolution_confidence", 0.0)
        ),
        "matched_kb_id": result.get("matched_kb_id", ""),
        "matched_title": result.get("matched_title", ""),
    }



def main():
    tickets = load_all_tickets()
    if not tickets:
        print("No tickets loaded - check data/eval/ files exist.")
        return

    print(f"Loaded {len(tickets)} tickets. Querying kb_articles collection...\n")

    results = [evaluate_ticket(t) for t in tickets]

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Full results saved to {RESULTS_PATH}\n")
    print_summary(results)


def print_summary(results: list[dict]):
    total = len(results)
    accuracy = sum(1 for r in results if r["correct"]) / total

    tp = sum(1 for r in results if r["predicted_deflectable"] and r["expected_deflectable"])
    fp = sum(1 for r in results if r["predicted_deflectable"] and not r["expected_deflectable"])
    fn = sum(1 for r in results if not r["predicted_deflectable"] and r["expected_deflectable"])

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total tickets: {total}")
    print(f"Accuracy (predicted deflectable == is_deflectable): {accuracy:.1%}")
    print(f"Precision: {precision:.1%}   Recall: {recall:.1%}   F1: {f1:.1%}")

    print("\nDeflection distribution:")
    deflected = sum(1 for r in results if r["predicted_deflectable"])
    not_deflected = total - deflected
    print(f"  DEFLECT:     {deflected}/{total} ({deflected/total:.1%})")
    print(f"  NO DEFLECT:  {not_deflected}/{total} ({not_deflected/total:.1%})")


    print("\nPer-category accuracy:")
    by_category: dict[str, list[dict]] = {}
    for r in results:
        by_category.setdefault(r["category"], []).append(r)
    for cat, rows in sorted(by_category.items()):
        correct = sum(1 for r in rows if r["correct"])
        print(f"  {cat}: {correct}/{len(rows)} ({correct/len(rows):.1%})")

    misses = [r for r in results if not r["correct"]]
    if misses:
        print(f"\nMisclassified tickets ({len(misses)}):")
        for r in misses[:15]:
            print(
                f"  {r['ticket_id']}: "
                f"expected_deflectable={r['expected_deflectable']}, "
                f"predicted_deflectable={r['predicted_deflectable']} "
                f"(confidence={r['resolution_confidence']})"
            )

        if len(misses) > 15:
            print(f"  ... and {len(misses) - 15} more (see CSV)")


if __name__ == "__main__":
    main()