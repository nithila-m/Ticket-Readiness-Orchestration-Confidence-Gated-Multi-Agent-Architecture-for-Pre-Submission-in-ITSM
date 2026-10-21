"""
Real-API evaluation of Agent 5 (Classify, Prioritise, Explain) against
the team's 150-ticket synthetic dataset.

Costs real API calls (Groq for Agent 5; Gemini too in Mode B) - not part
of the pytest suite. Run manually from the project root:
    python scripts/run_agent5_eval.py              # both modes
    python scripts/run_agent5_eval.py --mode A     # isolated only
    python scripts/run_agent5_eval.py --limit 10   # quick smoke run

Modes:
  A (isolated)   - state carries the ground-truth category (confidence
                   1.0) and the raw ticket text, no extracted fields.
                   Measures priority only; routing is trivially correct
                   here because Agent 1's category is trusted.
  B (end-to-end) - Agent 1 extraction + completeness scoring, then
                   Agent 5. Measures priority AND routing as the
                   pipeline would actually produce them.

LIMITATIONS (state these in the paper too):
  - Single run, no repeated trials: LLM output is non-deterministic.
  - Synthetic data: priority labels were set by the data generator.
  - The dataset has no P1 label. P1 is only produced by the
    INCIDENT_CLUSTER override, which never fires here (correlation is
    not run), so any P1 is counted as wrong and has no ground truth.
"""

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path

# Ensure project root is importable regardless of where this is run from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.classification_agent import ClassificationAgent, was_rule_fallback
from app.agents.information_extractor import InformationExtractor
from app.config.category_profiles import normalize_category_label
from app.config.routing import DATASET_PRIORITY_TO_P, team_for_category
from app.config.settings import settings
from app.providers.classification_base import ClassificationProvider
from app.providers.exceptions import LLMProviderError
from app.providers.gemini_provider import GeminiProvider
from app.providers.groq_classification_provider import GroqClassificationProvider
from app.schemas.classification import ClassificationLLMOutput
from app.schemas.conversation import ConversationState, Message
from app.schemas.extraction import CategoryPrediction
from app.services.ticket_analysis_service import TicketAnalysisService
import logging

logging.getLogger("google_genai").setLevel(logging.ERROR)
logging.getLogger("google_genai.types").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "eval"
RESULTS_CSV = ROOT / "results" / "agent5_eval_results.csv"
RESULTS_MD = ROOT / "results" / "agent5_eval_summary.md"

DATASET_FILES = [
    "wifi_internet.json",
    "ms_teams.json",
    "vit_email.json",
    "ad_account_creation.json",
    "printer_support.json",
]

# Same known dataset labeling-drift fixes as train_category_classifier.py
# (duplicated rather than imported: that module pulls in sklearn/pandas).
LABEL_FIX = {
    "AD Account Support": "AD Account Creation",
    "VIT Email": "VIT Email Support",
}

PRIORITY_LABELS = ["P2", "P3", "P4"]          # ground-truth classes
PREDICTED_LABELS = ["P1", "P2", "P3", "P4"]   # P1 possible, never correct here

# Free-tier RPM protection (Groq and Gemini). Adjust if on a paid tier.
SECONDS_BETWEEN_CALLS = 4.5
MAX_RETRIES = 2
# Agent 5's own timeout must outlast the 429 backoff below (10s + 20s).
AGENT5_TIMEOUT_SECONDS = 120.0


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


def ground_truth_category(ticket: dict) -> str | None:
    raw = ticket["true_category"]
    return normalize_category_label(LABEL_FIX.get(raw, raw))


def build_input_text(ticket: dict) -> str:
    """Same combined input as run_agent1_eval.py."""
    parts = [f"Subject: {ticket['subject']}", ticket["message"]]
    if ticket.get("ocr_text"):
        parts.append(f"[Screenshot text: {ticket['ocr_text']}]")
    return "\n".join(parts)


def _is_rate_limit(exc: Exception) -> bool:
    return "429" in str(exc) or "rate limit" in str(exc).lower()


class RetryingClassificationProvider(ClassificationProvider):
    """Eval-only wrapper: retries 429s with backoff before Agent 5 sees the error."""

    def __init__(self, inner: ClassificationProvider):
        self._inner = inner
        self.retries = 0

    async def classify_ticket(self, state: ConversationState) -> ClassificationLLMOutput:
        for attempt in range(MAX_RETRIES + 1):
            try:
                return await self._inner.classify_ticket(state)
            except LLMProviderError as exc:
                if _is_rate_limit(exc) and attempt < MAX_RETRIES:
                    wait = 10 * (attempt + 1)  # backoff: 10s, then 20s
                    print(f"    -> Groq 429, retrying in {wait}s...")
                    self.retries += 1
                    await asyncio.sleep(wait)
                    continue
                raise
        raise AssertionError("unreachable")


async def analyze_with_retry(service: TicketAnalysisService, text: str):
    for attempt in range(MAX_RETRIES + 1):
        try:
            return await service.analyze(text), None
        except LLMProviderError as exc:
            if _is_rate_limit(exc) and attempt < MAX_RETRIES:
                wait = 10 * (attempt + 1)
                print(f"    -> Gemini 429, retrying in {wait}s...")
                await asyncio.sleep(wait)
                continue
            return None, str(exc)
    return None, "unreachable"


def state_mode_a(ticket: dict, category: str | None) -> ConversationState:
    pred = (
        CategoryPrediction(value=category, confidence=1.0)
        if category
        else CategoryPrediction(value=None, confidence=0.0)
    )
    return ConversationState(
        conversation_id=f"eval-A-{ticket['ticket_id']}",
        raw_messages=[Message(role="user", content=build_input_text(ticket))],
        detected_category=pred,
    )


def state_mode_b(ticket: dict, analysis) -> ConversationState:
    return ConversationState(
        conversation_id=f"eval-B-{ticket['ticket_id']}",
        raw_messages=[Message(role="user", content=build_input_text(ticket))],
        detected_category=analysis.category,
        extracted_fields=analysis.extracted_fields,
        completeness_score=analysis.completeness_score,
        missing_or_uncertain_fields=analysis.missing_or_uncertain_fields,
    )


def _row(mode: str, ticket: dict, gt_category: str | None, result, error: str | None, agent1_category=None) -> dict:
    gt_priority = DATASET_PRIORITY_TO_P.get(ticket["priority"])
    expected_team = team_for_category(gt_category)
    return {
        "mode": mode,
        "ticket_id": ticket["ticket_id"],
        "gt_category": gt_category,
        "agent1_category": agent1_category,
        "predicted_category": result.category if result else None,
        "gt_priority_label": ticket["priority"],
        "gt_priority": gt_priority,
        "predicted_priority": result.priority if result else None,
        "priority_correct": (result.priority == gt_priority) if result else None,
        "expected_team": expected_team,
        "assignee_group": result.assignee_group if result else None,
        "routing_correct": (result.assignee_group == expected_team) if result else None,
        "confidence": round(result.confidence, 4) if result else None,
        "source": result.source if result else None,
        "fallback": was_rule_fallback(result) if result else None,
        "rationale": result.routing_rationale if result else None,
        "error": error,
    }


async def run_mode_a(tickets: list[dict], agent: ClassificationAgent) -> list[dict]:
    rows = []
    for i, ticket in enumerate(tickets, start=1):
        gt = ground_truth_category(ticket)
        result = await agent.classify(state_mode_a(ticket, gt))
        row = _row("A", ticket, gt, result, None, agent1_category=gt)
        rows.append(row)
        status = "FALLBACK" if row["fallback"] else ("OK" if row["priority_correct"] else "MISS")
        print(f"[A {i}/{len(tickets)}] {ticket['ticket_id']} - {status} ({result.priority})")
        if i < len(tickets):
            await asyncio.sleep(SECONDS_BETWEEN_CALLS)
    return rows


async def run_mode_b(
    tickets: list[dict], agent: ClassificationAgent, service: TicketAnalysisService
) -> list[dict]:
    rows = []
    for i, ticket in enumerate(tickets, start=1):
        gt = ground_truth_category(ticket)
        analysis, error = await analyze_with_retry(service, build_input_text(ticket))
        if analysis is None:
            rows.append(_row("B", ticket, gt, None, f"agent1: {error}"))
            print(f"[B {i}/{len(tickets)}] {ticket['ticket_id']} - AGENT1 ERROR")
        else:
            await asyncio.sleep(SECONDS_BETWEEN_CALLS)
            result = await agent.classify(state_mode_b(ticket, analysis))
            row = _row("B", ticket, gt, result, None, agent1_category=analysis.category.value)
            rows.append(row)
            status = "FALLBACK" if row["fallback"] else ("OK" if row["priority_correct"] else "MISS")
            route = "route-ok" if row["routing_correct"] else "route-MISS"
            print(f"[B {i}/{len(tickets)}] {ticket['ticket_id']} - {status} ({result.priority}), {route}")
        if i < len(tickets):
            await asyncio.sleep(SECONDS_BETWEEN_CALLS)
    return rows


# --- Metrics (no sklearn dependency) -----------------------------------------


def confusion(rows: list[dict]) -> dict[str, dict[str, int]]:
    matrix = {gt: {p: 0 for p in PREDICTED_LABELS} for gt in PRIORITY_LABELS}
    for r in rows:
        if r["gt_priority"] in matrix and r["predicted_priority"] in PREDICTED_LABELS:
            matrix[r["gt_priority"]][r["predicted_priority"]] += 1
    return matrix


def macro_f1(matrix: dict[str, dict[str, int]]) -> float:
    f1s = []
    for label in PRIORITY_LABELS:
        tp = matrix[label][label]
        fn = sum(matrix[label].values()) - tp
        fp = sum(matrix[gt][label] for gt in PRIORITY_LABELS) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(f1s) / len(f1s)


def summarize(rows: list[dict]) -> dict:
    scored = [r for r in rows if r["predicted_priority"] is not None]
    llm_only = [r for r in scored if not r["fallback"]]
    matrix = confusion(scored)

    def acc(subset, key):
        return sum(1 for r in subset if r[key]) / len(subset) if subset else 0.0

    return {
        "n_total": len(rows),
        "n_scored": len(scored),
        "n_upstream_errors": len(rows) - len(scored),
        "n_fallback": sum(1 for r in scored if r["fallback"]),
        "n_p1": sum(1 for r in scored if r["predicted_priority"] == "P1"),
        "priority_accuracy": acc(scored, "priority_correct"),
        "priority_accuracy_llm_only": acc(llm_only, "priority_correct"),
        "n_llm_only": len(llm_only),
        "macro_f1": macro_f1(matrix),
        "routing_accuracy": acc(scored, "routing_correct"),
        "baseline_accuracy": (
            sum(1 for r in scored if r["gt_priority"] == "P3") / len(scored) if scored else 0.0
        ),
        "matrix": matrix,
    }


def render_summary(summaries: dict[str, dict]) -> str:
    lines = [
        "# Agent 5 Evaluation Summary",
        "",
        f"Model: `{settings.classification_groq_model}` (Groq)"
        + (f"; Agent 1: `{settings.gemini_model}` (Gemini)" if "B" in summaries else ""),
        "",
        "## Caveats",
        "",
        "- **Single run.** No repeated trials; LLM output is non-deterministic, so "
        "treat these numbers as one sample, not a stable estimate.",
        "- **Synthetic data.** 150 generator-labelled tickets; priority labels "
        "(HIGH/MEDIUM/LOW -> P2/P3/P4) were set by the generator, not by "
        "independent human annotation.",
        "- **P1 has no ground truth.** The dataset has no P1 label and correlation "
        "is not run here, so the INCIDENT_CLUSTER -> P1 override is not exercised. "
        "Any P1 prediction is counted as incorrect.",
        "- Mode A routing is trivially correct: the ground-truth category is given "
        "at confidence 1.0, so Agent 5 always keeps it. Mode A also has no "
        "extracted fields or completeness score - priority is judged from the "
        "ticket text alone.",
        "",
    ]
    titles = {"A": "Mode A - isolated (ground-truth category)", "B": "Mode B - end-to-end (Agent 1 -> Agent 5)"}
    for mode, s in summaries.items():
        lines += [
            f"## {titles[mode]}",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Tickets | {s['n_total']} |",
            f"| Scored (Agent 5 returned a result) | {s['n_scored']} |",
        ]
        if mode == "B":
            lines.append(f"| Agent 1 errors (not scored) | {s['n_upstream_errors']} |")
        lines += [
            f"| Rule fallbacks (source=rule) | {s['n_fallback']} |",
            f"| Priority accuracy (all scored) | {s['priority_accuracy']:.1%} |",
            f"| Priority accuracy (LLM only, n={s['n_llm_only']}) | {s['priority_accuracy_llm_only']:.1%} |",
            f"| Priority macro-F1 (P2/P3/P4) | {s['macro_f1']:.3f} |",
            f"| Majority-class baseline (always P3) accuracy | {s['baseline_accuracy']:.1%} |",
            f"| Routing accuracy | {s['routing_accuracy']:.1%} |",
            f"| P1 predictions | {s['n_p1']} |",
            "",
            "Priority confusion matrix (rows = ground truth, columns = predicted):",
            "",
            "| GT \\ Pred | " + " | ".join(PREDICTED_LABELS) + " |",
            "|---|" + "---|" * len(PREDICTED_LABELS),
        ]
        for gt in PRIORITY_LABELS:
            lines.append(f"| {gt} | " + " | ".join(str(s["matrix"][gt][p]) for p in PREDICTED_LABELS) + " |")
        lines.append("")
    return "\n".join(lines)


async def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["A", "B", "both"], default="both")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N tickets")
    args = parser.parse_args()

    tickets = load_all_tickets()
    if args.limit:
        tickets = tickets[: args.limit]
    if not tickets:
        print("No tickets loaded - check data/eval/ files exist.")
        return

    print(f"Loaded {len(tickets)} tickets. Rate-limited to 1 call per {SECONDS_BETWEEN_CALLS}s.\n")

    groq = RetryingClassificationProvider(
        GroqClassificationProvider(api_key=settings.groq_api_key, model=settings.classification_groq_model)
    )
    agent = ClassificationAgent(groq, timeout_seconds=AGENT5_TIMEOUT_SECONDS)

    rows: list[dict] = []
    summaries: dict[str, dict] = {}

    if args.mode in ("A", "both"):
        mode_rows = await run_mode_a(tickets, agent)
        rows += mode_rows
        summaries["A"] = summarize(mode_rows)

    if args.mode in ("B", "both"):
        gemini = GeminiProvider(api_key=settings.gemini_api_key or "", model=settings.gemini_model)
        service = TicketAnalysisService(InformationExtractor(gemini))
        mode_rows = await run_mode_b(tickets, agent, service)
        rows += mode_rows
        summaries["B"] = summarize(mode_rows)

    RESULTS_CSV.parent.mkdir(exist_ok=True)
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary_md = render_summary(summaries)
    with open(RESULTS_MD, "w", encoding="utf-8") as f:
        f.write(summary_md)

    print("\n" + summary_md)
    print(f"Groq 429 retries: {groq.retries}")
    print(f"Results: {RESULTS_CSV}\nSummary: {RESULTS_MD}")


if __name__ == "__main__":
    asyncio.run(main())
