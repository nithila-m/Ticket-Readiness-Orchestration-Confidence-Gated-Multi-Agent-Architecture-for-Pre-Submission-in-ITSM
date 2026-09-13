"""
Agent 3: bridges the standalone, already-working retrieval logic in
TRO_Codes/agents/kb_retrieval_agent.py into app/'s world (typed results,
ConversationState-shaped output, injectable for tests) without moving or
rewriting that module. TRO_Codes/ has no __init__.py (not a formal
package), so this mirrors the same sys.path trick kb_retrieval_agent.py
already uses internally to import chroma_client1 - kept in exactly one
place so it doesn't drift.
"""

import asyncio
import sys
from pathlib import Path
from typing import Callable, Literal, NamedTuple

from app.config.category_profiles import category_key_to_kb_label
from app.config.settings import settings
from app.schemas.clarification import ClarificationDecision
from app.schemas.extraction import CategoryPrediction, ExtractedField

_TRO_CODES_DIR = Path(__file__).resolve().parent.parent.parent / "TRO_Codes"
if str(_TRO_CODES_DIR) not in sys.path:
    sys.path.insert(0, str(_TRO_CODES_DIR))

from agents.kb_retrieval_agent import deflect as _deflect  # noqa: E402


class KBRetrievalResult(NamedTuple):
    outcome: Literal["STRONG_MATCH", "WEAK_MATCH", "NO_MATCH"]
    similarity_score: float
    articles_checked: list[str]
    matched_kb_id: str | None
    matched_title: str | None
    offered_resolution: str | None


def build_kb_query(
    category: CategoryPrediction | None,
    extracted_fields: dict[str, ExtractedField],
) -> str:
    """
    Builds Agent 3's search string from the structured fields Agent 1
    already extracted, instead of the speaker-labeled multi-turn transcript
    used to re-run Agent 1 each turn (see transcript_formatter.py).

    KB articles are embedded at ingestion time as bare "title. symptoms"
    text (see TRO_Codes/kb_insertion.py::prepare_kb_records) - no dialogue
    framing, no "User:"/"Assistant asked:" labels. Searching with the raw
    transcript compares two differently-shaped strings and gets noisier
    every clarification turn; this keeps the query short and description-
    shaped like the thing it's being compared against.

    Only non-null field values are included - a field extracted with
    value=None carries no signal and would just add empty noise.

    category.value is Agent 1's internal snake_case key (e.g.
    "wifi_internet"), not a natural-language label - converted here via
    category_key_to_kb_label() so the embedded text reads like the KB
    articles it's being compared against ("Wifi/Internet Support") rather
    than a raw code identifier, which a sentence-transformer embeds less
    meaningfully alongside natural language.
    """
    lines: list[str] = []
    if category is not None and category.value:
        display_label = category_key_to_kb_label(category.value) or category.value
        lines.append(f"Category: {display_label}")
    for field_name, field in extracted_fields.items():
        if field.value is not None:
            lines.append(f"{field_name}: {field.value}")
    return "\n".join(lines)


def run_kb_retrieval(query_text: str, category: str | None = None) -> KBRetrievalResult:
    """
    Synchronous wrapper around TRO_Codes' deflect(). Kept synchronous on
    purpose (matches the underlying chromadb client) - KBDeflectionAgent
    below is what offloads this onto a thread for async callers.

    Thresholds now come from settings (tunable against
    results/agent3_eval_results.csv) instead of deflect()'s own hardcoded
    defaults, so TRO_Codes/ doesn't need to know about app/ config at all -
    this is the one seam where that translation happens.
    """
    raw = _deflect(
        query_text,
        category=category,
        strong_threshold=settings.kb_strong_threshold,
        weak_threshold=settings.kb_weak_threshold,
    )
    return KBRetrievalResult(
        outcome=raw["outcome"],
        similarity_score=raw.get("resolution_confidence", 0.0),
        articles_checked=raw.get("kb_articles_checked", []),
        matched_kb_id=raw.get("matched_kb_id"),
        matched_title=raw.get("matched_title"),
        offered_resolution=raw.get("offered_resolution"),
    )


def build_deflection_decision(kb_result: KBRetrievalResult) -> ClarificationDecision:
    """
    Turns a STRONG_MATCH Agent 3 result into the same ClarificationDecision
    shape Agent 2 would have returned, so the rest of the pipeline (API
    response, audit log, frontend) doesn't need to know Agent 2 never ran
    this turn. Only ever call this when kb_result.outcome == "STRONG_MATCH" -
    it does not check the outcome itself; the gate decision belongs in
    clarification_service.py, where the rest of the turn's control flow lives.
    """
    return ClarificationDecision(
        action="DEFLECTED",
        reasoning=(
            f"Matched knowledge base article '{kb_result.matched_title}' "
            f"(similarity={kb_result.similarity_score:.2f}), which resolves "
            f"this issue without needing a ticket."
        ),
        information_gap=None,
        question=None,
        expected_information_gain=0.0,
        affected_fields=[],
        priority="low",
        confidence=kb_result.similarity_score,
    )


class KBDeflectionAgent:
    """
    Agent 3. Thin async wrapper around a synchronous retrieval function -
    mirrors AdaptiveClarifier's role for Agent 2 (wraps a provider so
    ClarificationService depends on this class, not on TRO_Codes directly).
    Constructor-injectable so tests can swap in a fake without touching
    the real chromadb-backed collection.

    Takes the same structured state Agent 2 already sees (detected
    category + extracted fields) rather than a flat ticket_text string,
    so the query-building decision (build_kb_query, above) lives in one
    place instead of being duplicated by every caller.
    """

    def __init__(
        self,
        retrieve_fn: Callable[[str, str | None], KBRetrievalResult] = run_kb_retrieval,
    ):
        self._retrieve_fn = retrieve_fn

    async def check(
        self,
        category: CategoryPrediction | None,
        extracted_fields: dict[str, ExtractedField],
    ) -> KBRetrievalResult:
        query_text = build_kb_query(category, extracted_fields)
        # Convert internal key -> KB metadata display label before this
        # goes anywhere near the `where` filter. Passing the raw internal
        # key here was the original bug: kb_articles metadata never
        # contains "wifi_internet", so an unconverted filter would match
        # zero documents on every single call and silently force NO_MATCH.
        kb_category_label = (
            category_key_to_kb_label(category.value)
            if category is not None and category.value
            else None
        )
        return await asyncio.to_thread(self._retrieve_fn, query_text, kb_category_label)