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
import logging
import sys
from pathlib import Path
from typing import Callable, Literal, NamedTuple

from app.agents.request_text import build_kb_query  # noqa: F401 - re-exported for existing importers
from app.config.settings import settings
from app.schemas.clarification import ClarificationDecision

_TRO_CODES_DIR = Path(__file__).resolve().parent.parent.parent / "TRO_Codes"
if str(_TRO_CODES_DIR) not in sys.path:
    sys.path.insert(0, str(_TRO_CODES_DIR))

from agents.kb_retrieval_agent import deflect as _deflect  # noqa: E402

logger = logging.getLogger(__name__)


class KBRetrievalResult(NamedTuple):
    outcome: Literal["STRONG_MATCH", "WEAK_MATCH", "NO_MATCH"]
    similarity_score: float
    articles_checked: list[str]
    matched_kb_id: str | None
    matched_title: str | None
    offered_resolution: str | None


NO_MATCH_RESULT = KBRetrievalResult(
    outcome="NO_MATCH",
    similarity_score=0.0,
    articles_checked=[],
    matched_kb_id=None,
    matched_title=None,
    offered_resolution=None,
)


def run_kb_retrieval(query_text: str, category: str | None = None) -> KBRetrievalResult:
    """
    Synchronous wrapper around TRO_Codes' deflect(). Kept synchronous on
    purpose (matches the underlying chromadb client) - KBDeflectionAgent
    below is what offloads this onto a thread for async callers.
    Thresholds come from settings, so TRO_Codes/ never reads app/ config.
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
    """

    def __init__(
        self,
        retrieve_fn: Callable[[str, str | None], KBRetrievalResult] = run_kb_retrieval,
    ):
        self._retrieve_fn = retrieve_fn

    async def check(self, query_text: str, category: str | None) -> KBRetrievalResult:
        # Retrieval is an optional shortcut, not a required step: if Chroma
        # or the embedding model fails, log it and report NO_MATCH so the
        # turn falls through to Agent 2 instead of surfacing as a 500.
        try:
            return await asyncio.to_thread(self._retrieve_fn, query_text, category)
        except Exception:
            logger.exception("Agent 3 KB retrieval failed; treating as NO_MATCH")
            return NO_MATCH_RESULT
