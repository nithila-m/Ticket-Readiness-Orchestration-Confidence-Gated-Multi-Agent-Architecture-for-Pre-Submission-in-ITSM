"""
State carried through the LangGraph orchestrator.

`conversation` is the same ConversationState the rest of the app uses; the
nodes mutate it in place (via the existing service/pipeline step methods)
and return it, so nothing is copied or re-mapped. The other keys are
per-invocation scratch values the routers read.
"""

from typing import Literal, TypedDict

from app.agents.kb_deflection_agent import KBRetrievalResult
from app.schemas.clarification import ClarificationDecision
from app.schemas.conversation import ConversationState
from app.services.ticket_pipeline import GateDecision

Entry = Literal["message", "feedback", "duplicate_feedback", "review_approved"]


class GraphState(TypedDict, total=False):
    conversation: ConversationState
    entry: Entry
    user_message: str | None          # entry="message"
    resolved: bool | None             # entry="feedback"
    is_duplicate: bool | None         # entry="duplicate_feedback"
    kb_result: KBRetrievalResult | None
    decision: ClarificationDecision | None
    gate: GateDecision | None
    # configuration snapshot, so the routers stay pure functions of state
    pipeline_enabled: bool
    auto_dispatch: bool
