"""
Pure router functions for the LangGraph orchestrator: GraphState in, the
name of the next branch out. No I/O and no agent calls, so every decision
rule is unit-testable on its own.
"""

from typing import Literal

from app.orchestration.state import GraphState
from app.services.clarification_service import passes_completeness_gate

END_BRANCH = "end"


def entry_router(
    state: GraphState,
) -> Literal["message", "feedback", "duplicate_feedback", "review_approved"]:
    return state["entry"]


def completeness_gate(state: GraphState) -> Literal["kb", "clarify"]:
    """Agent 3 only runs once the request is complete enough to trust a KB match."""
    return "kb" if passes_completeness_gate(state["conversation"]) else "clarify"


def kb_gate(state: GraphState) -> Literal["deflect", "clarify"]:
    """A STRONG_MATCH deflects without Agent 2; anything weaker goes to Agent 2."""
    return "deflect" if state["kb_result"].outcome == "STRONG_MATCH" else "clarify"


def clarify_router(state: GraphState) -> Literal["correlate", "enqueue_review", "end"]:
    """READY -> ticket pipeline (when auto-dispatch is on); ESCALATE -> review."""
    if not state.get("pipeline_enabled"):
        return END_BRANCH
    action = state["decision"].action
    if action == "READY" and state.get("auto_dispatch"):
        return "correlate"
    if action == "ESCALATE":
        return "enqueue_review"
    return END_BRANCH


def feedback_router(state: GraphState) -> Literal["close_resolved", "correlate", "end"]:
    """Resolved -> SELF_RESOLVED; not resolved -> the user still needs a ticket."""
    if state["resolved"]:
        return "close_resolved"
    return "correlate" if state.get("pipeline_enabled") else END_BRANCH


def correlation_router(state: GraphState) -> Literal["confirm_duplicate", "classify"]:
    """A DUPLICATE is a suggestion: pause and ask the user before merging."""
    correlation = state["conversation"].correlation
    if correlation is not None and correlation.outcome == "DUPLICATE":
        return "confirm_duplicate"
    return "classify"


def confidence_gate(state: GraphState) -> Literal["dispatch", "enqueue_review"]:
    """
    Routes on the decision TicketPipeline.gate() already made (and audited)
    in the classify node - MERGE and DISPATCH both go to the Dispatcher.
    """
    return "enqueue_review" if state["gate"] == "REVIEW" else "dispatch"
