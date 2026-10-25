"""
Post-READY ticket pipeline: Agent 4 -> Agent 5 -> confidence gate -> Agent 6.

  correlate  Agent 4 first: Agent 5 reads state.correlation for its
             incident-cluster -> P1 rule.
  classify   Agent 5 always runs, even for a duplicate, so the merged
             ticket still records category/priority/team.
  gate       DUPLICATE -> MERGE (the gate does not apply to a merge);
             otherwise confidence < threshold -> REVIEW, else DISPATCH.
             Agent 5's rule fallback (confidence 0.3) therefore always
             lands in review.
  terminal   DISPATCHED / MERGED via the Dispatcher, or PENDING_REVIEW /
             ESCALATED via the review queue.

Every stage writes one AuditEvent ("dispatch" is written by the
Dispatcher itself), plus a final "terminal" event that /metrics counts.
Agents 4 and 5 never raise by contract, so run() only raises if the
store does - that is a real error and is allowed to propagate.
"""

import time
from typing import Literal

from app.agents.dispatcher import Dispatcher
from app.agents.interfaces import ClassificationAgent, CorrelationAgent, OpenTicketPool
from app.schemas.conversation import ConversationState
from app.schemas.pipeline import (
    AuditEvent,
    ClassificationResult,
    CorrelationResult,
    TerminalOutcome,
)
from app.storage.sqlite_store import SqliteStore

GateDecision = Literal["MERGE", "DISPATCH", "REVIEW"]

LOW_CONFIDENCE_REASON = "low_classification_confidence"


def confidence_gate(
    correlation: CorrelationResult | None,
    classification: ClassificationResult,
    threshold: float,
) -> GateDecision:
    """Pure decision rule - see the module docstring."""
    if correlation is not None and correlation.outcome == "DUPLICATE":
        return "MERGE"
    if classification.confidence < threshold:
        return "REVIEW"
    return "DISPATCH"


def _ms_since(start: float) -> float:
    return (time.perf_counter() - start) * 1000


class TicketPipeline:
    def __init__(
        self,
        correlation_agent: CorrelationAgent,
        pool: OpenTicketPool,
        classifier: ClassificationAgent,
        dispatcher: Dispatcher,
        store: SqliteStore,
        confidence_threshold: float,
    ):
        self._correlation_agent = correlation_agent
        self._pool = pool
        self._classifier = classifier
        self._dispatcher = dispatcher
        self._store = store
        self._threshold = confidence_threshold

    async def run(self, state: ConversationState) -> ConversationState:
        await self.correlate(state)
        await self.classify(state)
        if await self.gate(state) == "REVIEW":
            await self.enqueue_review(state, LOW_CONFIDENCE_REASON, "PENDING_REVIEW")
        else:
            await self.dispatch(state)
        return state

    async def escalate(self, state: ConversationState, reason: str) -> ConversationState:
        """Agent 2 gave up (or hit the turn budget): straight to human review."""
        await self.enqueue_review(state, reason, "ESCALATED")
        return state

    # --- individual steps (run() above, and the LangGraph nodes in
    #     app/orchestration/nodes.py, are both built from these) ----------

    async def correlate(self, state: ConversationState) -> CorrelationResult:
        start = time.perf_counter()
        correlation = await self._correlation_agent.check(state, self._pool)
        state.correlation = correlation
        await self._audit(state, "correlate", correlation.outcome, correlation.max_similarity, {
            "duplicate_of": correlation.duplicate_of,
            "cluster_id": correlation.cluster_id,
            "similar_tickets": len(correlation.similar_tickets),
        }, start)
        return correlation

    async def classify(self, state: ConversationState) -> ClassificationResult:
        start = time.perf_counter()
        classification = await self._classifier.classify(state)
        state.classification = classification
        await self._audit(state, "classify", classification.category, classification.confidence, {
            "priority": classification.priority,
            "assignee_group": classification.assignee_group,
            "source": classification.source,
        }, start)
        return classification

    async def gate(self, state: ConversationState) -> GateDecision:
        """Apply (and audit) the confidence gate. Needs classify() first."""
        start = time.perf_counter()
        decision = confidence_gate(state.correlation, state.classification, self._threshold)
        await self._audit(state, "confidence_gate", decision, state.classification.confidence, {
            "threshold": self._threshold,
        }, start)
        return decision

    async def dispatch(self, state: ConversationState) -> ConversationState:
        result = await self._dispatcher.dispatch(state)
        state.ticket = result.ticket
        state.terminal_outcome = result.outcome
        await self._audit_terminal(state)
        return state

    async def enqueue_review(
        self, state: ConversationState, reason: str, outcome: TerminalOutcome
    ) -> None:
        start = time.perf_counter()
        state.review_reason = reason
        state.terminal_outcome = outcome
        review_id = await self._store.enqueue_review(
            state.conversation_id, reason, state.model_dump_json()
        )
        state.review_item_id = review_id
        await self._audit(state, "review_queue", outcome, None, {
            "review_item_id": review_id,
            "reason": reason,
        }, start)
        await self._audit_terminal(state)

    async def _audit_terminal(self, state: ConversationState) -> None:
        await self._store.log_event(AuditEvent(
            conversation_id=state.conversation_id,
            turn=state.turn_count,
            stage="terminal",
            decision=state.terminal_outcome,
            payload={
                "ticket_id": state.ticket.ticket_id if state.ticket else None,
                "review_item_id": state.review_item_id,
            },
        ))

    async def _audit(
        self,
        state: ConversationState,
        stage: str,
        decision: str,
        confidence: float | None,
        payload: dict,
        start: float,
    ) -> None:
        await self._store.log_event(AuditEvent(
            conversation_id=state.conversation_id,
            turn=state.turn_count,
            stage=stage,
            decision=decision,
            confidence=confidence,
            payload=payload,
            latency_ms=_ms_since(start),
        ))
