"""
Orchestrates one conversational turn: append the user's message, rerun
Agent 1 on the full updated transcript, refresh extraction/completeness/
category on state, apply the completeness gate, run Agent 3 and the KB
decision gate when the gate passes, invoke Agent 2, log the decision,
persist.

Deliberately does not catch LLMProviderError from either agent - mirrors
ticket_analysis_service.py's own convention of letting provider failures
propagate to the API layer rather than being silently masked here.

Optional (Agents 4-6): with a TicketPipeline, READY runs the post-READY
pipeline (only when settings.auto_dispatch_on_ready), ESCALATE goes to the
review queue, and DEFLECTED waits for handle_deflection_feedback(). With a
SqliteStore, each turn's stages are written to the audit trail. Without
either, behaviour is exactly as before.
"""

import time

from app.agents.adaptive_clarifier import AdaptiveClarifier, was_forced_escalation
from app.agents.kb_deflection_agent import (
    KBDeflectionAgent,
    build_deflection_decision,
    build_kb_query,
)
from app.config.settings import settings
from app.repositories.base import ConversationRepository
from app.schemas.clarification import ClarificationDecision, ClarificationLogEntry
from app.schemas.conversation import ConversationState, Message
from app.schemas.pipeline import AuditEvent
from app.services.ticket_pipeline import TicketPipeline
from app.services.transcript_formatter import format_transcript_for_extraction
from app.services.ticket_analysis_service import TicketAnalysisService
from app.storage.sqlite_store import SqliteStore

ESCALATION_REASON = "clarification_escalated"


class ConversationNotFoundError(Exception):
    """No conversation with this id exists (API: 404)."""


class ConversationStateError(Exception):
    """The request is not valid in the conversation's current state (API: 409)."""


class ConversationClosedError(ConversationStateError):
    """The conversation already has a terminal outcome; it cannot take more messages."""


class ClarificationService:
    def __init__(
        self,
        extraction_service: TicketAnalysisService,
        clarifier: AdaptiveClarifier,
        repository: ConversationRepository,
        kb_agent: KBDeflectionAgent,
        ticket_pipeline: TicketPipeline | None = None,
        store: SqliteStore | None = None,
    ):
        self._extraction_service = extraction_service
        self._clarifier = clarifier
        self._repository = repository
        self._kb_agent = kb_agent
        self._ticket_pipeline = ticket_pipeline
        self._store = store

    async def handle_message(
        self, conversation_id: str, user_message: str
    ) -> tuple[ConversationState, ClarificationDecision]:
        state = await self._repository.get_or_create(conversation_id)
        if state.terminal_outcome is not None:
            raise ConversationClosedError(
                f"conversation {conversation_id!r} is closed ({state.terminal_outcome})"
            )
        turn = state.turn_count

        # 1. Append the user's message.
        state.raw_messages.append(Message(role="user", content=user_message))

        # 2. Rerun Agent 1 on the full updated transcript (not just this message) -
        #    this is what lets contradiction detection (RECHECK) work: Agent 1
        #    re-extracts from scratch each turn rather than merging deltas.
        start = time.perf_counter()
        transcript = format_transcript_for_extraction(state.raw_messages)
        analysis = await self._extraction_service.analyze(transcript)

        # 3. Refresh extraction-derived fields on state.
        state.extracted_fields = analysis.extracted_fields
        state.detected_category = analysis.category
        state.completeness_score = analysis.completeness_score
        state.missing_or_uncertain_fields = analysis.missing_or_uncertain_fields
        await self._audit(
            state, turn, "extract",
            (state.detected_category.value if state.detected_category else None) or "UNKNOWN",
            state.completeness_score,
            {"missing_or_uncertain_fields": state.missing_or_uncertain_fields}, start,
        )

        # 3.5. Completeness Gate + Agent 3 (KB retrieval). Below the
        #      threshold, the request is too vague to trust an embedding
        #      match against - kb_outcome stays None (never checked this
        #      turn) and we go straight to Agent 2, same as before Agent 3
        #      existed.
        if state.completeness_score >= settings.kb_gate_completeness_threshold:
            category = state.detected_category.value if state.detected_category else None
            kb_query = build_kb_query(category, state.extracted_fields)
            start = time.perf_counter()
            kb_result = await self._kb_agent.check(kb_query, category)
            state.kb_outcome = kb_result.outcome
            state.kb_similarity_score = kb_result.similarity_score
            state.kb_articles_checked = kb_result.articles_checked
            state.kb_matched_kb_id = kb_result.matched_kb_id
            state.kb_matched_title = kb_result.matched_title
            state.kb_offered_resolution = kb_result.offered_resolution
            await self._audit(
                state, turn, "kb", kb_result.outcome, kb_result.similarity_score,
                {"matched_kb_id": kb_result.matched_kb_id,
                 "articles_checked": kb_result.articles_checked}, start,
            )

            # 4. KB decision gate: a strong KB match short-circuits straight
            #    to DEFLECTED and Agent 2 never runs this turn - there's
            #    nothing left to clarify if the issue is already resolved
            #    by a KB article. Anything weaker (WEAK_MATCH/NO_MATCH)
            #    proceeds to Agent 2 exactly as before this step existed.
            start = time.perf_counter()
            if kb_result.outcome == "STRONG_MATCH":
                decision = build_deflection_decision(kb_result)
            else:
                decision = await self._clarifier.decide(state)
        else:
            start = time.perf_counter()
            decision = await self._clarifier.decide(state)
        await self._audit(
            state, turn, "clarify", decision.action, decision.confidence,
            {"forced_escalation": was_forced_escalation(decision),
             "affected_fields": decision.affected_fields}, start,
        )

        # 5. If Agent 2 asked a question, fold it into the transcript so the
        #    NEXT turn's Agent 1 rerun can label it correctly (see decision 2
        #    in the M5.6 design notes above).
        if decision.action == "ASK_CLARIFICATION" and decision.question:
            state.raw_messages.append(Message(role="assistant", content=decision.question))

        # 6. Append the audit log entry, increment turn, persist.
        state.clarification_log.append(
            ClarificationLogEntry(
                turn=state.turn_count,
                user_message=user_message,
                decision=decision,
            )
        )

        # 7. Agents 4-6 (optional). Runs before the turn increment so the
        #    pipeline's audit rows carry this turn's number. DEFLECTED sets
        #    nothing here - handle_deflection_feedback() decides.
        if self._ticket_pipeline is not None:
            if decision.action == "READY" and settings.auto_dispatch_on_ready:
                await self._ticket_pipeline.run(state)
            elif decision.action == "ESCALATE":
                await self._ticket_pipeline.escalate(state, ESCALATION_REASON)

        state.turn_count += 1
        await self._repository.save(state)

        return state, decision

    async def handle_deflection_feedback(
        self, conversation_id: str, resolved: bool
    ) -> ConversationState:
        """
        The user's answer to a DEFLECTED turn. Resolved -> SELF_RESOLVED
        (no ticket). Not resolved -> the user still needs help, so run the
        ticket pipeline (when one is configured).
        """
        state = await self._repository.get(conversation_id)
        if state is None:
            raise ConversationNotFoundError(f"conversation {conversation_id!r} not found")
        if state.terminal_outcome is not None:
            raise ConversationClosedError(
                f"conversation {conversation_id!r} is closed ({state.terminal_outcome})"
            )
        if not state.clarification_log or state.clarification_log[-1].decision.action != "DEFLECTED":
            raise ConversationStateError(
                f"conversation {conversation_id!r} was not deflected to a KB article"
            )

        state.kb_feedback = "RESOLVED" if resolved else "NOT_RESOLVED"
        # turn_count was already incremented by the DEFLECTED turn.
        turn = state.turn_count - 1
        await self._audit(
            state, turn, "deflection_feedback", state.kb_feedback, None,
            {"matched_kb_id": state.kb_matched_kb_id}, time.perf_counter(),
        )

        if resolved:
            state.terminal_outcome = "SELF_RESOLVED"
            await self._audit(state, turn, "terminal", "SELF_RESOLVED", None, {}, None)
        elif self._ticket_pipeline is not None:
            await self._ticket_pipeline.run(state)

        await self._repository.save(state)
        return state

    async def _audit(
        self,
        state: ConversationState,
        turn: int,
        stage: str,
        decision: str,
        confidence: float | None,
        payload: dict,
        start: float | None,
    ) -> None:
        if self._store is None:
            return
        await self._store.log_event(AuditEvent(
            conversation_id=state.conversation_id,
            turn=turn,
            stage=stage,
            decision=decision,
            confidence=confidence,
            payload=payload,
            latency_ms=(time.perf_counter() - start) * 1000 if start is not None else 0.0,
        ))