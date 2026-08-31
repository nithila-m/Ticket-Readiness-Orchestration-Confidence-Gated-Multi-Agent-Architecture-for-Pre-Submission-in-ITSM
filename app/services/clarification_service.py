"""
Orchestrates one conversational turn: append the user's message, rerun
Agent 1 on the full updated transcript, refresh extraction/completeness/
category on state, invoke Agent 2, log the decision, persist.

Deliberately does not catch LLMProviderError from either agent - mirrors
ticket_analysis_service.py's own convention of letting provider failures
propagate to the API layer rather than being silently masked here.
"""

from app.agents.adaptive_clarifier import AdaptiveClarifier
from app.repositories.base import ConversationRepository
from app.schemas.clarification import ClarificationDecision, ClarificationLogEntry
from app.schemas.conversation import ConversationState, Message
from app.services.transcript_formatter import format_transcript_for_extraction
from app.services.ticket_analysis_service import TicketAnalysisService


class ClarificationService:
    def __init__(
        self,
        extraction_service: TicketAnalysisService,
        clarifier: AdaptiveClarifier,
        repository: ConversationRepository,
    ):
        self._extraction_service = extraction_service
        self._clarifier = clarifier
        self._repository = repository

    async def handle_message(
        self, conversation_id: str, user_message: str
    ) -> tuple[ConversationState, ClarificationDecision]:
        state = await self._repository.get_or_create(conversation_id)

        # 1. Append the user's message.
        state.raw_messages.append(Message(role="user", content=user_message))

        # 2. Rerun Agent 1 on the full updated transcript (not just this message) -
        #    this is what lets contradiction detection (RECHECK) work: Agent 1
        #    re-extracts from scratch each turn rather than merging deltas.
        transcript = format_transcript_for_extraction(state.raw_messages)
        analysis = await self._extraction_service.analyze(transcript)

        # 3. Refresh extraction-derived fields on state.
        state.extracted_fields = analysis.extracted_fields
        state.detected_category = analysis.category
        state.completeness_score = analysis.completeness_score
        state.missing_or_uncertain_fields = analysis.missing_or_uncertain_fields

        # 4. Invoke Agent 2 (includes the turn-budget safeguard internally).
        decision = await self._clarifier.decide(state)

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
        state.turn_count += 1
        await self._repository.save(state)

        return state, decision