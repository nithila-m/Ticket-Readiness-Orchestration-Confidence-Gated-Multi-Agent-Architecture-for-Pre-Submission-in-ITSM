"""
Conversational endpoint for Agent 2 (Adaptive Clarifier).

Does not touch /analyze - that remains a standalone Agent 1 evaluation
endpoint. This router owns the multi-turn conversational loop, backed by
ClarificationService (Agent 1 rerun + Agent 2 decision + persistence).
All dependencies are wired through app/dependencies.py, reusing the same
Gemini-backed extraction service /analyze uses rather than constructing
a second one.
"""

from datetime import timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.dependencies import get_clarification_service
from app.providers.exceptions import LLMProviderError
from app.schemas.clarification import ClarificationDecision
from app.schemas.conversation import ConversationState
from app.schemas.pipeline import (
    ClassificationResult,
    CorrelationResult,
    DuplicateCandidate,
    TerminalOutcome,
    TicketRecord,
)
from app.services.clarification_service import ClarificationService, ConversationStateError

router = APIRouter(prefix="/conversations", tags=["clarification"])


class ConversationMessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class ConversationMessageResponse(BaseModel):
    conversation_id: str
    turn: int
    action: str
    question: str | None
    category: str | None
    completeness_score: float
    affected_fields: list[str]
    reasoning: str
    confidence: float
    kb_outcome: str | None = None
    kb_matched_title: str | None = None
    kb_offered_resolution: str | None = None
    # Agents 4-6 + terminal state (all optional; None until the pipeline runs)
    kb_similarity_score: float | None = None
    correlation: CorrelationResult | None = None
    classification: ClassificationResult | None = None
    ticket: TicketRecord | None = None
    terminal_outcome: TerminalOutcome | None = None
    review_item_id: int | None = None
    review_reason: str | None = None
    # Suspected duplicate waiting for POST .../duplicate-feedback. Only the
    # candidate's id, title and created time are exposed (DuplicateCandidate).
    awaiting_duplicate_confirmation: bool = False
    duplicate_candidate: DuplicateCandidate | None = None
    duplicate_question: str | None = None


def duplicate_question(candidate: DuplicateCandidate) -> str:
    """
    The yes/no question. Times are UTC here as a fallback; the frontend
    re-renders created_at in the browser's local time.
    """
    label = f"{candidate.ticket_id} - {candidate.title}" if candidate.title else candidate.ticket_id
    opened = ""
    if candidate.created_at is not None:
        created = candidate.created_at
        created = created if created.tzinfo else created.replace(tzinfo=timezone.utc)
        opened = f" (opened {created.astimezone(timezone.utc):%Y-%m-%d %H:%M} UTC)"
    return f"This looks like an existing ticket: {label}{opened}. Is this the same issue?"


def build_conversation_response(
    state: ConversationState, decision: ClarificationDecision
) -> ConversationMessageResponse:
    """Shared by the message, deflection-feedback and duplicate-feedback endpoints."""
    return ConversationMessageResponse(
        conversation_id=state.conversation_id,
        turn=state.turn_count,
        action=decision.action,
        question=decision.question,
        category=state.detected_category.value if state.detected_category else None,
        completeness_score=state.completeness_score,
        affected_fields=decision.affected_fields,
        reasoning=decision.reasoning,
        confidence=decision.confidence,
        kb_outcome=state.kb_outcome,
        kb_matched_title=state.kb_matched_title,
        kb_offered_resolution=state.kb_offered_resolution,
        kb_similarity_score=state.kb_similarity_score if state.kb_outcome else None,
        correlation=state.correlation,
        classification=state.classification,
        ticket=state.ticket,
        terminal_outcome=state.terminal_outcome,
        review_item_id=state.review_item_id,
        review_reason=state.review_reason,
        awaiting_duplicate_confirmation=state.awaiting_duplicate_confirmation,
        duplicate_candidate=state.duplicate_candidate if state.awaiting_duplicate_confirmation else None,
        duplicate_question=(
            duplicate_question(state.duplicate_candidate)
            if state.awaiting_duplicate_confirmation and state.duplicate_candidate else None
        ),
    )


@router.post("/{conversation_id}/messages", response_model=ConversationMessageResponse)
async def post_message(
        conversation_id: str,
        body: ConversationMessageRequest,
        service: ClarificationService = Depends(get_clarification_service),
    ) -> ConversationMessageResponse:
    try:
        state, decision = await service.handle_message(conversation_id, body.message)
    except LLMProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ConversationStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return build_conversation_response(state, decision)