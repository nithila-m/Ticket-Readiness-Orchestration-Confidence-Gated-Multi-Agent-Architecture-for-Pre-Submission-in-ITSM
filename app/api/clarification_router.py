"""
Conversational endpoint for Agent 2 (Adaptive Clarifier).

Does not touch /analyze - that remains a standalone Agent 1 evaluation
endpoint. This router owns the multi-turn conversational loop, backed by
ClarificationService (Agent 1 rerun + Agent 2 decision + persistence).
All dependencies are wired through app/dependencies.py, reusing the same
Gemini-backed extraction service /analyze uses rather than constructing
a second one.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.dependencies import get_clarification_service
from app.providers.exceptions import LLMProviderError
from app.services.clarification_service import ClarificationService

router = APIRouter(prefix="/conversations", tags=["clarification"])


class ConversationMessageRequest(BaseModel):
    message: str = Field(min_length=1)


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
    )