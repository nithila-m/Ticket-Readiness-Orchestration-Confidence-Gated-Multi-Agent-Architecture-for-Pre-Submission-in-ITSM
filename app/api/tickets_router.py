"""
Ticket, audit, review-queue and metrics endpoints (Agents 4-6 + reviewer).

The conversational loop itself stays in clarification_router.py; this
router adds what happens around it: the user's answer to a KB deflection,
the audit trail, the SQLite ticket store, the human review queue, and
aggregate metrics. All collaborators come from app/dependencies.py so
tests swap them with app.dependency_overrides.
"""

from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from pydantic import BaseModel, ValidationError

from app.agents.dispatcher import Dispatcher
from app.agents.interfaces import ClassificationAgent
from app.api.clarification_router import ConversationMessageResponse, build_conversation_response
from app.dependencies import (
    get_classification_agent,
    get_clarification_service,
    get_conversation_repository,
    get_dispatcher,
    get_store,
)
from app.providers.exceptions import LLMProviderError
from app.repositories.base import ConversationRepository
from app.schemas.conversation import ConversationState
from app.schemas.pipeline import (
    AuditEvent,
    ClassificationResult,
    Priority,
    ReviewStatus,
    TicketRecord,
    TicketStatus,
)
from app.services.clarification_service import (
    ClarificationService,
    ConversationNotFoundError,
    ConversationStateError,
)
from app.storage.sqlite_store import SqliteStore

router = APIRouter(tags=["tickets"])


# --- request / response models ------------------------------------------------

class DeflectionFeedbackRequest(BaseModel):
    resolved: bool


class ReviewApproveRequest(BaseModel):
    """Optional reviewer overrides; anything omitted keeps Agent 5's value."""
    priority: Priority | None = None
    assignee_group: str | None = None


class ReviewItemResponse(BaseModel):
    """A review row plus just enough of the state snapshot to decide on it."""
    id: int
    conversation_id: str
    reason: str
    status: ReviewStatus
    created_at: datetime
    resolved_at: datetime | None
    user_messages: list[str]
    detected_category: str | None
    completeness_score: float
    classification: ClassificationResult | None


class StageLatency(BaseModel):
    count: int
    avg_ms: float
    p95_ms: float


class MetricsResponse(BaseModel):
    conversations_completed: int
    conversations_by_outcome: dict[str, int]
    deflection_rate: float | None
    escalation_rate: float | None
    review_rate: float | None
    tickets_by_status: dict[str, int]
    reviews_by_status: dict[str, int]
    latency_by_stage: dict[str, StageLatency]


def _review_response(item) -> ReviewItemResponse:
    state = ConversationState.model_validate_json(item.state_json)
    return ReviewItemResponse(
        id=item.id,
        conversation_id=item.conversation_id,
        reason=item.reason,
        status=item.status,
        created_at=item.created_at,
        resolved_at=item.resolved_at,
        user_messages=[m.content for m in state.raw_messages if m.role == "user"],
        detected_category=state.detected_category.value if state.detected_category else None,
        completeness_score=state.completeness_score,
        classification=state.classification,
    )


async def _pending_review(store: SqliteStore, review_id: int):
    item = await store.get_review(review_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"review {review_id} not found")
    if item.status != "PENDING":
        raise HTTPException(status_code=409, detail=f"review {review_id} is already {item.status}")
    return item


async def _log_review_decision(store: SqliteStore, state: ConversationState, decision: str, payload: dict):
    await store.log_event(AuditEvent(
        conversation_id=state.conversation_id, turn=state.turn_count,
        stage="review_decision", decision=decision, payload=payload,
    ))


# --- conversations -------------------------------------------------------------

@router.post(
    "/conversations/{conversation_id}/deflection-feedback",
    response_model=ConversationMessageResponse,
)
async def deflection_feedback(
    conversation_id: str,
    body: DeflectionFeedbackRequest,
    service: ClarificationService = Depends(get_clarification_service),
) -> ConversationMessageResponse:
    try:
        state = await service.handle_deflection_feedback(conversation_id, body.resolved)
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ConversationStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except LLMProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return build_conversation_response(state, state.clarification_log[-1].decision)


@router.get("/conversations/{conversation_id}/audit", response_model=list[AuditEvent])
async def conversation_audit(
    conversation_id: str, store: SqliteStore = Depends(get_store)
) -> list[AuditEvent]:
    """Every audit row for the conversation, in the order written (empty if unknown)."""
    return await store.fetch_audit(conversation_id)


# --- tickets -----------------------------------------------------------------

@router.get("/tickets", response_model=list[TicketRecord])
async def list_tickets(
    status: TicketStatus | None = None,
    limit: int = Query(50, ge=1, le=500),
    store: SqliteStore = Depends(get_store),
) -> list[TicketRecord]:
    return await store.list_tickets(status, limit)


@router.get("/tickets/{ticket_id}", response_model=TicketRecord)
async def get_ticket(ticket_id: str, store: SqliteStore = Depends(get_store)) -> TicketRecord:
    ticket = await store.get_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail=f"ticket {ticket_id} not found")
    return ticket


# --- review queue ------------------------------------------------------------

@router.get("/review", response_model=list[ReviewItemResponse])
async def list_reviews(
    status: ReviewStatus | Literal["ALL"] = "PENDING",
    store: SqliteStore = Depends(get_store),
) -> list[ReviewItemResponse]:
    items = await store.list_reviews(None if status == "ALL" else status)
    return [_review_response(i) for i in items]


@router.post("/review/{review_id}/approve", response_model=TicketRecord)
async def approve_review(
    review_id: int,
    body: ReviewApproveRequest = Body(default_factory=ReviewApproveRequest),
    store: SqliteStore = Depends(get_store),
    dispatcher: Dispatcher = Depends(get_dispatcher),
    classifier: ClassificationAgent = Depends(get_classification_agent),
    repository: ConversationRepository = Depends(get_conversation_repository),
) -> TicketRecord:
    """
    Rehydrate the snapshot, apply overrides through a NEW ClassificationResult
    (so the schema validators run - an unknown team is a 422), dispatch,
    then mark the review APPROVED. An escalated conversation was never
    classified, so Agent 5 runs first.
    """
    item = await _pending_review(store, review_id)
    state = ConversationState.model_validate_json(item.state_json)

    base = state.classification or await classifier.classify(state)
    overrides = body.model_dump(exclude_none=True)
    if overrides:
        changes = ", ".join(f"{k}={v}" for k, v in overrides.items())
        overrides["routing_rationale"] = f"{base.routing_rationale} Reviewer override: {changes}."
    try:
        state.classification = ClassificationResult.model_validate({**base.model_dump(), **overrides})
    except ValidationError as exc:
        raise HTTPException(
            status_code=422, detail=exc.errors(include_url=False, include_context=False)
        ) from exc

    result = await dispatcher.dispatch(state)
    await store.resolve_review(review_id, "APPROVED")
    await _log_review_decision(store, state, "APPROVED", {
        "review_item_id": review_id, "ticket_id": result.ticket.ticket_id, "overrides": overrides,
    })
    await store.log_event(AuditEvent(
        conversation_id=state.conversation_id, turn=state.turn_count, stage="terminal",
        decision=result.outcome, payload={"ticket_id": result.ticket.ticket_id, "review_item_id": review_id},
    ))

    # Keep the live conversation (if it is still in memory) in step.
    live = await repository.get(state.conversation_id)
    if live is not None:
        live.classification = state.classification
        live.ticket = result.ticket
        live.terminal_outcome = result.outcome
        await repository.save(live)
    return result.ticket


@router.post("/review/{review_id}/reject", response_model=ReviewItemResponse)
async def reject_review(review_id: int, store: SqliteStore = Depends(get_store)) -> ReviewItemResponse:
    item = await _pending_review(store, review_id)
    resolved = await store.resolve_review(review_id, "REJECTED")
    state = ConversationState.model_validate_json(item.state_json)
    await _log_review_decision(store, state, "REJECTED", {"review_item_id": review_id})
    return _review_response(resolved)


# --- metrics -------------------------------------------------------------------

@router.get("/metrics", response_model=MetricsResponse)
async def metrics(store: SqliteStore = Depends(get_store)) -> MetricsResponse:
    return MetricsResponse(**await store.metrics())
