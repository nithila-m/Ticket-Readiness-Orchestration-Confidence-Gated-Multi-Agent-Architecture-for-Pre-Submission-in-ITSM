"""
Suspected-duplicate confirmation, run against BOTH orchestrators
(ClarificationService and the LangGraph GraphConversationService).
Real TicketPipeline + Dispatcher on a tmp_path SqliteStore; Agents 1-5 are
fakes - no LLM, no embedding model.
"""

from unittest.mock import AsyncMock

import pytest

from app.agents.dispatcher import Dispatcher
from app.agents.kb_deflection_agent import KBRetrievalResult
from app.config.routing import team_for_category
from app.config.settings import settings
from app.integrations.base import NullAdapter
from app.orchestration import GraphConversationService
from app.repositories.in_memory_conversation_repository import InMemoryConversationRepository
from app.schemas.analysis import AnalysisResult
from app.schemas.clarification import ClarificationDecision
from app.schemas.conversation import ConversationState
from app.schemas.extraction import CategoryPrediction, ExtractedField
from app.schemas.pipeline import (
    ClassificationResult,
    CorrelationResult,
    SimilarTicket,
    TicketRecord,
)
from app.services.clarification_service import (
    AWAITING_DUPLICATE_MESSAGE,
    ClarificationService,
    ConversationClosedError,
    ConversationNotFoundError,
    ConversationStateError,
)
from app.services.ticket_pipeline import TicketPipeline
from app.storage.sqlite_store import SqliteStore

TURN_STAGES = ["extract", "kb", "clarify"]


class SpyAdapter(NullAdapter):
    def __init__(self):
        self.linked: list[tuple[str, str]] = []
        self.created: list[str] = []

    async def create(self, ticket):
        self.created.append(ticket.ticket_id)
        return None

    async def link_duplicate(self, ticket, existing):
        self.linked.append((ticket.ticket_id, existing.ticket_id))


class DuplicateCorrelation:
    def __init__(self, candidate_id: str):
        self.candidate_id = candidate_id

    async def check(self, state, pool):
        return CorrelationResult(
            outcome="DUPLICATE", duplicate_of=self.candidate_id, max_similarity=0.97,
            similar_tickets=[
                SimilarTicket(ticket_id=self.candidate_id, similarity=0.97),
                SimilarTicket(ticket_id="TRO-OTHER", similarity=0.61),
            ],
        )


class FakeClassifier:
    async def classify(self, state):
        return ClassificationResult(
            category="wifi_internet", priority="P3", assignee_group=team_for_category("wifi_internet"),
            routing_rationale="Wifi drops.", confidence=0.85, source="stub",
        )


def _decision(action="READY"):
    return ClarificationDecision(
        action=action, reasoning=f"{action}.", expected_information_gain=0.0,
        question=None, priority="low", confidence=0.8,
    )


def _kb(outcome="NO_MATCH"):
    strong = outcome == "STRONG_MATCH"
    return KBRetrievalResult(
        outcome=outcome, similarity_score=0.9 if strong else 0.1, articles_checked=["KB0302"],
        matched_kb_id="KB0302" if strong else None,
        matched_title="Wi-Fi drops after sleep" if strong else None,
        offered_resolution="Disable fast startup." if strong else None,
    )


@pytest.fixture(autouse=True)
def auto_dispatch(monkeypatch):
    monkeypatch.setattr(settings, "auto_dispatch_on_ready", True)


@pytest.fixture(params=["service", "langgraph"])
def service_cls(request):
    return ClarificationService if request.param == "service" else GraphConversationService


@pytest.fixture
def store(tmp_path):
    s = SqliteStore(tmp_path / "dup.db")
    s.init_db()
    return s


@pytest.fixture
async def candidate(store):
    return await store.insert_ticket(TicketRecord(
        conversation_id="older", title="[Wifi/Internet Support] laptop drops wifi",
        description="private details of another user", category="wifi_internet",
        priority="P3", assignee_group=team_for_category("wifi_internet"),
        routing_rationale="r", confidence_score=0.9,
    ))


def make_service(service_cls, store, candidate, kb="NO_MATCH"):
    extraction = AsyncMock()
    extraction.analyze.return_value = AnalysisResult(
        category=CategoryPrediction(value="wifi_internet", confidence=0.9),
        extracted_fields={"symptom_type": ExtractedField(value="drops", confidence=0.9)},
        completeness_score=0.9, missing_or_uncertain_fields=[],
    )
    clarifier = AsyncMock()
    clarifier.decide.return_value = _decision()
    kb_agent = AsyncMock()
    kb_agent.check.return_value = _kb(kb)
    adapter = SpyAdapter()
    pipeline = TicketPipeline(
        DuplicateCorrelation(candidate.ticket_id), store, FakeClassifier(),
        Dispatcher(store, adapter), store, 0.6,
    )
    repository = InMemoryConversationRepository()
    service = service_cls(
        extraction, clarifier, repository, kb_agent, ticket_pipeline=pipeline, store=store,
    )
    return service, adapter, repository


async def stages(store, conv="c1"):
    return [e.stage for e in await store.fetch_audit(conv)]


async def test_duplicate_pauses_with_flag_and_no_ticket(service_cls, store, candidate):
    service, adapter, _ = make_service(service_cls, store, candidate)

    state, decision = await service.handle_message("c1", "laptop wifi drops")

    assert decision.action == "READY"
    assert state.awaiting_duplicate_confirmation
    assert state.terminal_outcome is None and state.ticket is None
    assert state.correlation.duplicate_of == candidate.ticket_id
    assert state.duplicate_candidate.ticket_id == candidate.ticket_id
    assert state.duplicate_candidate.title == candidate.title
    assert await stages(store) == TURN_STAGES + ["correlate"]
    assert adapter.linked == [] and adapter.created == []


async def test_confirmed_duplicate_is_merged_and_linked(service_cls, store, candidate):
    service, adapter, _ = make_service(service_cls, store, candidate)
    await service.handle_message("c1", "laptop wifi drops")

    state = await service.handle_duplicate_feedback("c1", is_duplicate=True)

    assert not state.awaiting_duplicate_confirmation
    assert state.terminal_outcome == "MERGED"
    assert state.ticket.duplicate_of == candidate.ticket_id
    assert adapter.linked == [(state.ticket.ticket_id, candidate.ticket_id)]
    assert await stages(store) == TURN_STAGES + [
        "correlate", "duplicate_confirmation", "classify", "confidence_gate", "dispatch", "terminal",
    ]
    [row] = [e for e in await store.fetch_audit("c1") if e.stage == "duplicate_confirmation"]
    assert row.decision == "CONFIRMED"
    assert row.payload == {"candidate_ticket_id": candidate.ticket_id, "similarity": 0.97}


async def test_rejected_duplicate_keeps_evidence_and_dispatches(service_cls, store, candidate):
    service, adapter, _ = make_service(service_cls, store, candidate)
    await service.handle_message("c1", "laptop wifi drops")

    state = await service.handle_duplicate_feedback("c1", is_duplicate=False)

    assert state.terminal_outcome == "DISPATCHED"
    assert state.correlation.outcome == "NO_MATCH"
    assert state.correlation.duplicate_of is None
    assert state.correlation.max_similarity == 0.97
    assert [s.ticket_id for s in state.correlation.similar_tickets] == [candidate.ticket_id, "TRO-OTHER"]
    assert state.ticket.duplicate_of is None and state.ticket.status == "OPEN"
    assert adapter.linked == [] and adapter.created == [state.ticket.ticket_id]
    rows = await store.fetch_audit("c1")
    assert [e.decision for e in rows if e.stage == "duplicate_confirmation"] == ["REJECTED"]
    assert [e.stage for e in rows].count("correlate") == 1  # cluster check not re-run


async def test_message_while_awaiting_is_rejected(service_cls, store, candidate):
    service, _, _ = make_service(service_cls, store, candidate)
    await service.handle_message("c1", "laptop wifi drops")

    with pytest.raises(ConversationStateError, match=AWAITING_DUPLICATE_MESSAGE):
        await service.handle_message("c1", "hello?")


async def test_duplicate_feedback_when_not_awaiting_is_rejected(service_cls, store, candidate):
    service, _, _ = make_service(service_cls, store, candidate)
    with pytest.raises(ConversationNotFoundError):
        await service.handle_duplicate_feedback("missing", is_duplicate=True)

    await service.handle_message("c1", "laptop wifi drops")
    await service.handle_duplicate_feedback("c1", is_duplicate=False)

    with pytest.raises(ConversationClosedError):  # terminal now
        await service.handle_duplicate_feedback("c1", is_duplicate=True)


async def test_duplicate_feedback_on_open_conversation_is_rejected(service_cls, store, candidate):
    service, _, repository = make_service(service_cls, store, candidate)
    await repository.save(ConversationState(conversation_id="c2"))

    with pytest.raises(ConversationStateError, match="not awaiting"):
        await service.handle_duplicate_feedback("c2", is_duplicate=True)


async def test_deflection_not_resolved_can_pause_on_duplicate(service_cls, store, candidate):
    service, _, _ = make_service(service_cls, store, candidate, kb="STRONG_MATCH")
    await service.handle_message("c1", "wifi drops after sleep")

    state = await service.handle_deflection_feedback("c1", resolved=False)
    assert state.awaiting_duplicate_confirmation

    with pytest.raises(ConversationStateError, match=AWAITING_DUPLICATE_MESSAGE):
        await service.handle_deflection_feedback("c1", resolved=False)

    state = await service.handle_duplicate_feedback("c1", is_duplicate=True)
    assert state.terminal_outcome == "MERGED"
