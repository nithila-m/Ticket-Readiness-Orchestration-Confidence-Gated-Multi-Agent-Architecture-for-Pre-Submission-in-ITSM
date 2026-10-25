"""
API tests for the suspected-duplicate question and
POST /conversations/{id}/duplicate-feedback. Real FastAPI app via
TestClient, real ClarificationService / GraphConversationService +
TicketPipeline + Dispatcher on a tmp_path SqliteStore; Agents 1-5 are
fakes. No Gemini, Groq, Jira or embedding model.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.agents.dispatcher import Dispatcher
from app.agents.kb_deflection_agent import KBRetrievalResult
from app.api.clarification_router import duplicate_question
from app.config.routing import team_for_category
from app.config.settings import settings
from app.dependencies import get_clarification_service, get_conversation_repository, get_store
from app.integrations.base import NullAdapter
from app.main import app
from app.orchestration import GraphConversationService
from app.repositories.in_memory_conversation_repository import InMemoryConversationRepository
from app.schemas.analysis import AnalysisResult
from app.schemas.clarification import ClarificationDecision
from app.schemas.extraction import CategoryPrediction, ExtractedField
from app.schemas.pipeline import (
    ClassificationResult,
    CorrelationResult,
    DuplicateCandidate,
    SimilarTicket,
    TicketRecord,
)
from app.services.clarification_service import ClarificationService
from app.services.ticket_pipeline import TicketPipeline
from app.storage.sqlite_store import SqliteStore

CANDIDATE_CREATED = datetime(2026, 10, 25, 14, 3, tzinfo=timezone.utc)
SECRET = "Room 4B, phone 98765, requester Priya"


class SwitchableCorrelation:
    def __init__(self):
        self.candidate_id: str | None = None

    async def check(self, state, pool):
        if self.candidate_id is None:
            return CorrelationResult(outcome="NO_MATCH")
        return CorrelationResult(
            outcome="DUPLICATE", duplicate_of=self.candidate_id, max_similarity=0.97,
            similar_tickets=[SimilarTicket(ticket_id=self.candidate_id, similarity=0.97)],
        )


class FakeClassifier:
    async def classify(self, state):
        return ClassificationResult(
            category="wifi_internet", priority="P3", assignee_group=team_for_category("wifi_internet"),
            routing_rationale="Wifi drops.", confidence=0.85, source="stub",
        )


@pytest.fixture(params=["service", "langgraph"])
def env(request, tmp_path, monkeypatch):
    service_cls = GraphConversationService if request.param == "langgraph" else ClarificationService
    monkeypatch.setattr(settings, "auto_dispatch_on_ready", True)
    store = SqliteStore(tmp_path / "api.db")
    store.init_db()
    candidate = store.insert_ticket_sync(TicketRecord(
        conversation_id="older", title="[Wifi/Internet Support] laptop drops wifi",
        description=SECRET, category="wifi_internet", priority="P3",
        assignee_group=team_for_category("wifi_internet"), routing_rationale="r",
        confidence_score=0.9, created_at=CANDIDATE_CREATED,
    ))
    repo = InMemoryConversationRepository()
    extraction = AsyncMock()
    extraction.analyze.return_value = AnalysisResult(
        category=CategoryPrediction(value="wifi_internet", confidence=0.9),
        extracted_fields={"symptom_type": ExtractedField(value="drops", confidence=0.9)},
        completeness_score=0.9, missing_or_uncertain_fields=[],
    )
    clarifier = AsyncMock()
    clarifier.decide.return_value = ClarificationDecision(
        action="READY", reasoning="READY.", expected_information_gain=0.0,
        priority="low", confidence=0.9,
    )
    kb = AsyncMock()
    kb.check.return_value = KBRetrievalResult(
        outcome="NO_MATCH", similarity_score=0.1, articles_checked=[],
        matched_kb_id=None, matched_title=None, offered_resolution=None,
    )
    correlation = SwitchableCorrelation()
    correlation.candidate_id = candidate.ticket_id
    pipeline = TicketPipeline(
        correlation, store, FakeClassifier(), Dispatcher(store, NullAdapter()), store, 0.6
    )
    service = service_cls(extraction, clarifier, repo, kb, ticket_pipeline=pipeline, store=store)

    overrides = {
        get_store: lambda: store,
        get_clarification_service: lambda: service,
        get_conversation_repository: lambda: repo,
    }
    app.dependency_overrides.update(overrides)
    try:
        with TestClient(app) as client:
            yield SimpleNamespace(client=client, store=store, candidate=candidate, correlation=correlation)
    finally:
        for dep in overrides:
            app.dependency_overrides.pop(dep, None)


def post_message(env, conv="c1", text="laptop wifi keeps dropping"):
    return env.client.post(f"/conversations/{conv}/messages", json={"message": text})


def feedback(env, conv="c1", **body):
    return env.client.post(f"/conversations/{conv}/duplicate-feedback", json=body)


def test_duplicate_reply_shows_only_id_title_and_time(env):
    r = post_message(env)
    assert r.status_code == 200
    data = r.json()

    assert data["awaiting_duplicate_confirmation"] is True
    assert data["terminal_outcome"] is None and data["ticket"] is None
    assert data["duplicate_candidate"] == {
        "ticket_id": env.candidate.ticket_id,
        "title": env.candidate.title,
        "created_at": "2026-10-25T14:03:00Z",
    }
    assert data["duplicate_question"] == (
        f"This looks like an existing ticket: {env.candidate.ticket_id} - {env.candidate.title} "
        "(opened 2026-10-25 14:03 UTC). Is this the same issue?"
    )
    assert SECRET not in r.text


def test_feedback_true_merges(env):
    post_message(env)
    r = feedback(env, is_duplicate=True)
    assert r.status_code == 200
    data = r.json()
    assert data["terminal_outcome"] == "MERGED"
    assert data["ticket"]["duplicate_of"] == env.candidate.ticket_id
    assert data["awaiting_duplicate_confirmation"] is False
    assert data["duplicate_question"] is None


def test_feedback_false_dispatches_with_no_match_evidence(env):
    post_message(env)
    data = feedback(env, is_duplicate=False).json()
    assert data["terminal_outcome"] == "DISPATCHED"
    assert data["correlation"]["outcome"] == "NO_MATCH"
    assert data["correlation"]["max_similarity"] == 0.97
    assert data["ticket"]["duplicate_of"] is None


def test_message_while_awaiting_is_409(env):
    post_message(env)
    r = post_message(env, text="hello?")
    assert r.status_code == 409
    assert r.json()["detail"] == "Please answer the duplicate question first."


def test_feedback_error_cases(env):
    assert feedback(env, conv="unknown", is_duplicate=True).status_code == 404

    env.correlation.candidate_id = None
    post_message(env, conv="c-closed")  # NO_MATCH -> DISPATCHED (terminal)
    assert feedback(env, conv="c-closed", is_duplicate=True).status_code == 409

    env.correlation.candidate_id = env.candidate.ticket_id
    post_message(env)
    assert feedback(env).status_code == 422
    assert feedback(env, is_duplicate=True).status_code == 200
    assert feedback(env, is_duplicate=True).status_code == 409  # already answered (terminal)


def test_duplicate_question_without_title_or_time():
    assert duplicate_question(DuplicateCandidate(ticket_id="TRO-1")) == (
        "This looks like an existing ticket: TRO-1. Is this the same issue?"
    )
