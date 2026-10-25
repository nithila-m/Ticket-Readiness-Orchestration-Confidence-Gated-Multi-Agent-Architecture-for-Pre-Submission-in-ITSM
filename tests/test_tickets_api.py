"""
API tests for the ticket / audit / review / metrics endpoints and the
extended conversation response. Real FastAPI app via TestClient, real
ClarificationService + TicketPipeline + Dispatcher(NullAdapter) on a
tmp_path SqliteStore; Agents 1-5 are fakes. No Gemini, Groq, Jira or
embedding model, and the real tro.db is never touched.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.agents.dispatcher import Dispatcher
from app.agents.kb_deflection_agent import KBRetrievalResult
from app.config.routing import team_for_category
from app.config.settings import settings
from app.dependencies import (
    get_classification_agent,
    get_clarification_service,
    get_conversation_repository,
    get_dispatcher,
    get_store,
)
from app.integrations.base import NullAdapter
from app.main import app
from app.orchestration import GraphConversationService
from app.providers.exceptions import LLMProviderError
from app.repositories.in_memory_conversation_repository import InMemoryConversationRepository
from app.schemas.analysis import AnalysisResult
from app.schemas.clarification import ClarificationDecision
from app.schemas.extraction import CategoryPrediction, ExtractedField
from app.schemas.pipeline import ClassificationResult, CorrelationResult
from app.services.clarification_service import ClarificationService
from app.services.ticket_pipeline import TicketPipeline
from app.storage.sqlite_store import SqliteStore


def _analysis() -> AnalysisResult:
    return AnalysisResult(
        category=CategoryPrediction(value="wifi_internet", confidence=0.9),
        extracted_fields={"symptom_type": ExtractedField(value="drops every few minutes", confidence=0.9)},
        completeness_score=0.9,
        missing_or_uncertain_fields=[],
    )


def _decision(action: str) -> ClarificationDecision:
    return ClarificationDecision(
        action=action, reasoning=f"{action} reasoning.", expected_information_gain=0.0,
        priority="low", confidence=0.9,
    )


def _kb(outcome: str) -> KBRetrievalResult:
    strong = outcome == "STRONG_MATCH"
    return KBRetrievalResult(
        outcome=outcome, similarity_score=0.9 if strong else 0.1, articles_checked=["KB0302"],
        matched_kb_id="KB0302" if strong else None,
        matched_title="Wi-Fi drops after sleep" if strong else None,
        offered_resolution="Disable fast startup." if strong else None,
    )


def _classification(confidence: float) -> ClassificationResult:
    return ClassificationResult(
        category="wifi_internet", priority="P3", assignee_group=team_for_category("wifi_internet"),
        routing_rationale="Wifi drops on several devices.", confidence=confidence, source="stub",
    )


class FakeCorrelation:
    async def check(self, state, pool):
        return CorrelationResult(outcome="NO_MATCH")


class FakeClassifier:
    def __init__(self, confidence: float = 0.85):
        self.confidence = confidence
        self.calls = 0

    async def classify(self, state):
        self.calls += 1
        return _classification(self.confidence)


@pytest.fixture(params=["service", "langgraph"])
def env(request, tmp_path, monkeypatch):
    # Every API test runs against both orchestrators (same constructor/methods).
    service_cls = GraphConversationService if request.param == "langgraph" else ClarificationService
    monkeypatch.setattr(settings, "auto_dispatch_on_ready", True)
    store = SqliteStore(tmp_path / "api.db")
    repo = InMemoryConversationRepository()
    extraction = AsyncMock()
    extraction.analyze.return_value = _analysis()
    clarifier = AsyncMock()
    clarifier.decide.return_value = _decision("READY")
    kb = AsyncMock()
    kb.check.return_value = _kb("NO_MATCH")
    classifier = FakeClassifier()
    dispatcher = Dispatcher(store, NullAdapter())
    pipeline = TicketPipeline(FakeCorrelation(), store, classifier, dispatcher, store, 0.6)
    service = service_cls(extraction, clarifier, repo, kb, ticket_pipeline=pipeline, store=store)

    overrides = {
        get_store: lambda: store,
        get_clarification_service: lambda: service,
        get_dispatcher: lambda: dispatcher,
        get_classification_agent: lambda: classifier,
        get_conversation_repository: lambda: repo,
    }
    app.dependency_overrides.update(overrides)
    try:
        with TestClient(app) as client:  # lifespan runs init_db on the tmp store
            yield SimpleNamespace(
                client=client, store=store, repo=repo, extraction=extraction,
                clarifier=clarifier, kb=kb, classifier=classifier,
            )
    finally:
        for dep in overrides:
            app.dependency_overrides.pop(dep, None)


def post_message(env, conv="conv-1", text="wifi keeps dropping"):
    return env.client.post(f"/conversations/{conv}/messages", json={"message": text})


def make_review(env, conv="conv-r"):
    """Drive one conversation into the review queue (low confidence)."""
    env.classifier.confidence = 0.3
    data = post_message(env, conv).json()
    env.classifier.confidence = 0.85
    assert data["terminal_outcome"] == "PENDING_REVIEW"
    return data["review_item_id"]


# --- POST /conversations/{id}/messages (extended response) -------------------

def test_ready_message_response_includes_pipeline_fields(env):
    r = post_message(env)
    assert r.status_code == 200
    data = r.json()
    assert data["action"] == "READY"
    assert data["terminal_outcome"] == "DISPATCHED"
    assert data["ticket"]["ticket_id"].startswith("TRO-")
    assert data["ticket"]["assignee_group"] == "Network Operations"
    assert data["classification"]["confidence"] == 0.85
    assert data["correlation"]["outcome"] == "NO_MATCH"
    assert data["kb_similarity_score"] == 0.1
    assert data["review_item_id"] is None


def test_low_confidence_message_returns_review_item(env):
    review_id = make_review(env)
    data = env.client.get("/review").json()
    assert [i["id"] for i in data] == [review_id]


def test_message_to_closed_conversation_is_409(env):
    post_message(env)
    r = post_message(env, text="any update?")
    assert r.status_code == 409
    assert "closed" in r.json()["detail"]


def test_message_over_2000_chars_is_422(env):
    assert post_message(env, text="x" * 2001).status_code == 422
    assert post_message(env, conv="c2", text="x" * 2000).status_code == 200


def test_message_provider_failure_is_503(env):
    env.extraction.analyze.side_effect = LLMProviderError("gemini down")
    assert post_message(env).status_code == 503


# --- POST /conversations/{id}/deflection-feedback ----------------------------

def _deflect(env, conv="conv-d"):
    env.kb.check.return_value = _kb("STRONG_MATCH")
    data = post_message(env, conv).json()
    env.kb.check.return_value = _kb("NO_MATCH")
    assert data["action"] == "DEFLECTED" and data["terminal_outcome"] is None


def test_feedback_resolved_returns_self_resolved(env):
    _deflect(env)
    r = env.client.post("/conversations/conv-d/deflection-feedback", json={"resolved": True})
    assert r.status_code == 200
    data = r.json()
    assert data["terminal_outcome"] == "SELF_RESOLVED"
    assert data["action"] == "DEFLECTED"
    assert data["ticket"] is None


def test_feedback_not_resolved_creates_ticket(env):
    _deflect(env)
    data = env.client.post("/conversations/conv-d/deflection-feedback", json={"resolved": False}).json()
    assert data["terminal_outcome"] == "DISPATCHED"
    assert data["ticket"]["ticket_id"].startswith("TRO-")


def test_feedback_error_cases(env):
    url = "/conversations/{}/deflection-feedback"
    assert env.client.post(url.format("unknown"), json={"resolved": True}).status_code == 404

    env.clarifier.decide.return_value = _decision("ESCALATE")
    post_message(env, "conv-esc")  # not deflected (and already closed)
    assert env.client.post(url.format("conv-esc"), json={"resolved": True}).status_code == 409

    _deflect(env)
    assert env.client.post(url.format("conv-d"), json={}).status_code == 422
    env.client.post(url.format("conv-d"), json={"resolved": True})
    assert env.client.post(url.format("conv-d"), json={"resolved": True}).status_code == 409


# --- GET /conversations/{id}/audit -------------------------------------------

def test_audit_trail_in_order(env):
    post_message(env)
    events = env.client.get("/conversations/conv-1/audit").json()
    assert [e["stage"] for e in events] == [
        "extract", "kb", "clarify", "correlate", "classify", "confidence_gate", "dispatch", "terminal",
    ]
    assert all("latency_ms" in e for e in events)
    assert env.client.get("/conversations/nobody/audit").json() == []


# --- /tickets ------------------------------------------------------------------

def test_list_and_get_tickets(env):
    a = post_message(env, "c1").json()["ticket"]["ticket_id"]
    b = post_message(env, "c2").json()["ticket"]["ticket_id"]

    listed = env.client.get("/tickets").json()
    assert [t["ticket_id"] for t in listed] == [b, a]  # newest first
    assert len(env.client.get("/tickets", params={"limit": 1}).json()) == 1
    assert env.client.get("/tickets", params={"status": "MERGED"}).json() == []
    assert env.client.get("/tickets", params={"status": "BOGUS"}).status_code == 422

    got = env.client.get(f"/tickets/{a}")
    assert got.status_code == 200 and got.json()["ticket_id"] == a
    assert env.client.get("/tickets/TRO-2026-99999").status_code == 404


# --- /review -----------------------------------------------------------------

def test_list_reviews_includes_context(env):
    review_id = make_review(env)
    [item] = env.client.get("/review").json()
    assert item["id"] == review_id
    assert item["reason"] == "low_classification_confidence"
    assert item["user_messages"] == ["wifi keeps dropping"]
    assert item["classification"]["confidence"] == 0.3
    assert env.client.get("/review", params={"status": "APPROVED"}).json() == []
    assert len(env.client.get("/review", params={"status": "ALL"}).json()) == 1
    assert env.client.get("/review", params={"status": "NOPE"}).status_code == 422


def test_approve_with_overrides_dispatches_and_resolves(env):
    review_id = make_review(env)
    r = env.client.post(f"/review/{review_id}/approve",
                        json={"priority": "P1", "assignee_group": "Desktop Support"})
    assert r.status_code == 200
    ticket = r.json()
    assert ticket["priority"] == "P1"
    assert ticket["assignee_group"] == "Desktop Support"
    assert "Reviewer override" in ticket["routing_rationale"]
    assert env.client.get(f"/tickets/{ticket['ticket_id']}").status_code == 200

    assert env.client.get("/review", params={"status": "APPROVED"}).json()[0]["id"] == review_id
    live = env.client.get("/conversations/conv-r/audit").json()
    assert [e["stage"] for e in live][-2:] == ["review_decision", "terminal"]
    live_state = asyncio.run(env.repo.get("conv-r"))  # sync test: no loop running here
    assert live_state.terminal_outcome == "DISPATCHED"
    assert live_state.ticket.ticket_id == ticket["ticket_id"]

    again = env.client.post(f"/review/{review_id}/approve", json={})
    assert again.status_code == 409


def test_approve_without_body_keeps_agent5_values(env):
    review_id = make_review(env)
    r = env.client.post(f"/review/{review_id}/approve")
    assert r.status_code == 200
    assert r.json()["priority"] == "P3"
    assert "Reviewer override" not in r.json()["routing_rationale"]


def test_approve_invalid_team_is_422_and_leaves_review_pending(env):
    review_id = make_review(env)
    r = env.client.post(f"/review/{review_id}/approve", json={"assignee_group": "Printer Wizards"})
    assert r.status_code == 422
    assert env.client.get("/review").json()[0]["id"] == review_id
    assert env.client.get("/tickets").json() == []
    assert env.client.post(f"/review/{review_id}/approve", json={"priority": "P9"}).status_code == 422


def test_approve_escalated_review_classifies_first(env):
    env.clarifier.decide.return_value = _decision("ESCALATE")
    data = post_message(env, "conv-esc").json()
    assert data["terminal_outcome"] == "ESCALATED" and data["classification"] is None
    calls_before = env.classifier.calls

    r = env.client.post(f"/review/{data['review_item_id']}/approve", json={})

    assert r.status_code == 200
    assert env.classifier.calls == calls_before + 1
    assert r.json()["category"] == "wifi_internet"


def test_approve_and_reject_unknown_review_is_404(env):
    assert env.client.post("/review/999/approve", json={}).status_code == 404
    assert env.client.post("/review/999/reject").status_code == 404


def test_reject_resolves_review(env):
    review_id = make_review(env)
    r = env.client.post(f"/review/{review_id}/reject")
    assert r.status_code == 200 and r.json()["status"] == "REJECTED"
    assert env.client.get("/review").json() == []
    assert env.client.get("/tickets").json() == []
    assert env.client.post(f"/review/{review_id}/reject").status_code == 409


# --- /metrics ----------------------------------------------------------------

def test_metrics_empty(env):
    data = env.client.get("/metrics").json()
    assert data["conversations_completed"] == 0
    assert data["deflection_rate"] is None
    assert data["latency_by_stage"] == {}


def test_metrics_after_mixed_outcomes(env):
    post_message(env, "c-dispatch")                                 # DISPATCHED
    review_id = make_review(env, "c-review")                        # PENDING_REVIEW
    _deflect(env, "c-kb")
    env.client.post("/conversations/c-kb/deflection-feedback", json={"resolved": True})  # SELF_RESOLVED
    env.client.post(f"/review/{review_id}/approve", json={})       # later approval

    data = env.client.get("/metrics").json()
    assert data["conversations_completed"] == 3
    assert data["conversations_by_outcome"] == {"DISPATCHED": 1, "PENDING_REVIEW": 1, "SELF_RESOLVED": 1}
    assert data["deflection_rate"] == pytest.approx(1 / 3)
    assert data["review_rate"] == pytest.approx(1 / 3)
    assert data["escalation_rate"] == 0
    assert data["tickets_by_status"] == {"OPEN": 2}
    assert data["reviews_by_status"] == {"APPROVED": 1}
    stages = data["latency_by_stage"]
    assert {"extract", "kb", "clarify", "classify", "dispatch"} <= set(stages)
    assert "terminal" not in stages
    assert stages["extract"]["count"] == 3
    assert stages["extract"]["p95_ms"] >= 0
