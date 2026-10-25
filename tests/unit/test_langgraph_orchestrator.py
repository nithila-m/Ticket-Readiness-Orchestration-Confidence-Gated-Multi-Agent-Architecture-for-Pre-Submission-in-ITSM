"""
LangGraph orchestrator (app/orchestration): one test per path through the
graph, asserting the terminal outcome and the audit stage order, plus the
pure router functions. Real TicketPipeline + Dispatcher(NullAdapter) on a
tmp_path SqliteStore; Agents 1-5 are fakes - no LLM, no embedding model.
"""

from unittest.mock import AsyncMock

import pytest

from app.agents.dispatcher import Dispatcher
from app.agents.kb_deflection_agent import KBDeflectionAgent, KBRetrievalResult
from app.config.routing import team_for_category
from app.config.settings import settings
from app.integrations.base import NullAdapter
from app.orchestration import GraphConversationService, routing
from app.repositories.in_memory_conversation_repository import InMemoryConversationRepository
from app.schemas.analysis import AnalysisResult
from app.schemas.clarification import ClarificationDecision
from app.schemas.conversation import ConversationState
from app.schemas.extraction import CategoryPrediction, ExtractedField
from app.schemas.pipeline import ClassificationResult, CorrelationResult
from app.services.ticket_pipeline import TicketPipeline
from app.storage.sqlite_store import SqliteStore

TURN_STAGES = ["extract", "kb", "clarify"]


def _analysis(completeness: float = 0.9) -> AnalysisResult:
    return AnalysisResult(
        category=CategoryPrediction(value="wifi_internet", confidence=0.9),
        extracted_fields={"symptom_type": ExtractedField(value="drops", confidence=0.9)},
        completeness_score=completeness,
        missing_or_uncertain_fields=[],
    )


def _decision(action: str) -> ClarificationDecision:
    return ClarificationDecision(
        action=action, reasoning=f"{action}.", expected_information_gain=0.0,
        question="One device or several?" if action == "ASK_CLARIFICATION" else None,
        priority="low", confidence=0.8,
    )


def _kb(outcome: str) -> KBRetrievalResult:
    strong = outcome == "STRONG_MATCH"
    return KBRetrievalResult(
        outcome=outcome, similarity_score=0.9 if strong else 0.1, articles_checked=["KB0302"],
        matched_kb_id="KB0302" if strong else None,
        matched_title="Wi-Fi drops after sleep" if strong else None,
        offered_resolution="Disable fast startup." if strong else None,
    )


class FakeCorrelation:
    async def check(self, state, pool):
        return CorrelationResult(outcome="NO_MATCH")


class FakeClassifier:
    def __init__(self, confidence=0.85):
        self.confidence = confidence

    async def classify(self, state):
        return ClassificationResult(
            category="wifi_internet", priority="P3", assignee_group=team_for_category("wifi_internet"),
            routing_rationale="Wifi drops.", confidence=self.confidence, source="stub",
        )


@pytest.fixture(autouse=True)
def auto_dispatch(monkeypatch):
    monkeypatch.setattr(settings, "auto_dispatch_on_ready", True)


@pytest.fixture
def store(tmp_path):
    s = SqliteStore(tmp_path / "graph.db")
    s.init_db()
    return s


def make_service(store, decision="READY", kb="NO_MATCH", completeness=0.9, confidence=0.85, kb_agent=None):
    extraction = AsyncMock()
    extraction.analyze.return_value = _analysis(completeness)
    clarifier = AsyncMock()
    clarifier.decide.return_value = _decision(decision)
    if kb_agent is None:
        kb_agent = AsyncMock()
        kb_agent.check.return_value = _kb(kb)
    pipeline = TicketPipeline(
        FakeCorrelation(), store, FakeClassifier(confidence), Dispatcher(store, NullAdapter()), store, 0.6
    )
    service = GraphConversationService(
        extraction, clarifier, InMemoryConversationRepository(), kb_agent,
        ticket_pipeline=pipeline, store=store,
    )
    return service, clarifier


async def stages(store, conv="c1"):
    return [e.stage for e in await store.fetch_audit(conv)]


# --- one test per path ----------------------------------------------------------

async def test_path_ask_question(store):
    service, _ = make_service(store, decision="ASK_CLARIFICATION")
    state, decision = await service.handle_message("c1", "wifi broken")
    assert decision.action == "ASK_CLARIFICATION"
    assert state.terminal_outcome is None
    assert state.raw_messages[-1].role == "assistant"
    assert await stages(store) == TURN_STAGES


async def test_path_below_completeness_gate_skips_kb(store):
    service, _ = make_service(store, decision="ASK_CLARIFICATION", completeness=0.2)
    await service.handle_message("c1", "wifi")
    assert await stages(store) == ["extract", "clarify"]


async def test_path_deflect(store):
    service, clarifier = make_service(store, kb="STRONG_MATCH")
    state, decision = await service.handle_message("c1", "wifi drops after sleep")
    assert decision.action == "DEFLECTED"
    assert state.terminal_outcome is None
    clarifier.decide.assert_not_called()
    assert await stages(store) == TURN_STAGES


async def test_path_feedback_resolved(store):
    service, _ = make_service(store, kb="STRONG_MATCH")
    await service.handle_message("c1", "wifi drops after sleep")
    state = await service.handle_deflection_feedback("c1", resolved=True)
    assert state.terminal_outcome == "SELF_RESOLVED"
    assert await stages(store) == TURN_STAGES + ["deflection_feedback", "terminal"]


async def test_path_feedback_not_resolved_dispatches(store):
    service, _ = make_service(store, kb="STRONG_MATCH")
    await service.handle_message("c1", "wifi drops after sleep")
    state = await service.handle_deflection_feedback("c1", resolved=False)
    assert state.terminal_outcome == "DISPATCHED"
    assert state.ticket is not None
    assert await stages(store) == TURN_STAGES + [
        "deflection_feedback", "correlate", "classify", "confidence_gate", "dispatch", "terminal",
    ]


async def test_path_ready_dispatches(store):
    service, _ = make_service(store, decision="READY")
    state, _ = await service.handle_message("c1", "wifi drops on all devices")
    assert state.terminal_outcome == "DISPATCHED"
    assert await store.get_ticket(state.ticket.ticket_id) == state.ticket
    assert state.turn_count == 1
    assert await stages(store) == TURN_STAGES + [
        "correlate", "classify", "confidence_gate", "dispatch", "terminal",
    ]


async def test_path_ready_without_auto_dispatch_stops(store, monkeypatch):
    monkeypatch.setattr(settings, "auto_dispatch_on_ready", False)
    service, _ = make_service(store, decision="READY")
    state, _ = await service.handle_message("c1", "wifi drops on all devices")
    assert state.terminal_outcome is None
    assert await stages(store) == TURN_STAGES


async def test_path_low_confidence_goes_to_review(store):
    service, _ = make_service(store, decision="READY", confidence=0.3)
    state, _ = await service.handle_message("c1", "wifi drops")
    assert state.terminal_outcome == "PENDING_REVIEW"
    assert state.review_reason == "low_classification_confidence"
    assert [r.id for r in await store.list_reviews()] == [state.review_item_id]
    assert await stages(store) == TURN_STAGES + [
        "correlate", "classify", "confidence_gate", "review_queue", "terminal",
    ]


async def test_path_escalate(store):
    service, _ = make_service(store, decision="ESCALATE")
    state, _ = await service.handle_message("c1", "I don't know")
    assert state.terminal_outcome == "ESCALATED"
    assert state.review_reason == "clarification_escalated"
    assert state.classification is None
    assert await stages(store) == TURN_STAGES + ["review_queue", "terminal"]


async def test_path_kb_exception_still_clarifies(store):
    def broken_retrieve(query_text, category):
        raise RuntimeError("embedding model failed to load")

    service, clarifier = make_service(
        store, decision="ASK_CLARIFICATION", kb_agent=KBDeflectionAgent(retrieve_fn=broken_retrieve)
    )
    state, decision = await service.handle_message("c1", "wifi broken")
    assert state.kb_outcome == "NO_MATCH"
    assert decision.action == "ASK_CLARIFICATION"
    clarifier.decide.assert_called_once()
    assert await stages(store) == TURN_STAGES


async def test_review_approved_entry_dispatches(store):
    service, _ = make_service(store)
    state = ConversationState(
        conversation_id="c-approved",
        detected_category=CategoryPrediction(value="wifi_internet", confidence=0.9),
        classification=await FakeClassifier(0.3).classify(None),
    )
    await service.dispatch_approved(state)
    assert state.terminal_outcome == "DISPATCHED"
    assert await stages(store, "c-approved") == ["dispatch", "terminal"]


def test_graph_has_the_documented_nodes(store):
    service, _ = make_service(store)
    nodes = set(service.graph.get_graph().nodes)
    assert {
        "extract", "kb", "deflect", "clarify", "apply_feedback", "close_resolved",
        "correlate", "confirm_duplicate", "apply_duplicate_feedback",
        "classify", "enqueue_review", "dispatch",
    } <= nodes


# --- pure routers ----------------------------------------------------------------

def _gs(**kw):
    base = {"pipeline_enabled": True, "auto_dispatch": True}
    base.update(kw)
    return base


@pytest.mark.parametrize("entry", ["message", "feedback", "duplicate_feedback", "review_approved"])
def test_entry_router(entry):
    assert routing.entry_router({"entry": entry}) == entry


def test_completeness_gate():
    low = ConversationState(conversation_id="c", completeness_score=settings.kb_gate_completeness_threshold - 0.01)
    high = ConversationState(conversation_id="c", completeness_score=settings.kb_gate_completeness_threshold)
    assert routing.completeness_gate({"conversation": low}) == "clarify"
    assert routing.completeness_gate({"conversation": high}) == "kb"


@pytest.mark.parametrize("outcome, expected", [
    ("STRONG_MATCH", "deflect"), ("WEAK_MATCH", "clarify"), ("NO_MATCH", "clarify"),
])
def test_kb_gate(outcome, expected):
    assert routing.kb_gate({"kb_result": _kb(outcome)}) == expected


@pytest.mark.parametrize("action, overrides, expected", [
    ("READY", {}, "correlate"),
    ("READY", {"auto_dispatch": False}, "end"),
    ("ESCALATE", {"auto_dispatch": False}, "enqueue_review"),
    ("ASK_CLARIFICATION", {}, "end"),
    ("RECHECK", {}, "end"),
    ("READY", {"pipeline_enabled": False}, "end"),
    ("ESCALATE", {"pipeline_enabled": False}, "end"),
])
def test_clarify_router(action, overrides, expected):
    assert routing.clarify_router(_gs(decision=_decision(action), **overrides)) == expected


@pytest.mark.parametrize("resolved, enabled, expected", [
    (True, True, "close_resolved"), (False, True, "correlate"), (False, False, "end"),
])
def test_feedback_router(resolved, enabled, expected):
    assert routing.feedback_router(_gs(resolved=resolved, pipeline_enabled=enabled)) == expected


@pytest.mark.parametrize("gate, expected", [
    ("DISPATCH", "dispatch"), ("MERGE", "dispatch"), ("REVIEW", "enqueue_review"),
])
def test_confidence_gate_router(gate, expected):
    assert routing.confidence_gate({"gate": gate}) == expected


@pytest.mark.parametrize("outcome, expected", [
    ("DUPLICATE", "confirm_duplicate"),
    ("INCIDENT_CLUSTER", "classify"),
    ("NO_MATCH", "classify"),
])
def test_correlation_router(outcome, expected):
    conv = ConversationState(conversation_id="c", correlation=CorrelationResult(outcome=outcome))
    assert routing.correlation_router({"conversation": conv}) == expected
