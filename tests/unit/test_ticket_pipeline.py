"""
Unit tests for app/services/ticket_pipeline.py. Real SqliteStore +
Dispatcher(NullAdapter) on tmp_path; Agents 4 and 5 are fakes (or the
real ClassificationAgent on FakeClassificationProvider) - no Groq, no
embedding model.
"""

import pytest

from app.agents.classification_agent import ClassificationAgent
from app.agents.dispatcher import Dispatcher
from app.config.routing import team_for_category
from app.integrations.base import NullAdapter
from app.providers.fake_classification_provider import FakeClassificationProvider
from app.schemas.classification import ClassificationLLMOutput
from app.schemas.conversation import ConversationState, Message
from app.schemas.extraction import CategoryPrediction, ExtractedField
from app.schemas.pipeline import ClassificationResult, CorrelationResult, SimilarTicket, TicketRecord
from app.services.ticket_pipeline import TicketPipeline, confidence_gate
from app.storage.sqlite_store import SqliteStore

THRESHOLD = 0.6


class FakeCorrelation:
    def __init__(self, result: CorrelationResult):
        self.result = result
        self.pools = []

    async def check(self, state, pool):
        self.pools.append(pool)
        return self.result


class FakeClassifier:
    def __init__(self, result: ClassificationResult):
        self.result = result
        self.seen_correlation = "not called"

    async def classify(self, state):
        self.seen_correlation = state.correlation
        return self.result


def classification(confidence: float = 0.85, category: str = "wifi_internet") -> ClassificationResult:
    return ClassificationResult(
        category=category, priority="P3", assignee_group=team_for_category(category),
        routing_rationale="Wifi drops on several devices.", confidence=confidence, source="stub",
    )


def make_state(conversation_id: str = "conv-1") -> ConversationState:
    return ConversationState(
        conversation_id=conversation_id,
        turn_count=1,
        raw_messages=[Message(role="user", content="wifi keeps dropping in SJT")],
        extracted_fields={
            "symptom_type": ExtractedField(value="drops", confidence=0.9),
            "building": ExtractedField(value="SJT", confidence=0.9),
        },
        detected_category=CategoryPrediction(value="wifi_internet", confidence=0.95),
    )


@pytest.fixture
def store(tmp_path) -> SqliteStore:
    s = SqliteStore(tmp_path / "pipeline.db")
    s.init_db()
    return s


def make_pipeline(store, correlation: CorrelationResult, classifier) -> TicketPipeline:
    return TicketPipeline(
        correlation_agent=FakeCorrelation(correlation),
        pool=store,
        classifier=classifier,
        dispatcher=Dispatcher(store, NullAdapter()),
        store=store,
        confidence_threshold=THRESHOLD,
    )


NO_MATCH = CorrelationResult(outcome="NO_MATCH")


# --- pure gate -----------------------------------------------------------------

@pytest.mark.parametrize("outcome, conf, expected", [
    ("NO_MATCH", 0.9, "DISPATCH"),
    ("NO_MATCH", 0.6, "DISPATCH"),        # threshold itself passes
    ("NO_MATCH", 0.59, "REVIEW"),
    ("INCIDENT_CLUSTER", 0.3, "REVIEW"),
    ("DUPLICATE", 0.1, "MERGE"),          # gate does not apply to a merge
])
def test_confidence_gate(outcome, conf, expected):
    corr = CorrelationResult(outcome=outcome, duplicate_of="TRO-2026-00001" if outcome == "DUPLICATE" else None)
    assert confidence_gate(corr, classification(conf), THRESHOLD) == expected


# --- run() ------------------------------------------------------------------

async def test_no_match_high_confidence_dispatches(store):
    classifier = FakeClassifier(classification(0.85))
    pipeline = make_pipeline(store, NO_MATCH, classifier)

    state = await pipeline.run(make_state())

    assert state.terminal_outcome == "DISPATCHED"
    assert state.ticket.status == "OPEN"
    assert await store.get_ticket(state.ticket.ticket_id) == state.ticket
    assert state.correlation == NO_MATCH
    assert state.classification.confidence == 0.85
    assert classifier.seen_correlation == NO_MATCH  # Agent 4 ran before Agent 5
    assert await store.list_reviews() == []


async def test_pool_passed_to_agent4_is_the_store(store):
    pipeline = make_pipeline(store, NO_MATCH, FakeClassifier(classification()))
    await pipeline.run(make_state())
    assert pipeline._correlation_agent.pools == [store]


async def test_low_confidence_goes_to_review_without_ticket(store):
    pipeline = make_pipeline(store, NO_MATCH, FakeClassifier(classification(0.3)))

    state = await pipeline.run(make_state())

    assert state.terminal_outcome == "PENDING_REVIEW"
    assert state.ticket is None
    assert state.review_reason == "low_classification_confidence"
    assert await store.list_tickets() == []
    [item] = await store.list_reviews()
    assert item.id == state.review_item_id
    assert item.reason == "low_classification_confidence"
    snapshot = ConversationState.model_validate_json(item.state_json)
    assert snapshot.classification.confidence == 0.3  # reviewer sees Agent 5's output


async def test_duplicate_is_merged_even_with_low_confidence(store):
    original = await store.insert_ticket(TicketRecord(
        conversation_id="older", title="[Wifi/Internet Support] drops", description="d",
        category="wifi_internet", priority="P3", assignee_group="Network Operations",
        routing_rationale="r", confidence_score=0.9,
    ))
    dup = CorrelationResult(outcome="DUPLICATE", duplicate_of=original.ticket_id, max_similarity=0.97)
    pipeline = make_pipeline(store, dup, FakeClassifier(classification(0.2)))

    state = await pipeline.run(make_state())

    assert state.terminal_outcome == "MERGED"
    assert state.ticket.status == "MERGED"
    assert state.ticket.duplicate_of == original.ticket_id
    assert state.classification is not None  # still classified for the record
    assert await store.list_reviews() == []


async def test_incident_cluster_raises_priority_to_p1(store):
    cluster = CorrelationResult(
        outcome="INCIDENT_CLUSTER", cluster_id="CLU-abc",
        similar_tickets=[SimilarTicket(ticket_id="TRO-2026-00001", similarity=0.8),
                         SimilarTicket(ticket_id="TRO-2026-00002", similarity=0.75)],
        max_similarity=0.8,
    )
    agent5 = ClassificationAgent(provider=FakeClassificationProvider(output=ClassificationLLMOutput(
        category="wifi_internet", priority="P3", confidence=0.9,
        rationale="Wifi outage reported in SJT.", impact_signal="multiple_users",
    )))
    pipeline = make_pipeline(store, cluster, agent5)

    state = await pipeline.run(make_state())

    assert state.terminal_outcome == "DISPATCHED"
    assert state.ticket.priority == "P1"
    assert state.ticket.incident_cluster_id == "CLU-abc"
    assert "P1" in state.ticket.routing_rationale


async def test_rule_fallback_always_goes_to_review(store):
    agent5 = ClassificationAgent(provider=FakeClassificationProvider(error=RuntimeError("groq down")))
    pipeline = make_pipeline(store, NO_MATCH, agent5)

    state = await pipeline.run(make_state())

    assert state.classification.source == "rule"
    assert state.terminal_outcome == "PENDING_REVIEW"


# --- escalate() --------------------------------------------------------------

async def test_escalate_enqueues_review(store):
    pipeline = make_pipeline(store, NO_MATCH, FakeClassifier(classification()))

    state = await pipeline.escalate(make_state(), "clarification_escalated")

    assert state.terminal_outcome == "ESCALATED"
    assert state.review_reason == "clarification_escalated"
    [item] = await store.list_reviews()
    assert item.id == state.review_item_id and item.reason == "clarification_escalated"
    assert await store.list_tickets() == []
    assert [e.stage for e in await store.fetch_audit("conv-1")] == ["review_queue", "terminal"]


# --- audit -----------------------------------------------------------------------

async def test_dispatch_audit_stages_in_order_with_latency(store):
    pipeline = make_pipeline(store, NO_MATCH, FakeClassifier(classification(0.85)))
    await pipeline.run(make_state())

    events = await store.fetch_audit("conv-1")
    assert [e.stage for e in events] == ["correlate", "classify", "confidence_gate", "dispatch", "terminal"]
    assert [e.decision for e in events] == ["NO_MATCH", "wifi_internet", "DISPATCH", "DISPATCHED", "DISPATCHED"]
    assert all(e.turn == 1 for e in events)
    assert all(e.latency_ms >= 0 for e in events)
    assert events[2].payload == {"threshold": THRESHOLD}


async def test_review_audit_stages_in_order(store):
    pipeline = make_pipeline(store, NO_MATCH, FakeClassifier(classification(0.3)))
    state = await pipeline.run(make_state())

    events = await store.fetch_audit("conv-1")
    assert [e.stage for e in events] == ["correlate", "classify", "confidence_gate", "review_queue", "terminal"]
    assert events[3].payload["review_item_id"] == state.review_item_id
    assert events[4].decision == "PENDING_REVIEW"
