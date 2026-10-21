"""
Unit tests for app/agents/dispatcher.py. Real SqliteStore on tmp_path,
small fake adapters - no Groq, Gemini or Jira.
"""

import pytest

from app.agents.dispatcher import Dispatcher
from app.config.routing import team_for_category
from app.integrations.base import ExternalRef, ExternalTicketAdapter, NullAdapter
from app.schemas.conversation import ConversationState, Message
from app.schemas.extraction import CategoryPrediction, ExtractedField
from app.schemas.pipeline import ClassificationResult, CorrelationResult
from app.storage.sqlite_store import SqliteStore


@pytest.fixture
def store(tmp_path) -> SqliteStore:
    s = SqliteStore(tmp_path / "dispatch.db")
    s.init_db()
    return s


def make_state(conversation_id: str = "conv-1", correlation: CorrelationResult | None = None) -> ConversationState:
    return ConversationState(
        conversation_id=conversation_id,
        turn_count=1,
        raw_messages=[Message(role="user", content="printer on floor 3 jams every page")],
        extracted_fields={"symptom": ExtractedField(value="paper jam", confidence=0.9)},
        detected_category=CategoryPrediction(value="printer_support", confidence=0.9),
        classification=ClassificationResult(
            category="printer_support", priority="P3",
            assignee_group=team_for_category("printer_support"),
            routing_rationale="Printer jam on a shared printer.", confidence=0.8, source="stub",
        ),
        correlation=correlation,
    )


class RecordingAdapter:
    name = "fake"

    def __init__(self, ref: ExternalRef | None = None, error: Exception | None = None):
        self.ref = ref
        self.error = error
        self.created = []
        self.linked = []

    async def create(self, ticket):
        self.created.append(ticket)
        if self.error:
            raise self.error
        return self.ref

    async def link_duplicate(self, ticket, existing):
        self.linked.append((ticket, existing))


def test_adapters_satisfy_protocol():
    assert isinstance(NullAdapter(), ExternalTicketAdapter)
    assert isinstance(RecordingAdapter(), ExternalTicketAdapter)


async def test_null_adapter_dispatches_open_ticket_with_one_audit_row(store):
    result = await Dispatcher(store, NullAdapter()).dispatch(make_state())

    assert result.outcome == "DISPATCHED"
    assert result.external_error is None
    # created_at defaults to now(), so only the format is fixed, not the year
    assert result.ticket.ticket_id.startswith("TRO-") and result.ticket.ticket_id.endswith("-00001")
    assert result.ticket.status == "OPEN"
    assert result.ticket.external_key is None
    assert await store.get_ticket(result.ticket.ticket_id) == result.ticket

    audit = await store.fetch_audit("conv-1")
    assert len(audit) == 1
    assert audit[0].stage == "dispatch"
    assert audit[0].decision == "DISPATCHED"
    assert audit[0].confidence == 0.8
    assert audit[0].payload == {
        "ticket_id": result.ticket.ticket_id, "adapter": "local",
        "external_key": None, "external_error": None,
    }
    assert audit[0].latency_ms >= 0


async def test_external_ref_is_persisted(store):
    adapter = RecordingAdapter(ref=ExternalRef("TRO-12", "https://x.atlassian.net/browse/TRO-12"))
    result = await Dispatcher(store, adapter).dispatch(make_state())

    assert result.ticket.external_key == "TRO-12"
    stored = await store.get_ticket(result.ticket.ticket_id)
    assert stored.external_key == "TRO-12"
    assert stored.external_url == "https://x.atlassian.net/browse/TRO-12"
    assert (await store.fetch_audit("conv-1"))[0].payload["external_key"] == "TRO-12"


async def test_adapter_failure_still_saves_ticket_and_audits_error(store):
    adapter = RecordingAdapter(error=ConnectionError("jira unreachable"))
    result = await Dispatcher(store, adapter).dispatch(make_state())

    assert result.outcome == "DISPATCHED"
    assert result.external_error == "ConnectionError: jira unreachable"
    stored = await store.get_ticket(result.ticket.ticket_id)
    assert stored is not None and stored.status == "OPEN"
    assert stored.external_key is None

    audit = await store.fetch_audit("conv-1")
    assert len(audit) == 1
    assert audit[0].payload["external_error"] == "ConnectionError: jira unreachable"


async def test_duplicate_of_existing_ticket_is_merged_and_linked(store):
    original = (await Dispatcher(store, NullAdapter()).dispatch(make_state("conv-orig"))).ticket

    adapter = RecordingAdapter(ref=ExternalRef("SHOULD-NOT", "u"))
    corr = CorrelationResult(outcome="DUPLICATE", duplicate_of=original.ticket_id, max_similarity=0.95)
    result = await Dispatcher(store, adapter).dispatch(make_state("conv-dup", corr))

    assert result.outcome == "MERGED"
    assert result.external_error is None
    assert result.ticket.status == "MERGED"
    assert result.ticket.duplicate_of == original.ticket_id
    assert adapter.created == []
    assert len(adapter.linked) == 1
    linked_ticket, existing = adapter.linked[0]
    assert linked_ticket.ticket_id == result.ticket.ticket_id
    assert existing == original

    # merged tickets don't enter Agent 4's open pool
    assert (await store.get_ticket(result.ticket.ticket_id)).status == "MERGED"
    assert (await store.fetch_audit("conv-dup"))[0].decision == "MERGED"


async def test_duplicate_of_missing_ticket_merges_without_linking(store):
    adapter = RecordingAdapter()
    corr = CorrelationResult(outcome="DUPLICATE", duplicate_of="TRO-2026-99999", max_similarity=0.9)
    result = await Dispatcher(store, adapter).dispatch(make_state(correlation=corr))

    assert result.outcome == "MERGED"
    assert result.external_error is None
    assert result.ticket.status == "MERGED"
    assert adapter.linked == []
    assert adapter.created == []
    assert len(await store.fetch_audit("conv-1")) == 1
