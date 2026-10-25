"""Unit tests for SqliteStore's review queue (tmp_path database only)."""

import sqlite3

import pytest

from app.schemas.pipeline import TicketRecord
from app.storage.sqlite_store import SqliteStore


@pytest.fixture
def store(tmp_path) -> SqliteStore:
    s = SqliteStore(tmp_path / "review.db")
    s.init_db()
    return s


def test_init_db_creates_review_queue_table(store):
    with sqlite3.connect(store.db_path) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "review_queue" in tables


def test_enqueue_and_get_round_trip(store):
    review_id = store.enqueue_review_sync("conv-1", "low_classification_confidence", '{"a": 1}')
    item = store.get_review_sync(review_id)
    assert item.id == review_id
    assert item.conversation_id == "conv-1"
    assert item.reason == "low_classification_confidence"
    assert item.status == "PENDING"
    assert item.state_json == '{"a": 1}'
    assert item.created_at.tzinfo is not None
    assert item.resolved_at is None


def test_get_missing_review_returns_none(store):
    assert store.get_review_sync(999) is None


def test_list_reviews_pending_oldest_first(store):
    a = store.enqueue_review_sync("c1", "r", "{}")
    b = store.enqueue_review_sync("c2", "r", "{}")
    c = store.enqueue_review_sync("c3", "r", "{}")
    store.resolve_review_sync(b, "REJECTED")

    assert [i.id for i in store.list_reviews_sync()] == [a, c]
    assert [i.id for i in store.list_reviews_sync(status="REJECTED")] == [b]
    assert [i.id for i in store.list_reviews_sync(status=None)] == [a, b, c]


@pytest.mark.parametrize("status", ["APPROVED", "REJECTED"])
def test_resolve_sets_status_and_resolved_at(store, status):
    review_id = store.enqueue_review_sync("c1", "r", "{}")
    item = store.resolve_review_sync(review_id, status)
    assert item.status == status
    assert item.resolved_at is not None and item.resolved_at >= item.created_at


def test_resolve_twice_is_rejected(store):
    review_id = store.enqueue_review_sync("c1", "r", "{}")
    store.resolve_review_sync(review_id, "APPROVED")
    with pytest.raises(ValueError, match="already APPROVED"):
        store.resolve_review_sync(review_id, "REJECTED")


@pytest.mark.parametrize("bad", ["PENDING", "DONE", ""])
def test_resolve_rejects_invalid_status(store, bad):
    review_id = store.enqueue_review_sync("c1", "r", "{}")
    with pytest.raises(ValueError):
        store.resolve_review_sync(review_id, bad)
    assert store.get_review_sync(review_id).status == "PENDING"


def test_resolve_missing_review_raises_key_error(store):
    with pytest.raises(KeyError):
        store.resolve_review_sync(42, "APPROVED")


async def test_async_wrappers(store):
    review_id = await store.enqueue_review("c1", "clarification_escalated", "{}")
    assert (await store.get_review(review_id)).reason == "clarification_escalated"
    assert len(await store.list_reviews()) == 1
    assert (await store.resolve_review(review_id, "APPROVED")).status == "APPROVED"
    assert await store.list_reviews() == []


def test_clear_tickets_restarts_numbering(store):
    def ticket():
        return TicketRecord(
            conversation_id="c", title="t", description="d", category="general", priority="P3",
            assignee_group="Service Desk L1", routing_rationale="r", confidence_score=0.5,
        )

    store.insert_ticket_sync(ticket())
    store.insert_ticket_sync(ticket())
    assert store.clear_tickets_sync() == 2
    assert store.list_tickets_sync() == []
    assert store.insert_ticket_sync(ticket()).ticket_id.endswith("-00001")
