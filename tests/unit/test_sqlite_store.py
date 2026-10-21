"""
Unit tests for app/storage/sqlite_store.py (Agent 6 ticket store + audit).
Every test gets a throwaway database under pytest's tmp_path - the real
tro.db is never touched.
"""

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.agents.interfaces import OpenTicketPool
from app.schemas.pipeline import AuditEvent, TicketRecord
from app.storage.sqlite_store import SqliteStore, format_ticket_id, to_db_timestamp

NOW = datetime(2026, 10, 21, 8, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path) -> SqliteStore:
    s = SqliteStore(tmp_path / "test_tro.db")
    s.init_db()
    return s


def make_ticket(**overrides) -> TicketRecord:
    data = dict(
        conversation_id="conv-1",
        title="[Wifi/Internet Support] drops every few minutes",
        description="Reported by user:\nwifi keeps dropping in SJT-412",
        category="wifi_internet",
        priority="P2",
        assignee_group="Network Operations",
        routing_rationale="Matched wifi symptom fields.",
        confidence_score=0.87,
        kb_articles_checked=["KB0302"],
        building="SJT",
        created_at=NOW,
    )
    data.update(overrides)
    return TicketRecord(**data)


# --- schema / init -----------------------------------------------------

def test_init_db_is_idempotent_and_enables_wal(store):
    store.init_db()  # second call must not fail
    with sqlite3.connect(store.db_path) as conn:
        mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert mode.lower() == "wal"
    assert {"tickets", "audit_events"} <= tables


def test_init_db_creates_missing_parent_directory(tmp_path):
    s = SqliteStore(tmp_path / "nested" / "dir" / "tro.db")
    s.init_db()
    assert s.db_path.exists()


# --- helpers -------------------------------------------------------------

def test_to_db_timestamp_is_fixed_width_utc_even_with_zero_microseconds():
    ist = timezone(timedelta(hours=5, minutes=30))
    assert to_db_timestamp(datetime(2026, 10, 21, 13, 30, tzinfo=ist)) == "2026-10-21T08:00:00.000000+00:00"
    # naive datetimes are treated as UTC
    assert to_db_timestamp(datetime(2026, 10, 21, 8, 0)) == "2026-10-21T08:00:00.000000+00:00"


def test_format_ticket_id():
    assert format_ticket_id(NOW, 1) == "TRO-2026-00001"
    assert format_ticket_id(NOW, 123) == "TRO-2026-00123"


# --- insert / get ---------------------------------------------------------

def test_insert_assigns_sequential_ticket_ids(store):
    a = store.insert_ticket_sync(make_ticket())
    b = store.insert_ticket_sync(make_ticket(conversation_id="conv-2"))
    assert a.ticket_id == "TRO-2026-00001"
    assert b.ticket_id == "TRO-2026-00002"


def test_insert_ignores_caller_supplied_ticket_id(store):
    saved = store.insert_ticket_sync(make_ticket(ticket_id="TRO-9999-99999"))
    assert saved.ticket_id == "TRO-2026-00001"


def test_get_ticket_round_trips_every_field(store):
    saved = store.insert_ticket_sync(make_ticket(duplicate_of="TRO-2026-00000", incident_cluster_id="cl-1"))
    loaded = store.get_ticket_sync(saved.ticket_id)
    assert loaded == saved


def test_get_ticket_missing_returns_none(store):
    assert store.get_ticket_sync("TRO-2026-99999") is None


# --- list ---------------------------------------------------------------------

def test_list_tickets_newest_first_and_status_filter(store):
    old = store.insert_ticket_sync(make_ticket(created_at=NOW - timedelta(hours=3)))
    new = store.insert_ticket_sync(make_ticket(created_at=NOW))
    merged = store.insert_ticket_sync(make_ticket(status="MERGED"))

    all_ids = [t.ticket_id for t in store.list_tickets_sync()]
    assert all_ids.index(new.ticket_id) < all_ids.index(old.ticket_id)
    assert len(all_ids) == 3

    merged_only = store.list_tickets_sync(status="MERGED")
    assert [t.ticket_id for t in merged_only] == [merged.ticket_id]


def test_list_tickets_respects_limit(store):
    for i in range(5):
        store.insert_ticket_sync(make_ticket(conversation_id=f"c{i}"))
    assert len(store.list_tickets_sync(limit=2)) == 2


def test_list_open_filters_status_window_and_category(store):
    recent = store.insert_ticket_sync(make_ticket(created_at=NOW - timedelta(minutes=30)))
    store.insert_ticket_sync(make_ticket(created_at=NOW - timedelta(hours=5)))         # too old
    store.insert_ticket_sync(make_ticket(status="MERGED"))                              # not OPEN
    printer = store.insert_ticket_sync(make_ticket(category="printer_support"))         # other category

    since = NOW - timedelta(hours=2)
    wifi_open = store.list_open_sync(since, category="wifi_internet")
    assert [t.ticket_id for t in wifi_open] == [recent.ticket_id]

    any_open = {t.ticket_id for t in store.list_open_sync(since)}
    assert any_open == {recent.ticket_id, printer.ticket_id}


def test_list_open_window_boundary_is_inclusive_across_timezones(store):
    t = store.insert_ticket_sync(make_ticket(created_at=NOW))
    ist = timezone(timedelta(hours=5, minutes=30))
    since_same_instant_in_ist = NOW.astimezone(ist)
    assert [x.ticket_id for x in store.list_open_sync(since_same_instant_in_ist)] == [t.ticket_id]


# --- updates -------------------------------------------------------------

def test_update_external_ref_persists_to_record_and_column(store):
    saved = store.insert_ticket_sync(make_ticket())
    updated = store.update_external_ref_sync(saved.ticket_id, "TRO-12", "https://x.atlassian.net/browse/TRO-12")
    assert updated.external_key == "TRO-12"

    reloaded = store.get_ticket_sync(saved.ticket_id)
    assert reloaded.external_url == "https://x.atlassian.net/browse/TRO-12"
    with sqlite3.connect(store.db_path) as conn:
        col = conn.execute("SELECT external_key FROM tickets WHERE ticket_id=?", (saved.ticket_id,)).fetchone()[0]
    assert col == "TRO-12"


def test_update_ticket_status_removes_ticket_from_open_pool(store):
    saved = store.insert_ticket_sync(make_ticket())
    store.update_ticket_status_sync(saved.ticket_id, "CLOSED")
    assert store.get_ticket_sync(saved.ticket_id).status == "CLOSED"
    assert store.list_open_sync(NOW - timedelta(days=1)) == []


def test_update_ticket_status_rejects_invalid_status(store):
    saved = store.insert_ticket_sync(make_ticket())
    with pytest.raises(ValidationError):
        store.update_ticket_status_sync(saved.ticket_id, "RESOLVED")
    assert store.get_ticket_sync(saved.ticket_id).status == "OPEN"  # unchanged


def test_update_missing_ticket_raises_key_error(store):
    with pytest.raises(KeyError):
        store.update_external_ref_sync("TRO-2026-99999", "K", "u")


# --- audit ------------------------------------------------------------------

def test_audit_events_round_trip_in_insertion_order(store):
    for i, stage in enumerate(["extract", "kb", "clarify", "dispatch"]):
        store.log_event_sync(AuditEvent(
            conversation_id="conv-1", turn=i, stage=stage, decision=f"D{i}",
            confidence=0.5 + i / 10, payload={"i": i, "nested": {"ok": True}}, latency_ms=12.5,
        ))
    store.log_event_sync(AuditEvent(conversation_id="other", turn=0, stage="extract", decision="X"))

    events = store.fetch_audit_sync("conv-1")
    assert [e.stage for e in events] == ["extract", "kb", "clarify", "dispatch"]
    assert events[3].payload == {"i": 3, "nested": {"ok": True}}
    assert events[0].latency_ms == 12.5
    assert events[0].created_at.tzinfo is not None


def test_audit_event_with_null_confidence(store):
    store.log_event_sync(AuditEvent(conversation_id="c", turn=0, stage="dispatch", decision="DISPATCHED"))
    assert store.fetch_audit_sync("c")[0].confidence is None


# --- async wrappers + protocol ----------------------------------------------

def test_store_satisfies_open_ticket_pool_protocol(store):
    assert isinstance(store, OpenTicketPool)


async def test_async_wrappers_end_to_end(store):
    saved = await store.insert_ticket(make_ticket())
    assert (await store.get_ticket(saved.ticket_id)) == saved
    assert len(await store.list_tickets()) == 1
    assert len(await store.list_open(NOW - timedelta(hours=1), "wifi_internet")) == 1

    updated = await store.update_external_ref(saved.ticket_id, "TRO-1", "https://x/browse/TRO-1")
    assert updated.external_key == "TRO-1"
    assert (await store.update_ticket_status(saved.ticket_id, "MERGED")).status == "MERGED"

    await store.log_event(AuditEvent(conversation_id="conv-1", turn=0, stage="dispatch", decision="DISPATCHED"))
    assert len(await store.fetch_audit("conv-1")) == 1
