"""scripts/seed_ticket_store.py: seeding into a tmp_path DB (never tro.db)."""

from datetime import datetime, timedelta, timezone

from app.config.category_profiles import VALID_CATEGORIES
from app.storage.sqlite_store import SqliteStore
from scripts.seed_ticket_store import load_seed, seed_tickets

NOW = datetime(2026, 10, 25, 9, 0, tzinfo=timezone.utc)


def test_seed_produces_open_tickets_with_valid_categories(tmp_path):
    store = SqliteStore(tmp_path / "seed.db")
    store.init_db()
    raw = load_seed()

    counts = seed_tickets(store, raw, now=NOW)

    tickets = store.list_tickets_sync(limit=10_000)
    assert len(tickets) == len(raw) == sum(counts.values())
    assert all(t.status == "OPEN" for t in tickets)
    assert all(t.category in VALID_CATEGORIES for t in tickets)
    assert all(t.ticket_id.startswith("TRO-") for t in tickets)  # store-issued, not dataset ids
    assert all(t.created_at <= NOW for t in tickets)


def test_seed_respects_limit(tmp_path):
    store = SqliteStore(tmp_path / "seed.db")
    store.init_db()
    seed_tickets(store, load_seed(), now=NOW, limit=25)
    assert len(store.list_tickets_sync(limit=10_000)) == 25


def test_seeded_tickets_fall_in_cluster_window_right_after_seeding(tmp_path):
    # offsets are resolved against seed time, so tickets with offsets under
    # 6 h are inside Agent 4's incident-cluster window immediately
    store = SqliteStore(tmp_path / "seed.db")
    store.init_db()
    raw = load_seed()
    seed_tickets(store, raw, now=NOW)

    expected = sum(1 for r in raw if r.get("created_at_offset_minutes", 0) >= -6 * 60)
    assert expected > 0
    assert len(store.list_open_sync(NOW - timedelta(hours=6))) == expected
