"""JsonOpenTicketPool unit tests (temporary JSON files, plus one load of the bundled dataset)."""

import json
from datetime import datetime, timedelta, timezone

from app.agents.seed_ticket_pool import SEED_RATIONALE, JsonOpenTicketPool
from app.config.routing import team_for_category

NOW = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)


def _seed_file(tmp_path, tickets):
    path = tmp_path / "seed.json"
    path.write_text(json.dumps(tickets), encoding="utf-8")
    return path


def _raw(tid, offset, **extra):
    base = {
        "ticket_id": tid, "true_category": "VIT Email", "category_selected": "VIT Email Support",
        "building": "SJT", "priority": "HIGH", "subject": "No mail", "message": "Nothing arrives.",
        "status": "OPEN", "created_at_offset_minutes": offset,
        "label_type": "cluster_member", "gold_cluster_id": "G-1", "duplicate_of": "X-1",
    }
    base.update(extra)
    return base


async def test_seed_pool_converts_to_valid_records_without_leaking_labels(tmp_path):
    pool = JsonOpenTicketPool(_seed_file(tmp_path, [_raw("E1", -30)]), clock=lambda: NOW)
    [record] = await pool.list_open(since=NOW - timedelta(hours=1))

    assert record.category == "vit_email"  # alias label normalised
    assert record.assignee_group == team_for_category("vit_email")
    assert record.priority == "P2"
    assert record.title == "No mail" and record.description == "Nothing arrives."
    assert record.routing_rationale == SEED_RATIONALE
    assert record.created_at == NOW - timedelta(minutes=30)
    assert record.incident_cluster_id is None and record.duplicate_of is None


async def test_seed_pool_filters_since_category_status_and_sorts_newest_first(tmp_path):
    pool = JsonOpenTicketPool(_seed_file(tmp_path, [
        _raw("OLD", -200 * 60), _raw("A", -120), _raw("B", -10),
        _raw("CLOSED", -5, status="CLOSED"),
        _raw("P", -15, true_category="Printer Support"),
    ]), clock=lambda: NOW)

    rows = await pool.list_open(since=NOW - timedelta(hours=72))
    assert [r.ticket_id for r in rows] == ["B", "P", "A"]

    email_only = await pool.list_open(since=NOW - timedelta(hours=72), category="vit_email")
    assert [r.ticket_id for r in email_only] == ["B", "A"]


def test_bundled_seed_dataset_loads_as_valid_records():
    records = JsonOpenTicketPool(clock=lambda: NOW).records()
    assert len(records) >= 150
    assert all(r.ticket_id for r in records)
