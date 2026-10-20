"""Contract tests for app/schemas/pipeline.py (Agents 4, 5, 6 + audit)."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.schemas.pipeline import (
    AuditEvent,
    ClassificationResult,
    CorrelationResult,
    TicketRecord,
)


def _ticket(**overrides) -> TicketRecord:
    data = dict(
        conversation_id="conv-1",
        title="[Wifi/Internet Support] drops every few minutes",
        description="Reported by user:\nwifi keeps dropping in SJT-412",
        category="wifi_internet",
        priority="P2",
        assignee_group="Network Operations",
        routing_rationale="Matched wifi symptom fields.",
        confidence_score=0.87,
        kb_articles_checked=["KB0302", "KB0305"],
        building="SJT",
    )
    data.update(overrides)
    return TicketRecord(**data)


# --- TicketRecord ------------------------------------------------------

def test_ticket_record_json_round_trip_preserves_all_fields():
    original = _ticket(ticket_id="TRO-2026-00001", external_key="TRO-12")
    restored = TicketRecord.model_validate_json(original.model_dump_json())
    assert restored == original
    assert restored.created_at.tzinfo is not None  # stays timezone-aware


def test_ticket_record_defaults():
    t = _ticket()
    assert t.ticket_id is None
    assert t.status == "OPEN"
    assert t.duplicate_of is None
    assert t.external_key is None and t.external_url is None
    assert t.created_at <= datetime.now(timezone.utc)


def test_ticket_record_rejects_title_over_255_chars():
    with pytest.raises(ValidationError):
        _ticket(title="x" * 300)


def test_ticket_record_rejects_empty_title():
    with pytest.raises(ValidationError):
        _ticket(title="")


@pytest.mark.parametrize("bad_priority", ["P5", "P0", "high", ""])
def test_ticket_record_rejects_unknown_priority(bad_priority):
    with pytest.raises(ValidationError):
        _ticket(priority=bad_priority)


def test_ticket_record_rejects_unknown_status():
    with pytest.raises(ValidationError):
        _ticket(status="RESOLVED")


@pytest.mark.parametrize("bad_conf", [-0.1, 1.1])
def test_ticket_record_rejects_confidence_out_of_range(bad_conf):
    with pytest.raises(ValidationError):
        _ticket(confidence_score=bad_conf)


# --- ClassificationResult (Agent 5) -------------------------------------

def test_classification_result_valid():
    r = ClassificationResult(
        category="printer_support", priority="P3", assignee_group="Desktop Support",
        routing_rationale="Printer symptom on one device.", confidence=0.8, source="llm",
    )
    assert r.source == "llm"


def test_classification_result_requires_non_empty_rationale():
    with pytest.raises(ValidationError):
        ClassificationResult(
            category="printer_support", priority="P3", assignee_group="Desktop Support",
            routing_rationale="", confidence=0.8, source="llm",
        )


def test_classification_result_rejects_unknown_source():
    with pytest.raises(ValidationError):
        ClassificationResult(
            category="printer_support", priority="P3", assignee_group="Desktop Support",
            routing_rationale="x", confidence=0.8, source="gpt",
        )


# --- CorrelationResult (Agent 4) ----------------------------------------

def test_correlation_result_defaults_for_no_match():
    r = CorrelationResult(outcome="NO_MATCH")
    assert r.similar_tickets == []
    assert r.max_similarity == 0.0


def test_correlation_result_rejects_unknown_outcome():
    with pytest.raises(ValidationError):
        CorrelationResult(outcome="MAYBE")


# --- AuditEvent ----------------------------------------------------------

def test_audit_event_payload_defaults_to_independent_dicts():
    a = AuditEvent(conversation_id="c", turn=0, stage="dispatch", decision="DISPATCHED")
    b = AuditEvent(conversation_id="c", turn=1, stage="dispatch", decision="DISPATCHED")
    a.payload["k"] = 1
    assert b.payload == {}