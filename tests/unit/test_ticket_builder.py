"""Unit tests for app/agents/ticket_builder.py (pure - no DB, no LLM)."""

import pytest

from app.agents.ticket_builder import (
    TITLE_MAX_CHARS,
    build_description,
    build_ticket,
    build_title,
    pick_primary_field,
)
from app.config.routing import team_for_category
from app.schemas.conversation import ConversationState, Message
from app.schemas.extraction import CategoryPrediction, ExtractedField
from app.schemas.pipeline import ClassificationResult, CorrelationResult, TicketRecord


def make_state(category: str = "wifi_internet", **overrides) -> ConversationState:
    data = dict(
        conversation_id="conv-1",
        turn_count=2,
        raw_messages=[
            Message(role="user", content="wifi   keeps dropping\nin SJT-412"),
            Message(role="assistant", content="Is it one device or several?"),
            Message(role="user", content="all my devices"),
        ],
        extracted_fields={
            "symptom_type": ExtractedField(value="drops every few minutes", confidence=0.92),
            "single_or_multiple_devices": ExtractedField(value="multiple", confidence=0.8),
            "ssid": ExtractedField(value=None, confidence=0.0),
            "building": ExtractedField(value="SJT", confidence=0.9),
        },
        detected_category=CategoryPrediction(value=category, confidence=0.95),
        completeness_score=0.81,
        kb_outcome="NO_MATCH",
        kb_articles_checked=["KB0302", "KB0305"],
        classification=ClassificationResult(
            category=category, priority="P2", assignee_group=team_for_category(category),
            routing_rationale="Matched wifi symptom fields.", confidence=0.87, source="stub",
        ),
    )
    data.update(overrides)
    return ConversationState(**data)


def test_pick_primary_field_is_highest_weight():
    assert pick_primary_field("wifi_internet") == "symptom_type"
    assert pick_primary_field("printer_support") == "symptom"
    assert pick_primary_field("general") == "symptom_or_error"


def test_title_uses_display_label_and_primary_field():
    assert build_title(make_state()) == "[Wifi/Internet Support] drops every few minutes"


def test_title_general_category_label():
    state = make_state(
        "general",
        extracted_fields={"symptom_or_error": ExtractedField(value="laptop slow", confidence=0.7)},
    )
    assert build_title(state) == "[General] laptop slow"


def test_title_falls_back_to_first_user_message_with_collapsed_whitespace():
    state = make_state(extracted_fields={"symptom_type": ExtractedField(value=None, confidence=0.0)})
    assert build_title(state) == "[Wifi/Internet Support] wifi keeps dropping in SJT-412"


def test_title_is_truncated_to_max_chars():
    long = "x" * 500
    state = make_state(extracted_fields={"symptom_type": ExtractedField(value=long, confidence=0.9)})
    title = build_title(state)
    assert len(title) == TITLE_MAX_CHARS
    assert title.startswith("[Wifi/Internet Support] xxx")


def test_description_excludes_assistant_messages():
    desc = build_description(make_state())
    reported = desc.split("\n\n")[0]
    assert reported.startswith("Reported by user:")
    assert "all my devices" in reported
    assert "Is it one device or several?" not in desc


def test_description_sections_and_non_null_fields_only():
    desc = build_description(make_state())
    assert "- symptom_type: drops every few minutes (conf 0.92)" in desc
    assert "ssid" not in desc
    assert "Completeness: 0.81" in desc
    assert "Clarification turns: 2" in desc
    assert "Knowledge base: NO_MATCH (articles checked: KB0302, KB0305)" in desc


def test_build_ticket_without_classification_raises():
    with pytest.raises(ValueError, match="classification"):
        build_ticket(make_state(classification=None))


def test_build_ticket_copies_classification_and_state():
    t = build_ticket(make_state())
    assert isinstance(t, TicketRecord)
    assert t.ticket_id is None
    assert t.category == "wifi_internet"
    assert t.priority == "P2"
    assert t.assignee_group == "Network Operations"
    assert t.routing_rationale == "Matched wifi symptom fields."
    assert t.confidence_score == 0.87
    assert t.kb_articles_checked == ["KB0302", "KB0305"]
    assert t.building == "SJT"
    assert t.duplicate_of is None and t.incident_cluster_id is None


def test_build_ticket_building_none_when_not_extracted():
    state = make_state(extracted_fields={})
    assert build_ticket(state).building is None


def test_build_ticket_duplicate_copies_duplicate_of_and_cluster():
    state = make_state(correlation=CorrelationResult(
        outcome="DUPLICATE", duplicate_of="TRO-2026-00007", cluster_id="cl-3", max_similarity=0.93,
    ))
    t = build_ticket(state)
    assert t.duplicate_of == "TRO-2026-00007"
    assert t.incident_cluster_id == "cl-3"


def test_built_ticket_revalidates_against_schema():
    t = build_ticket(make_state("printer_support"))
    assert TicketRecord.model_validate(t.model_dump()) == t
