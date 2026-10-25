import logging

import pytest

from app.agents import kb_deflection_agent
from app.agents.kb_deflection_agent import (
    KBDeflectionAgent,
    KBRetrievalResult,
    build_deflection_decision,
    build_kb_query,
    run_kb_retrieval,
)
from app.config.settings import settings
from app.schemas.extraction import ExtractedField

# Importable only after app.agents.kb_deflection_agent has put TRO_Codes/ on sys.path.
import agents.kb_retrieval_agent as kb_retrieval_agent  # noqa: E402


def test_build_deflection_decision_shapes_valid_clarification_decision():
    kb_result = KBRetrievalResult(
        outcome="STRONG_MATCH", similarity_score=0.88, articles_checked=["KB0302"],
        matched_kb_id="KB0302",
        matched_title="Wi-Fi disconnects repeatedly after device wakes from sleep",
        offered_resolution="1. Disable fast startup\n2. Forget and re-join Wi-Fi",
    )
    decision = build_deflection_decision(kb_result)

    assert decision.action == "DEFLECTED"
    assert decision.question is None
    assert decision.confidence == 0.88
    assert "Wi-Fi disconnects repeatedly after device wakes from sleep" in decision.reasoning


def test_build_deflection_decision_reasoning_includes_similarity_score():
    kb_result = KBRetrievalResult(
        outcome="STRONG_MATCH", similarity_score=0.7321, articles_checked=["KB0301"],
        matched_kb_id="KB0301", matched_title="VPN disconnects after laptop wakes from sleep",
        offered_resolution="Update VPN client to latest version.",
    )
    decision = build_deflection_decision(kb_result)

    assert "0.73" in decision.reasoning


# --- build_kb_query ---------------------------------------------------------

def test_build_kb_query_one_line_per_field_plus_display_label():
    fields = {
        "symptom_type": ExtractedField(value="drops every few minutes", confidence=0.9),
        "device_type": ExtractedField(value="laptop", confidence=0.8),
    }
    query = build_kb_query("wifi_internet", fields)

    assert query.splitlines() == [
        "symptom_type: drops every few minutes",
        "device_type: laptop",
        "Category: Wifi/Internet Support",
    ]


def test_build_kb_query_skips_null_fields():
    fields = {
        "failure_type": ExtractedField(value="no new mail", confidence=0.9),
        "error_signal": ExtractedField(value=None, confidence=0.0),
    }
    query = build_kb_query("vit_email", fields)

    assert "error_signal" not in query
    assert "None" not in query
    assert "Category: VIT Email Support" in query  # canonical label, not the "VIT Email" alias


def test_build_kb_query_without_category_has_no_category_line():
    fields = {"symptom": ExtractedField(value="paper jam", confidence=0.9)}
    assert build_kb_query(None, fields) == "symptom: paper jam"


def test_build_kb_query_unknown_key_used_as_is():
    assert build_kb_query("general", {}) == "Category: general"


# --- KBDeflectionAgent.check --------------------------------------------------

def _strong(kb_id="KB0302") -> KBRetrievalResult:
    return KBRetrievalResult("STRONG_MATCH", 0.9, [kb_id], kb_id, "title", "fix")


async def test_check_passes_query_and_category_to_retrieve_fn():
    calls = []

    def fake_retrieve(query_text, category):
        calls.append((query_text, category))
        return _strong()

    result = await KBDeflectionAgent(retrieve_fn=fake_retrieve).check("symptom: x", "wifi_internet")

    assert calls == [("symptom: x", "wifi_internet")]
    assert result.outcome == "STRONG_MATCH"


async def test_check_returns_no_match_when_retrieval_raises(caplog):
    def broken_retrieve(query_text, category):
        raise RuntimeError("chroma unavailable")

    with caplog.at_level(logging.ERROR, logger=kb_deflection_agent.__name__):
        result = await KBDeflectionAgent(retrieve_fn=broken_retrieve).check("q", "ms_teams")

    assert result.outcome == "NO_MATCH"
    assert result.similarity_score == 0.0
    assert result.articles_checked == []
    assert result.matched_kb_id is None
    assert "chroma unavailable" in caplog.text


# --- run_kb_retrieval -> deflect() ---------------------------------------------

def test_run_kb_retrieval_passes_category_and_settings_thresholds(monkeypatch):
    captured = {}

    def fake_deflect(query_text, **kwargs):
        captured["query_text"] = query_text
        captured.update(kwargs)
        return {"outcome": "WEAK_MATCH", "resolution_confidence": 0.5,
                "kb_articles_checked": ["KB0503"], "matched_kb_id": "KB0503",
                "matched_title": "t", "offered_resolution": "r"}

    monkeypatch.setattr(kb_deflection_agent, "_deflect", fake_deflect)
    result = run_kb_retrieval("failure_type: no mail", "vit_email")

    assert captured == {
        "query_text": "failure_type: no mail",
        "category": "vit_email",
        "strong_threshold": settings.kb_strong_threshold,
        "weak_threshold": settings.kb_weak_threshold,
    }
    assert result.outcome == "WEAK_MATCH"
    assert result.matched_kb_id == "KB0503"


# --- deflect() with a fake collection (no real Chroma) -------------------------

class _FakeCollection:
    def __init__(self, distance: float):
        self._distance = distance
        self.query_kwargs = None

    def query(self, **kwargs):
        self.query_kwargs = kwargs
        return {
            "ids": [["KB0502"]],
            "distances": [[self._distance]],
            "metadatas": [[{"title": "Not receiving emails", "resolution": "Check spam."}]],
        }


@pytest.fixture
def fake_collection(monkeypatch):
    def install(distance: float) -> _FakeCollection:
        collection = _FakeCollection(distance)
        monkeypatch.setattr(kb_retrieval_agent, "get_collection", lambda: collection)
        return collection
    return install


def test_deflect_without_category_sends_no_where_filter(fake_collection):
    collection = fake_collection(distance=0.2)
    kb_retrieval_agent.deflect("no new mail")
    assert "where" not in collection.query_kwargs


def test_deflect_category_filter_includes_alias_labels(fake_collection):
    collection = fake_collection(distance=0.2)
    kb_retrieval_agent.deflect("no new mail", category="vit_email")
    assert collection.query_kwargs["where"] == {
        "category": {"$in": ["VIT Email Support", "VIT Email"]}
    }


def test_deflect_category_without_kb_label_is_unfiltered(fake_collection):
    collection = fake_collection(distance=0.2)
    kb_retrieval_agent.deflect("something", category="general")
    assert "where" not in collection.query_kwargs


@pytest.mark.parametrize(
    "distance, expected",
    [(0.27, "STRONG_MATCH"), (0.30, "WEAK_MATCH"), (0.54, "WEAK_MATCH"), (0.60, "NO_MATCH")],
)
def test_deflect_default_thresholds_unchanged(fake_collection, distance, expected):
    fake_collection(distance=distance)
    assert kb_retrieval_agent.deflect("q")["outcome"] == expected


def test_deflect_custom_thresholds(fake_collection):
    fake_collection(distance=0.4)  # similarity 0.6
    result = kb_retrieval_agent.deflect("q", strong_threshold=0.6, weak_threshold=0.3)
    assert result["outcome"] == "STRONG_MATCH"
    assert result["deflect"] is True


def test_build_kb_query_still_importable_from_kb_deflection_agent():
    from app.agents import request_text

    assert build_kb_query is request_text.build_kb_query
