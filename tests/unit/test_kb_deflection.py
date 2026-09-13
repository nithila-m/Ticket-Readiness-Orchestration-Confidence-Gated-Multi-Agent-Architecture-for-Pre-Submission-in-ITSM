import pytest

from app.agents.kb_deflection_agent import (
    KBDeflectionAgent,
    KBRetrievalResult,
    build_deflection_decision,
    build_kb_query,
)
from app.schemas.extraction import CategoryPrediction, ExtractedField


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


# --- build_kb_query ----------------------------------------------------
# These cover the fix for Agent 3 searching with the raw multi-turn
# transcript instead of the structured fields Agent 1 already extracted.
# See clarification_service.py and kb_deflection_agent.py::build_kb_query.
#
# IMPORTANT: CategoryPrediction.value below is always an internal snake_case
# key ("wifi_internet"), NOT a display label ("Wifi/Internet Support").
# That's what Agent 1's extractor actually returns - it's prompted with
# VALID_CATEGORIES (see category_profiles.py / information_extractor.py),
# never with display labels. Using a display label here would exercise a
# code path real traffic never hits and hide bugs in the key->label
# conversion (see test_build_kb_query_converts_internal_key_to_kb_label
# below, which is a regression guard for exactly that).

def test_build_kb_query_includes_category_and_non_null_fields():
    category = CategoryPrediction(value="wifi_internet", confidence=0.9)
    extracted_fields = {
        "symptom_type": ExtractedField(value="connection drops repeatedly", confidence=0.8),
        "device_type": ExtractedField(value="laptop", confidence=0.7),
        "ssid": ExtractedField(value=None, confidence=0.0),  # not extracted this turn
    }

    query = build_kb_query(category, extracted_fields)

    # displayed as the human-readable KB label, not the raw internal key
    assert "Category: Wifi/Internet Support" in query
    assert "symptom_type: connection drops repeatedly" in query
    assert "device_type: laptop" in query
    # a field with no extracted value carries no signal and must not appear
    assert "ssid" not in query


def test_build_kb_query_converts_internal_key_to_kb_label():
    """
    Regression guard for the actual bug found during review: Agent 1 only
    ever produces internal keys ("wifi_internet"), but that string never
    appears in KB metadata (which stores "Wifi/Internet Support"). If this
    conversion silently stopped happening, the query text would embed a
    raw code identifier instead of natural language, and (separately, see
    test_check_converts_category_key_for_the_where_filter below) category
    filtering would match zero documents on every call.
    """
    category = CategoryPrediction(value="ad_account_creation", confidence=0.9)

    query = build_kb_query(category, {})

    assert query == "Category: AD Account Creation"
    assert "ad_account_creation" not in query


def test_build_kb_query_handles_missing_category():
    # detected_category can be None early in a conversation, before Agent 1
    # is confident enough to assign one - build_kb_query must not crash.
    query = build_kb_query(
        None, {"symptom_type": ExtractedField(value="drops", confidence=0.6)}
    )

    assert "Category:" not in query
    assert "symptom_type: drops" in query


def test_build_kb_query_with_no_extracted_fields_returns_category_only():
    category = CategoryPrediction(value="printer_support", confidence=0.85)

    query = build_kb_query(category, {})

    assert query == "Category: Printer Support"


def test_build_kb_query_never_includes_dialogue_labels():
    # Regression guard for the exact bug this change fixes: the query must
    # never look like the transcript format ("User:", "Assistant asked:").
    category = CategoryPrediction(value="ms_teams", confidence=0.9)
    extracted_fields = {"failure_type": ExtractedField(value="audio", confidence=0.9)}

    query = build_kb_query(category, extracted_fields)

    assert "User:" not in query
    assert "Assistant asked:" not in query


def test_build_kb_query_unrecognized_category_key_falls_back_to_raw_value():
    # Defensive path: if Agent 1 ever returns something outside
    # VALID_CATEGORIES (shouldn't happen - information_extractor.py
    # sanitizes this - but build_kb_query must not crash either way),
    # fall back to the raw value rather than dropping the category line.
    category = CategoryPrediction(value="some_unmapped_key", confidence=0.5)

    query = build_kb_query(category, {})

    assert query == "Category: some_unmapped_key"


# --- KBDeflectionAgent.check: category key -> KB label for the where filter

@pytest.mark.asyncio
async def test_check_converts_category_key_for_the_where_filter():
    """
    Regression guard for the actual bug found during review (see above):
    KBDeflectionAgent.check() must pass the KB *display label* to the
    retrieval function's category filter, never the raw internal key -
    otherwise every category-filtered query silently matches zero KB
    documents and Agent 3 always returns NO_MATCH.
    """
    captured = {}

    def fake_retrieve_fn(query_text: str, category: str | None = None) -> KBRetrievalResult:
        captured["query_text"] = query_text
        captured["category"] = category
        return KBRetrievalResult(
            outcome="NO_MATCH", similarity_score=0.0, articles_checked=[],
            matched_kb_id=None, matched_title=None, offered_resolution=None,
        )

    agent = KBDeflectionAgent(retrieve_fn=fake_retrieve_fn)
    category = CategoryPrediction(value="vit_email", confidence=0.9)

    await agent.check(category, {})

    assert captured["category"] == "VIT Email Support"
    assert captured["category"] != "vit_email"


@pytest.mark.asyncio
async def test_check_passes_none_category_when_no_category_detected():
    captured = {}

    def fake_retrieve_fn(query_text: str, category: str | None = None) -> KBRetrievalResult:
        captured["category"] = category
        return KBRetrievalResult(
            outcome="NO_MATCH", similarity_score=0.0, articles_checked=[],
            matched_kb_id=None, matched_title=None, offered_resolution=None,
        )

    agent = KBDeflectionAgent(retrieve_fn=fake_retrieve_fn)

    await agent.check(None, {"symptom_type": ExtractedField(value="drops", confidence=0.6)})

    assert captured["category"] is None