import pytest
from pydantic import ValidationError

from app.schemas.extraction import CategoryPrediction, ExtractedField, ExtractionResult
from app.schemas.analysis import AnalysisResult
from app.schemas.conversation import ConversationState, Message, UserMessageRequest


def test_extracted_field_valid():
    field = ExtractedField(value="VPN", confidence=0.98)
    assert field.value == "VPN"
    assert field.confidence == 0.98


def test_extracted_field_null_value_allowed():
    field = ExtractedField(value=None, confidence=0.0)
    assert field.value is None
    assert field.confidence == 0.0


def test_extracted_field_confidence_out_of_range_rejected():
    with pytest.raises(ValidationError):
        ExtractedField(value="x", confidence=1.5)
    with pytest.raises(ValidationError):
        ExtractedField(value="x", confidence=-0.1)


def test_category_prediction_valid():
    cat = CategoryPrediction(value="network_vpn", confidence=0.96)
    assert cat.value == "network_vpn"


def test_extraction_result_vpn_example():
    result = ExtractionResult(
        category=CategoryPrediction(value="network_vpn", confidence=0.95),
        extracted_fields={
            "affected_system": ExtractedField(value="VPN", confidence=0.98),
            "trigger": ExtractedField(value="after laptop wakes from sleep", confidence=0.91),
            "frequency": ExtractedField(value=None, confidence=0.0),
        },
    )
    assert result.category.value == "network_vpn"
    assert result.extracted_fields["frequency"].value is None
    assert len(result.extracted_fields) == 3


def test_analysis_result_full_example():
    result = AnalysisResult(
        category=CategoryPrediction(value="network_vpn", confidence=0.96),
        extracted_fields={
            "affected_system": ExtractedField(value="VPN", confidence=0.99),
            "trigger": ExtractedField(value="laptop wakes from sleep", confidence=0.95),
            "frequency": ExtractedField(value="every time", confidence=0.98),
            "error_message": ExtractedField(value=None, confidence=0.0),
        },
        completeness_score=0.74,
        missing_or_uncertain_fields=["error_message", "network_context"],
    )
    assert result.completeness_score == 0.74
    assert "error_message" in result.missing_or_uncertain_fields


def test_user_message_request_valid():
    req = UserMessageRequest(message="My VPN disconnects every time my laptop wakes from sleep.")
    assert req.message.startswith("My VPN")


def test_user_message_request_rejects_empty():
    with pytest.raises(ValidationError):
        UserMessageRequest(message="")


def test_conversation_state_defaults():
    state = ConversationState(conversation_id="conv-001")
    assert state.turn_count == 0
    assert state.raw_messages == []
    assert state.detected_category is None
    assert state.completeness_score == 0.0
    assert state.clarification_log == []


def test_conversation_state_populated_by_agent1():
    state = ConversationState(
        conversation_id="conv-002",
        raw_messages=[Message(role="user", content="Outlook crashes when I open an attachment.")],
        extracted_fields={
            "application": ExtractedField(value="Outlook", confidence=0.99),
            "trigger": ExtractedField(value="opening an attachment", confidence=0.95),
        },
        detected_category=CategoryPrediction(value="software", confidence=0.95),
        completeness_score=0.68,
        missing_or_uncertain_fields=["error_message"],
    )
    assert state.detected_category.value == "software"
    assert state.extracted_fields["application"].value == "Outlook"

def test_extracted_field_rejects_value_with_zero_confidence_mismatch():
    with pytest.raises(ValidationError):
        ExtractedField(value=None, confidence=0.5)


def test_extracted_field_allows_value_with_low_confidence():
    field = ExtractedField(value="maybe VPN", confidence=0.2)
    assert field.confidence == 0.2


def test_category_prediction_allows_none_when_undetermined():
    cat = CategoryPrediction(value=None, confidence=0.0)
    assert cat.value is None


def test_category_prediction_rejects_none_with_nonzero_confidence():
    with pytest.raises(ValidationError):
        CategoryPrediction(value=None, confidence=0.3)