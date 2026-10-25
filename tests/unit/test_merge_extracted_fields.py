"""
Unit tests for merge_extracted_fields - the pure carry-forward merge that
ClarificationService.extract() applies between Agent 1 reruns.
"""

from app.schemas.extraction import ExtractedField
from app.services.clarification_service import merge_extracted_fields


def _field(value, confidence=0.9) -> ExtractedField:
    return ExtractedField(value=value, confidence=confidence)


def test_field_omitted_in_new_result_is_carried_forward():
    previous = {"symptom_type": _field("drops"), "device_type": _field("laptop")}
    new = {"device_type": _field("laptop")}

    merged, carried = merge_extracted_fields(previous, "wifi_internet", new, "wifi_internet")

    assert merged["symptom_type"].value == "drops"
    assert merged["device_type"].value == "laptop"
    assert carried == ["symptom_type"]


def test_changed_value_in_new_result_wins():
    previous = {"symptom_type": _field("drops")}
    new = {"symptom_type": _field("slow", confidence=0.7)}

    merged, carried = merge_extracted_fields(previous, "wifi_internet", new, "wifi_internet")

    assert merged["symptom_type"].value == "slow"
    assert merged["symptom_type"].confidence == 0.7
    assert carried == []


def test_category_change_does_not_carry_forward():
    previous = {"symptom_type": _field("drops")}
    new = {"failure_type": _field("audio")}

    merged, carried = merge_extracted_fields(previous, "wifi_internet", new, "ms_teams")

    assert merged == new
    assert carried == []
