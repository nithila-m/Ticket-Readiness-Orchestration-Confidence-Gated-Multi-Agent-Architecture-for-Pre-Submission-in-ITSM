from app.agents.completeness_scorer import score_completeness
from app.schemas.extraction import ExtractedField


def test_all_fields_present_high_confidence_scores_near_one():
    fields = {
        "affected_system": ExtractedField(value="wifi", confidence=0.95),
        "location": ExtractedField(value="library", confidence=0.9),
        "trigger": ExtractedField(value="after connecting", confidence=0.85),
        "frequency": ExtractedField(value="every time", confidence=0.9),
        "error_message": ExtractedField(value="no internet access", confidence=0.8),
        "device": ExtractedField(value="laptop", confidence=0.85),
    }
    result = score_completeness("wifi_internet", fields)
    assert result.score > 0.85
    assert result.missing_or_uncertain_fields == []


def test_almost_no_information_scores_near_zero():
    fields = {
        "affected_system": ExtractedField(value=None, confidence=0.0),
    }
    result = score_completeness("wifi_internet", fields)
    assert result.score < 0.2
    # every profile field except the one explicitly-null one should be flagged
    assert "affected_system" in result.missing_or_uncertain_fields
    assert "location" in result.missing_or_uncertain_fields
    assert len(result.missing_or_uncertain_fields) == 6  # all fields in wifi_internet profile


def test_field_absent_entirely_counts_as_missing():
    # 'location' never appears in the dict at all
    fields = {
        "affected_system": ExtractedField(value="wifi", confidence=0.9),
    }
    result = score_completeness("wifi_internet", fields)
    assert "location" in result.missing_or_uncertain_fields


def test_low_confidence_field_counts_as_uncertain_even_with_value():
    fields = {
        "affected_system": ExtractedField(value="maybe wifi", confidence=0.2),
    }
    result = score_completeness("wifi_internet", fields)
    assert "affected_system" in result.missing_or_uncertain_fields


def test_high_confidence_field_not_flagged():
    fields = {
        "affected_system": ExtractedField(value="wifi", confidence=0.6),
    }
    result = score_completeness("wifi_internet", fields)
    assert "affected_system" not in result.missing_or_uncertain_fields


def test_unknown_category_falls_back_to_general_profile():
    fields = {
        "affected_system": ExtractedField(value="something", confidence=0.9),
    }
    result = score_completeness("not_a_real_category", fields)
    general_field_names = {"affected_system", "error_message", "location"}
    assert set(result.missing_or_uncertain_fields).issubset(general_field_names)


def test_none_category_falls_back_to_general_profile():
    fields = {}
    result = score_completeness(None, fields)
    assert result.score == 0.0
    assert "affected_system" in result.missing_or_uncertain_fields


def test_ad_account_creation_full_example():
    fields = {
        "requester_role": ExtractedField(value="student", confidence=0.95),
        "department": ExtractedField(value="SCOPE", confidence=0.9),
        "account_type": ExtractedField(value="student AD login", confidence=0.85),
        "required_by_date": ExtractedField(value=None, confidence=0.0),
        "approval_reference": ExtractedField(value=None, confidence=0.0),
    }
    result = score_completeness("ad_account_creation", fields)
    assert 0.5 < result.score < 0.9
    assert "required_by_date" in result.missing_or_uncertain_fields
    assert "approval_reference" in result.missing_or_uncertain_fields
    assert "requester_role" not in result.missing_or_uncertain_fields


def test_printer_support_partial_example():
    fields = {
        "affected_system": ExtractedField(value="HP LaserJet lab printer", confidence=0.9),
        "error_message": ExtractedField(value="paper jam error", confidence=0.88),
    }
    result = score_completeness("printer_support", fields)
    assert "location" in result.missing_or_uncertain_fields
    assert "affected_system" not in result.missing_or_uncertain_fields


def test_score_is_bounded_between_zero_and_one():
    fields = {
        "affected_system": ExtractedField(value="teams", confidence=1.0),
        "trigger": ExtractedField(value="joining a call", confidence=1.0),
        "error_message": ExtractedField(value="app freezes", confidence=1.0),
        "device": ExtractedField(value="desktop", confidence=1.0),
        "frequency": ExtractedField(value="always", confidence=1.0),
    }
    result = score_completeness("ms_teams", fields)
    assert 0.0 <= result.score <= 1.0