from app.agents.completeness_scorer import score_completeness
from app.schemas.extraction import ExtractedField


def test_all_fields_present_high_confidence_scores_near_one():
    fields = {
        "symptom_type": ExtractedField(value="drops intermittently", confidence=0.95),
        "single_or_multiple_devices": ExtractedField(value="multiple", confidence=0.9),
        "when_started": ExtractedField(value="since last night", confidence=0.85),
        "device_type": ExtractedField(value="laptop", confidence=0.9),
        "ssid": ExtractedField(value="VIT-WiFi", confidence=0.9),
    }
    result = score_completeness("wifi_internet", fields)
    assert result.score > 0.85
    assert result.missing_or_uncertain_fields == []


def test_almost_no_information_scores_near_zero():
    fields = {"symptom_type": ExtractedField(value=None, confidence=0.0)}
    result = score_completeness("wifi_internet", fields)
    assert result.score < 0.2
    assert len(result.missing_or_uncertain_fields) == 5  # all wifi_internet fields


def test_field_absent_entirely_counts_as_missing():
    fields = {"symptom_type": ExtractedField(value="won't connect", confidence=0.9)}
    result = score_completeness("wifi_internet", fields)
    assert "ssid" in result.missing_or_uncertain_fields


def test_low_confidence_field_counts_as_uncertain_even_with_value():
    fields = {"symptom_type": ExtractedField(value="maybe drops?", confidence=0.2)}
    result = score_completeness("wifi_internet", fields)
    assert "symptom_type" in result.missing_or_uncertain_fields


def test_high_confidence_field_not_flagged():
    fields = {"symptom_type": ExtractedField(value="won't connect", confidence=0.6)}
    result = score_completeness("wifi_internet", fields)
    assert "symptom_type" not in result.missing_or_uncertain_fields


def test_unknown_category_falls_back_to_general_profile():
    fields = {"symptom_or_error": ExtractedField(value="something", confidence=0.9)}
    result = score_completeness("not_a_real_category", fields)
    general_fields = {"symptom_or_error", "when_started", "scope"}
    assert set(result.missing_or_uncertain_fields).issubset(general_fields)


def test_none_category_falls_back_to_general_profile():
    result = score_completeness(None, {})
    assert result.score == 0.0
    assert "symptom_or_error" in result.missing_or_uncertain_fields


def test_ad_account_creation_full_example():
    fields = {
        "error_or_symptom": ExtractedField(value="locked out", confidence=0.95),
        "username_domain": ExtractedField(value="EMP2291", confidence=0.9),
        "when_started": ExtractedField(value="this morning", confidence=0.85),
        "device_context": ExtractedField(value=None, confidence=0.0),
        "troubleshooting_done": ExtractedField(value=None, confidence=0.0),
    }
    result = score_completeness("ad_account_creation", fields)
    assert 0.5 < result.score < 0.9
    assert "device_context" in result.missing_or_uncertain_fields
    assert "troubleshooting_done" in result.missing_or_uncertain_fields
    assert "error_or_symptom" not in result.missing_or_uncertain_fields


def test_printer_support_partial_example():
    fields = {
        "symptom": ExtractedField(value="offline", confidence=0.9),
        "error_message": ExtractedField(value="Printer Status: Offline", confidence=0.88),
    }
    result = score_completeness("printer_support", fields)
    assert "scope" in result.missing_or_uncertain_fields
    assert "symptom" not in result.missing_or_uncertain_fields


def test_score_is_bounded_between_zero_and_one():
    fields = {
        "failure_type": ExtractedField(value="can't hear audio", confidence=1.0),
        "scope": ExtractedField(value="every meeting", confidence=1.0),
        "error_signal": ExtractedField(value=None, confidence=0.0),
        "device_platform": ExtractedField(value="desktop app", confidence=1.0),
    }
    result = score_completeness("ms_teams", fields)
    assert 0.0 <= result.score <= 1.0


def test_teams_and_email_score_identically_for_same_field_pattern():
    # Confirms the intentional shared vocabulary actually behaves
    # identically in the scorer, not just in the config.
    fields = {
        "failure_type": ExtractedField(value="won't send", confidence=0.9),
        "scope": ExtractedField(value="all outgoing mail", confidence=0.85),
    }
    teams_result = score_completeness("ms_teams", fields)
    email_result = score_completeness("vit_email", fields)
    assert teams_result.score == email_result.score