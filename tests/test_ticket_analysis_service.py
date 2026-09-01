import pytest

from app.agents.information_extractor import InformationExtractor
from app.providers.base import LLMProvider
from app.schemas.extraction import CategoryPrediction, ExtractedField, ExtractionResult
from app.services.ticket_analysis_service import TicketAnalysisService


class StubProvider(LLMProvider):
    def __init__(self, canned_result: ExtractionResult):
        self._canned_result = canned_result

    async def extract_information(self, issue_text: str) -> ExtractionResult:
        return self._canned_result


def make_service(canned: ExtractionResult) -> TicketAnalysisService:
    extractor = InformationExtractor(StubProvider(canned))
    return TicketAnalysisService(extractor)


@pytest.mark.asyncio
async def test_wifi_example_end_to_end():
    canned = ExtractionResult(
        category=CategoryPrediction(value="wifi_internet", confidence=0.95),
        extracted_fields={
            "symptom_type": ExtractedField(value="drops intermittently", confidence=0.9),
            "single_or_multiple_devices": ExtractedField(value="multiple", confidence=0.88),
            "when_started": ExtractedField(value="since last night", confidence=0.85),
            "ssid": ExtractedField(value=None, confidence=0.0),
        },
    )
    service = make_service(canned)
    result = await service.analyze(
        "None of my devices are getting internet since last night — laptop and phone both."
    )
    assert result.category.value == "wifi_internet"
    assert 0.0 < result.completeness_score < 1.0
    assert "ssid" in result.missing_or_uncertain_fields
    assert "device_type" in result.missing_or_uncertain_fields  # never mentioned at all


@pytest.mark.asyncio
async def test_vague_message_scores_low_completeness():
    canned = ExtractionResult(
        category=CategoryPrediction(value=None, confidence=0.0),
        extracted_fields={},
    )
    service = make_service(canned)
    result = await service.analyze("wifi not working pls help")
    assert result.category.value is None
    assert result.completeness_score == 0.0
    assert len(result.missing_or_uncertain_fields) > 0


@pytest.mark.asyncio
async def test_ad_account_creation_mostly_complete():
    canned = ExtractionResult(
        category=CategoryPrediction(value="ad_account_creation", confidence=0.9),
        extracted_fields={
            "error_or_symptom": ExtractedField(value="account locked", confidence=0.95),
            "username_domain": ExtractedField(value="EMP2291", confidence=0.9),
            "when_started": ExtractedField(value="this morning", confidence=0.85),
        },
    )
    service = make_service(canned)
    result = await service.analyze(
        "My AD account got locked this morning, username EMP2291, can't log in."
    )
    assert result.category.value == "ad_account_creation"
    assert result.completeness_score > 0.5
    assert "device_context" in result.missing_or_uncertain_fields
    assert "troubleshooting_done" in result.missing_or_uncertain_fields
    assert "error_or_symptom" not in result.missing_or_uncertain_fields


@pytest.mark.asyncio
async def test_hallucinated_category_is_sanitized_before_scoring():
    # If the extractor's sanitization step failed, this would score against
    # a made-up category with no profile - this test proves the whole
    # pipeline (not just InformationExtractor in isolation) is protected.
    canned = ExtractionResult(
        category=CategoryPrediction(value="not_a_real_category", confidence=0.7),
        extracted_fields={"symptom_type": ExtractedField(value="something", confidence=0.8)},
    )
    service = make_service(canned)
    result = await service.analyze("Something vague happened with my system.")
    assert result.category.value is None
    # falls back to 'general' profile fields internally
    general_fields = {"symptom_or_error", "when_started", "scope"}
    assert set(result.missing_or_uncertain_fields).issubset(general_fields)


@pytest.mark.asyncio
async def test_printer_support_example():
    canned = ExtractionResult(
        category=CategoryPrediction(value="printer_support", confidence=0.93),
        extracted_fields={
            "symptom": ExtractedField(value="showing offline", confidence=0.9),
            "printer_model": ExtractedField(value="HP LaserJet P1108", confidence=0.88),
            "connection_type": ExtractedField(value="network", confidence=0.85),
            "when_started": ExtractedField(value="this morning around 9am", confidence=0.85),
        },
    )
    service = make_service(canned)
    result = await service.analyze(
        "The HP LaserJet P1108 has been showing offline since this morning, connected over network."
    )
    # weighted_sum = (1.0*0.9)+(0.6*0.88)+(0.6*0.85)+(0.6*0.85) = 0.9+0.528+0.51+0.51 = 2.448
    # total_weight (printer_support) = 1.0+0.8+0.7+0.6+0.6+0.6+0.5 = 4.8
    # score = 2.448 / 4.8 = 0.51
    assert 0.45 < result.completeness_score < 0.55
    assert "scope" in result.missing_or_uncertain_fields
    assert "error_message" in result.missing_or_uncertain_fields
    assert "troubleshooting_done" in result.missing_or_uncertain_fields
    assert "symptom" not in result.missing_or_uncertain_fields


@pytest.mark.asyncio
async def test_ms_teams_and_vit_email_score_consistently_for_same_pattern():
    # Confirms end-to-end (not just the scorer in isolation) that the
    # intentionally shared vocabulary between these two categories
    # produces consistent behavior through the full service.
    fields = {
        "failure_type": ExtractedField(value="can't send", confidence=0.9),
        "scope": ExtractedField(value="all outgoing", confidence=0.85),
    }
    teams_canned = ExtractionResult(
        category=CategoryPrediction(value="ms_teams", confidence=0.9), extracted_fields=fields
    )
    email_canned = ExtractionResult(
        category=CategoryPrediction(value="vit_email", confidence=0.9), extracted_fields=fields
    )
    teams_result = await make_service(teams_canned).analyze("Teams message.")
    email_result = await make_service(email_canned).analyze("Email message.")
    assert teams_result.completeness_score == email_result.completeness_score