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
            "affected_system": ExtractedField(value="wifi", confidence=0.98),
            "location": ExtractedField(value="library", confidence=0.9),
            "trigger": ExtractedField(value="after laptop wakes from sleep", confidence=0.91),
            "frequency": ExtractedField(value=None, confidence=0.0),
        },
    )
    service = make_service(canned)
    result = await service.analyze(
        "My wifi in the library keeps disconnecting after I wake my laptop from sleep."
    )
    assert result.category.value == "wifi_internet"
    assert 0.0 < result.completeness_score < 1.0
    assert "frequency" in result.missing_or_uncertain_fields
    assert "error_message" in result.missing_or_uncertain_fields  # never mentioned at all


@pytest.mark.asyncio
async def test_vague_message_scores_low_completeness():
    canned = ExtractionResult(
        category=CategoryPrediction(value=None, confidence=0.0),
        extracted_fields={},
    )
    service = make_service(canned)
    result = await service.analyze("My thing is broken.")
    assert result.category.value is None
    assert result.completeness_score == 0.0
    assert len(result.missing_or_uncertain_fields) > 0


@pytest.mark.asyncio
async def test_ad_account_creation_mostly_complete():
    canned = ExtractionResult(
        category=CategoryPrediction(value="ad_account_creation", confidence=0.9),
        extracted_fields={
            "requester_role": ExtractedField(value="student", confidence=0.95),
            "department": ExtractedField(value="SCOPE", confidence=0.9),
            "account_type": ExtractedField(value="AD login", confidence=0.85),
        },
    )
    service = make_service(canned)
    result = await service.analyze(
        "I'm a SCOPE student and I need an AD login account created."
    )
    assert result.category.value == "ad_account_creation"
    assert result.completeness_score > 0.5
    assert "required_by_date" in result.missing_or_uncertain_fields
    assert "approval_reference" in result.missing_or_uncertain_fields
    assert "requester_role" not in result.missing_or_uncertain_fields


@pytest.mark.asyncio
async def test_hallucinated_category_is_sanitized_before_scoring():
    # If the extractor's sanitization step failed, this would score against
    # a made-up category with no profile - this test proves the whole
    # pipeline (not just InformationExtractor in isolation) is protected.
    canned = ExtractionResult(
        category=CategoryPrediction(value="not_a_real_category", confidence=0.7),
        extracted_fields={"affected_system": ExtractedField(value="something", confidence=0.8)},
    )
    service = make_service(canned)
    result = await service.analyze("Something vague happened with my system.")
    assert result.category.value is None
    # falls back to 'general' profile fields internally
    general_fields = {"affected_system", "error_message", "location"}
    assert set(result.missing_or_uncertain_fields).issubset(general_fields)


@pytest.mark.asyncio
async def test_printer_support_example():
    canned = ExtractionResult(
        category=CategoryPrediction(value="printer_support", confidence=0.93),
        extracted_fields={
            "affected_system": ExtractedField(value="HP LaserJet, 2nd floor lab", confidence=0.9),
            "error_message": ExtractedField(value="paper jam", confidence=0.88),
            "location": ExtractedField(value="CS lab", confidence=0.85),
        },
    )
    service = make_service(canned)
    result = await service.analyze("The HP LaserJet in the CS lab has a paper jam.")
    assert 0.6 < result.completeness_score < 0.7
    assert "network_context" in result.missing_or_uncertain_fields