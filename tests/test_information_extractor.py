import pytest

from app.agents.information_extractor import InformationExtractor
from app.providers.base import LLMProvider
from app.schemas.extraction import CategoryPrediction, ExtractedField, ExtractionResult


class StubProvider(LLMProvider):
    def __init__(self, canned_result: ExtractionResult):
        self._canned_result = canned_result

    async def extract_information(self, issue_text: str) -> ExtractionResult:
        return self._canned_result


@pytest.mark.asyncio
async def test_valid_category_passes_through_unchanged():
    canned = ExtractionResult(
        category=CategoryPrediction(value="wifi_internet", confidence=0.92),
        extracted_fields={"symptom_type": ExtractedField(value="drops", confidence=0.9)},
    )
    extractor = InformationExtractor(StubProvider(canned))
    result = await extractor.extract("My wifi keeps dropping in the library.")
    assert result.category.value == "wifi_internet"


@pytest.mark.asyncio
async def test_unrecognized_category_is_sanitized_to_none():
    canned = ExtractionResult(
        category=CategoryPrediction(value="totally_made_up_category", confidence=0.8),
        extracted_fields={},
    )
    extractor = InformationExtractor(StubProvider(canned))
    result = await extractor.extract("Something ambiguous happened.")
    assert result.category.value is None
    assert result.category.confidence == 0.0


@pytest.mark.asyncio
async def test_general_is_rejected_as_llm_output_since_its_internal_only():
    canned = ExtractionResult(
        category=CategoryPrediction(value="general", confidence=0.5),
        extracted_fields={},
    )
    extractor = InformationExtractor(StubProvider(canned))
    result = await extractor.extract("My thing is broken.")
    assert result.category.value is None


@pytest.mark.asyncio
async def test_none_category_passes_through_unchanged():
    canned = ExtractionResult(
        category=CategoryPrediction(value=None, confidence=0.0),
        extracted_fields={},
    )
    extractor = InformationExtractor(StubProvider(canned))
    result = await extractor.extract("asdkjf")
    assert result.category.value is None


@pytest.mark.asyncio
async def test_extracted_fields_pass_through_untouched():
    canned = ExtractionResult(
        category=CategoryPrediction(value="ms_teams", confidence=0.9),
        extracted_fields={
            "failure_type": ExtractedField(value="can't hear audio", confidence=0.88),
            "error_signal": ExtractedField(value=None, confidence=0.0),
        },
    )
    extractor = InformationExtractor(StubProvider(canned))
    result = await extractor.extract("Teams audio doesn't work for me.")
    assert result.extracted_fields["failure_type"].value == "can't hear audio"
    assert result.extracted_fields["error_signal"].value is None


@pytest.mark.asyncio
async def test_realistic_ad_account_message_end_to_end_with_stub():
    canned = ExtractionResult(
        category=CategoryPrediction(value="ad_account_creation", confidence=0.9),
        extracted_fields={
            "error_or_symptom": ExtractedField(value="account locked", confidence=0.95),
            "when_started": ExtractedField(value=None, confidence=0.0),
        },
    )
    extractor = InformationExtractor(StubProvider(canned))
    result = await extractor.extract("my account is locked not able to login")
    assert result.category.value == "ad_account_creation"
    assert result.extracted_fields["when_started"].value is None