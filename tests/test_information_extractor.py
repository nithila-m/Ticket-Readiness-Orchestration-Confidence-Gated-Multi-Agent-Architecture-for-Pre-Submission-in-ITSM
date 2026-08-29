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
        extracted_fields={"affected_system": ExtractedField(value="wifi", confidence=0.9)},
    )
    extractor = InformationExtractor(StubProvider(canned))
    result = await extractor.extract("My wifi keeps dropping in the library.")
    assert result.category.value == "wifi_internet"
    assert result.category.confidence == 0.92


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
            "trigger": ExtractedField(value="joining a call", confidence=0.88),
            "error_message": ExtractedField(value=None, confidence=0.0),
        },
    )
    extractor = InformationExtractor(StubProvider(canned))
    result = await extractor.extract("Teams freezes whenever I join a call.")
    assert result.extracted_fields["trigger"].value == "joining a call"
    assert result.extracted_fields["error_message"].value is None


@pytest.mark.asyncio
async def test_realistic_wifi_style_message_end_to_end_with_stub():
    canned = ExtractionResult(
        category=CategoryPrediction(value="wifi_internet", confidence=0.95),
        extracted_fields={
            "affected_system": ExtractedField(value="wifi", confidence=0.98),
            "trigger": ExtractedField(value="after laptop wakes from sleep", confidence=0.91),
            "frequency": ExtractedField(value=None, confidence=0.0),
        },
    )
    extractor = InformationExtractor(StubProvider(canned))
    result = await extractor.extract(
        "My wifi keeps disconnecting after I wake my laptop from sleep."
    )
    assert result.category.value == "wifi_internet"
    assert result.extracted_fields["frequency"].value is None