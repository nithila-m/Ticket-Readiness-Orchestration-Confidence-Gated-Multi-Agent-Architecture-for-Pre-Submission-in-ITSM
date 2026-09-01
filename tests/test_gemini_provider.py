from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.providers.exceptions import LLMProviderError
from app.providers.gemini_provider import GeminiProvider, WireExtractedField, WireExtractionResponse
from app.schemas.extraction import CategoryPrediction


def make_provider() -> GeminiProvider:
    with patch("app.providers.gemini_provider.genai.Client"):
        return GeminiProvider(api_key="fake-key", model="gemini-3.5-flash-lite")


def test_missing_api_key_raises_immediately():
    with pytest.raises(LLMProviderError):
        GeminiProvider(api_key="", model="gemini-2.5-flash-lite")


@pytest.mark.asyncio
async def test_successful_extraction_converts_wire_format_to_extraction_result():
    wire = WireExtractionResponse(
        category=CategoryPrediction(value="wifi_internet", confidence=0.9),
        fields=[WireExtractedField(field_name="symptom_type", value="drops", confidence=0.95)],
    )
    mock_response = MagicMock()
    mock_response.parsed = wire

    provider = make_provider()
    provider._client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    result = await provider.extract_information("My wifi keeps dropping.")
    assert result.category.value == "wifi_internet"
    assert result.extracted_fields["symptom_type"].value == "drops"


@pytest.mark.asyncio
async def test_falls_back_to_manual_json_parse_when_unparsed():
    raw_json = (
        '{"category": {"value": "printer_support", "confidence": 0.85}, '
        '"fields": [{"field_name": "symptom", "value": "paper jam", "confidence": 0.9}]}'
    )
    mock_response = MagicMock()
    mock_response.parsed = None
    mock_response.text = raw_json

    provider = make_provider()
    provider._client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    result = await provider.extract_information("Printer keeps jamming.")
    assert result.category.value == "printer_support"
    assert result.extracted_fields["symptom"].value == "paper jam"


@pytest.mark.asyncio
async def test_sdk_exception_raises_llm_provider_error():
    provider = make_provider()
    provider._client.aio.models.generate_content = AsyncMock(side_effect=RuntimeError("network down"))

    with pytest.raises(LLMProviderError):
        await provider.extract_information("Something broke.")


@pytest.mark.asyncio
async def test_malformed_json_raises_llm_provider_error():
    mock_response = MagicMock()
    mock_response.parsed = None
    mock_response.text = "not valid json at all"

    provider = make_provider()
    provider._client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    with pytest.raises(LLMProviderError):
        await provider.extract_information("Something vague.")


@pytest.mark.asyncio
async def test_internally_inconsistent_field_raises_llm_provider_error():
    # Simulates Gemini ignoring the "omit rather than null" instruction and
    # returning a null value with nonzero confidence - should surface as a
    # clear provider error, not an unhandled crash.
    wire = WireExtractionResponse(
        category=CategoryPrediction(value="wifi_internet", confidence=0.9),
        fields=[WireExtractedField(field_name="symptom_type", value=None, confidence=0.4)],
    )
    mock_response = MagicMock()
    mock_response.parsed = wire

    provider = make_provider()
    provider._client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    with pytest.raises(LLMProviderError):
        await provider.extract_information("Something vague.")