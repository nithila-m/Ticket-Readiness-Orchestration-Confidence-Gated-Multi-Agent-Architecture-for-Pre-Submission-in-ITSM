import json

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError

from app.config.category_profiles import VALID_CATEGORIES
from app.providers.base import LLMProvider
from app.providers.exceptions import LLMProviderError
from app.schemas.extraction import CategoryPrediction, ExtractedField, ExtractionResult

_LLM_CATEGORIES = ", ".join(c for c in VALID_CATEGORIES if c != "general")

EXTRACTION_SYSTEM_PROMPT = f"""You are an information extraction system for VIT's IT service desk.

Given a user's raw description of an IT issue, extract:
1. A category — MUST be exactly one of: {_LLM_CATEGORIES}.
   If none of these clearly apply, return null. Do NOT invent a category name.
2. Any of the following fields that are ACTUALLY MENTIONED in the text.
   Only extract fields relevant to the detected category. Return each as an
   object with field_name, value, and confidence:

   For wifi_internet: device_type, when_started, symptom_type,
     single_or_multiple_devices, ssid
   For ms_teams / vit_email: failure_type, scope, error_signal, device_platform
   For ad_account_creation: username_domain, when_started, error_or_symptom,
     device_context, troubleshooting_done
   For printer_support: printer_model, when_started, symptom, scope,
     connection_type, error_message, troubleshooting_done

STRICT RULES:
- Do NOT invent, guess, or infer information not present in the text.
- Do not output a field at all if it is not mentioned in the text — omit it
  from the fields list entirely rather than including it with a null value.
- confidence reflects how certain you are about an extracted value (0.0-1.0),
  based only on textual evidence, not plausibility.
"""


class WireExtractedField(BaseModel):
    """LLM-facing shape for a single field. Fixed properties only -
    Gemini's schema validator rejects open-ended dict/map types."""

    field_name: str
    value: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class WireExtractionResponse(BaseModel):
    """
    LLM-facing response schema. Gemini's structured-output mode does not
    support Pydantic's `dict[str, X]` fields (they compile to JSON Schema's
    `additionalProperties`, which the Gemini API rejects outright). This
    flat list-of-objects shape is a fixed schema Gemini can enforce, and
    gets converted into the real ExtractionResult (dict-keyed) domain
    model after parsing - callers of this provider never see this format.
    """

    category: CategoryPrediction
    fields: list[WireExtractedField] = Field(default_factory=list)


def _wire_to_extraction_result(wire: WireExtractionResponse) -> ExtractionResult:
    extracted_fields = {
        item.field_name: ExtractedField(value=item.value, confidence=item.confidence)
        for item in wire.fields
    }
    return ExtractionResult(category=wire.category, extracted_fields=extracted_fields)


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise LLMProviderError("Gemini API key is missing. Set GEMINI_API_KEY in .env.")
        self._client = genai.Client(api_key=api_key)
        self._model = model

    async def extract_information(self, issue_text: str) -> ExtractionResult:
        try:
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=f"{EXTRACTION_SYSTEM_PROMPT}\n\nUser issue:\n{issue_text}",
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=WireExtractionResponse,
                ),
            )
        except Exception as exc:
            raise LLMProviderError(f"Gemini API call failed: {exc}", cause=exc) from exc

        wire = response.parsed
        if wire is None:
            try:
                data = json.loads(response.text)
                wire = WireExtractionResponse.model_validate(data)
            except Exception as exc:
                raise LLMProviderError(
                    f"Gemini returned a response that failed schema validation: {exc}",
                    cause=exc,
                ) from exc

        try:
            return _wire_to_extraction_result(wire)
        except ValidationError as exc:
            raise LLMProviderError(
                f"Gemini returned an internally inconsistent field "
                f"(e.g. a null value with nonzero confidence): {exc}",
                cause=exc,
            ) from exc