import json

from google import genai
from google.genai import types

from app.providers.base import LLMProvider
from app.providers.exceptions import LLMProviderError
from app.schemas.extraction import ExtractionResult

from app.config.category_profiles import VALID_CATEGORIES

_LLM_CATEGORIES = ", ".join(c for c in VALID_CATEGORIES if c != "general")

EXTRACTION_SYSTEM_PROMPT = f"""You are an information extraction system for VIT's IT service desk.

Given a user's raw description of an IT issue, extract:
1. A category — MUST be exactly one of: {_LLM_CATEGORIES}.
   If none of these clearly apply, return null. Do NOT invent a category name.
2. Any of the following fields that are ACTUALLY MENTIONED in the text.
   Only extract fields relevant to the detected category:

   For wifi_internet: device_type, when_started, symptom_type,
     single_or_multiple_devices, ssid
   For ms_teams / vit_email: failure_type, scope, error_signal, device_platform
   For ad_account_creation: username_domain, when_started, error_or_symptom,
     device_context, troubleshooting_done
   For printer_support: printer_model, when_started, symptom, scope,
     connection_type, error_message, troubleshooting_done

STRICT RULES:
- Do NOT invent, guess, or infer information not present in the text.
- If a field is not mentioned, its value must be null and confidence must be 0.0.
- confidence reflects how certain you are about an extracted value (0.0-1.0),
  based only on textual evidence, not plausibility.
- Only include fields in extracted_fields that are relevant to what was said —
  do not pad the output with every possible field name.
"""
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
                    response_schema=ExtractionResult,
                ),
            )
        except Exception as exc:  # SDK raises its own error types; normalize them here
            raise LLMProviderError(f"Gemini API call failed: {exc}", cause=exc) from exc

        if response.parsed is not None:
            return response.parsed

        # Fallback: some SDK versions/paths return raw text needing manual parsing
        try:
            data = json.loads(response.text)
            return ExtractionResult.model_validate(data)
        except Exception as exc:
            raise LLMProviderError(
                f"Gemini returned a response that failed schema validation: {exc}",
                cause=exc,
            ) from exc