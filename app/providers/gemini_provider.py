import json

from google import genai
from google.genai import types

from app.providers.base import LLMProvider
from app.providers.exceptions import LLMProviderError
from app.schemas.extraction import ExtractionResult

EXTRACTION_SYSTEM_PROMPT = """You are an information extraction system for an IT service desk.

Given a user's raw description of an IT issue, extract:
1. A category for the issue (or null if genuinely unclear).
2. Any of the following fields that are ACTUALLY MENTIONED in the text:
   affected_system, location, trigger, frequency, error_message, device,
   action_attempted, requester_role, department, account_type,
   required_by_date, approval_reference, network_context.

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