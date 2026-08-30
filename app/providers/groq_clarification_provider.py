"""
Groq provider for Agent 2 (Adaptive Clarifier) - free tier.

Uses Groq's strict json_schema structured-output mode (constrained
decoding), currently confirmed only on openai/gpt-oss-20b and
openai/gpt-oss-120b. Strict mode requires every schema property to be
listed in "required" (nullable fields use a ["type","null"] union
instead of being omitted) and "additionalProperties": false throughout -
this is handled by _to_strict_schema below rather than trusting
Pydantic's default model_json_schema() output as-is.

Defensive fallback mirrors gemini_provider.py: if structured output is
silently ignored (a known occasional Groq issue) and the model returns
free-form text instead, we attempt a manual json.loads + Pydantic
validate before giving up.

The prompt and state-serialization below are placeholders (M5.3 scope
is proving the wire works end-to-end) - both get replaced by the real
system prompt and a dedicated serializer in M5.4.
"""

import json
from typing import Any

from groq import AsyncGroq
from pydantic import BaseModel, ValidationError

from app.providers.clarification_base import ClarificationProvider
from app.providers.exceptions import LLMProviderError
from app.schemas.clarification import ClarificationDecision
from app.schemas.conversation import ConversationState

# PLACEHOLDER - replaced by the real prompt in M5.4.
_PLACEHOLDER_SYSTEM_PROMPT = """You are Agent 2, an adaptive clarification
agent for VIT's IT service desk. Given the conversation state below,
decide the next best action: ASK_CLARIFICATION, READY, RECHECK, or ESCALATE.
Do not ask for information already present. Prefer one high-value question
over several low-value ones. Never invent facts not present in the text.
Respond with a single JSON object matching the required schema exactly."""


def _placeholder_serialize_state(state: ConversationState) -> str:
    """PLACEHOLDER - replaced by a proper serializer in M5.4."""
    messages = "\n".join(f"{m.role}: {m.content}" for m in state.raw_messages)
    category = state.detected_category.value if state.detected_category else None
    return (
        f"Conversation so far:\n{messages}\n\n"
        f"Detected category: {category}\n"
        f"Completeness score: {state.completeness_score}\n"
        f"Missing or uncertain fields: {state.missing_or_uncertain_fields}\n"
        f"Turn count: {state.turn_count}"
    )


def _to_strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """
    Convert a Pydantic model's JSON schema into Groq's strict-mode shape:
    every property required (nullable fields get a type union instead of
    being optional), additionalProperties: false at every object level.
    """
    schema = model.model_json_schema()
    schema["required"] = list(schema.get("properties", {}).keys())
    schema["additionalProperties"] = False
    for prop in schema.get("properties", {}).values():
        # Optional[str]-style fields compile to anyOf: [{"type":"string"},{"type":"null"}]
        # already valid for strict mode as-is - no change needed there.
        if "$ref" in prop or "$defs" in schema:
            pass  # nested $defs (e.g. enums) don't need additionalProperties patching here
    return schema


class GroqClarificationProvider(ClarificationProvider):
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise LLMProviderError(
                "Groq API key is missing. Set GROQ_API_KEY in .env "
                "(free, no credit card - console.groq.com)."
            )
        self._client = AsyncGroq(api_key=api_key)
        self._model = model
        self._schema = _to_strict_schema(ClarificationDecision)

    async def decide_clarification(self, state: ConversationState) -> ClarificationDecision:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _PLACEHOLDER_SYSTEM_PROMPT},
                    {"role": "user", "content": _placeholder_serialize_state(state)},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "clarification_decision",
                        "schema": self._schema,
                        "strict": True,
                    },
                },
            )
        except Exception as exc:
            raise LLMProviderError(f"Groq API call failed: {exc}", cause=exc) from exc

        raw_content = response.choices[0].message.content
        if raw_content is None:
            raise LLMProviderError("Groq returned an empty response.")

        try:
            data = json.loads(raw_content)
            return ClarificationDecision.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise LLMProviderError(
                f"Groq returned a response that didn't match the schema "
                f"(structured output may have been silently ignored): {exc}",
                cause=exc,
            ) from exc