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

Prompt and state serialization live in adaptive_clarifier_prompts.py and
clarification_context.py respectively (M5.4) - this file only owns the
API call, schema enforcement, and response parsing.
"""

import json
from typing import Any

from groq import AsyncGroq
from pydantic import BaseModel, ValidationError

from app.agents.adaptive_clarifier_prompts import SYSTEM_PROMPT
from app.agents.clarification_context import build_agent2_context
from app.config.settings import settings
from app.providers.clarification_base import ClarificationProvider
from app.providers.exceptions import LLMProviderError
from app.schemas.clarification import ClarificationDecision
from app.schemas.conversation import ConversationState


def _to_strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """
    Convert a Pydantic model's JSON schema into Groq's strict-mode shape:
    every property required (nullable fields get a type union instead of
    being optional), additionalProperties: false at every object level.
    """
    schema = model.model_json_schema()
    schema["required"] = list(schema.get("properties", {}).keys())
    schema["additionalProperties"] = False
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
        user_context = build_agent2_context(
            state, max_turns=settings.max_clarification_turns
        )

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_context},
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