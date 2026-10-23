"""
Groq provider for Agent 5 (Classify, Prioritise, Explain) - free tier.

Same pattern as groq_clarification_provider.py: Groq strict json_schema
structured output (constrained decoding, confirmed on openai/gpt-oss-20b
and openai/gpt-oss-120b), with every property required and
additionalProperties: false, plus a manual json.loads + Pydantic
validate in case structured output is silently ignored.

Prompt and state serialization live in classification_prompts.py - this
file only owns the API call, schema enforcement, and response parsing.
Every failure surfaces as LLMProviderError; ClassificationAgent turns
that into a rule-based fallback rather than letting it propagate.
"""

import json

from groq import AsyncGroq
from pydantic import ValidationError

from app.agents.classification_prompts import SYSTEM_PROMPT, build_agent5_context
from app.providers.classification_base import ClassificationProvider
from app.providers.exceptions import LLMProviderError
from app.providers.groq_clarification_provider import _to_strict_schema
from app.schemas.classification import ClassificationLLMOutput
from app.schemas.conversation import ConversationState


class GroqClassificationProvider(ClassificationProvider):
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise LLMProviderError(
                "Groq API key is missing. Set GROQ_API_KEY in .env "
                "(free, no credit card - console.groq.com)."
            )
        self._client = AsyncGroq(api_key=api_key)
        self._model = model
        self._schema = _to_strict_schema(ClassificationLLMOutput)

    async def classify_ticket(self, state: ConversationState) -> ClassificationLLMOutput:
        user_context = build_agent5_context(state)

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
                        "name": "classification_output",
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
            return ClassificationLLMOutput.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise LLMProviderError(
                f"Groq returned a response that didn't match the schema "
                f"(structured output may have been silently ignored): {exc}",
                cause=exc,
            ) from exc
