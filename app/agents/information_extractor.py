"""
Agent 1 orchestration: Information Extractor.

Wraps an LLMProvider to produce a validated ExtractionResult. Prompting
alone (see gemini_provider.py) is not a guarantee — this adds a second,
code-level check that rejects any category the LLM returns outside TRO's
known set, rather than trusting instruction-following blindly.
"""

from app.config.category_profiles import VALID_CATEGORIES
from app.providers.base import LLMProvider
from app.schemas.extraction import CategoryPrediction, ExtractionResult

# "general" is an internal fallback profile for the scorer - it has no
# real signal in the input text, so the LLM should never claim to have
# detected it. Treat it the same as any other out-of-vocabulary label.
_LLM_ASSIGNABLE_CATEGORIES = {c for c in VALID_CATEGORIES if c != "general"}


class InformationExtractor:
    def __init__(self, provider: LLMProvider):
        self._provider = provider

    async def extract(self, issue_text: str) -> ExtractionResult:
        result = await self._provider.extract_information(issue_text)
        return self._sanitize_category(result)

    @staticmethod
    def _sanitize_category(result: ExtractionResult) -> ExtractionResult:
        category = result.category
        if category.value is not None and category.value not in _LLM_ASSIGNABLE_CATEGORIES:
            return result.model_copy(
                update={"category": CategoryPrediction(value=None, confidence=0.0)}
            )
        return result