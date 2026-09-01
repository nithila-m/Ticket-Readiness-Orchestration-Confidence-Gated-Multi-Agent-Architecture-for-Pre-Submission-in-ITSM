from abc import ABC, abstractmethod

from app.schemas.extraction import ExtractionResult


class LLMProvider(ABC):
    """Abstract interface for any LLM-based information extractor."""

    @abstractmethod
    async def extract_information(self, issue_text: str) -> ExtractionResult:
        """
        Extract structured fields + category from raw user issue text.
        Must never invent values — missing info should come back as
        value=None, confidence=0.0 (enforced downstream by ExtractionResult's
        own validators).
        """
        raise NotImplementedError