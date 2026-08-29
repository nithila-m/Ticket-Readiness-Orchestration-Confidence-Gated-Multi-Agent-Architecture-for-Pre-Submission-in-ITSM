"""
Ticket Analysis Service — combines Agent 1's two halves:
InformationExtractor (LLM-based extraction) and score_completeness
(deterministic scoring) into the single Agent 1 result.

Deliberately does NOT catch LLMProviderError - failures propagate to
the caller (the API layer) so they can be turned into a proper HTTP
response instead of being silently masked here.
"""

from app.agents.completeness_scorer import score_completeness
from app.agents.information_extractor import InformationExtractor
from app.schemas.analysis import AnalysisResult


class TicketAnalysisService:
    def __init__(self, extractor: InformationExtractor):
        self._extractor = extractor

    async def analyze(self, issue_text: str) -> AnalysisResult:
        extraction = await self._extractor.extract(issue_text)

        completeness = score_completeness(
            category=extraction.category.value,
            extracted_fields=extraction.extracted_fields,
        )

        return AnalysisResult(
            category=extraction.category,
            extracted_fields=extraction.extracted_fields,
            completeness_score=completeness.score,
            missing_or_uncertain_fields=completeness.missing_or_uncertain_fields,
        )