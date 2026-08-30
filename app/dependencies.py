from functools import lru_cache

from app.agents.information_extractor import InformationExtractor
from app.config.settings import settings
from app.providers.gemini_provider import GeminiProvider
from app.services.ticket_analysis_service import TicketAnalysisService


@lru_cache
def get_ticket_analysis_service() -> TicketAnalysisService:
    """
    Builds the real, Gemini-backed TicketAnalysisService.
    Cached as a singleton - safe because GeminiProvider holds no
    per-request state. Only instantiated on first actual call (i.e.
    the first /analyze request), not at app startup, so /health
    doesn't require an API key to work.
    """
    provider = GeminiProvider(api_key=settings.gemini_api_key or "", model=settings.gemini_model)
    extractor = InformationExtractor(provider)
    return TicketAnalysisService(extractor)