from functools import lru_cache

from app.agents.adaptive_clarifier import AdaptiveClarifier
from app.agents.classification_agent import ClassificationAgent
from app.agents.information_extractor import InformationExtractor
from app.agents.kb_deflection_agent import KBDeflectionAgent
from app.config.settings import settings
from app.providers.gemini_provider import GeminiProvider
from app.providers.groq_classification_provider import GroqClassificationProvider
from app.providers.groq_clarification_provider import GroqClarificationProvider
from app.repositories.in_memory_conversation_repository import InMemoryConversationRepository
from app.services.clarification_service import ClarificationService
from app.services.ticket_analysis_service import TicketAnalysisService
from app.storage.sqlite_store import SqliteStore


@lru_cache
def get_store() -> SqliteStore:
    """
    Singleton SQLite store for tickets (Agent 6) and the audit trail.
    The path comes from settings.sqlite_path (repo-root tro.db by default).
    Tables are created in app.main's lifespan handler via init_db(), not
    here, so importing this module never touches the disk.
    Tests override this with app.dependency_overrides[get_store].
    """
    return SqliteStore(settings.sqlite_path)


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


@lru_cache
def get_conversation_repository() -> InMemoryConversationRepository:
    """
    Singleton in-memory store for ConversationState, keyed by conversation_id.
    Cached so all requests across the app's lifetime share the same store -
    a fresh instance per request would make multi-turn conversations
    impossible, since state would never persist between calls.
    """
    return InMemoryConversationRepository()


@lru_cache
def get_kb_deflection_agent() -> KBDeflectionAgent:
    """
    Builds the real, ChromaDB-backed KBDeflectionAgent (Agent 3). Uses the
    module's default retrieve_fn (TRO_Codes' deflect(), via the
    run_kb_retrieval bridge) - only tests override this with a fake.
    """
    return KBDeflectionAgent()


@lru_cache
def get_classification_agent() -> ClassificationAgent:
    """
    Builds the real, Groq-backed ClassificationAgent (Agent 5). Cached as a
    singleton; only instantiated on first actual call, not at app startup,
    so the app boots without GROQ_API_KEY. Not wired into
    ClarificationService - the orchestrator / Agent 6 own that call.
    """
    provider = GroqClassificationProvider(
        api_key=settings.groq_api_key, model=settings.classification_groq_model
    )
    return ClassificationAgent(provider=provider)


@lru_cache
def get_clarification_service() -> ClarificationService:
    """
    Builds the real, Groq-backed ClarificationService for Agent 2, plus
    Agent 3's KBDeflectionAgent.
    Reuses get_ticket_analysis_service() for Agent 1 rather than
    constructing a second GeminiProvider/extractor - one extraction
    pipeline shared by both /analyze and the conversational endpoint.
    Cached as a singleton; only instantiated on first actual call to the
    conversational endpoint, not at app startup - mirrors
    get_ticket_analysis_service's own no-key-needed-at-startup property.
    """
    extraction_service = get_ticket_analysis_service()

    groq_provider = GroqClarificationProvider(
        api_key=settings.groq_api_key, model=settings.groq_model
    )
    clarifier = AdaptiveClarifier(
        provider=groq_provider, max_turns=settings.max_clarification_turns
    )

    repository = get_conversation_repository()
    kb_agent = get_kb_deflection_agent()

    return ClarificationService(extraction_service, clarifier, repository, kb_agent)