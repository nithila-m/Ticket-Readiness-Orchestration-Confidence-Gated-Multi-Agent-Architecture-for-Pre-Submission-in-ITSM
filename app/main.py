from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.clarification_router import router as clarification_router
from app.config.settings import settings
from app.dependencies import get_dispatcher, get_store, get_ticket_analysis_service
from app.providers.exceptions import LLMProviderError
from app.schemas.analysis import AnalysisResult
from app.schemas.conversation import UserMessageRequest
from app.services.ticket_analysis_service import TicketAnalysisService


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create the SQLite tables (tickets, audit_events) once at startup.
    # Idempotent - safe on every restart. Goes through app.dependency_overrides
    # if a test has swapped the store, so tests never touch the real tro.db.
    store_factory = app.dependency_overrides.get(get_store, get_store)
    store_factory().init_db()
    yield
    # Close the Jira HTTP client, but only if a dispatcher was ever built -
    # never construct one (or touch Jira settings) just to shut it down.
    if get_dispatcher.cache_info().currsize:
        aclose = getattr(get_dispatcher().adapter, "aclose", None)
        if aclose is not None:
            await aclose()


app = FastAPI(
    title="TRO - Ticket Readiness Orchestration",
    version="0.2.0",
    lifespan=lifespan,
)

# Demo-only CORS: allows the frontend to be opened from a different origin
# (e.g. a separate dev server) if you choose not to use the static mount
# below. Same-origin requests (the default /chat mount) don't need this,
# but it's harmless to leave on for a local demo.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(clarification_router)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "environment": settings.app_env,
    }


@app.post("/analyze", response_model=AnalysisResult)
async def analyze_ticket(
    request: UserMessageRequest,
    service: TicketAnalysisService = Depends(get_ticket_analysis_service),
) -> AnalysisResult:
    try:
        return await service.analyze(request.message)
    except LLMProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


# Serves frontend/index.html, style.css, app.js at /chat/*.
# Placed after the API routes so it can never shadow them.
app.mount("/chat", StaticFiles(directory="frontend", html=True), name="frontend")