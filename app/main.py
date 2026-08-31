from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.clarification_router import router as clarification_router
from app.config.settings import settings
from app.dependencies import get_ticket_analysis_service
from app.providers.exceptions import LLMProviderError
from app.schemas.analysis import AnalysisResult
from app.schemas.conversation import UserMessageRequest
from app.services.ticket_analysis_service import TicketAnalysisService

app = FastAPI(title="TRO - Ticket Readiness Orchestration", version="0.1.0")

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