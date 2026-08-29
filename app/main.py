from fastapi import Depends, FastAPI, HTTPException

from app.core.settings import settings
from app.dependencies import get_ticket_analysis_service
from app.providers.exceptions import LLMProviderError
from app.schemas.analysis import AnalysisResult
from app.schemas.conversation import UserMessageRequest
from app.services.ticket_analysis_service import TicketAnalysisService

app = FastAPI(title="TRO - Ticket Readiness Orchestration", version="0.1.0")


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