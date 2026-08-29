from fastapi import FastAPI

from app.core.settings import settings

app = FastAPI(title="TRO - Ticket Readiness Orchestration", version="0.1.0")


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "environment": settings.app_env,
    }