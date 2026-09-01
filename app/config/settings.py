from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    llm_provider: str = "gemini"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.5-flash-lite"

    # Groq - Agent 2 (Adaptive Clarifier) - free tier, no credit card required
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"  # strict structured-output support confirmed
    max_clarification_turns: int = 3


settings = Settings()