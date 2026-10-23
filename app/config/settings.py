from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

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

    # Agent 3 (KB Retrieval / Deflection) - Completeness Gate
    # Agent 3 only runs once completeness_score reaches this floor. Below
    # it, a message is too vague to trust an embedding match against (e.g.
    # a bare "wifi not working" could coincidentally sit close to some KB
    # article by wording alone) - defer to Agent 2 clarifying first.
    kb_gate_completeness_threshold: float = 0.5
    # Cosine-similarity cut-offs for deflect()'s three outcomes:
    # >= strong -> STRONG_MATCH (deflect), >= weak -> WEAK_MATCH (Agent 2
    # still runs), below weak -> NO_MATCH. Defaults reproduce deflect()'s
    # own hardcoded defaults; tune with scripts/sweep_agent3_thresholds.py.
    kb_strong_threshold: float = 0.72
    kb_weak_threshold: float = 0.45

    # --- Agent 5 (Classify, Prioritise, Explain) ---
    # Empty = use groq_model (resolved in _default_classification_model), so
    # overriding GROQ_MODEL alone still moves both agents together.
    classification_groq_model: str = ""
    # Owned by the orchestrator's confidence gate, NOT by Agent 5: Agent 5
    # always returns a result and never compares against this itself.
    # Below it, the orchestrator should route to PENDING_REVIEW instead
    # of dispatching.
    classification_confidence_threshold: float = 0.6

    # --- Persistence (Agent 6 + audit) ---
    sqlite_path: Path = _PROJECT_ROOT / "tro.db"

    # --- Agent 6: external dispatch (Jira Service Management) ---
    jira_enabled: bool = False
    jira_base_url: str = ""          # https://<site>.atlassian.net
    jira_email: str = ""
    jira_api_token: str = ""
    jira_project_key: str = "TRO"
    jira_issue_type: str = "Task"    # set from createmeta (Phase 4.1)
    jira_priority_map: dict[str, str] = {"P1": "Highest", "P2": "High", "P3": "Medium", "P4": "Low"}
    jira_timeout_seconds: float = 10.0

    # Interim wiring until the LangGraph orchestrator replaces ClarificationService
    auto_dispatch_on_ready: bool = False

    @model_validator(mode="after")
    def _default_classification_model(self):
        if not self.classification_groq_model:
            self.classification_groq_model = self.groq_model
        return self


settings = Settings()