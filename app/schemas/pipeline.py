from datetime import datetime, timezone
from typing import Literal
from pydantic import BaseModel, Field, field_validator

from app.config.category_profiles import VALID_CATEGORIES
from app.config.routing import CATEGORY_TO_TEAM, DEFAULT_TEAM

TerminalOutcome = Literal["SELF_RESOLVED", "DISPATCHED", "MERGED", "PENDING_REVIEW", "ESCALATED"]
TicketStatus = Literal["OPEN", "MERGED", "PENDING_REVIEW", "CLOSED"]
Priority = Literal["P1", "P2", "P3", "P4"]

_VALID_TEAMS = set(CATEGORY_TO_TEAM.values()) | {DEFAULT_TEAM}


def _check_category(value: str) -> str:
    # Internal snake_case keys only; display labels must be converted first.
    if value not in VALID_CATEGORIES:
        raise ValueError(
            f"category {value!r} is not an internal category key {VALID_CATEGORIES}; "
            "convert display labels with "
            "app.config.category_profiles.normalize_category_label()"
        )
    return value


def _check_team(value: str) -> str:
    # assignee_group always comes from app.config.routing, never free text.
    if value not in _VALID_TEAMS:
        raise ValueError(
            f"assignee_group {value!r} is not a team in app.config.routing "
            f"{sorted(_VALID_TEAMS)}"
        )
    return value

class SimilarTicket(BaseModel):
    ticket_id: str
    similarity: float = Field(ge=0.0, le=1.0)
    created_at: datetime | None = None

class CorrelationResult(BaseModel):                 # Agent 4 output (Medhini)
    outcome: Literal["DUPLICATE", "INCIDENT_CLUSTER", "NO_MATCH"]
    duplicate_of: str | None = None
    cluster_id: str | None = None
    similar_tickets: list[SimilarTicket] = Field(default_factory=list)
    max_similarity: float = Field(default=0.0, ge=0.0, le=1.0)

class ClassificationResult(BaseModel):              # Agent 5 output (Trisha)
    category: str
    priority: Priority
    assignee_group: str
    routing_rationale: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    source: Literal["llm", "rule", "stub"]

    validate_category = field_validator("category")(_check_category)
    validate_team = field_validator("assignee_group")(_check_team)

class TicketRecord(BaseModel):                      # Agent 6 output (canonical schema)
    ticket_id: str | None = None                    # assigned by the store on insert
    conversation_id: str
    title: str = Field(min_length=1, max_length=255)
    description: str
    category: str
    priority: Priority
    assignee_group: str
    routing_rationale: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    kb_articles_checked: list[str] = Field(default_factory=list)
    duplicate_of: str | None = None
    incident_cluster_id: str | None = None
    building: str | None = None
    status: TicketStatus = "OPEN"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    external_key: str | None = None                 # e.g. Jira "TRO-12"
    external_url: str | None = None

    validate_category = field_validator("category")(_check_category)

ReviewStatus = Literal["PENDING", "APPROVED", "REJECTED"]

class ReviewItem(BaseModel):                        # human review queue (confidence gate / escalation)
    id: int
    conversation_id: str
    reason: str
    status: ReviewStatus = "PENDING"
    state_json: str                                 # ConversationState snapshot at enqueue time
    created_at: datetime
    resolved_at: datetime | None = None

class AuditEvent(BaseModel):
    conversation_id: str
    turn: int
    stage: str
    decision: str
    confidence: float | None = None
    payload: dict = Field(default_factory=dict)
    latency_ms: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))