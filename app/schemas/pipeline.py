from datetime import datetime, timezone
from typing import Literal
from pydantic import BaseModel, Field

TerminalOutcome = Literal["SELF_RESOLVED", "DISPATCHED", "MERGED", "PENDING_REVIEW", "ESCALATED"]
TicketStatus = Literal["OPEN", "MERGED", "PENDING_REVIEW", "CLOSED"]
Priority = Literal["P1", "P2", "P3", "P4"]

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
    source: Literal["llm", "hgb", "rule", "stub"]

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

class AuditEvent(BaseModel):
    conversation_id: str
    turn: int
    stage: str
    decision: str
    confidence: float | None = None
    payload: dict = Field(default_factory=dict)
    latency_ms: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))