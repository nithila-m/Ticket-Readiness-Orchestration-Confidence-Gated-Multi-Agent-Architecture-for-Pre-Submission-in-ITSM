from pydantic import BaseModel, Field

from app.schemas.extraction import CategoryPrediction, ExtractedField


class Message(BaseModel):
    role: str  # "user" or "system"
    content: str


class UserMessageRequest(BaseModel):
    """Incoming request body for POST /analyze."""

    message: str = Field(min_length=1)


class ConversationState(BaseModel):
    """
    Full shared state, forward-compatible with Agent 2.
    Agent 1 populates: raw_messages, extracted_fields, detected_category,
    completeness_score, missing_or_uncertain_fields.
    Agent 2 will later append to clarification_log and increment turn_count.
    """

    conversation_id: str
    turn_count: int = 0
    raw_messages: list[Message] = Field(default_factory=list)
    extracted_fields: dict[str, ExtractedField] = Field(default_factory=dict)
    detected_category: CategoryPrediction | None = None
    completeness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    missing_or_uncertain_fields: list[str] = Field(default_factory=list)
    clarification_log: list[str] = Field(default_factory=list)