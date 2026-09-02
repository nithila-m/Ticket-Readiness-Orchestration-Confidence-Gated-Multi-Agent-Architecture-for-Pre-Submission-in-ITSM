from pydantic import BaseModel, Field
from typing import Literal

from app.schemas.clarification import ClarificationLogEntry
from app.schemas.extraction import CategoryPrediction, ExtractedField


class Message(BaseModel):
    # "assistant" covers Agent 2's clarifying questions when they're
    # folded back into the transcript that gets re-sent to Agent 1's
    # extractor each turn. "system" is reserved, currently unused.
    role: Literal["user", "assistant", "system"]
    content: str


class UserMessageRequest(BaseModel):
    """Incoming request body for POST /analyze."""

    message: str = Field(min_length=1)


class ConversationState(BaseModel):
    """
    Full shared state, forward-compatible with Agent 2 and Agent 3.
    Agent 1 populates: raw_messages, extracted_fields, detected_category,
    completeness_score, missing_or_uncertain_fields.
    Agent 2 appends to clarification_log and increments turn_count.
    Agent 3 populates: kb_outcome, kb_similarity_score, kb_articles_checked,
    and (only for STRONG_MATCH/WEAK_MATCH) kb_matched_kb_id/title/resolution.
    """

    conversation_id: str
    turn_count: int = 0
    raw_messages: list[Message] = Field(default_factory=list)
    extracted_fields: dict[str, ExtractedField] = Field(default_factory=dict)
    detected_category: CategoryPrediction | None = None
    completeness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    missing_or_uncertain_fields: list[str] = Field(default_factory=list)
    clarification_log: list[ClarificationLogEntry] = Field(default_factory=list)

    # --- Agent 3 (KB Retrieval / Deflection) ---
    # None = Agent 3 hasn't run yet this conversation. Once it has, this is
    # always one of STRONG_MATCH / WEAK_MATCH / NO_MATCH - deliberately no
    # separate "kb_match_found" bool, since that's just kb_outcome != NO_MATCH
    # and a second field would only risk drifting out of sync with this one.
    kb_outcome: Literal["STRONG_MATCH", "WEAK_MATCH", "NO_MATCH"] | None = None
    kb_similarity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    kb_articles_checked: list[str] = Field(default_factory=list)
    kb_matched_kb_id: str | None = None
    kb_matched_title: str | None = None
    kb_offered_resolution: str | None = None