"""
Agent 2 (Adaptive Clarifier) output schema.

ClarificationDecision is the LLM's structured judgment call, not a
generated string. See ADAPTIVE_CLARIFIER_SYSTEM_PROMPT for the reasoning
this schema is meant to capture.
"""

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ClarificationDecision(BaseModel):
    """A single Agent 2 decision: what to do next, and why."""

    action: Literal["ASK_CLARIFICATION", "READY", "RECHECK", "ESCALATE"]
    reasoning: str = Field(min_length=1)
    information_gap: str | None = None
    question: str | None = None
    expected_information_gain: float = Field(ge=0.0, le=1.0)
    affected_fields: list[str] = Field(default_factory=list)
    priority: Literal["low", "medium", "high"]
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def enforce_question_matches_action(self):
        if self.action == "ASK_CLARIFICATION" and not self.question:
            raise ValueError("ASK_CLARIFICATION requires a non-empty question")
        if self.action != "ASK_CLARIFICATION" and self.question is not None:
            raise ValueError(f"question must be null when action is {self.action}")
        return self


class ClarificationLogEntry(BaseModel):
    """
    One turn's audit record. Stored on ConversationState.clarification_log.
    This is what makes the 'genuinely agentic' claim demonstrable rather
    than just asserted (digest.txt Section 19).
    """

    turn: int
    user_message: str
    decision: ClarificationDecision
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))