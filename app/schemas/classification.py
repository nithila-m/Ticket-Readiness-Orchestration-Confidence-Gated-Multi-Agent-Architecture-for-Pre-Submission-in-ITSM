"""
Agent 5 (Classify, Prioritise, Explain) LLM wire schema.

ClassificationLLMOutput is what the model is asked to return - NOT the
pipeline's ClassificationResult. It is deliberately flat (no dict
fields: Gemini/Groq structured-output schemas reject or mishandle them)
and deliberately has no assignee_group: the team is always looked up in
app/config/routing.py, never chosen by the model.

priority excludes P1 on purpose - P1 is only ever set by Agent 5's
deterministic INCIDENT_CLUSTER override, never by the model.
"""

from typing import Literal

from pydantic import BaseModel, Field


class ClassificationLLMOutput(BaseModel):
    """The model's raw classification judgment, before Agent 5's overrides."""

    category: str
    priority: Literal["P2", "P3", "P4"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)
    impact_signal: Literal["single_user", "multiple_users", "unknown"]
