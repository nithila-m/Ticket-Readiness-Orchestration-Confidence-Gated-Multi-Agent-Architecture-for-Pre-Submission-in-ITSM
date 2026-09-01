from typing import Any
from pydantic import BaseModel, Field, model_validator


class ExtractedField(BaseModel):
    """A single extracted piece of information with its confidence."""

    value: Any | None = None
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def enforce_null_value_zero_confidence(self):
        if self.value is None and self.confidence != 0.0:
            raise ValueError("Confidence must be 0 when extracted value is None")
        return self


class CategoryPrediction(BaseModel):
    """The detected issue category with confidence. None = extractor couldn't determine it."""

    value: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def enforce_null_value_zero_confidence(self):
        if self.value is None and self.confidence != 0.0:
            raise ValueError("Confidence must be 0 when category is None")
        return self


class ExtractionResult(BaseModel):
    """Raw output from the Information Extractor (LLM call), before scoring."""

    category: CategoryPrediction
    extracted_fields: dict[str, ExtractedField] = Field(default_factory=dict)