from pydantic import BaseModel, Field

from app.schemas.extraction import CategoryPrediction, ExtractedField


class AnalysisResult(BaseModel):
    """Final Agent 1 output: extraction + completeness score."""

    category: CategoryPrediction
    extracted_fields: dict[str, ExtractedField] = Field(default_factory=dict)
    completeness_score: float = Field(ge=0.0, le=1.0)
    missing_or_uncertain_fields: list[str] = Field(default_factory=list)