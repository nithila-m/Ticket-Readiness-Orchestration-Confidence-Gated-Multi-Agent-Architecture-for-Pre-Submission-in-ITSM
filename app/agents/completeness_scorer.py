"""
Deterministic completeness scoring for Agent 1.

completeness_score = sum(weight * confidence) / sum(weight)
                      over all fields defined in the category's profile.

A field counts as "missing_or_uncertain" if:
  - it was never extracted at all (absent from extracted_fields), OR
  - its value is None, OR
  - its confidence is below CONFIDENCE_THRESHOLD

This logic is intentionally pure and LLM-free: it only reasons over
already-extracted structured data, so it's independently testable
without any API calls.
"""

from typing import NamedTuple

from app.config.category_profiles import get_profile
from app.schemas.extraction import ExtractedField

CONFIDENCE_THRESHOLD = 0.5


class CompletenessResult(NamedTuple):
    score: float
    missing_or_uncertain_fields: list[str]


def score_completeness(
    category: str | None,
    extracted_fields: dict[str, ExtractedField],
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> CompletenessResult:
    profile = get_profile(category)

    if not profile:
        return CompletenessResult(score=0.0, missing_or_uncertain_fields=[])

    total_weight = 0.0
    weighted_sum = 0.0
    missing_or_uncertain: list[str] = []

    for field_name, weight in profile.items():
        field = extracted_fields.get(field_name)
        total_weight += weight

        if field is None:
            missing_or_uncertain.append(field_name)
            continue

        weighted_sum += weight * field.confidence

        if field.value is None or field.confidence < confidence_threshold:
            missing_or_uncertain.append(field_name)

    score = weighted_sum / total_weight if total_weight > 0 else 0.0

    return CompletenessResult(
        score=round(score, 4),
        missing_or_uncertain_fields=missing_or_uncertain,
    )