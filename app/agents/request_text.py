"""
Request text shared by Agent 3 (KB query) and Agent 4 (correlation query
and each ticket's stored correlation_text).

Pure, no I/O: kept apart from kb_deflection_agent.py, which imports
TRO_Codes/chromadb at module level, so pure modules such as
ticket_builder.py can build the same text without pulling that in.
"""

from app.config.category_profiles import category_key_to_display
from app.schemas.extraction import ExtractedField


def build_kb_query(category: str | None, extracted_fields: dict[str, ExtractedField]) -> str:
    """
    Builds the retrieval query from Agent 1's structured output instead of
    the speaker-labelled multi-turn transcript. KB articles are embedded as
    plain "title. symptoms" text (TRO_Codes/kb_insertion.py), so a short,
    description-shaped query compares like with like; the raw transcript
    adds "User:"/"Assistant asked:" framing that gets noisier every turn.

    One "field: value" line per non-null field, then a "Category: <label>"
    line using the canonical display label (a sentence-transformer embeds
    "Wifi/Internet Support" more meaningfully than "wifi_internet").
    An unknown key is used as-is; None adds no category line.
    """
    lines = [
        f"{name}: {field.value}"
        for name, field in extracted_fields.items()
        if field.value is not None
    ]
    if category:
        lines.append(f"Category: {category_key_to_display(category) or category}")
    return "\n".join(lines)
