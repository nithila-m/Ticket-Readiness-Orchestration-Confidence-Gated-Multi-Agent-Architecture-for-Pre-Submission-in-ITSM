"""
Pure functions that turn a ConversationState into the context Agent 2's
prompt needs. No LLM calls here - everything is deterministic and
independently unit-testable (see tests/unit/test_clarification_context.py).

This module deliberately re-derives the missing-vs-uncertain distinction
that completeness_scorer.py collapses into one list, reusing its own
CONFIDENCE_THRESHOLD as the single source of truth for what counts as
"uncertain" rather than hardcoding a second copy of that number here.
"""

from app.agents.completeness_scorer import CONFIDENCE_THRESHOLD
from app.config.category_profiles import get_profile
from app.schemas.clarification import ClarificationLogEntry
from app.schemas.conversation import ConversationState, Message
from app.schemas.extraction import ExtractedField


def classify_field_gaps(
    category: str | None,
    extracted_fields: dict[str, ExtractedField],
) -> tuple[list[str], list[str]]:
    """
    Split a category's profile fields into (never_extracted, uncertain).
    never_extracted: the field is absent from extracted_fields entirely.
    uncertain: the field is present but value is None or confidence is low.
    """
    profile = get_profile(category)
    never_extracted: list[str] = []
    uncertain: list[str] = []

    for field_name in profile:
        field = extracted_fields.get(field_name)
        if field is None:
            never_extracted.append(field_name)
        elif field.value is None or field.confidence < CONFIDENCE_THRESHOLD:
            uncertain.append(field_name)

    return never_extracted, uncertain


def format_category_profile(category: str | None) -> str:
    """Render the detected category's field-weight table for the prompt."""
    profile = get_profile(category)
    label = category or "general (category undetermined)"
    lines = [f"Field importance weights for category '{label}':"]
    for field_name, weight in sorted(profile.items(), key=lambda kv: -kv[1]):
        lines.append(f"  - {field_name}: {weight}")
    return "\n".join(lines)


def format_extraction_snapshot(
    category: str | None,
    extracted_fields: dict[str, ExtractedField],
) -> str:
    """Render every profile field's current known status."""
    profile = get_profile(category)
    never_extracted, uncertain = classify_field_gaps(category, extracted_fields)
    lines = ["Current extraction state:"]

    for field_name in profile:
        if field_name in never_extracted:
            lines.append(f"  - {field_name}: NOT MENTIONED YET")
        elif field_name in uncertain:
            field = extracted_fields[field_name]
            lines.append(
                f"  - {field_name}: value={field.value!r}, "
                f"confidence={field.confidence:.2f} (LOW CONFIDENCE)"
            )
        else:
            field = extracted_fields[field_name]
            lines.append(
                f"  - {field_name}: value={field.value!r}, "
                f"confidence={field.confidence:.2f} (resolved)"
            )

    return "\n".join(lines)


def format_conversation_transcript(raw_messages: list[Message]) -> str:
    if not raw_messages:
        return "(no messages yet)"
    lines = []
    for msg in raw_messages:
        speaker = "User" if msg.role == "user" else "Agent"
        lines.append(f"{speaker}: {msg.content}")
    return "\n".join(lines)


def format_clarification_history(clarification_log: list[ClarificationLogEntry]) -> str:
    """
    Pair each prior question with the user's next message (its de facto
    answer), so the model can check semantically whether something it's
    about to ask has already been resolved - per digest.txt Section 15,
    this must not collapse into a string-equality check; giving the model
    the actual Q&A pairs is what makes semantic redundancy checking possible.
    """
    asked_entries = [
        (i, entry) for i, entry in enumerate(clarification_log) if entry.decision.question
    ]
    if not asked_entries:
        return "No clarification questions have been asked yet this conversation."

    lines = []
    for i, entry in asked_entries:
        answer = (
            clarification_log[i + 1].user_message
            if i + 1 < len(clarification_log)
            else "(not yet answered)"
        )
        lines.append(f'  Q ("{entry.decision.affected_fields}"): "{entry.decision.question}"')
        lines.append(f'  A: "{answer}"')
    return "\n".join(lines)


def build_agent2_context(state: ConversationState, max_turns: int) -> str:
    """Assemble the full user-turn context block sent to the LLM."""
    category = state.detected_category.value if state.detected_category else None
    category_confidence = state.detected_category.confidence if state.detected_category else 0.0

    return f"""=== CONVERSATION TRANSCRIPT ===
{format_conversation_transcript(state.raw_messages)}

=== DETECTED CATEGORY ===
{category or "undetermined"} (confidence: {category_confidence:.2f})

=== COMPLETENESS SCORE (evidence, not a threshold decision) ===
{state.completeness_score:.2f}

{format_category_profile(category)}

{format_extraction_snapshot(category, state.extracted_fields)}

=== PRIOR CLARIFICATION Q&A ===
{format_clarification_history(state.clarification_log)}

=== TURN BUDGET ===
Current turn: {state.turn_count} / max {max_turns} clarification turns.
"""