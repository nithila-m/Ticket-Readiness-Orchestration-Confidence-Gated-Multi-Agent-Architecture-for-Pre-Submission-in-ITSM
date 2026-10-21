"""
Static system prompt and pure context builder for Agent 5 (Classify,
Prioritise, Explain).

Mirrors Agent 2's split (adaptive_clarifier_prompts.py +
clarification_context.py): the provider only owns the API call; what the
model sees is built here by deterministic, independently testable
functions.

The prompt never shows the model a team roster - routing is a table
lookup in app/config/routing.py, so the model has no reason (and no
data) to name a team.
"""

from app.agents.clarification_context import format_conversation_transcript
from app.config.category_profiles import VALID_CATEGORIES
from app.schemas.conversation import ConversationState
from app.schemas.extraction import ExtractedField

SYSTEM_PROMPT = f"""You are Agent 5, the Classify / Prioritise / Explain \
agent for VIT's IT service desk (TRO system).

The ticket has already been gathered and checked by earlier agents. Your \
job is to confirm its category, assign a priority, and write a short \
routing rationale a human support agent can audit.

You will receive: the conversation transcript, the category detected by \
the extraction agent with its confidence, the extracted fields with their \
confidences, a completeness score, the knowledge-base articles the user was \
already shown, and the outcome of duplicate / incident-cluster correlation.

CATEGORY
- Return exactly one of these keys: {", ".join(VALID_CATEGORIES)}.
- Keep the detected category unless the evidence clearly contradicts it.
- Use "general" only if no other key fits.

PRIORITY (return P2, P3 or P4 only - never P1; P1 is set by the system)
- P2: the user cannot work at all with no workaround, OR several users / a \
whole room, floor or service are affected, OR there is a hard deadline \
stated (exam, class, submission).
- P3: a single user is degraded or blocked on one function but has a \
workaround, or the impact is unclear.
- P4: a cosmetic issue, a how-to question, a request with no current \
disruption, or the user says it is not urgent.
- An incident-cluster correlation outcome is handled by the system; judge \
priority from this ticket's own evidence.

IMPACT SIGNAL
- single_user / multiple_users only when the transcript or the extracted \
fields say so; otherwise unknown.

CONFIDENCE
- 0 to 1: how sure you are of the category AND priority together. Lower it \
when key fields are missing or low-confidence.

RATIONALE (1-3 sentences)
- Cite the specific extracted fields (by name and value) that drove the \
category and priority, e.g. "scope='whole floor' and symptom='won't connect' \
indicate P2".
- Only state facts present in the transcript or extracted fields. Do not \
invent devices, locations, error codes, user counts or deadlines.
- Do not name any support team, person or group - routing is decided \
separately.
"""


def format_extracted_fields(extracted_fields: dict[str, ExtractedField]) -> str:
    """Render every extracted field with its confidence, nulls marked."""
    if not extracted_fields:
        return "(no fields extracted)"
    lines = []
    for name, field in sorted(extracted_fields.items()):
        if field.value is None:
            lines.append(f"  - {name}: NOT PROVIDED")
        else:
            lines.append(f"  - {name}: value={field.value!r}, confidence={field.confidence:.2f}")
    return "\n".join(lines)


def format_kb_checked(state: ConversationState) -> str:
    """Render which KB articles the user already saw and whether one matched."""
    if state.kb_outcome is None:
        return "KB retrieval has not run for this ticket."
    articles = ", ".join(state.kb_articles_checked) if state.kb_articles_checked else "none"
    lines = [
        f"KB outcome: {state.kb_outcome} (similarity {state.kb_similarity_score:.2f})",
        f"Articles already checked: {articles}",
    ]
    if state.kb_feedback is not None:
        lines.append(f"User feedback on offered article: {state.kb_feedback}")
    return "\n".join(lines)


def format_correlation(state: ConversationState) -> str:
    """Render the correlation outcome, or say it hasn't run."""
    corr = state.correlation
    if corr is None:
        return "Correlation has not run for this ticket."
    lines = [f"Outcome: {corr.outcome} (max similarity {corr.max_similarity:.2f})"]
    if corr.similar_tickets:
        lines.append(f"Similar open tickets: {len(corr.similar_tickets)}")
    return "\n".join(lines)


def build_agent5_context(state: ConversationState) -> str:
    """Assemble the full user-turn context block sent to the LLM."""
    category = state.detected_category.value if state.detected_category else None
    category_confidence = state.detected_category.confidence if state.detected_category else 0.0

    return f"""=== CONVERSATION TRANSCRIPT ===
{format_conversation_transcript(state.raw_messages)}

=== DETECTED CATEGORY (extraction agent) ===
{category or "undetermined"} (confidence: {category_confidence:.2f})

=== EXTRACTED FIELDS ===
{format_extracted_fields(state.extracted_fields)}

=== COMPLETENESS SCORE ===
{state.completeness_score:.2f}

=== KNOWLEDGE BASE ===
{format_kb_checked(state)}

=== CORRELATION ===
{format_correlation(state)}
"""
