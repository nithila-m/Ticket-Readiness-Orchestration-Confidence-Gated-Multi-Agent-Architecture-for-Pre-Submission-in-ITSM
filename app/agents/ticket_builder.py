"""
Agent 6 - canonical ticket builder.

Pure functions, no I/O: ConversationState in, TicketRecord out. The
Dispatcher (app/agents/dispatcher.py) owns persistence and external sinks;
keeping this side-effect free means the exact ticket text is unit-testable
without a database or network.

The ticket's routing decision (category, priority, team, rationale,
confidence) is copied from Agent 5's ClassificationResult unchanged - the
builder never re-derives it.
"""

import re

from app.agents.request_text import build_kb_query
from app.config.category_profiles import CATEGORY_DISPLAY_TO_KEY, get_profile
from app.schemas.conversation import ConversationState
from app.schemas.pipeline import TicketRecord

TITLE_MAX_CHARS = 120

# Inverted dataset label map, plus "general" which has no dataset label.
CATEGORY_KEY_TO_DISPLAY: dict[str, str] = {
    **{key: label for label, key in CATEGORY_DISPLAY_TO_KEY.items()},
    "general": "General",
}


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def pick_primary_field(category: str | None) -> str:
    """The highest-weight field in the category's profile (e.g. wifi -> symptom_type)."""
    profile = get_profile(category)
    return max(profile, key=profile.get)


def _first_user_message(state: ConversationState) -> str:
    for m in state.raw_messages:
        if m.role == "user" and m.content.strip():
            return m.content
    return ""


def build_title(state: ConversationState) -> str:
    """
    "[<display label>] <primary field value>", falling back to the first
    user message when the primary field wasn't extracted. Whitespace is
    collapsed and the result truncated to TITLE_MAX_CHARS.
    """
    category = state.classification.category if state.classification else None
    label = CATEGORY_KEY_TO_DISPLAY.get(category, "General")

    field = state.extracted_fields.get(pick_primary_field(category))
    summary = str(field.value) if field is not None and field.value is not None else ""
    if not _collapse(summary):
        summary = _first_user_message(state)
    # TicketRecord.title must be non-empty, so never return a bare label + "".
    summary = _collapse(summary) or "(no description provided)"

    return f"[{label}] {summary}"[:TITLE_MAX_CHARS]


def build_description(state: ConversationState) -> str:
    """Plain-text ticket body; sections are separated by blank lines."""
    user_lines = [m.content.strip() for m in state.raw_messages if m.role == "user"]
    detail_lines = [
        f"- {name}: {f.value} (conf {f.confidence:.2f})"
        for name, f in state.extracted_fields.items()
        if f.value is not None
    ]
    kb_articles = ", ".join(state.kb_articles_checked) or "none"

    sections = [
        "Reported by user:\n" + ("\n".join(user_lines) or "(no user messages)"),
        "Extracted details:\n" + ("\n".join(detail_lines) or "(none)"),
        f"Completeness: {state.completeness_score:.2f}",
        f"Clarification turns: {state.turn_count}",
        f"Knowledge base: {state.kb_outcome or 'NOT_RUN'} (articles checked: {kb_articles})",
    ]
    return "\n\n".join(sections)


def build_correlation_text(state: ConversationState) -> str | None:
    """
    The text Agent 4 compares later requests against: the same
    build_kb_query() format, from the same source (Agent 1's category and
    fields), as the correlation engine's incoming query. None when no
    field was extracted - a bare "Category: ..." line would make every
    field-less request in that category an exact match - so the engine
    falls back to title + description.
    """
    if not any(f.value is not None for f in state.extracted_fields.values()):
        return None
    category = state.detected_category.value if state.detected_category else None
    return build_kb_query(category, state.extracted_fields) or None


def build_ticket(state: ConversationState) -> TicketRecord:
    """
    Assemble the canonical TicketRecord. ticket_id stays None - only the
    store issues ids. Raises ValueError if Agent 5 hasn't classified yet.
    """
    c = state.classification
    if c is None:
        raise ValueError(
            f"conversation {state.conversation_id!r} has no classification; "
            "Agent 5 must run before a ticket can be built"
        )

    corr = state.correlation
    building = state.extracted_fields.get("building")

    return TicketRecord(
        conversation_id=state.conversation_id,
        title=build_title(state),
        description=build_description(state),
        category=c.category,
        priority=c.priority,
        assignee_group=c.assignee_group,
        routing_rationale=c.routing_rationale,
        confidence_score=c.confidence,
        kb_articles_checked=list(state.kb_articles_checked),
        duplicate_of=corr.duplicate_of if corr else None,
        incident_cluster_id=corr.cluster_id if corr else None,
        building=(
            str(building.value) if building is not None and building.value is not None else None
        ),
        correlation_text=build_correlation_text(state),
    )
