"""
Formats a ConversationState's raw_messages into the single issue_text
string Agent 1's existing extractor expects (it has no native concept of
multi-turn dialogue - see ticket_analysis_service.py's analyze(issue_text)
signature). Kept separate from clarification_service.py so this format
can be unit-tested without touching any provider or repository.
"""

from app.schemas.conversation import Message


def format_transcript_for_extraction(raw_messages: list[Message]) -> str:
    """
    Labels each message by speaker so Agent 1's extractor doesn't mistake
    Agent 2's own clarifying questions for user-reported symptoms.
    """
    lines = []
    for msg in raw_messages:
        if msg.role == "user":
            lines.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            lines.append(f"Assistant asked: {msg.content}")
        else:
            lines.append(f"System: {msg.content}")
    return "\n".join(lines)