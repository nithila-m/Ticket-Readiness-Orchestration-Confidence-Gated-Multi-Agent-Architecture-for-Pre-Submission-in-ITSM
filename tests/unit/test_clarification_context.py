"""
Unit tests for clarification_context.py's pure formatting functions.
No LLM calls, no ConversationState mutation - just deterministic
input -> output checks. Unaffected by prompt wording changes.
"""

from app.agents.clarification_context import (
    classify_field_gaps,
    format_clarification_history,
)
from app.schemas.clarification import ClarificationDecision, ClarificationLogEntry
from app.schemas.extraction import ExtractedField


def test_classify_field_gaps_splits_missing_vs_uncertain():
    extracted = {
        "failure_type": ExtractedField(value="audio", confidence=0.9),
        "scope": ExtractedField(value="multiple users", confidence=0.3),  # low confidence
        # error_signal, device_platform never extracted at all
    }
    never_extracted, uncertain = classify_field_gaps("ms_teams", extracted)

    assert "scope" in uncertain
    assert "failure_type" not in uncertain
    assert "failure_type" not in never_extracted
    assert set(never_extracted) == {"error_signal", "device_platform"}


def test_classify_field_gaps_handles_null_value():
    extracted = {
        "failure_type": ExtractedField(value=None, confidence=0.0),
    }
    never_extracted, uncertain = classify_field_gaps("ms_teams", extracted)

    # Present in the dict but null value -> uncertain, not never_extracted.
    assert "failure_type" in uncertain
    assert "failure_type" not in never_extracted


def test_clarification_history_pairs_question_with_next_answer():
    log = [
        ClarificationLogEntry(
            turn=0,
            user_message="Teams isn't working.",
            decision=ClarificationDecision(
                action="ASK_CLARIFICATION",
                reasoning="Scope unknown.",
                information_gap="scope",
                question="Is this happening to just you?",
                expected_information_gain=0.7,
                affected_fields=["scope"],
                priority="high",
                confidence=0.8,
            ),
        ),
        ClarificationLogEntry(
            turn=1,
            user_message="No, two others too.",
            decision=ClarificationDecision(
                action="READY",
                reasoning="Sufficient.",
                information_gap=None,
                question=None,
                expected_information_gain=0.0,
                affected_fields=[],
                priority="low",
                confidence=0.9,
            ),
        ),
    ]
    formatted = format_clarification_history(log)

    assert "Is this happening to just you?" in formatted
    assert "No, two others too." in formatted


def test_clarification_history_empty_log():
    formatted = format_clarification_history([])
    assert "No clarification questions" in formatted