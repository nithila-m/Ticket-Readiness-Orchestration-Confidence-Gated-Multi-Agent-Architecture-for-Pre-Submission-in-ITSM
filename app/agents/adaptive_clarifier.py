"""
Agent 2: Adaptive Clarifier.

Thin wrapper around a ClarificationProvider with exactly one deterministic
safeguard layered on top: if the LLM still wants to ask another question
after the turn budget is exhausted, force ESCALATE instead. This is a
safety net against the documented ClarifyMT-Bench Refusal-persona failure
mode (digest.txt Section 17/17-equivalent), not a decision-making
mechanism - the model's actual judgment is never second-guessed for any
other reason.

The override is deliberately a standalone pure function, not inlined into
decide(), so it can be unit-tested directly with hand-built
ClarificationDecision objects (see tests/unit/test_adaptive_clarifier.py)
without touching a provider, live or fake.
"""

from app.providers.clarification_base import ClarificationProvider
from app.schemas.clarification import ClarificationDecision
from app.schemas.conversation import ConversationState

FORCED_ESCALATION_TAG = "[turn budget exceeded - forced escalation]"


def _apply_turn_budget_safeguard(
    decision: ClarificationDecision,
    turn_count: int,
    max_turns: int,
) -> ClarificationDecision:
    """
    If the model wants to ask another question but the turn budget is
    already exhausted, override to ESCALATE. Every other action (READY,
    RECHECK, ESCALATE) passes through untouched - this only fires on the
    one combination that could otherwise loop indefinitely.
    """
    if decision.action != "ASK_CLARIFICATION" or turn_count < max_turns:
        return decision

    return ClarificationDecision(
        action="ESCALATE",
        reasoning=f"{FORCED_ESCALATION_TAG} Original model reasoning: {decision.reasoning}",
        information_gap=decision.information_gap,
        question=None,
        expected_information_gain=decision.expected_information_gain,
        affected_fields=decision.affected_fields,
        priority="high",
        confidence=decision.confidence,
    )


def was_forced_escalation(decision: ClarificationDecision) -> bool:
    """
    True if this decision was produced by the turn-budget safeguard
    rather than the model's own judgment. Used by eval/audit code to
    separate genuine model decisions from safety-net overrides -
    conflating the two would misrepresent how often the agent itself
    chooses to stop asking.
    """
    return decision.action == "ESCALATE" and decision.reasoning.startswith(
        FORCED_ESCALATION_TAG
    )


class AdaptiveClarifier:
    """Agent 2. Wraps a ClarificationProvider with the turn-budget safeguard."""

    def __init__(self, provider: ClarificationProvider, max_turns: int):
        self._provider = provider
        self._max_turns = max_turns

    async def decide(self, state: ConversationState) -> ClarificationDecision:
        decision = await self._provider.decide_clarification(state)
        return _apply_turn_budget_safeguard(
            decision, turn_count=state.turn_count, max_turns=self._max_turns
        )