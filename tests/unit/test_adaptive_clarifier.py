"""
Unit tests for AdaptiveClarifier's turn-budget safeguard.
Uses FakeClarificationProvider - no live API calls, fully deterministic,
and unaffected by any changes to the real system prompt.
"""

import pytest

from app.agents.adaptive_clarifier import AdaptiveClarifier, was_forced_escalation
from app.providers.fake_clarification_provider import FakeClarificationProvider
from app.schemas.clarification import ClarificationDecision
from app.schemas.conversation import ConversationState, Message

MAX_TURNS = 3


def _scripted_ask_decision() -> ClarificationDecision:
    return ClarificationDecision(
        action="ASK_CLARIFICATION",
        reasoning="Device platform is still unknown.",
        information_gap="device_platform",
        question="What device are you using?",
        expected_information_gain=0.7,
        affected_fields=["device_platform"],
        priority="medium",
        confidence=0.8,
    )


def _make_state(turn_count: int) -> ConversationState:
    return ConversationState(
        conversation_id=f"test-{turn_count}",
        turn_count=turn_count,
        raw_messages=[Message(role="user", content="My Teams isn't working.")],
    )


@pytest.mark.asyncio
async def test_ask_passes_through_below_budget():
    agent = AdaptiveClarifier(
        provider=FakeClarificationProvider(_scripted_ask_decision()),
        max_turns=MAX_TURNS,
    )
    decision = await agent.decide(_make_state(turn_count=1))

    assert decision.action == "ASK_CLARIFICATION"
    assert not was_forced_escalation(decision)


@pytest.mark.asyncio
async def test_ask_forced_to_escalate_at_budget():
    agent = AdaptiveClarifier(
        provider=FakeClarificationProvider(_scripted_ask_decision()),
        max_turns=MAX_TURNS,
    )
    decision = await agent.decide(_make_state(turn_count=MAX_TURNS))

    assert decision.action == "ESCALATE"
    assert was_forced_escalation(decision)
    assert "Device platform is still unknown." in decision.reasoning


@pytest.mark.asyncio
async def test_ask_stays_forced_past_budget():
    agent = AdaptiveClarifier(
        provider=FakeClarificationProvider(_scripted_ask_decision()),
        max_turns=MAX_TURNS,
    )
    decision = await agent.decide(_make_state(turn_count=MAX_TURNS + 1))

    assert decision.action == "ESCALATE"
    assert was_forced_escalation(decision)


@pytest.mark.asyncio
async def test_ready_passes_through_even_at_budget():
    """
    The safeguard must ONLY override ASK_CLARIFICATION. READY, RECHECK,
    and genuine ESCALATE decisions from the model must never be touched,
    even when the turn budget is exhausted.
    """
    ready_decision = ClarificationDecision(
        action="READY",
        reasoning="All required fields resolved.",
        information_gap=None,
        question=None,
        expected_information_gain=0.0,
        affected_fields=[],
        priority="low",
        confidence=0.95,
    )
    agent = AdaptiveClarifier(
        provider=FakeClarificationProvider(ready_decision),
        max_turns=MAX_TURNS,
    )
    decision = await agent.decide(_make_state(turn_count=MAX_TURNS))

    assert decision.action == "READY"
    assert not was_forced_escalation(decision)


@pytest.mark.asyncio
async def test_recheck_passes_through_even_at_budget():
    recheck_decision = ClarificationDecision(
        action="RECHECK",
        reasoning="User's latest message contradicts earlier extraction.",
        information_gap=None,
        question=None,
        expected_information_gain=0.5,
        affected_fields=["scope"],
        priority="medium",
        confidence=0.7,
    )
    agent = AdaptiveClarifier(
        provider=FakeClarificationProvider(recheck_decision),
        max_turns=MAX_TURNS,
    )
    decision = await agent.decide(_make_state(turn_count=MAX_TURNS + 5))

    assert decision.action == "RECHECK"
    assert not was_forced_escalation(decision)


@pytest.mark.asyncio
async def test_genuine_escalate_not_double_tagged():
    """
    A genuine model-chosen ESCALATE (not forced by budget) must NOT be
    mistaken for a forced escalation by was_forced_escalation - otherwise
    eval code can't distinguish "the model gave up" from "we cut it off."
    """
    genuine_escalate = ClarificationDecision(
        action="ESCALATE",
        reasoning="User is refusing to provide any diagnostic detail.",
        information_gap=None,
        question=None,
        expected_information_gain=0.0,
        affected_fields=[],
        priority="high",
        confidence=0.85,
    )
    agent = AdaptiveClarifier(
        provider=FakeClarificationProvider(genuine_escalate),
        max_turns=MAX_TURNS,
    )
    decision = await agent.decide(_make_state(turn_count=1))

    assert decision.action == "ESCALATE"
    assert not was_forced_escalation(decision)