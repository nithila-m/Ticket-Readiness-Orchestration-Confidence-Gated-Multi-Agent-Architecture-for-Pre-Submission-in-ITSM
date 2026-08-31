"""
Manual smoke test for AdaptiveClarifier's turn-budget safeguard.
Run from the project root: python -m scripts.smoke_test_adaptive_clarifier

Uses FakeClarificationProvider (scripted, deterministic) rather than the
live Groq provider - the safeguard logic being tested here doesn't
depend on what the model actually reasons, only on how the agent reacts
to a given decision once the turn budget is exhausted. Testing this
against a live model would make the test flaky for no benefit.
"""

import asyncio

from app.agents.adaptive_clarifier import AdaptiveClarifier, was_forced_escalation
from app.providers.fake_clarification_provider import FakeClarificationProvider
from app.schemas.clarification import ClarificationDecision
from app.schemas.conversation import ConversationState, Message

MAX_TURNS = 3

# The provider will "want" to keep asking, every time, regardless of state.
scripted_decision = ClarificationDecision(
    action="ASK_CLARIFICATION",
    reasoning="Device platform is still unknown.",
    information_gap="device_platform",
    question="What device are you using?",
    expected_information_gain=0.7,
    affected_fields=["device_platform"],
    priority="medium",
    confidence=0.8,
)


async def run_at_turn(turn_count: int) -> None:
    agent = AdaptiveClarifier(
        provider=FakeClarificationProvider(scripted_decision),
        max_turns=MAX_TURNS,
    )
    state = ConversationState(
        conversation_id=f"budget-test-turn-{turn_count}",
        turn_count=turn_count,
        raw_messages=[Message(role="user", content="My Teams isn't working.")],
    )
    decision = await agent.decide(state)
    print(f"\nturn_count={turn_count} (max={MAX_TURNS}) -> action={decision.action}")
    print(f"  forced_escalation={was_forced_escalation(decision)}")
    print(f"  reasoning={decision.reasoning}")


async def main():
    # Below budget: provider's ASK_CLARIFICATION should pass through untouched.
    await run_at_turn(turn_count=1)
    await run_at_turn(turn_count=2)

    # At/over budget: same scripted ASK_CLARIFICATION must be overridden to ESCALATE.
    await run_at_turn(turn_count=3)
    await run_at_turn(turn_count=4)


if __name__ == "__main__":
    asyncio.run(main())