"""
Manual smoke tests for GroqClarificationProvider - not part of pytest.
Run from the project root: python -m scripts.smoke_test_groq_provider

Scenario 1 (ASK_CLARIFICATION path): mirrors digest.txt's Teams "audio
stopped during class, two classmates have the same issue" example.
scope and failure_type are already resolved with high confidence - checks
that Agent 2 does NOT re-ask about scope.

Scenario 2 (READY path): mirrors digest.txt Example D - "My Teams camera
isn't working. It works in Zoom but not Teams." The failure is specific
and app-isolated (Teams-only, not a general hardware problem), but
device_platform and scope remain unresolved. Checks whether Agent 2 can
judge the ticket actionable despite incomplete profile fields, rather
than mechanically asking for every missing field. If it asks anyway,
that's not necessarily wrong - but worth reading its reasoning closely.
"""

import asyncio

from app.config.settings import settings
from app.providers.groq_clarification_provider import GroqClarificationProvider
from app.schemas.conversation import ConversationState, Message
from app.schemas.extraction import CategoryPrediction, ExtractedField


async def run_scenario(name: str, state: ConversationState) -> None:
    provider = GroqClarificationProvider(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
    )
    print(f"\n{'=' * 60}\nSCENARIO: {name}\n{'=' * 60}")
    decision = await provider.decide_clarification(state)
    print(decision.model_dump_json(indent=2))
    return decision


async def main():
    # --- Scenario 1: ASK_CLARIFICATION, avoid redundancy ---
    scenario_1_state = ConversationState(
        conversation_id="smoke-test-2",
        turn_count=1,
        raw_messages=[
            Message(
                role="user",
                content=(
                    "Teams audio stopped during my class and two classmates "
                    "have the same problem."
                ),
            ),
        ],
        extracted_fields={
            "failure_type": ExtractedField(value="audio", confidence=0.85),
            "scope": ExtractedField(value="multiple users", confidence=0.8),
        },
        detected_category=CategoryPrediction(value="ms_teams", confidence=0.9),
        completeness_score=0.55,
        missing_or_uncertain_fields=["error_signal", "device_platform"],
    )
    decision_1 = await run_scenario("ASK_CLARIFICATION path", scenario_1_state)
    if decision_1.question and "multiple" in decision_1.question.lower():
        print(
            "\n⚠️  WARNING: question mentions 'multiple' - check it isn't "
            "re-asking about scope, which is already resolved."
        )

    # --- Scenario 2: READY despite incomplete profile fields ---
    scenario_2_state = ConversationState(
        conversation_id="smoke-test-3",
        turn_count=1,
        raw_messages=[
            Message(
                role="user",
                content="My Teams camera isn't working. It works in Zoom but not Teams.",
            ),
        ],
        extracted_fields={
            "failure_type": ExtractedField(value="camera", confidence=0.9),
            "error_signal": ExtractedField(
                value="isolated to Teams app, works fine in Zoom", confidence=0.85
            ),
        },
        detected_category=CategoryPrediction(value="ms_teams", confidence=0.9),
        completeness_score=0.45,  # below a naive threshold - deliberately, per the test
        missing_or_uncertain_fields=["scope", "device_platform"],
    )
    decision_2 = await run_scenario("READY path (Example D)", scenario_2_state)
    if decision_2.action != "READY":
        print(
            f"\nℹ️  Note: returned {decision_2.action} instead of READY. "
            "Not automatically wrong - read the reasoning field above and "
            "judge whether the remaining gap is genuinely load-bearing."
        )


if __name__ == "__main__":
    asyncio.run(main())