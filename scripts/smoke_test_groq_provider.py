"""
Manual smoke test for GroqClarificationProvider - not part of pytest.
Run from the project root: python -m scripts.smoke_test_groq_provider

This state mirrors digest.txt's Teams "audio stopped during class, two
classmates have the same issue" example. scope and failure_type are
already resolved with high confidence - the point of this test is to
visually confirm Agent 2 does NOT re-ask about scope (e.g. "is this
affecting multiple people?"), since that's already known. If it does
re-ask, the prompt/context in M5.4 needs another pass.
"""

import asyncio

from app.config.settings import settings
from app.providers.groq_clarification_provider import GroqClarificationProvider
from app.schemas.conversation import ConversationState, Message
from app.schemas.extraction import CategoryPrediction, ExtractedField


async def main():
    provider = GroqClarificationProvider(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
    )

    state = ConversationState(
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

    decision = await provider.decide_clarification(state)
    print(decision.model_dump_json(indent=2))

    if decision.question and "multiple" in decision.question.lower():
        print(
            "\n⚠️  WARNING: question mentions 'multiple' - check it isn't "
            "re-asking about scope, which is already resolved."
        )


if __name__ == "__main__":
    asyncio.run(main())