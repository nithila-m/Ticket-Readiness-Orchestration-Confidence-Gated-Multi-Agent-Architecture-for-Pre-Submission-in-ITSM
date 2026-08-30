"""
Manual smoke test for GroqClarificationProvider - not part of pytest.
Run directly: python scripts/smoke_test_groq_provider.py
"""

import asyncio

from app.config.settings import settings  # adjust import to your actual settings instance
from app.providers.groq_clarification_provider import GroqClarificationProvider
from app.schemas.conversation import ConversationState, Message
from app.schemas.extraction import CategoryPrediction, ExtractedField


async def main():
    provider = GroqClarificationProvider(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
    )

    state = ConversationState(
        conversation_id="smoke-test-1",
        turn_count=1,
        raw_messages=[Message(role="user", content="My Teams isn't working.")],
        extracted_fields={
            "failure_type": ExtractedField(value=None, confidence=0.0),
        },
        detected_category=CategoryPrediction(value="ms_teams", confidence=0.7),
        completeness_score=0.3,
        missing_or_uncertain_fields=["failure_type", "scope", "error_signal", "device_platform"],
    )

    decision = await provider.decide_clarification(state)
    print(decision.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())