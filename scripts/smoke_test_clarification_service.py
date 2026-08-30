"""
Manual end-to-end smoke test for ClarificationService.
Run from the project root: python -m scripts.smoke_test_clarification_service

Simulates a 2-3 turn conversation, using the REAL Gemini extractor and
REAL Groq clarifier (not fakes) - this is the first test exercising the
full Agent 1 -> Agent 2 -> persistence loop together. Costs real API
calls (still free tier), so don't run this in a tight loop.
"""

import asyncio

from app.agents.adaptive_clarifier import AdaptiveClarifier
from app.agents.information_extractor import InformationExtractor  # adjust import if named differently
from app.config.settings import settings
from app.providers.gemini_provider import GeminiProvider
from app.providers.groq_clarification_provider import GroqClarificationProvider
from app.repositories.in_memory_conversation_repository import InMemoryConversationRepository
from app.services.clarification_service import ClarificationService
from app.services.ticket_analysis_service import TicketAnalysisService


async def main():
    gemini = GeminiProvider(api_key=settings.gemini_api_key, model=settings.gemini_model)
    extractor = InformationExtractor(provider=gemini)  # adjust constructor if different
    extraction_service = TicketAnalysisService(extractor=extractor)

    groq_provider = GroqClarificationProvider(
        api_key=settings.groq_api_key, model=settings.groq_model
    )
    clarifier = AdaptiveClarifier(
        provider=groq_provider, max_turns=settings.max_clarification_turns
    )

    repository = InMemoryConversationRepository()
    service = ClarificationService(extraction_service, clarifier, repository)

    conversation_id = "e2e-smoke-1"
    turns = [
        "Teams isn't working.",
        "Audio drops during meetings, happens to me and two other people in the same call.",
        "I'm on Windows 11 desktop app, no error message shown.",
    ]

    for i, user_message in enumerate(turns, start=1):
        print(f"\n{'=' * 60}\nTURN {i}: \"{user_message}\"\n{'=' * 60}")
        state, decision = await service.handle_message(conversation_id, user_message)
        print(f"Category: {state.detected_category}")
        print(f"Completeness: {state.completeness_score}")
        print(f"Decision: {decision.action}")
        if decision.question:
            print(f"Question: {decision.question}")
        print(f"Reasoning: {decision.reasoning}")

        if decision.action in ("READY", "ESCALATE"):
            print(f"\nConversation terminated at turn {i} with action={decision.action}.")
            break


if __name__ == "__main__":
    asyncio.run(main())