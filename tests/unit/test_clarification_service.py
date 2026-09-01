"""
Unit tests for ClarificationService's orchestration loop.
Both the extraction service and the clarifier are faked - this tests
ONLY the service's own logic (message append, state refresh, log append,
turn increment, persistence), not any model's actual judgment, and is
unaffected by prompt wording changes.
"""

from unittest.mock import AsyncMock

import pytest

from app.repositories.in_memory_conversation_repository import InMemoryConversationRepository
from app.schemas.analysis import AnalysisResult
from app.schemas.clarification import ClarificationDecision
from app.schemas.extraction import CategoryPrediction, ExtractedField
from app.services.clarification_service import ClarificationService


def _make_analysis_result(completeness: float) -> AnalysisResult:
    return AnalysisResult(
        category=CategoryPrediction(value="ms_teams", confidence=0.9),
        extracted_fields={"failure_type": ExtractedField(value="audio", confidence=0.9)},
        completeness_score=completeness,
        missing_or_uncertain_fields=["scope"],
    )


def _make_ask_decision() -> ClarificationDecision:
    return ClarificationDecision(
        action="ASK_CLARIFICATION",
        reasoning="Scope unknown.",
        information_gap="scope",
        question="Is this happening to just you?",
        expected_information_gain=0.7,
        affected_fields=["scope"],
        priority="high",
        confidence=0.8,
    )


def _make_ready_decision() -> ClarificationDecision:
    return ClarificationDecision(
        action="READY",
        reasoning="Sufficient.",
        information_gap=None,
        question=None,
        expected_information_gain=0.0,
        affected_fields=[],
        priority="low",
        confidence=0.9,
    )


@pytest.fixture
def fake_extraction_service():
    service = AsyncMock()
    service.analyze.return_value = _make_analysis_result(completeness=0.5)
    return service


@pytest.fixture
def fake_clarifier():
    agent = AsyncMock()
    agent.decide.return_value = _make_ask_decision()
    return agent


@pytest.fixture
def repository():
    return InMemoryConversationRepository()


@pytest.mark.asyncio
async def test_first_turn_creates_state_and_increments_turn(
    fake_extraction_service, fake_clarifier, repository
):
    service = ClarificationService(fake_extraction_service, fake_clarifier, repository)

    state, decision = await service.handle_message("conv-1", "Teams isn't working.")

    assert state.turn_count == 1
    assert state.raw_messages[0].role == "user"
    assert state.raw_messages[0].content == "Teams isn't working."
    assert decision.action == "ASK_CLARIFICATION"


@pytest.mark.asyncio
async def test_extraction_refreshes_state_fields(
    fake_extraction_service, fake_clarifier, repository
):
    service = ClarificationService(fake_extraction_service, fake_clarifier, repository)

    state, _ = await service.handle_message("conv-1", "Teams isn't working.")

    assert state.completeness_score == 0.5
    assert state.detected_category.value == "ms_teams"
    assert "scope" in state.missing_or_uncertain_fields


@pytest.mark.asyncio
async def test_ask_clarification_question_appended_to_transcript(
    fake_extraction_service, fake_clarifier, repository
):
    service = ClarificationService(fake_extraction_service, fake_clarifier, repository)

    state, decision = await service.handle_message("conv-1", "Teams isn't working.")

    assert len(state.raw_messages) == 2
    assert state.raw_messages[1].role == "assistant"
    assert state.raw_messages[1].content == decision.question


@pytest.mark.asyncio
async def test_ready_does_not_append_assistant_message(
    fake_extraction_service, fake_clarifier, repository
):
    fake_clarifier.decide.return_value = _make_ready_decision()
    service = ClarificationService(fake_extraction_service, fake_clarifier, repository)

    state, decision = await service.handle_message("conv-1", "Windows 11, no error.")

    assert decision.action == "READY"
    assert len(state.raw_messages) == 1  # only the user message, no question to append


@pytest.mark.asyncio
async def test_clarification_log_entry_recorded(
    fake_extraction_service, fake_clarifier, repository
):
    service = ClarificationService(fake_extraction_service, fake_clarifier, repository)

    state, decision = await service.handle_message("conv-1", "Teams isn't working.")

    assert len(state.clarification_log) == 1
    log_entry = state.clarification_log[0]
    assert log_entry.turn == 0  # logged BEFORE increment
    assert log_entry.user_message == "Teams isn't working."
    assert log_entry.decision.action == "ASK_CLARIFICATION"

@pytest.mark.asyncio
async def test_state_persists_across_calls(fake_extraction_service, fake_clarifier, repository):
    service = ClarificationService(fake_extraction_service, fake_clarifier, repository)

    await service.handle_message("conv-1", "Teams isn't working.")
    state_after_turn_2, _ = await service.handle_message("conv-1", "Just me, on Windows.")

    assert state_after_turn_2.turn_count == 2
    assert len(state_after_turn_2.raw_messages) == 4  # user, assistant, user, assistant
    assert len(state_after_turn_2.clarification_log) == 2


@pytest.mark.asyncio
async def test_different_conversation_ids_are_isolated(
    fake_extraction_service, fake_clarifier, repository
):
    service = ClarificationService(fake_extraction_service, fake_clarifier, repository)

    state_a, _ = await service.handle_message("conv-a", "Teams isn't working.")
    state_b, _ = await service.handle_message("conv-b", "Printer is offline.")

    assert state_a.turn_count == 1
    assert state_b.turn_count == 1
    assert state_a.conversation_id != state_b.conversation_id


@pytest.mark.asyncio
async def test_extraction_service_receives_labeled_transcript(
    fake_extraction_service, fake_clarifier, repository
):
    """
    Confirms the M5.6 transcript-formatting decision: Agent 1 must
    receive a speaker-labeled transcript, not just the raw latest message,
    so it re-extracts against the FULL conversation each turn.
    """
    service = ClarificationService(fake_extraction_service, fake_clarifier, repository)

    await service.handle_message("conv-1", "Teams isn't working.")
    await service.handle_message("conv-1", "Just me, on Windows.")

    second_call_args = fake_extraction_service.analyze.call_args_list[1]
    transcript_sent = second_call_args.args[0]

    assert "User: Teams isn't working." in transcript_sent
    assert "Assistant asked:" in transcript_sent
    assert "User: Just me, on Windows." in transcript_sent