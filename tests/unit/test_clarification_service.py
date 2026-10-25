"""
Unit tests for ClarificationService's orchestration loop.
Both the extraction service and the clarifier are faked - this tests
ONLY the service's own logic (message append, state refresh, log append,
turn increment, persistence), not any model's actual judgment, and is
unaffected by prompt wording changes.
"""

from unittest.mock import AsyncMock

import pytest

from app.agents.kb_deflection_agent import KBDeflectionAgent, KBRetrievalResult
from app.config.settings import settings
from app.orchestration import GraphConversationService
from app.repositories.in_memory_conversation_repository import InMemoryConversationRepository
from app.schemas.analysis import AnalysisResult
from app.schemas.clarification import ClarificationDecision
from app.schemas.extraction import CategoryPrediction, ExtractedField
from app.services.clarification_service import (
    ClarificationService,
    ConversationClosedError,
    ConversationNotFoundError,
    ConversationStateError,
)
from app.storage.sqlite_store import SqliteStore


def _make_no_match_kb_result() -> KBRetrievalResult:
    return KBRetrievalResult(
        outcome="NO_MATCH",
        similarity_score=0.1,
        articles_checked=["KB0001"],
        matched_kb_id=None,
        matched_title=None,
        offered_resolution=None,
    )


def _make_strong_match_kb_result() -> KBRetrievalResult:
    return KBRetrievalResult(
        outcome="STRONG_MATCH",
        similarity_score=0.91,
        articles_checked=["KB0302"],
        matched_kb_id="KB0302",
        matched_title="Wi-Fi disconnects repeatedly after device wakes from sleep",
        offered_resolution="1. Disable fast startup\n2. Forget and re-join Wi-Fi",
    )


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


@pytest.fixture(autouse=True, params=["service", "langgraph"])
def implementation(request, monkeypatch):
    """
    Runs every test in this module twice: against the hand-written
    ClarificationService and against the LangGraph-backed
    GraphConversationService (same constructor, same methods), by swapping
    the module-level name the tests construct.
    """
    if request.param == "langgraph":
        monkeypatch.setitem(globals(), "ClarificationService", GraphConversationService)
    return request.param


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
def fake_kb_agent():
    agent = AsyncMock()
    agent.check.return_value = _make_no_match_kb_result()
    return agent


@pytest.fixture
def repository():
    return InMemoryConversationRepository()


@pytest.mark.asyncio
async def test_first_turn_creates_state_and_increments_turn(
    fake_extraction_service, fake_clarifier, fake_kb_agent, repository
):
    service = ClarificationService(fake_extraction_service, fake_clarifier, repository, fake_kb_agent)

    state, decision = await service.handle_message("conv-1", "Teams isn't working.")

    assert state.turn_count == 1
    assert state.raw_messages[0].role == "user"
    assert state.raw_messages[0].content == "Teams isn't working."
    assert decision.action == "ASK_CLARIFICATION"


@pytest.mark.asyncio
async def test_extraction_refreshes_state_fields(
    fake_extraction_service, fake_clarifier, fake_kb_agent, repository
):
    service = ClarificationService(fake_extraction_service, fake_clarifier, repository, fake_kb_agent)

    state, _ = await service.handle_message("conv-1", "Teams isn't working.")

    assert state.completeness_score == 0.5
    assert state.detected_category.value == "ms_teams"
    assert "scope" in state.missing_or_uncertain_fields


@pytest.mark.asyncio
async def test_ask_clarification_question_appended_to_transcript(
    fake_extraction_service, fake_clarifier, fake_kb_agent, repository
):
    service = ClarificationService(fake_extraction_service, fake_clarifier, repository, fake_kb_agent)

    state, decision = await service.handle_message("conv-1", "Teams isn't working.")

    assert len(state.raw_messages) == 2
    assert state.raw_messages[1].role == "assistant"
    assert state.raw_messages[1].content == decision.question


@pytest.mark.asyncio
async def test_ready_does_not_append_assistant_message(
    fake_extraction_service, fake_clarifier, fake_kb_agent, repository
):
    fake_clarifier.decide.return_value = _make_ready_decision()
    service = ClarificationService(fake_extraction_service, fake_clarifier, repository, fake_kb_agent)

    state, decision = await service.handle_message("conv-1", "Windows 11, no error.")

    assert decision.action == "READY"
    assert len(state.raw_messages) == 1  # only the user message, no question to append


@pytest.mark.asyncio
async def test_clarification_log_entry_recorded(
    fake_extraction_service, fake_clarifier, fake_kb_agent, repository
):
    service = ClarificationService(fake_extraction_service, fake_clarifier, repository, fake_kb_agent)

    state, decision = await service.handle_message("conv-1", "Teams isn't working.")

    assert len(state.clarification_log) == 1
    log_entry = state.clarification_log[0]
    assert log_entry.turn == 0  # logged BEFORE increment
    assert log_entry.user_message == "Teams isn't working."
    assert log_entry.decision.action == "ASK_CLARIFICATION"

@pytest.mark.asyncio
async def test_state_persists_across_calls(fake_extraction_service, fake_clarifier, fake_kb_agent, repository):
    service = ClarificationService(fake_extraction_service, fake_clarifier, repository, fake_kb_agent)

    await service.handle_message("conv-1", "Teams isn't working.")
    state_after_turn_2, _ = await service.handle_message("conv-1", "Just me, on Windows.")

    assert state_after_turn_2.turn_count == 2
    assert len(state_after_turn_2.raw_messages) == 4  # user, assistant, user, assistant
    assert len(state_after_turn_2.clarification_log) == 2


@pytest.mark.asyncio
async def test_different_conversation_ids_are_isolated(
    fake_extraction_service, fake_clarifier, fake_kb_agent, repository
):
    service = ClarificationService(fake_extraction_service, fake_clarifier, repository, fake_kb_agent)

    state_a, _ = await service.handle_message("conv-a", "Teams isn't working.")
    state_b, _ = await service.handle_message("conv-b", "Printer is offline.")

    assert state_a.turn_count == 1
    assert state_b.turn_count == 1
    assert state_a.conversation_id != state_b.conversation_id


@pytest.mark.asyncio
async def test_extraction_service_receives_labeled_transcript(
    fake_extraction_service, fake_clarifier, fake_kb_agent, repository
):
    """
    Confirms the M5.6 transcript-formatting decision: Agent 1 must
    receive a speaker-labeled transcript, not just the raw latest message,
    so it re-extracts against the FULL conversation each turn.
    """
    service = ClarificationService(fake_extraction_service, fake_clarifier, repository, fake_kb_agent)

    await service.handle_message("conv-1", "Teams isn't working.")
    await service.handle_message("conv-1", "Just me, on Windows.")

    second_call_args = fake_extraction_service.analyze.call_args_list[1]
    transcript_sent = second_call_args.args[0]

    assert "User: Teams isn't working." in transcript_sent
    assert "Assistant asked:" in transcript_sent
    assert "User: Just me, on Windows." in transcript_sent


@pytest.mark.asyncio
async def test_strong_kb_match_deflects_and_skips_agent2(
    fake_extraction_service, fake_clarifier, fake_kb_agent, repository
):
    fake_kb_agent.check.return_value = _make_strong_match_kb_result()
    service = ClarificationService(fake_extraction_service, fake_clarifier, repository, fake_kb_agent)

    state, decision = await service.handle_message("conv-1", "Wifi drops after my laptop sleeps.")

    assert decision.action == "DEFLECTED"
    assert state.kb_outcome == "STRONG_MATCH"
    assert state.kb_matched_kb_id == "KB0302"
    fake_clarifier.decide.assert_not_called()
    # DEFLECTED has no question, so nothing extra gets folded into the transcript
    assert len(state.raw_messages) == 1


@pytest.mark.asyncio
async def test_weak_kb_match_still_calls_agent2(
    fake_extraction_service, fake_clarifier, fake_kb_agent, repository
):
    fake_kb_agent.check.return_value = KBRetrievalResult(
        outcome="WEAK_MATCH",
        similarity_score=0.5,
        articles_checked=["KB0303"],
        matched_kb_id="KB0303",
        matched_title="Captive portal login page not loading",
        offered_resolution="Open a plain HTTP page to trigger the captive portal.",
    )
    service = ClarificationService(fake_extraction_service, fake_clarifier, repository, fake_kb_agent)

    state, decision = await service.handle_message("conv-1", "Teams isn't working.")

    assert state.kb_outcome == "WEAK_MATCH"
    assert decision.action == "ASK_CLARIFICATION"  # unchanged: fake_clarifier's default
    fake_clarifier.decide.assert_called_once()


@pytest.mark.asyncio
async def test_kb_agent_receives_structured_query_and_category_key_not_transcript(
    fake_extraction_service, fake_clarifier, fake_kb_agent, repository
):
    service = ClarificationService(fake_extraction_service, fake_clarifier, repository, fake_kb_agent)

    await service.handle_message("conv-1", "Teams isn't working.")

    query_text, category = fake_kb_agent.check.call_args.args
    assert query_text == "failure_type: audio\nCategory: Microsoft Teams Support"
    assert category == "ms_teams"
    assert "User:" not in query_text


@pytest.mark.asyncio
async def test_below_completeness_gate_skips_kb_agent(
    fake_extraction_service, fake_clarifier, fake_kb_agent, repository
):
    fake_extraction_service.analyze.return_value = _make_analysis_result(
        completeness=settings.kb_gate_completeness_threshold - 0.01
    )
    service = ClarificationService(fake_extraction_service, fake_clarifier, repository, fake_kb_agent)

    state, decision = await service.handle_message("conv-1", "Teams isn't working.")

    fake_kb_agent.check.assert_not_called()
    assert state.kb_outcome is None
    fake_clarifier.decide.assert_called_once()


@pytest.mark.asyncio
async def test_kb_retrieval_failure_falls_through_to_agent2(
    fake_extraction_service, fake_clarifier, repository
):
    def broken_retrieve(query_text, category):
        raise RuntimeError("embedding model failed to load")

    kb_agent = KBDeflectionAgent(retrieve_fn=broken_retrieve)
    service = ClarificationService(fake_extraction_service, fake_clarifier, repository, kb_agent)

    state, decision = await service.handle_message("conv-1", "Teams isn't working.")

    assert state.kb_outcome == "NO_MATCH"
    assert state.kb_similarity_score == 0.0
    assert decision.action == "ASK_CLARIFICATION"
    fake_clarifier.decide.assert_called_once()


# --- Agents 4-6 wiring (TicketPipeline + audit) -------------------------------

class FakePipeline:
    """Records calls; sets the terminal outcome the real pipeline would."""

    def __init__(self):
        self.run_calls = []
        self.escalate_calls = []

    async def run(self, state):
        self.run_calls.append(state.conversation_id)
        state.terminal_outcome = "DISPATCHED"
        return state

    async def escalate(self, state, reason):
        self.escalate_calls.append(reason)
        state.terminal_outcome = "ESCALATED"
        return state

    # Step methods - what the LangGraph nodes call instead of run()/escalate().
    # They record into the same lists so each test asserts the same thing
    # against both orchestrators.
    async def correlate(self, state):
        self.run_calls.append(state.conversation_id)

    async def classify(self, state):
        pass

    async def gate(self, state):
        return "DISPATCH"

    async def dispatch(self, state):
        state.terminal_outcome = "DISPATCHED"
        return state

    async def enqueue_review(self, state, reason, outcome):
        self.escalate_calls.append(reason)
        state.terminal_outcome = outcome


def _make_escalate_decision() -> ClarificationDecision:
    return ClarificationDecision(
        action="ESCALATE",
        reasoning="User cannot provide the needed details.",
        expected_information_gain=0.0,
        priority="high",
        confidence=0.6,
    )


@pytest.fixture
def pipeline():
    return FakePipeline()


@pytest.fixture
def store(tmp_path):
    s = SqliteStore(tmp_path / "svc.db")
    s.init_db()
    return s


@pytest.fixture
def auto_dispatch(monkeypatch):
    monkeypatch.setattr(settings, "auto_dispatch_on_ready", True)


async def test_ready_with_flag_on_runs_pipeline_once(
    fake_extraction_service, fake_clarifier, fake_kb_agent, repository, pipeline, auto_dispatch
):
    fake_clarifier.decide.return_value = _make_ready_decision()
    service = ClarificationService(
        fake_extraction_service, fake_clarifier, repository, fake_kb_agent, ticket_pipeline=pipeline
    )

    state, decision = await service.handle_message("conv-1", "Windows 11, no error.")

    assert decision.action == "READY"
    assert pipeline.run_calls == ["conv-1"]
    assert state.terminal_outcome == "DISPATCHED"
    assert (await repository.get("conv-1")).terminal_outcome == "DISPATCHED"  # persisted


async def test_message_after_terminal_outcome_is_rejected_and_not_redispatched(
    fake_extraction_service, fake_clarifier, fake_kb_agent, repository, pipeline, auto_dispatch
):
    fake_clarifier.decide.return_value = _make_ready_decision()
    service = ClarificationService(
        fake_extraction_service, fake_clarifier, repository, fake_kb_agent, ticket_pipeline=pipeline
    )
    await service.handle_message("conv-1", "Windows 11, no error.")

    with pytest.raises(ConversationClosedError):
        await service.handle_message("conv-1", "Any update?")

    assert pipeline.run_calls == ["conv-1"]
    assert fake_extraction_service.analyze.call_count == 1
    assert (await repository.get("conv-1")).turn_count == 1


async def test_ready_with_flag_off_leaves_behaviour_unchanged(
    fake_extraction_service, fake_clarifier, fake_kb_agent, repository, pipeline, monkeypatch
):
    monkeypatch.setattr(settings, "auto_dispatch_on_ready", False)
    fake_clarifier.decide.return_value = _make_ready_decision()
    service = ClarificationService(
        fake_extraction_service, fake_clarifier, repository, fake_kb_agent, ticket_pipeline=pipeline
    )

    state, _ = await service.handle_message("conv-1", "Windows 11, no error.")

    assert pipeline.run_calls == []
    assert state.terminal_outcome is None
    await service.handle_message("conv-1", "still broken")  # conversation not closed


async def test_escalate_calls_pipeline_escalate(
    fake_extraction_service, fake_clarifier, fake_kb_agent, repository, pipeline
):
    fake_clarifier.decide.return_value = _make_escalate_decision()
    service = ClarificationService(
        fake_extraction_service, fake_clarifier, repository, fake_kb_agent, ticket_pipeline=pipeline
    )

    state, _ = await service.handle_message("conv-1", "I do not know.")

    assert pipeline.escalate_calls == ["clarification_escalated"]
    assert pipeline.run_calls == []
    assert state.terminal_outcome == "ESCALATED"


async def test_deflected_turn_waits_for_feedback(
    fake_extraction_service, fake_clarifier, fake_kb_agent, repository, pipeline, auto_dispatch
):
    fake_kb_agent.check.return_value = _make_strong_match_kb_result()
    service = ClarificationService(
        fake_extraction_service, fake_clarifier, repository, fake_kb_agent, ticket_pipeline=pipeline
    )

    state, decision = await service.handle_message("conv-1", "Wifi drops after sleep.")

    assert decision.action == "DEFLECTED"
    assert state.terminal_outcome is None
    assert pipeline.run_calls == [] and pipeline.escalate_calls == []


async def test_feedback_resolved_closes_as_self_resolved(
    fake_extraction_service, fake_clarifier, fake_kb_agent, repository, pipeline, store
):
    fake_kb_agent.check.return_value = _make_strong_match_kb_result()
    service = ClarificationService(
        fake_extraction_service, fake_clarifier, repository, fake_kb_agent,
        ticket_pipeline=pipeline, store=store,
    )
    await service.handle_message("conv-1", "Wifi drops after sleep.")

    state = await service.handle_deflection_feedback("conv-1", resolved=True)

    assert state.kb_feedback == "RESOLVED"
    assert state.terminal_outcome == "SELF_RESOLVED"
    assert pipeline.run_calls == []
    assert (await repository.get("conv-1")).terminal_outcome == "SELF_RESOLVED"
    stages = [(e.stage, e.decision) for e in await store.fetch_audit("conv-1")]
    assert stages[-2:] == [("deflection_feedback", "RESOLVED"), ("terminal", "SELF_RESOLVED")]


async def test_feedback_not_resolved_runs_ticket_pipeline(
    fake_extraction_service, fake_clarifier, fake_kb_agent, repository, pipeline
):
    fake_kb_agent.check.return_value = _make_strong_match_kb_result()
    service = ClarificationService(
        fake_extraction_service, fake_clarifier, repository, fake_kb_agent, ticket_pipeline=pipeline
    )
    await service.handle_message("conv-1", "Wifi drops after sleep.")

    state = await service.handle_deflection_feedback("conv-1", resolved=False)

    assert state.kb_feedback == "NOT_RESOLVED"
    assert pipeline.run_calls == ["conv-1"]
    assert state.terminal_outcome == "DISPATCHED"


async def test_feedback_invalid_states(
    fake_extraction_service, fake_clarifier, fake_kb_agent, repository, pipeline
):
    service = ClarificationService(
        fake_extraction_service, fake_clarifier, repository, fake_kb_agent, ticket_pipeline=pipeline
    )

    with pytest.raises(ConversationNotFoundError):
        await service.handle_deflection_feedback("nope", resolved=True)

    await service.handle_message("conv-ask", "Teams isn't working.")  # ASK, not DEFLECTED
    with pytest.raises(ConversationStateError):
        await service.handle_deflection_feedback("conv-ask", resolved=True)

    fake_kb_agent.check.return_value = _make_strong_match_kb_result()
    await service.handle_message("conv-kb", "Wifi drops after sleep.")
    await service.handle_deflection_feedback("conv-kb", resolved=True)
    with pytest.raises(ConversationClosedError):  # feedback twice
        await service.handle_deflection_feedback("conv-kb", resolved=False)
    assert issubclass(ConversationClosedError, ConversationStateError)  # both map to 409


async def test_turn_stages_are_audited_when_store_is_set(
    fake_extraction_service, fake_clarifier, fake_kb_agent, repository, store
):
    service = ClarificationService(
        fake_extraction_service, fake_clarifier, repository, fake_kb_agent, store=store
    )

    await service.handle_message("conv-1", "Teams isn't working.")

    events = await store.fetch_audit("conv-1")
    assert [(e.stage, e.decision) for e in events] == [
        ("extract", "ms_teams"), ("kb", "NO_MATCH"), ("clarify", "ASK_CLARIFICATION"),
    ]
    assert all(e.turn == 0 and e.latency_ms >= 0 for e in events)
    assert events[2].payload["forced_escalation"] is False


async def test_kb_stage_not_audited_below_completeness_gate(
    fake_extraction_service, fake_clarifier, fake_kb_agent, repository, store
):
    fake_extraction_service.analyze.return_value = _make_analysis_result(
        completeness=settings.kb_gate_completeness_threshold - 0.01
    )
    service = ClarificationService(
        fake_extraction_service, fake_clarifier, repository, fake_kb_agent, store=store
    )

    await service.handle_message("conv-1", "Teams isn't working.")

    assert [e.stage for e in await store.fetch_audit("conv-1")] == ["extract", "clarify"]
