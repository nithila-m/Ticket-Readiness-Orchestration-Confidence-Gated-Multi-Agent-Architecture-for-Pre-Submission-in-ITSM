"""
End-to-end tests for POST /conversations/{id}/messages, hit through the
real FastAPI app (TestClient) with get_clarification_service overridden -
same dependency-override pattern tests/test_api.py already uses for
/analyze. Every collaborator (extraction, clarifier, KB agent) is a
lightweight stub, not a mock of internals, so this exercises the real
ClarificationService.handle_message() control flow: completeness gate ->
Agent 3 -> KB decision gate -> Agent 2 (or not) -> response shape.

Maps directly onto the five TEST cases from the integration plan:
TEST 1 - complete request + strong KB match -> deflection
TEST 2 - complete request + weak/no KB match -> continues downstream
TEST 3 - incomplete request -> Agent 3 must NOT run prematurely
TEST 4 - multi-turn clarification -> Agent 3 only runs once complete
TEST 5 - existing chat behavior (response shape) unaffected by Agent 3
"""

import pytest
from fastapi.testclient import TestClient

from app.agents.kb_deflection_agent import KBRetrievalResult
from app.dependencies import get_clarification_service
from app.main import app
from app.repositories.in_memory_conversation_repository import InMemoryConversationRepository
from app.schemas.analysis import AnalysisResult
from app.schemas.clarification import ClarificationDecision
from app.schemas.extraction import CategoryPrediction, ExtractedField
from app.services.clarification_service import ClarificationService

client = TestClient(app)


def _analysis(completeness: float) -> AnalysisResult:
    return AnalysisResult(
        category=CategoryPrediction(value="wifi_internet", confidence=0.9),
        extracted_fields={"symptom_type": ExtractedField(value="drops", confidence=0.9)},
        completeness_score=completeness,
        missing_or_uncertain_fields=[] if completeness > 0.8 else ["when_started"],
    )


def _ask_decision(question: str = "Is this on one device or several?") -> ClarificationDecision:
    return ClarificationDecision(
        action="ASK_CLARIFICATION",
        reasoning="Need scope.",
        information_gap="scope",
        question=question,
        expected_information_gain=0.6,
        affected_fields=["scope"],
        priority="high",
        confidence=0.7,
    )


def _ready_decision() -> ClarificationDecision:
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


def _strong_kb_result() -> KBRetrievalResult:
    return KBRetrievalResult(
        outcome="STRONG_MATCH",
        similarity_score=0.9,
        articles_checked=["KB0302"],
        matched_kb_id="KB0302",
        matched_title="Wi-Fi disconnects repeatedly after device wakes from sleep",
        offered_resolution="1. Disable fast startup\n2. Forget and re-join Wi-Fi",
    )


def _no_match_kb_result() -> KBRetrievalResult:
    return KBRetrievalResult(
        outcome="NO_MATCH",
        similarity_score=0.1,
        articles_checked=["KB0001"],
        matched_kb_id=None,
        matched_title=None,
        offered_resolution=None,
    )


class StubExtractionService:
    """Returns queued results in order; repeats the last one once exhausted."""

    def __init__(self, results: list[AnalysisResult]):
        self._results = results
        self._calls = 0

    async def analyze(self, transcript: str) -> AnalysisResult:
        i = min(self._calls, len(self._results) - 1)
        self._calls += 1
        return self._results[i]


class StubClarifier:
    def __init__(self, decisions: list[ClarificationDecision]):
        self._decisions = decisions
        self.calls = 0

    async def decide(self, state) -> ClarificationDecision:
        i = min(self.calls, len(self._decisions) - 1)
        self.calls += 1
        return self._decisions[i]


class StubKBAgent:
    def __init__(self, result: KBRetrievalResult):
        self._result = result
        self.calls = 0

    async def check(self, ticket_text: str) -> KBRetrievalResult:
        self.calls += 1
        return self._result


def _build_service(extraction_results, decisions, kb_result):
    """Builds one ClarificationService instance shared across every request
    in a test, so multi-turn state actually persists (mirrors how the real
    app's singleton, cached ClarificationService behaves)."""
    extraction = StubExtractionService(extraction_results)
    clarifier = StubClarifier(decisions)
    kb_agent = StubKBAgent(kb_result)
    repository = InMemoryConversationRepository()
    service = ClarificationService(extraction, clarifier, repository, kb_agent)
    return service, clarifier, kb_agent


@pytest.fixture(autouse=True)
def clear_overrides():
    yield
    app.dependency_overrides.clear()


def test_1_complete_request_strong_kb_match_deflects():
    service, clarifier, kb_agent = _build_service(
        extraction_results=[_analysis(0.9)],
        decisions=[_ready_decision()],  # should never be reached
        kb_result=_strong_kb_result(),
    )
    app.dependency_overrides[get_clarification_service] = lambda: service

    response = client.post(
        "/conversations/test-1/messages",
        json={"message": "Wifi disconnects every time my laptop wakes from sleep."},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["action"] == "DEFLECTED"
    assert body["kb_outcome"] == "STRONG_MATCH"
    assert "Wi-Fi disconnects repeatedly" in body["kb_matched_title"]
    assert clarifier.calls == 0  # Agent 2 never ran


def test_2_complete_request_weak_kb_match_continues_downstream():
    service, clarifier, kb_agent = _build_service(
        extraction_results=[_analysis(0.9)],
        decisions=[_ready_decision()],
        kb_result=_no_match_kb_result(),
    )
    app.dependency_overrides[get_clarification_service] = lambda: service

    response = client.post(
        "/conversations/test-2/messages",
        json={"message": "Wifi drops every few minutes on my laptop, started yesterday."},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["action"] == "READY"
    assert body["kb_outcome"] == "NO_MATCH"
    assert kb_agent.calls == 1
    assert clarifier.calls == 1  # Agent 2 ran as normal


def test_3_incomplete_request_agent3_does_not_run_prematurely():
    service, clarifier, kb_agent = _build_service(
        extraction_results=[_analysis(0.2)],  # below kb_gate_completeness_threshold
        decisions=[_ask_decision()],
        kb_result=_strong_kb_result(),  # would deflect if the gate didn't stop it
    )
    app.dependency_overrides[get_clarification_service] = lambda: service

    response = client.post("/conversations/test-3/messages", json={"message": "wifi broken"})

    assert response.status_code == 200
    body = response.json()
    assert body["action"] == "ASK_CLARIFICATION"
    assert body["kb_outcome"] is None
    assert kb_agent.calls == 0  # Agent 3 never invoked - the gate held


def test_4_multi_turn_clarification_then_agent3_runs():
    service, clarifier, kb_agent = _build_service(
        extraction_results=[_analysis(0.2), _analysis(0.9)],
        decisions=[_ask_decision(), _ready_decision()],
        kb_result=_no_match_kb_result(),
    )
    app.dependency_overrides[get_clarification_service] = lambda: service

    turn1 = client.post("/conversations/test-4/messages", json={"message": "wifi broken"})
    assert turn1.json()["action"] == "ASK_CLARIFICATION"
    assert turn1.json()["kb_outcome"] is None
    assert kb_agent.calls == 0

    turn2 = client.post(
        "/conversations/test-4/messages",
        json={"message": "Just my laptop, started this morning."},
    )
    body2 = turn2.json()
    assert body2["turn"] == 2
    assert body2["action"] == "READY"
    assert body2["kb_outcome"] == "NO_MATCH"
    assert kb_agent.calls == 1  # only ran once completeness crossed the gate


def test_5_response_shape_unaffected_by_agent3_fields():
    """
    Confirms every pre-Agent-3 response field is still present and
    correctly populated - i.e. adding Agent 3 didn't change Agent 1/2's
    contract with the API layer, it only added new optional fields.
    """
    service, clarifier, kb_agent = _build_service(
        extraction_results=[_analysis(0.2)],
        decisions=[_ask_decision(question="Just you, or others too?")],
        kb_result=_no_match_kb_result(),
    )
    app.dependency_overrides[get_clarification_service] = lambda: service

    response = client.post("/conversations/test-5/messages", json={"message": "wifi broken"})
    body = response.json()

    for field in (
        "conversation_id",
        "turn",
        "action",
        "question",
        "category",
        "completeness_score",
        "affected_fields",
        "reasoning",
        "confidence",
    ):
        assert field in body

    assert body["conversation_id"] == "test-5"
    assert body["question"] == "Just you, or others too?"
    assert body["category"] == "wifi_internet"
    # New Agent 3 fields present but null - didn't run, gate held
    assert body["kb_outcome"] is None
    assert body["kb_matched_title"] is None