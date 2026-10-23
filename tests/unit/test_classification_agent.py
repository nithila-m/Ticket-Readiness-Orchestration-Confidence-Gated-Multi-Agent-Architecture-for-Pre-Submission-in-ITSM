import asyncio

import pytest

from app.agents.classification_agent import (
    CLUSTER_OVERRIDE_TAG,
    DISAGREEMENT_PENALTY,
    RULE_CONFIDENCE,
    RULE_FALLBACK_TAG,
    ClassificationAgent,
    adjust_confidence,
    resolve_category,
)
from app.agents.classification_prompts import build_agent5_context
from app.agents.interfaces import ClassificationAgent as ClassificationAgentProtocol
from app.config.category_profiles import VALID_CATEGORIES
from app.config.routing import CATEGORY_TO_TEAM, DEFAULT_TEAM
from app.providers.classification_base import ClassificationProvider
from app.providers.exceptions import LLMProviderError
from app.providers.fake_classification_provider import FakeClassificationProvider
from app.schemas.classification import ClassificationLLMOutput
from app.schemas.conversation import ConversationState, Message
from app.schemas.extraction import CategoryPrediction, ExtractedField
from app.schemas.pipeline import CorrelationResult, SimilarTicket


def _state(
    category: str | None = "wifi_internet",
    confidence: float = 0.9,
    correlation: CorrelationResult | None = None,
) -> ConversationState:
    pred = (
        CategoryPrediction(value=category, confidence=confidence)
        if category
        else CategoryPrediction(value=None, confidence=0.0)
    )
    return ConversationState(
        conversation_id="c1",
        raw_messages=[Message(role="user", content="wifi keeps dropping on my laptop in SJT")],
        detected_category=pred,
        extracted_fields={
            "symptom_type": ExtractedField(value="drops", confidence=0.9),
            "device_type": ExtractedField(value="laptop", confidence=0.8),
            "ssid": ExtractedField(value=None, confidence=0.0),
        },
        completeness_score=0.6,
        correlation=correlation,
    )


def _output(**overrides) -> ClassificationLLMOutput:
    data = {
        "category": "wifi_internet",
        "priority": "P3",
        "confidence": 0.8,
        "rationale": "symptom_type='drops' on device_type='laptop' for one user indicates P3.",
        "impact_signal": "single_user",
    }
    data.update(overrides)
    return ClassificationLLMOutput(**data)


def _agent(output=None, error=None, timeout_seconds: float = 20.0) -> ClassificationAgent:
    return ClassificationAgent(
        FakeClassificationProvider(output=output, error=error), timeout_seconds=timeout_seconds
    )


def _cluster() -> CorrelationResult:
    return CorrelationResult(
        outcome="INCIDENT_CLUSTER",
        cluster_id="CL-7",
        similar_tickets=[
            SimilarTicket(ticket_id="T1", similarity=0.9),
            SimilarTicket(ticket_id="T2", similarity=0.85),
        ],
        max_similarity=0.9,
    )


# --- Contract ---------------------------------------------------------------


def test_satisfies_protocol():
    assert isinstance(_agent(_output()), ClassificationAgentProtocol)


def test_routing_table_covers_every_category():
    assert set(CATEGORY_TO_TEAM) == set(VALID_CATEGORIES)


# --- Normal LLM path --------------------------------------------------------


async def test_llm_path_happy():
    result = await _agent(_output()).classify(_state())
    assert result.source == "llm"
    assert result.category == "wifi_internet"
    assert result.priority == "P3"
    assert result.assignee_group == "Network Operations"
    assert "symptom_type" in result.routing_rationale


async def test_team_always_from_table_even_if_llm_returns_team_like_string():
    # A team name in the category slot is not a valid key -> ignored.
    out = _output(
        category="Network Operations",
        rationale="Assign to Campus Infra Team; symptom_type='drops'.",
    )
    result = await _agent(out).classify(_state("printer_support", 0.9))
    assert result.category == "printer_support"
    assert result.assignee_group == CATEGORY_TO_TEAM["printer_support"]
    assert "Routed to Desktop Support via category routing table" in result.routing_rationale


async def test_invalid_llm_category_falls_back_to_agent1_category():
    # Agent 1 low-confidence, so the LLM *could* override - but only with a valid key.
    result = await _agent(_output(category="networking")).classify(_state("wifi_internet", 0.3))
    assert result.category == "wifi_internet"
    assert result.assignee_group == "Network Operations"


async def test_invalid_llm_category_and_no_agent1_category_gives_general():
    result = await _agent(_output(category="networking")).classify(_state(None))
    assert result.category == "general"
    assert result.assignee_group == DEFAULT_TEAM


async def test_llm_can_override_low_confidence_agent1():
    result = await _agent(_output(category="ms_teams", confidence=0.9)).classify(
        _state("wifi_internet", 0.4)
    )
    assert result.category == "ms_teams"
    assert result.assignee_group == "Collaboration Tools Support"
    assert result.confidence == pytest.approx(0.9 * DISAGREEMENT_PENALTY)
    assert "Category changed" in result.routing_rationale


async def test_llm_can_override_general():
    result = await _agent(_output(category="vit_email")).classify(_state("general", 0.9))
    assert result.category == "vit_email"


async def test_llm_cannot_override_confident_agent1():
    result = await _agent(_output(category="ms_teams")).classify(_state("wifi_internet", 0.9))
    assert result.category == "wifi_internet"
    assert result.assignee_group == "Network Operations"


# --- Confidence -------------------------------------------------------------


async def test_confidence_capped_at_agent1_when_agreeing():
    result = await _agent(_output(confidence=0.95)).classify(_state("wifi_internet", 0.7))
    assert result.confidence == pytest.approx(0.7)


async def test_confidence_not_raised_when_llm_lower():
    result = await _agent(_output(confidence=0.5)).classify(_state("wifi_internet", 0.9))
    assert result.confidence == pytest.approx(0.5)


def test_adjust_confidence_clamped():
    for llm, a1, agreed, overrode in [
        (1.0, 1.0, True, False),
        (0.0, 0.0, False, True),
        (1.0, 0.0, False, False),
    ]:
        assert 0.0 <= adjust_confidence(llm, a1, agreed, overrode) <= 1.0


def test_resolve_category_threshold_boundary():
    assert resolve_category("wifi_internet", 0.5, "ms_teams") == ("wifi_internet", False)
    assert resolve_category("wifi_internet", 0.49, "ms_teams") == ("ms_teams", True)


# --- Priority / correlation -------------------------------------------------


async def test_incident_cluster_forces_p1_and_rationale_mentions_cluster():
    result = await _agent(_output(priority="P4")).classify(_state(correlation=_cluster()))
    assert result.priority == "P1"
    assert result.source == "llm"
    assert CLUSTER_OVERRIDE_TAG in result.routing_rationale
    assert "cluster" in result.routing_rationale.lower()
    assert "CL-7" in result.routing_rationale


@pytest.mark.parametrize("outcome", ["NO_MATCH", "DUPLICATE"])
async def test_non_cluster_correlation_keeps_llm_priority(outcome):
    result = await _agent(_output(priority="P2")).classify(
        _state(correlation=CorrelationResult(outcome=outcome))
    )
    assert result.priority == "P2"


async def test_correlation_none_works():
    state = _state(correlation=None)
    result = await _agent(_output(priority="P2")).classify(state)
    assert result.priority == "P2"
    assert "Correlation has not run" in build_agent5_context(state)


# --- Never raise ------------------------------------------------------------


async def test_provider_raises_gives_rule_fallback():
    result = await _agent(error=LLMProviderError("Groq API call failed: 429")).classify(_state())
    assert result.source == "rule"
    assert result.priority == "P3"
    assert result.confidence == RULE_CONFIDENCE
    assert result.category == "wifi_internet"
    assert result.assignee_group == "Network Operations"
    assert RULE_FALLBACK_TAG in result.routing_rationale


async def test_unexpected_exception_gives_rule_fallback():
    result = await _agent(error=RuntimeError("boom")).classify(_state(None))
    assert result.source == "rule"
    assert result.category == "general"
    assert result.assignee_group == DEFAULT_TEAM


async def test_invalid_output_gives_rule_fallback():
    result = await _agent(output={"category": "wifi_internet", "priority": "P9"}).classify(_state())
    assert result.source == "rule"


async def test_timeout_gives_rule_fallback():
    class _SlowProvider(ClassificationProvider):
        async def classify_ticket(self, state):
            await asyncio.sleep(5)

    agent = ClassificationAgent(_SlowProvider(), timeout_seconds=0.01)
    result = await agent.classify(_state())
    assert result.source == "rule"
    assert "timed out" in result.routing_rationale


async def test_rule_fallback_still_applies_cluster_override():
    result = await _agent(error=LLMProviderError("down")).classify(_state(correlation=_cluster()))
    assert result.source == "rule"
    assert result.priority == "P1"


# --- Invariants -------------------------------------------------------------


@pytest.mark.parametrize(
    "output,error,state",
    [
        (_output(), None, _state()),
        (_output(confidence=1.0), None, _state(None)),
        (_output(confidence=0.0, category="general"), None, _state("general", 0.2)),
        (_output(rationale=" "), None, _state()),
        (None, LLMProviderError("x"), _state(correlation=_cluster())),
    ],
)
async def test_rationale_never_empty_and_confidence_in_range(output, error, state):
    result = await _agent(output, error).classify(state)
    assert result.routing_rationale.strip()
    assert 0.0 <= result.confidence <= 1.0
