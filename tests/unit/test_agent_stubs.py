from datetime import datetime, timezone

from app.agents.interfaces import ClassificationAgent, CorrelationAgent, OpenTicketPool
from app.agents.stubs import StubClassificationAgent, StubCorrelationAgent
from app.schemas.conversation import ConversationState
from app.schemas.extraction import CategoryPrediction


class _EmptyPool:
    async def list_open(self, since, category=None):
        return []


def _state(category: str | None) -> ConversationState:
    pred = (
        CategoryPrediction(value=category, confidence=0.95)
        if category
        else CategoryPrediction(value=None, confidence=0.0)
    )
    return ConversationState(conversation_id="c1", detected_category=pred)


def test_stubs_satisfy_protocols():
    assert isinstance(StubClassificationAgent(), ClassificationAgent)
    assert isinstance(StubCorrelationAgent(), CorrelationAgent)
    assert isinstance(_EmptyPool(), OpenTicketPool)


async def test_stub_classifier_maps_category_to_team():
    result = await StubClassificationAgent().classify(_state("wifi_internet"))
    assert result.category == "wifi_internet"
    assert result.assignee_group == "Network Operations"
    assert result.priority == "P3"
    assert result.source == "stub"
    assert 0.0 <= result.confidence <= 1.0
    assert result.routing_rationale


async def test_stub_classifier_falls_back_to_general_when_no_category():
    result = await StubClassificationAgent().classify(_state(None))
    assert result.category == "general"
    assert result.assignee_group == "Service Desk L1"


async def test_stub_correlation_always_no_match():
    result = await StubCorrelationAgent().check(_state("printer_support"), _EmptyPool())
    assert result.outcome == "NO_MATCH"
    assert result.duplicate_of is None