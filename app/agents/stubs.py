"""
Placeholder agents so the orchestrator and Agent 6 can be built and
demoed before the real Agent 4 (correlation) and Agent 5 (classification)
exist. Both satisfy the Protocols in app/agents/interfaces.py.

Swapping in a real agent is a one-line change in app/dependencies.py.
Delete this file once both real agents are merged.
"""

from app.agents.interfaces import OpenTicketPool
from app.config.routing import CATEGORY_TO_TEAM, DEFAULT_TEAM
from app.schemas.conversation import ConversationState
from app.schemas.pipeline import ClassificationResult, CorrelationResult

# Kept as an alias so existing imports keep working; the real table now
# lives in app/config/routing.py (owned by Agent 5).
STUB_CATEGORY_TO_TEAM: dict[str, str] = CATEGORY_TO_TEAM

STUB_PRIORITY = "P3"
STUB_CONFIDENCE = 0.9
STUB_RATIONALE = "Routed by category mapping (placeholder until Agent 5)."


class StubClassificationAgent:
    """Stand-in for Agent 5. Deterministic: category -> team, fixed priority."""

    async def classify(self, state: ConversationState) -> ClassificationResult:
        category = (
            state.detected_category.value
            if state.detected_category is not None and state.detected_category.value
            else "general"
        )
        team = STUB_CATEGORY_TO_TEAM.get(category, DEFAULT_TEAM)
        return ClassificationResult(
            category=category,
            priority=STUB_PRIORITY,
            assignee_group=team,
            routing_rationale=f"{STUB_RATIONALE} Category '{category}' -> {team}.",
            confidence=STUB_CONFIDENCE,
            source="stub",
        )


class StubCorrelationAgent:
    """Stand-in for Agent 4. Never finds a duplicate or cluster."""

    async def check(
        self, state: ConversationState, pool: OpenTicketPool
    ) -> CorrelationResult:
        return CorrelationResult(outcome="NO_MATCH")
