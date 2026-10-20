"""
Placeholder agents so the orchestrator and Agent 6 can be built and
demoed before the real Agent 4 (correlation) and Agent 5 (classification)
exist. Both satisfy the Protocols in app/agents/interfaces.py.

Swapping in a real agent is a one-line change in app/dependencies.py.
Delete this file once both real agents are merged.
"""

from app.agents.interfaces import OpenTicketPool
from app.schemas.conversation import ConversationState
from app.schemas.pipeline import ClassificationResult, CorrelationResult

# Internal category key (Agent 1 / category_profiles.py) -> support team.
# Placeholder routing table; Agent 5 owns the real one.
STUB_CATEGORY_TO_TEAM: dict[str, str] = {
    "wifi_internet": "Network Operations",
    "ms_teams": "Collaboration Tools Support",
    "vit_email": "Messaging Support",
    "ad_account_creation": "Identity & Access Management",
    "printer_support": "Desktop Support",
    "general": "Service Desk L1",
}
DEFAULT_TEAM = "Service Desk L1"

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