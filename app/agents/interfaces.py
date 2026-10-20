"""
Contracts between the orchestrator / Agent 6 and the agents owned by
other team members (Agent 4 - Medhini, Agent 5 - Trisha).

These are typing.Protocol classes (structural typing): an agent does NOT
need to inherit from them. Any class with a matching async method
signature satisfies the contract, so each owner can build and test their
agent in isolation and it plugs into dependencies.py without changes.

If a signature here needs to change, change it in a PR all three review -
never adapt it silently inside the graph or the dispatcher.
"""

from datetime import datetime
from typing import Protocol, runtime_checkable

from app.schemas.conversation import ConversationState
from app.schemas.pipeline import ClassificationResult, CorrelationResult, TicketRecord


@runtime_checkable
class OpenTicketPool(Protocol):
    """
    Read-only view of currently open tickets, used by Agent 4 for
    duplicate / incident-cluster detection.

    Implemented by SqliteStore.list_open() (Agent 6's ticket store).
    Tests can pass any small fake class with the same method.
    """

    async def list_open(
        self,
        since: datetime,
        category: str | None = None,
    ) -> list[TicketRecord]:
        """
        Return tickets with status OPEN created at or after `since`
        (timezone-aware UTC), optionally filtered by internal category
        key (e.g. "wifi_internet"). Newest first.
        """
        ...


@runtime_checkable
class CorrelationAgent(Protocol):
    """Agent 4 - Correlation Engine (duplicate + temporal cluster)."""

    async def check(
        self,
        state: ConversationState,
        pool: OpenTicketPool,
    ) -> CorrelationResult:
        ...


@runtime_checkable
class ClassificationAgent(Protocol):
    """Agent 5 - Classify, Prioritise, Explain."""

    async def classify(self, state: ConversationState) -> ClassificationResult:
        """
        Must always return a ClassificationResult with a non-empty
        routing_rationale and a confidence in [0, 1]. Low confidence is
        handled by the confidence gate, not by raising.
        """
        ...