"""
Contract for external ticket systems (Jira in Phase 4).

SQLite is the source of truth and is always written first; an adapter is
a best-effort sink the Dispatcher calls afterwards. Adapters may raise -
the Dispatcher catches and audits the error so an external outage never
fails the user's turn.
"""

from typing import NamedTuple, Protocol, runtime_checkable

from app.schemas.pipeline import TicketRecord


class ExternalRef(NamedTuple):
    key: str    # e.g. Jira "TRO-12"
    url: str


@runtime_checkable
class ExternalTicketAdapter(Protocol):
    name: str

    async def create(self, ticket: TicketRecord) -> ExternalRef | None:
        """Create the ticket externally. None = nothing was created."""
        ...

    async def link_duplicate(self, ticket: TicketRecord, existing: TicketRecord) -> None:
        """Record `ticket` as a duplicate of the already-stored `existing`."""
        ...


class NullAdapter:
    """Local-only mode: tickets live in SQLite and nowhere else."""

    name = "local"

    async def create(self, ticket: TicketRecord) -> ExternalRef | None:
        return None

    async def link_duplicate(self, ticket: TicketRecord, existing: TicketRecord) -> None:
        return None
