"""
Agent 6 - local-first Dispatcher.

Order of operations is the whole design:
  1. build the canonical ticket (pure, app/agents/ticket_builder.py)
  2. insert it into SQLite - the source of truth, always first
  3. best-effort external sink (Jira later; NullAdapter for now) inside
     try/except, so an external failure is recorded but never raised
  4. one "dispatch" AuditEvent, success or not

Only the store's async methods are used, so the event loop is never
blocked by disk I/O.
"""

import time
from typing import Literal, NamedTuple

from app.agents.ticket_builder import build_ticket
from app.integrations.base import ExternalTicketAdapter
from app.schemas.conversation import ConversationState
from app.schemas.pipeline import AuditEvent, TicketRecord
from app.storage.sqlite_store import SqliteStore


class DispatchResult(NamedTuple):
    ticket: TicketRecord
    outcome: Literal["DISPATCHED", "MERGED"]
    external_error: str | None


class Dispatcher:
    def __init__(self, store: SqliteStore, adapter: ExternalTicketAdapter):
        self._store = store
        self._adapter = adapter

    async def dispatch(self, state: ConversationState) -> DispatchResult:
        start = time.perf_counter()

        is_duplicate = state.correlation is not None and state.correlation.outcome == "DUPLICATE"
        outcome = "MERGED" if is_duplicate else "DISPATCHED"
        ticket = build_ticket(state).model_copy(
            update={"status": "MERGED" if is_duplicate else "OPEN"}
        )

        saved = await self._store.insert_ticket(ticket)

        external_error: str | None = None
        try:
            if is_duplicate:
                existing = (
                    await self._store.get_ticket(saved.duplicate_of)
                    if saved.duplicate_of
                    else None
                )
                if existing is not None:
                    await self._adapter.link_duplicate(saved, existing)
            else:
                ref = await self._adapter.create(saved)
                if ref is not None:
                    saved = await self._store.update_external_ref(saved.ticket_id, ref.key, ref.url)
        except Exception as exc:  # external sinks must never fail the turn
            external_error = f"{type(exc).__name__}: {exc}"

        await self._store.log_event(AuditEvent(
            conversation_id=state.conversation_id,
            turn=state.turn_count,
            stage="dispatch",
            decision=outcome,
            confidence=saved.confidence_score,
            payload={
                "ticket_id": saved.ticket_id,
                "adapter": self._adapter.name,
                "external_key": saved.external_key,
                "external_error": external_error,
            },
            latency_ms=(time.perf_counter() - start) * 1000,
        ))
        return DispatchResult(saved, outcome, external_error)
