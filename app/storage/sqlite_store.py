"""
SQLite persistence for Agent 6 (tickets) and the pipeline audit trail.

Design rules (carried over from tro-backend/db.py on feature/orchestrator):
  - one short-lived connection per call - crash-safe and simple
  - parameterised `?` SQL only, never string-formatted values
  - sqlite3.Row for dict-like rows

Every public method has a synchronous `*_sync` version (used by scripts
and tests) and an async version that offloads it with asyncio.to_thread,
the same pattern KBDeflectionAgent uses for chromadb, so FastAPI's event
loop is never blocked by disk I/O.

The full TicketRecord is stored as JSON in `record_json` (source of
truth); a few columns are duplicated out of it purely so they can be
filtered/indexed in SQL (status, category, created_at, ...).

Datetimes are stored as fixed-width UTC strings
("2026-10-21T08:15:00.000000+00:00") so plain string comparison in SQL
orders them correctly - list_open()'s time window depends on this.

SqliteStore also satisfies the OpenTicketPool protocol
(app/agents/interfaces.py) via list_open(), which Agent 4 consumes.
"""

import asyncio
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from app.schemas.pipeline import AuditEvent, TicketRecord, TicketStatus

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id        TEXT UNIQUE,
    conversation_id  TEXT NOT NULL,
    category         TEXT,
    priority         TEXT,
    status           TEXT NOT NULL,
    building         TEXT,
    created_at       TEXT NOT NULL,
    external_key     TEXT,
    record_json      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tickets_open
    ON tickets(status, category, created_at);
CREATE INDEX IF NOT EXISTS idx_tickets_conversation
    ON tickets(conversation_id);

CREATE TABLE IF NOT EXISTS audit_events (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id  TEXT NOT NULL,
    turn             INTEGER NOT NULL,
    stage            TEXT NOT NULL,
    decision         TEXT NOT NULL,
    confidence       REAL,
    payload          TEXT NOT NULL,
    latency_ms       REAL NOT NULL,
    created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_conversation
    ON audit_events(conversation_id, id);
"""

_TS_FORMAT = "%Y-%m-%dT%H:%M:%S.%f+00:00"


def to_db_timestamp(dt: datetime) -> str:
    """Normalise any datetime to a fixed-width UTC string. Naive = assumed UTC."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime(_TS_FORMAT)


def format_ticket_id(created_at: datetime, row_id: int) -> str:
    """TRO-<year>-<5-digit row id>, e.g. TRO-2026-00001."""
    return f"TRO-{created_at.year}-{row_id:05d}"


class SqliteStore:
    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path)

    @property
    def db_path(self) -> Path:
        return self._db_path

    # ------------------------------------------------------------------
    # connection + schema
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        # timeout: wait (rather than fail instantly) if another process -
        # an eval script, DB Browser for SQLite - holds a write lock.
        conn = sqlite3.connect(self._db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        """Create tables/indexes if missing and enable WAL. Idempotent."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            # WAL lets readers (eval scripts, DB browser) run while the
            # API writes. It's persistent - set once, stays on the file.
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.executescript(SCHEMA)
            conn.commit()

    # ------------------------------------------------------------------
    # tickets
    # ------------------------------------------------------------------

    def insert_ticket_sync(self, ticket: TicketRecord) -> TicketRecord:
        """
        Insert a ticket and assign its ticket_id from the autoincrement
        row id, inside one transaction. Returns the stored record (with
        ticket_id set). Any ticket_id already on the input is ignored -
        the store is the only thing that issues ids.
        """
        created = to_db_timestamp(ticket.created_at)
        with closing(self._connect()) as conn:
            with conn:  # commits on success, rolls back on exception
                cur = conn.execute(
                    """
                    INSERT INTO tickets (conversation_id, category, priority,
                                         status, building, created_at, record_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ticket.conversation_id,
                        ticket.category,
                        ticket.priority,
                        ticket.status,
                        ticket.building,
                        created,
                        "{}",  # placeholder until the id is known
                    ),
                )
                row_id = cur.lastrowid
                ticket_id = format_ticket_id(ticket.created_at, row_id)
                stored = ticket.model_copy(update={"ticket_id": ticket_id})
                conn.execute(
                    "UPDATE tickets SET ticket_id = ?, record_json = ? WHERE id = ?",
                    (ticket_id, stored.model_dump_json(), row_id),
                )
        return stored

    def get_ticket_sync(self, ticket_id: str) -> TicketRecord | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT record_json FROM tickets WHERE ticket_id = ?", (ticket_id,)
            ).fetchone()
        return TicketRecord.model_validate_json(row["record_json"]) if row else None

    def list_tickets_sync(
        self, status: TicketStatus | None = None, limit: int = 50
    ) -> list[TicketRecord]:
        """Newest first. `status=None` returns all statuses."""
        sql = "SELECT record_json FROM tickets"
        params: list = []
        if status is not None:
            sql += " WHERE status = ?"
            params.append(status)
        sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(limit)
        with closing(self._connect()) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [TicketRecord.model_validate_json(r["record_json"]) for r in rows]

    def list_open_sync(
        self, since: datetime, category: str | None = None
    ) -> list[TicketRecord]:
        """OPEN tickets created at/after `since`, optionally one category. Newest first."""
        sql = "SELECT record_json FROM tickets WHERE status = 'OPEN' AND created_at >= ?"
        params: list = [to_db_timestamp(since)]
        if category is not None:
            sql += " AND category = ?"
            params.append(category)
        sql += " ORDER BY created_at DESC, id DESC"
        with closing(self._connect()) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [TicketRecord.model_validate_json(r["record_json"]) for r in rows]

    def update_external_ref_sync(
        self, ticket_id: str, external_key: str, external_url: str
    ) -> TicketRecord:
        """Record the external (Jira) key/url on both the column and the JSON."""
        return self._update_record_sync(
            ticket_id,
            {"external_key": external_key, "external_url": external_url},
            column_updates={"external_key": external_key},
        )

    def update_ticket_status_sync(self, ticket_id: str, status: TicketStatus) -> TicketRecord:
        return self._update_record_sync(
            ticket_id, {"status": status}, column_updates={"status": status}
        )

    def _update_record_sync(
        self, ticket_id: str, record_updates: dict, column_updates: dict
    ) -> TicketRecord:
        with closing(self._connect()) as conn:
            with conn:
                row = conn.execute(
                    "SELECT record_json FROM tickets WHERE ticket_id = ?", (ticket_id,)
                ).fetchone()
                if row is None:
                    raise KeyError(f"ticket {ticket_id!r} not found")
                current = TicketRecord.model_validate_json(row["record_json"])
                # model_validate (not model_copy) so updates are re-validated,
                # e.g. an invalid status is rejected rather than stored.
                updated = TicketRecord.model_validate(
                    {**current.model_dump(), **record_updates}
                )
                # Column names come only from this module's own call sites,
                # never from user input, so interpolating them is safe.
                set_sql = ", ".join(f"{col} = ?" for col in column_updates)
                conn.execute(
                    f"UPDATE tickets SET {set_sql}, record_json = ? WHERE ticket_id = ?",
                    (*column_updates.values(), updated.model_dump_json(), ticket_id),
                )
        return updated

    # ------------------------------------------------------------------
    # audit trail
    # ------------------------------------------------------------------

    def log_event_sync(self, event: AuditEvent) -> int:
        """Append one audit row. Returns its row id."""
        with closing(self._connect()) as conn:
            with conn:
                cur = conn.execute(
                    """
                    INSERT INTO audit_events (conversation_id, turn, stage, decision,
                                              confidence, payload, latency_ms, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.conversation_id,
                        event.turn,
                        event.stage,
                        event.decision,
                        event.confidence,
                        json.dumps(event.payload, default=str),
                        event.latency_ms,
                        to_db_timestamp(event.created_at),
                    ),
                )
                return cur.lastrowid

    def fetch_audit_sync(self, conversation_id: str) -> list[AuditEvent]:
        """All audit rows for one conversation, in the order they were written."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM audit_events WHERE conversation_id = ? ORDER BY id",
                (conversation_id,),
            ).fetchall()
        return [
            AuditEvent(
                conversation_id=r["conversation_id"],
                turn=r["turn"],
                stage=r["stage"],
                decision=r["decision"],
                confidence=r["confidence"],
                payload=json.loads(r["payload"]),
                latency_ms=r["latency_ms"],
                created_at=datetime.fromisoformat(r["created_at"]),
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # async wrappers (used by the API, Dispatcher and orchestrator)
    # ------------------------------------------------------------------

    async def insert_ticket(self, ticket: TicketRecord) -> TicketRecord:
        return await asyncio.to_thread(self.insert_ticket_sync, ticket)

    async def get_ticket(self, ticket_id: str) -> TicketRecord | None:
        return await asyncio.to_thread(self.get_ticket_sync, ticket_id)

    async def list_tickets(
        self, status: TicketStatus | None = None, limit: int = 50
    ) -> list[TicketRecord]:
        return await asyncio.to_thread(self.list_tickets_sync, status, limit)

    async def list_open(
        self, since: datetime, category: str | None = None
    ) -> list[TicketRecord]:
        """OpenTicketPool implementation (consumed by Agent 4)."""
        return await asyncio.to_thread(self.list_open_sync, since, category)

    async def update_external_ref(
        self, ticket_id: str, external_key: str, external_url: str
    ) -> TicketRecord:
        return await asyncio.to_thread(
            self.update_external_ref_sync, ticket_id, external_key, external_url
        )

    async def update_ticket_status(self, ticket_id: str, status: TicketStatus) -> TicketRecord:
        return await asyncio.to_thread(self.update_ticket_status_sync, ticket_id, status)

    async def log_event(self, event: AuditEvent) -> int:
        return await asyncio.to_thread(self.log_event_sync, event)

    async def fetch_audit(self, conversation_id: str) -> list[AuditEvent]:
        return await asyncio.to_thread(self.fetch_audit_sync, conversation_id)
