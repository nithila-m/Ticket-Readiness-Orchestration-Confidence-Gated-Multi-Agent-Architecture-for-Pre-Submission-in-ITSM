"""
SQLite audit log for TRO.

Provides:
  - init_db()   : creates audit.db and the audit_log table if missing
  - log_event() : writes one audit row (called from graph nodes later)
  - fetch_recent(): reads recent rows back (for verification and debugging)

Uses Python's built-in sqlite3 module — no external dependencies.
"""

import sqlite3
from typing import Optional

# Location of the SQLite audit database file.
DB_PATH = "audit.db"


# Helper to open a fresh database connection.
def _connect() -> sqlite3.Connection:
    """Open a new connection. Callers are responsible for closing it."""
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    """Create the audit_log table if it doesn't exist. Idempotent."""
    conn = _connect()
    try:
        # Schema: each row captures one step in the TRO workflow for auditing and replay.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT,
                stage       TEXT,
                decision    TEXT,
                confidence  REAL,
                timestamp   TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def log_event(
    session_id: str,
    stage: str,
    decision: str,
    confidence: float,
) -> int:
    """Insert one audit row. Returns the new row's id."""
    # Write audit entry so each workflow step is recorded for later review or debugging.
    conn = _connect()
    try:
        cursor = conn.execute(
            """
            INSERT INTO audit_log (session_id, stage, decision, confidence)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, stage, decision, confidence),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def fetch_recent(limit: int = 10, session_id: Optional[str] = None) -> list[dict]:
    """Return up to `limit` recent audit rows, newest first.

    If session_id is given, only rows for that session are returned.
    """
    # Open a connection and configure it to return rows as dict-like objects.
    conn = _connect()
    conn.row_factory = sqlite3.Row  # rows behave like dicts
    try:
        if session_id is None:
            # Fetch global audit trail, newest entries first.
            rows = conn.execute(
                "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            # Fetch audit trail for one session only.
            rows = conn.execute(
                "SELECT * FROM audit_log WHERE session_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


if __name__ == "__main__":
    # Quick sanity check: initialize the database and display recent audit entries.
    # Useful for manual testing or verifying the schema.
    init_db()
    print(f"Initialized {DB_PATH}")
    recent = fetch_recent(limit=5)
    print(f"Most recent {len(recent)} rows:")
    for row in recent:
        print(f"  {row}")

'''
Design notes:
* Parameterized queries (? placeholders) — never string-format SQL, even for a demo. 
  Good habit to lock in early.
* sqlite3.Row gives you dict-like access to rows, so fetch_recent() returns 
  clean dicts you can serialize to JSON later.
* _connect() returns a fresh connection per call. Slightly wasteful for high throughput, 
  but crash-safe and dead simple — right trade-off for Day 1.
* log_event() returns the new row's id, useful if you later want to link an audit row back to a ticket.
'''