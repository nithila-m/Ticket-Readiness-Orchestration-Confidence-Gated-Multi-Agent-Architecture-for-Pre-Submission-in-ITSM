"""
Startup wiring: the FastAPI lifespan handler must create the SQLite
tables, and must use the overridden store in tests (never the real tro.db).
"""

import sqlite3

from fastapi.testclient import TestClient

from app.dependencies import get_store
from app.main import app
from app.storage.sqlite_store import SqliteStore


def test_lifespan_initialises_overridden_store(tmp_path):
    db_path = tmp_path / "lifespan.db"
    app.dependency_overrides[get_store] = lambda: SqliteStore(db_path)
    try:
        # Using TestClient as a context manager runs the lifespan handler.
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200
        with sqlite3.connect(db_path) as conn:
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"tickets", "audit_events"} <= tables
    finally:
        app.dependency_overrides.pop(get_store, None)
