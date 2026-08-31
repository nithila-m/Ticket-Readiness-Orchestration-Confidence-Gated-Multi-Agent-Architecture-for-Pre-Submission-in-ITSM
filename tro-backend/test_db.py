"""One-off script: write a dummy audit row and read it back."""

from db import init_db, log_event, fetch_recent

init_db()

new_id = log_event(
    session_id="test_session_1",
    stage="stub_stage",
    decision="dummy_decision",
    confidence=0.42,
)
print(f"Inserted row id: {new_id}")

rows = fetch_recent(limit=5)
print(f"Rows in audit_log ({len(rows)} total):")
for row in rows:
    print(f"  {row}")