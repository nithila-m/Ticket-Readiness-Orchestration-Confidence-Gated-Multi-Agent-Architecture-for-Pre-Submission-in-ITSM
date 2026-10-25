"""
Seed the SQLite ticket store with the synthetic open-ticket pool so Agent 4
(Correlation Engine) has something to match against.

Run from the project root:
    python -m scripts.seed_ticket_store            # add to settings.sqlite_path (tro.db)
    python -m scripts.seed_ticket_store --reset    # delete existing tickets first (asks to confirm)
    python -m scripts.seed_ticket_store --db other.db --limit 50

Each record goes through seed_to_record() from app/agents/seed_ticket_pool.py,
the same conversion JsonOpenTicketPool uses. created_at offsets are resolved
against the clock NOW, at seed time, so the incident-cluster window (6 h)
holds right after seeding - re-seed with --reset if the data gets stale.
Ticket ids are issued by the store (TRO-<year>-<n>); the dataset's own ids
are not kept.
"""

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from app.agents.seed_ticket_pool import DEFAULT_SEED_PATH, seed_to_record
from app.config.settings import settings
from app.storage.sqlite_store import SqliteStore


def seed_tickets(
    store: SqliteStore,
    raw_tickets: list[dict],
    now: datetime | None = None,
    limit: int | None = None,
) -> Counter:
    """Insert seed tickets; returns a count per category."""
    now = now or datetime.now(timezone.utc)
    counts: Counter = Counter()
    for raw in raw_tickets[:limit]:
        record = seed_to_record(raw, now)
        store.insert_ticket_sync(record)
        counts[record.category] += 1
    return counts


def load_seed(path: Path = DEFAULT_SEED_PATH) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the TRO SQLite ticket store.")
    parser.add_argument("--db", type=Path, default=settings.sqlite_path,
                        help=f"SQLite file (default: {settings.sqlite_path})")
    parser.add_argument("--reset", action="store_true",
                        help="delete ALL existing tickets first (asks for confirmation)")
    parser.add_argument("--limit", type=int, default=None, help="seed only the first N tickets")
    parser.add_argument("--seed-file", type=Path, default=DEFAULT_SEED_PATH)
    args = parser.parse_args()

    store = SqliteStore(args.db)
    store.init_db()

    if args.reset:
        existing = len(store.list_tickets_sync(limit=1_000_000))
        answer = input(f"Delete all {existing} tickets in {args.db}? Type 'yes' to confirm: ")
        if answer.strip().lower() != "yes":
            print("Aborted - nothing changed.")
            return
        print(f"Deleted {store.clear_tickets_sync()} tickets.")

    counts = seed_tickets(store, load_seed(args.seed_file), limit=args.limit)
    print(f"Seeded {sum(counts.values())} tickets into {args.db}:")
    for category, n in sorted(counts.items()):
        print(f"  {category:<22} {n}")
    print(f"Open tickets now in store: {len(store.list_tickets_sync(status='OPEN', limit=1_000_000))}")


if __name__ == "__main__":
    main()
