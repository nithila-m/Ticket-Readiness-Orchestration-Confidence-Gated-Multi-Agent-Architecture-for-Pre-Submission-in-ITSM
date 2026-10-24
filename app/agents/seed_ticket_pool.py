"""
JsonOpenTicketPool: an OpenTicketPool backed by data/eval/tickets_augmented.json.

Stand-in for Agent 6's SqliteStore.list_open() so Agent 4 can be built,
evaluated and demoed before real tickets exist. Both satisfy the same
protocol (app/agents/interfaces.py), so swapping is a one-line change in
app/dependencies.py.

Each JSON ticket becomes a valid TicketRecord: category from the dataset
label via normalize_category_label, team from app.config.routing, priority
via DATASET_PRIORITY_TO_P, title = subject, description = message,
created_at = load time + created_at_offset_minutes. The dataset's eval
labels (label_type, gold_cluster_id, duplicate_of, hard_negative_of) are
deliberately NOT copied onto the records, so Agent 4 never sees the answers.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from app.config.category_profiles import normalize_category_label
from app.config.routing import DATASET_PRIORITY_TO_P, team_for_category
from app.schemas.pipeline import TicketRecord

DEFAULT_SEED_PATH = Path(__file__).resolve().parents[2] / "data" / "eval" / "tickets_augmented.json"
SEED_RATIONALE = "Seed data from data/eval/tickets_augmented.json; not classified by Agent 5."


def seed_to_record(raw: dict, now: datetime) -> TicketRecord:
    category = (
        normalize_category_label(raw.get("true_category"))
        or normalize_category_label(raw.get("category_selected"))
        or "general"
    )
    return TicketRecord(
        ticket_id=raw["ticket_id"],
        conversation_id=f"seed-{raw['ticket_id']}",
        title=(raw.get("subject") or raw["ticket_id"])[:255],
        description=raw.get("message") or "",
        category=category,
        priority=DATASET_PRIORITY_TO_P.get(raw.get("priority"), "P3"),
        assignee_group=team_for_category(category),
        routing_rationale=SEED_RATIONALE,
        confidence_score=1.0,
        building=raw.get("building"),
        status=raw.get("status", "OPEN"),
        created_at=now + timedelta(minutes=raw.get("created_at_offset_minutes", 0)),
    )


class JsonOpenTicketPool:
    """
    Loads the JSON lazily on first list_open(); created_at offsets are
    resolved against the clock at that moment and then kept fixed.
    """

    def __init__(
        self,
        path: Path = DEFAULT_SEED_PATH,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        self._path = Path(path)
        self._clock = clock
        self._records: list[TicketRecord] | None = None

    def records(self) -> list[TicketRecord]:
        if self._records is None:
            now = self._clock()
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._records = [seed_to_record(t, now) for t in raw]
        return self._records

    async def list_open(self, since: datetime, category: str | None = None) -> list[TicketRecord]:
        rows = [
            r for r in self.records()
            if r.status == "OPEN"
            and r.created_at >= since
            and (category is None or r.category == category)
        ]
        return sorted(rows, key=lambda r: r.created_at, reverse=True)
