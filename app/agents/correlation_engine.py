"""
Agent 4: Correlation Engine (duplicate + incident-cluster detection).

Deterministic: embeds the incoming request and the open tickets, compares
them by cosine similarity and applies fixed threshold rules. There is no
LLM call and no learned model beyond the sentence embedder.

Decision order:
  1. DUPLICATE        best eligible open ticket scores >= dup_threshold
                      (see the category rule below) -> duplicate_of it.
  2. INCIDENT_CLUSTER at least cluster_min_count tickets, counting the
                      incoming one, that score >= cluster_threshold, are in
                      the same building and were created within
                      cluster_window_hours of now -> cluster_id.
  3. NO_MATCH         otherwise, and on an empty pool, an empty query or
                      any failure (never raises).

Compared text: the query is build_kb_query(Agent 1's category, extracted
fields) - short "field: value" lines plus "Category: <label>". Each ticket
is compared by its stored correlation_text, built at dispatch in the same
format from the same source (ticket_builder.build_correlation_text), so
like is compared with like; the ticket body's boilerplate ("Reported by
user:", "Completeness: ...", ...) would otherwise drag identical requests
below the duplicate threshold. Tickets without correlation_text (seed
tickets, tickets created before it existed, or no extracted fields) fall
back to "title\ndescription".

Minimum-field gate (DUPLICATE only): a query with fewer than
dup_min_fields non-null extracted fields skips the DUPLICATE step, and a
candidate whose correlation_text has fewer than dup_min_fields
"field: value" lines (the "Category:" line does not count) cannot be the
duplicate. One-field texts are mostly the category line, so unrelated
thin requests collapse onto each other. Candidates without
correlation_text (title + description fallback) stay eligible. The
cluster step ignores this gate.

Building rule: if the incoming building and a candidate's building are
both known and differ, the candidate can be neither a duplicate nor a
cluster member. Agent 1 does not currently extract a building, so when the
incoming building is unknown a cluster is still allowed, but only among
candidates that share one known building (the largest such group).
Duplicates do not need a building.

Category rule: Agent 4 runs before Agent 5, so Agent 1's category may
differ from the category stored on open tickets. Category is therefore a
scoring signal, not a filter: when both are known and agree, the score is
the cosine similarity + category_agreement_bonus (capped at 1.0);
otherwise the score is the plain similarity. A duplicate across different
categories additionally needs raw similarity >= cross_category_dup_threshold.

Evidence: similar_tickets carries up to 5 tickets with their RAW cosine
similarity (the bonus only affects ranking and decisions) - the cluster
members for INCIDENT_CLUSTER, otherwise the top eligible candidates;
max_similarity is the highest raw similarity among eligible candidates.
"""

import asyncio
import hashlib
import logging
import math
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Callable

from app.agents.interfaces import OpenTicketPool
from app.agents.request_text import build_kb_query
from app.config.buildings import normalize_building
from app.config.settings import settings
from app.schemas.conversation import ConversationState
from app.schemas.pipeline import CorrelationResult, SimilarTicket, TicketRecord

logger = logging.getLogger(__name__)

EmbedFn = Callable[[list[str]], list[list[float]]]

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # same model Agent 3's KB index uses
BUILDING_FIELD_NAMES = ("building", "location")
MAX_EVIDENCE = 5


class _LazySentenceTransformer:
    """Loads all-MiniLM-L6-v2 on first call; caches embeddings by text."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self._model_name = model_name
        self._model = None
        self._cache: dict[str, list[float]] = {}

    def __call__(self, texts: list[str]) -> list[list[float]]:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self._model_name)
            vectors = self._model.encode(missing, normalize_embeddings=True)
            for text, vector in zip(missing, vectors):
                self._cache[text] = [float(x) for x in vector]
        return [self._cache[t] for t in texts]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def query_field_count(state: ConversationState) -> int:
    """Non-null extracted fields - the "field: value" lines of the query."""
    return sum(1 for f in state.extracted_fields.values() if f.value is not None)


def correlation_field_lines(text: str | None) -> int | None:
    """
    "field: value" lines in a stored correlation_text, not counting the
    "Category:" line; None when the ticket has no correlation_text.
    """
    if not text:
        return None
    return sum(
        1 for line in text.splitlines()
        if line.strip() and not line.startswith("Category:")
    )


def candidate_text(ticket: TicketRecord) -> str:
    """Stored correlation_text when set; else title + description (see module docstring)."""
    if ticket.correlation_text:
        return ticket.correlation_text
    return f"{ticket.title}\n{ticket.description}"


def state_building(state: ConversationState) -> str | None:
    """Normalised building from Agent 1's extracted fields, if any."""
    for name in BUILDING_FIELD_NAMES:
        field = state.extracted_fields.get(name)
        if field is not None and field.value:
            return normalize_building(str(field.value))
    return None


def _building_compatible(incoming: str | None, candidate: str | None) -> bool:
    """False only when both buildings are known and differ."""
    candidate = normalize_building(candidate)
    return incoming is None or candidate is None or candidate == incoming


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _cluster_id(building: str, members: list[TicketRecord]) -> str:
    """
    Reuse the members' existing incident_cluster_id when they have one;
    otherwise derive one from the building and the earliest member, so the
    id stays stable as later reports join the same cluster.
    """
    existing = Counter(m.incident_cluster_id for m in members if m.incident_cluster_id)
    if existing:
        return existing.most_common(1)[0][0]
    earliest = min(members, key=lambda m: (_as_utc(m.created_at), m.ticket_id))
    digest = hashlib.sha1(f"{building}|{earliest.ticket_id}".encode()).hexdigest()[:10]
    return f"CLU-{digest}"


class CorrelationEngine:
    """Agent 4. Satisfies app.agents.interfaces.CorrelationAgent."""

    def __init__(
        self,
        embed_fn: EmbedFn | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        dup_threshold: float | None = None,
        cluster_threshold: float | None = None,
        cluster_min_count: int | None = None,
        cluster_window_hours: float | None = None,
        lookback_hours: float | None = None,
        category_agreement_bonus: float | None = None,
        cross_category_dup_threshold: float | None = None,
        dup_min_fields: int | None = None,
    ):
        """Thresholds left as None come from app.config.settings (corr_*)."""

        def pick(value, default):
            return default if value is None else value

        self._embed_fn = embed_fn if embed_fn is not None else _LazySentenceTransformer()
        self._clock = clock
        self.dup_threshold = pick(dup_threshold, settings.corr_dup_threshold)
        self.cluster_threshold = pick(cluster_threshold, settings.corr_cluster_threshold)
        self.cluster_min_count = pick(cluster_min_count, settings.corr_cluster_min_count)
        self.cluster_window = timedelta(
            hours=pick(cluster_window_hours, settings.corr_cluster_window_hours)
        )
        self.lookback = timedelta(hours=pick(lookback_hours, settings.corr_lookback_hours))
        self.category_agreement_bonus = pick(
            category_agreement_bonus, settings.corr_category_agreement_bonus
        )
        self.cross_category_dup_threshold = pick(
            cross_category_dup_threshold, settings.corr_cross_category_dup_threshold
        )
        self.dup_min_fields = pick(dup_min_fields, settings.corr_dup_min_fields)

    async def check(self, state: ConversationState, pool: OpenTicketPool) -> CorrelationResult:
        try:
            return await self._check(state, pool)
        except Exception:
            logger.exception("Agent 4 correlation failed; treating as NO_MATCH")
            return CorrelationResult(outcome="NO_MATCH")

    async def _check(self, state: ConversationState, pool: OpenTicketPool) -> CorrelationResult:
        category = state.detected_category.value if state.detected_category else None
        query = build_kb_query(category, state.extracted_fields)
        if not query:
            return CorrelationResult(outcome="NO_MATCH")

        now = self._clock()
        building = state_building(state)
        # No category filter: category only adjusts the score (see module docstring).
        pool_tickets = await pool.list_open(since=now - self.lookback)
        candidates = [
            t for t in pool_tickets
            if t.ticket_id
            and _as_utc(t.created_at) <= now  # ignore anything "created" after the request
            and _building_compatible(building, t.building)
        ]
        if not candidates:
            return CorrelationResult(outcome="NO_MATCH")

        vectors = await asyncio.to_thread(
            self._embed_fn, [query] + [candidate_text(t) for t in candidates]
        )
        scored = []
        for ticket, vector in zip(candidates, vectors[1:]):
            raw = max(0.0, min(1.0, _cosine(vectors[0], vector)))
            agrees = category is not None and ticket.category == category
            score = min(1.0, raw + self.category_agreement_bonus) if agrees else raw
            scored.append((score, raw, agrees, ticket))
        scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
        max_similarity = round(max(s[1] for s in scored), 4)

        def evidence(rows) -> list[SimilarTicket]:
            return [
                SimilarTicket(ticket_id=t.ticket_id, similarity=round(raw, 4), created_at=t.created_at)
                for _, raw, _, t in rows[:MAX_EVIDENCE]
            ]

        # 1. DUPLICATE (skipped for a thin query - see the minimum-field gate)
        dup_candidates = scored if query_field_count(state) >= self.dup_min_fields else []
        for score, raw, agrees, ticket in dup_candidates:
            if score < self.dup_threshold:
                break
            field_lines = correlation_field_lines(ticket.correlation_text)
            if field_lines is not None and field_lines < self.dup_min_fields:
                continue
            categories_differ = category is not None and not agrees
            if categories_differ and raw < self.cross_category_dup_threshold:
                continue
            return CorrelationResult(
                outcome="DUPLICATE",
                duplicate_of=ticket.ticket_id,
                similar_tickets=evidence(scored),
                max_similarity=max_similarity,
            )

        # 2. INCIDENT_CLUSTER
        recent = [
            s for s in scored
            if s[0] >= self.cluster_threshold and _as_utc(s[3].created_at) >= now - self.cluster_window
        ]
        groups: dict[str, list] = {}
        for row in recent:
            member_building = normalize_building(row[3].building)
            if member_building is not None and (building is None or member_building == building):
                groups.setdefault(member_building, []).append(row)
        if groups:
            cluster_building, members = max(groups.items(), key=lambda kv: (len(kv[1]), kv[1][0][0]))
            if len(members) + 1 >= self.cluster_min_count:
                return CorrelationResult(
                    outcome="INCIDENT_CLUSTER",
                    cluster_id=_cluster_id(cluster_building, [m[3] for m in members]),
                    similar_tickets=evidence(members),
                    max_similarity=max_similarity,
                )

        # 3. NO_MATCH
        return CorrelationResult(
            outcome="NO_MATCH",
            similar_tickets=evidence(scored),
            max_similarity=max_similarity,
        )
