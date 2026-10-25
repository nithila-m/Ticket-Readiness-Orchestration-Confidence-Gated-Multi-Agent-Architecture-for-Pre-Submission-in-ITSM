"""
Agent 4 unit tests. Fake embedder + fake pool only: no model, no files.

The fake embedder maps a marker like "[s=0.75]" found in a text to a unit
vector whose cosine similarity with the query vector ("[q]") is exactly
that value, so each test controls similarities precisely.
"""

import math
import re
from datetime import datetime, timedelta, timezone


from app.agents.classification_agent import apply_cluster_override
from app.agents.correlation_engine import CorrelationEngine, candidate_text, correlation_field_lines
from app.agents.ticket_builder import build_correlation_text, build_description
from app.config.settings import settings
from app.agents.interfaces import CorrelationAgent, OpenTicketPool
from app.agents.seed_ticket_pool import JsonOpenTicketPool
from app.config.routing import team_for_category
from app.schemas.conversation import ConversationState
from app.schemas.extraction import CategoryPrediction, ExtractedField
from app.schemas.pipeline import TicketRecord

NOW = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)
_MARKER = re.compile(r"\[s=([0-9.]+)\]")


def fake_embed(texts: list[str]) -> list[list[float]]:
    vectors = []
    for text in texts:
        if "[q]" in text:
            vectors.append([1.0, 0.0])
        elif (m := _MARKER.search(text)) is not None:
            c = float(m.group(1))
            vectors.append([c, math.sqrt(max(0.0, 1 - c * c))])
        else:
            vectors.append([0.0, 1.0])  # unrelated: similarity 0
    return vectors


class FakePool:
    def __init__(self, tickets: list[TicketRecord]):
        self._tickets = tickets
        self.calls: list[tuple[datetime, str | None]] = []

    async def list_open(self, since: datetime, category: str | None = None) -> list[TicketRecord]:
        self.calls.append((since, category))
        rows = [t for t in self._tickets if t.created_at >= since and (category is None or t.category == category)]
        return sorted(rows, key=lambda t: t.created_at, reverse=True)


def ticket(tid, sim, *, category="wifi_internet", building="SJT", hours_ago=1.0, cluster_id=None):
    return TicketRecord(
        ticket_id=tid,
        conversation_id=f"c-{tid}",
        title=f"ticket {tid} [s={sim}]",
        description="details",
        category=category,
        priority="P3",
        assignee_group=team_for_category(category),
        routing_rationale="test",
        confidence_score=0.9,
        building=building,
        incident_cluster_id=cluster_id,
        created_at=NOW - timedelta(hours=hours_ago),
    )


def state(category="wifi_internet", building="SJT", symptom="wifi down [q]"):
    fields = {}
    if symptom is not None:
        fields["symptom_type"] = ExtractedField(value=symptom, confidence=0.9)
    if building is not None:
        fields["building"] = ExtractedField(value=building, confidence=0.9)
    return ConversationState(
        conversation_id="conv-1",
        extracted_fields=fields,
        detected_category=CategoryPrediction(value=category, confidence=0.9) if category else None,
    )


def engine(embed_fn=fake_embed, **overrides) -> CorrelationEngine:
    params = dict(
        dup_threshold=0.85, cluster_threshold=0.70, cluster_min_count=3,
        cluster_window_hours=6, lookback_hours=72,
        category_agreement_bonus=0.05, cross_category_dup_threshold=0.95,
    )
    params.update(overrides)
    return CorrelationEngine(embed_fn=embed_fn, clock=lambda: NOW, **params)


# --- contract ------------------------------------------------------------------

def test_engine_satisfies_correlation_agent_protocol():
    assert isinstance(engine(), CorrelationAgent)


def test_seed_pool_satisfies_open_ticket_pool_protocol():
    assert isinstance(JsonOpenTicketPool(), OpenTicketPool)


async def test_pool_queried_with_lookback_and_no_category_filter():
    pool = FakePool([])
    await engine().check(state(), pool)
    assert pool.calls == [(NOW - timedelta(hours=72), None)]


# --- DUPLICATE --------------------------------------------------------------------

async def test_exact_duplicate_is_duplicate():
    result = await engine().check(state(), FakePool([ticket("T1", 1.0), ticket("T2", 0.3)]))
    assert result.outcome == "DUPLICATE"
    assert result.duplicate_of == "T1"
    assert result.max_similarity == 1.0
    assert result.similar_tickets[0].ticket_id == "T1"
    assert result.similar_tickets[0].created_at == NOW - timedelta(hours=1)


async def test_same_text_in_different_building_is_no_match():
    result = await engine().check(state(building="SJT"), FakePool([ticket("T1", 1.0, building="TT")]))
    assert result.outcome == "NO_MATCH"
    assert result.duplicate_of is None
    assert result.similar_tickets == []


async def test_building_spelling_variants_count_as_same_building():
    pool = FakePool([ticket("T1", 1.0, building="Men's Hostel A-Block")])
    result = await engine().check(state(building="Men's Hostel A"), pool)
    assert result.outcome == "DUPLICATE"


async def test_duplicate_allowed_when_incoming_building_unknown():
    no_building = state(building=None)
    # A second field keeps the query above the minimum-field gate; this test is about buildings.
    no_building.extracted_fields["device_type"] = ExtractedField(value="laptop", confidence=0.9)
    result = await engine().check(no_building, FakePool([ticket("T1", 1.0, building="TT")]))
    assert result.outcome == "DUPLICATE"


# --- category: signal, not filter ----------------------------------------------------

async def test_category_agreement_bonus_can_lift_score_over_dup_threshold():
    result = await engine().check(state(), FakePool([ticket("T1", 0.82)]))
    assert result.outcome == "DUPLICATE"  # 0.82 + 0.05 bonus >= 0.85
    assert result.similar_tickets[0].similarity == 0.82  # evidence reports raw similarity


async def test_cross_category_duplicate_needs_very_high_similarity():
    disagree = FakePool([ticket("T1", 0.90, category="vit_email")])
    assert (await engine().check(state(), disagree)).outcome != "DUPLICATE"

    very_high = FakePool([ticket("T1", 0.97, category="vit_email")])
    result = await engine().check(state(), very_high)
    assert result.outcome == "DUPLICATE"
    assert result.duplicate_of == "T1"


async def test_category_disagreement_still_counts_toward_cluster():
    pool = FakePool([ticket("T1", 0.75, category="ms_teams"), ticket("T2", 0.75)])
    result = await engine().check(state(), pool)
    assert result.outcome == "INCIDENT_CLUSTER"
    assert {t.ticket_id for t in result.similar_tickets} == {"T1", "T2"}


# --- INCIDENT_CLUSTER -----------------------------------------------------------------

async def test_similar_tickets_same_building_within_window_form_cluster():
    pool = FakePool([ticket("T1", 0.75, hours_ago=1), ticket("T2", 0.78, hours_ago=3), ticket("X", 0.2)])
    result = await engine().check(state(), pool)
    assert result.outcome == "INCIDENT_CLUSTER"
    assert result.cluster_id and result.cluster_id.startswith("CLU-")
    assert [t.ticket_id for t in result.similar_tickets] == ["T2", "T1"]  # members only, best first


async def test_cluster_needs_min_count_including_incoming_ticket():
    result = await engine().check(state(), FakePool([ticket("T1", 0.75)]))
    assert result.outcome == "NO_MATCH"  # 1 candidate + incoming = 2 < 3


async def test_same_tickets_outside_window_are_not_a_cluster():
    pool = FakePool([ticket("T1", 0.75, hours_ago=7), ticket("T2", 0.75, hours_ago=9)])
    result = await engine().check(state(), pool)
    assert result.outcome == "NO_MATCH"
    assert len(result.similar_tickets) == 2  # still reported as evidence


async def test_tickets_in_other_buildings_do_not_form_cluster():
    pool = FakePool([ticket("T1", 0.75, building="TT"), ticket("T2", 0.75, building="TT")])
    assert (await engine().check(state(building="SJT"), pool)).outcome == "NO_MATCH"


async def test_unknown_incoming_building_clusters_within_one_shared_building():
    pool = FakePool([
        ticket("S1", 0.75, building="SJT"), ticket("S2", 0.76, building="SJT"),
        ticket("T1", 0.72, building="TT"),
    ])
    result = await engine().check(state(building=None), pool)
    assert result.outcome == "INCIDENT_CLUSTER"
    assert {t.ticket_id for t in result.similar_tickets} == {"S1", "S2"}


async def test_cluster_reuses_members_existing_cluster_id():
    pool = FakePool([ticket("T1", 0.75, cluster_id="INC-42"), ticket("T2", 0.75)])
    assert (await engine().check(state(), pool)).cluster_id == "INC-42"


async def test_generated_cluster_id_is_deterministic():
    tickets = [ticket("T1", 0.75, hours_ago=2), ticket("T2", 0.75, hours_ago=1)]
    first = await engine().check(state(), FakePool(tickets))
    grown = await engine().check(state(), FakePool(tickets + [ticket("T3", 0.75, hours_ago=0.5)]))
    assert first.cluster_id == grown.cluster_id  # anchored on the earliest member


# --- time handling --------------------------------------------------------------------

async def test_ticket_older_than_lookback_is_not_considered():
    result = await engine().check(state(), FakePool([ticket("OLD", 1.0, hours_ago=80)]))
    assert result.outcome == "NO_MATCH"
    assert result.similar_tickets == []


async def test_tickets_created_after_now_are_ignored():
    result = await engine().check(state(), FakePool([ticket("FUTURE", 1.0, hours_ago=-1)]))
    assert result.outcome == "NO_MATCH"


# --- robustness ---------------------------------------------------------------------

async def test_empty_pool_is_no_match():
    result = await engine().check(state(), FakePool([]))
    assert result.outcome == "NO_MATCH"
    assert result.max_similarity == 0.0


async def test_embedder_failure_is_no_match():
    def broken(texts):
        raise RuntimeError("model failed to load")

    result = await engine(embed_fn=broken).check(state(), FakePool([ticket("T1", 1.0)]))
    assert result.outcome == "NO_MATCH"


async def test_empty_query_is_no_match_without_embedding():
    calls = []

    def recording(texts):
        calls.append(texts)
        return fake_embed(texts)

    result = await engine(embed_fn=recording).check(
        state(category=None, building=None, symptom=None), FakePool([ticket("T1", 1.0)])
    )
    assert result.outcome == "NO_MATCH"
    assert calls == []


async def test_evidence_capped_at_five():
    pool = FakePool([ticket(f"T{i}", 0.3 + i / 100, building="SJT") for i in range(8)])
    result = await engine().check(state(), pool)
    assert len(result.similar_tickets) == 5
    assert result.max_similarity == 0.37


# --- feeds Agent 5 ------------------------------------------------------------------

async def test_incident_cluster_feeds_agent5_p1_override():
    pool = FakePool([ticket("T1", 0.75), ticket("T2", 0.75)])
    correlation = await engine().check(state(), pool)

    priority, rationale = apply_cluster_override("P3", "Routed to Network Operations.", correlation)

    assert correlation.outcome == "INCIDENT_CLUSTER"
    assert priority == "P1"
    assert correlation.cluster_id in rationale
    assert "2 similar open ticket(s)" in rationale


# --- correlation_text -----------------------------------------------------------

def test_candidate_text_uses_correlation_text_when_set():
    t = ticket("T1", 0.5).model_copy(update={"correlation_text": "symptom_type: drops"})
    assert candidate_text(t) == "symptom_type: drops"


def test_candidate_text_falls_back_to_title_and_description():
    for value in (None, ""):
        t = ticket("T1", 0.5).model_copy(update={"correlation_text": value})
        assert candidate_text(t) == f"{t.title}\ndetails"


def exact_match_embed(texts: list[str]) -> list[list[float]]:
    """Identical strings -> identical vectors; any two different strings -> orthogonal."""
    index = {t: i for i, t in enumerate(dict.fromkeys(texts))}
    return [[1.0 if i == index[t] else 0.0 for i in range(len(index))] for t in texts]


async def test_repeat_request_is_duplicate_of_ticket_with_boilerplate_body():
    s = state(symptom="drops every few minutes")
    stored = ticket("T1", 0.0).model_copy(update={
        "description": build_description(s),  # full ticket body: "Reported by user:", ...
        "correlation_text": build_correlation_text(s),
    })
    eng = engine(embed_fn=exact_match_embed, dup_threshold=settings.corr_dup_threshold)

    result = await eng.check(s, FakePool([stored]))

    assert "Completeness:" in stored.description
    assert result.outcome == "DUPLICATE"
    assert result.duplicate_of == "T1"
    # Without correlation_text the same ticket is compared by its body instead.
    legacy = stored.model_copy(update={"correlation_text": None})
    assert (await eng.check(s, FakePool([legacy]))).outcome == "NO_MATCH"


# --- minimum-field gate (DUPLICATE only) -----------------------------------------

def with_correlation_text(t: TicketRecord, field_lines: int) -> TicketRecord:
    lines = [f"f{i}: v [s={_MARKER.search(t.title).group(1)}]" for i in range(field_lines)]
    return t.model_copy(update={"correlation_text": "\n".join(lines + ["Category: Wifi/Internet Support"])})


async def test_query_with_one_field_is_never_duplicate():
    one_field = state(building=None)  # only symptom_type
    result = await engine(dup_min_fields=2).check(one_field, FakePool([ticket("T1", 1.0)]))
    assert result.outcome == "NO_MATCH"
    assert result.max_similarity == 1.0  # still compared, just not a duplicate


async def test_candidate_with_one_field_line_is_skipped_for_duplicate():
    thin = with_correlation_text(ticket("T1", 1.0), field_lines=1)
    rich = with_correlation_text(ticket("T2", 0.9), field_lines=2)
    result = await engine(dup_min_fields=2).check(state(), FakePool([thin, rich]))
    assert result.outcome == "DUPLICATE"
    assert result.duplicate_of == "T2"


async def test_thin_candidates_still_count_for_clusters():
    thin = [with_correlation_text(ticket(f"T{i}", 0.80), field_lines=1) for i in range(2)]
    result = await engine(dup_min_fields=2, dup_threshold=0.75).check(state(), FakePool(thin))
    assert result.outcome == "INCIDENT_CLUSTER"
    assert {s.ticket_id for s in result.similar_tickets} == {"T0", "T1"}


async def test_candidate_without_correlation_text_stays_eligible():
    seed = ticket("T1", 1.0)
    assert seed.correlation_text is None
    result = await engine(dup_min_fields=2).check(state(), FakePool([seed]))
    assert result.outcome == "DUPLICATE" and result.duplicate_of == "T1"


def test_correlation_field_lines_ignores_category_line():
    assert correlation_field_lines(None) is None
    assert correlation_field_lines("symptom: drops\nCategory: Wifi/Internet Support") == 1
    assert correlation_field_lines("a: 1\nb: 2\nCategory: X") == 2
