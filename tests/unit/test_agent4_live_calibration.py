"""scripts/run_agent4_live_calibration.py: tiny in-memory data, fake embed_fn,
no model, no files from data/."""

import pytest

from app.agents.correlation_engine import candidate_text
from scripts.run_agent4_eval import REF_NOW, expected_outcomes
from scripts.run_agent4_live_calibration import (
    build_pool_records,
    cached_state,
    query_text,
    recommend,
    run_sweep,
    suggestion_rule,
)


def ticket(tid, label, offset, *, subject="Wifi drops", message="Laptop wifi drops.",
           category="Wifi/Internet Support", building="TT", duplicate_of=None, hard_negative_of=None):
    return {
        "ticket_id": tid, "category_selected": category, "true_category": category,
        "building": building, "room": None, "priority": "LOW", "subject": subject,
        "message": message, "status": "OPEN", "created_at_offset_minutes": offset,
        "label_type": label, "gold_cluster_id": None, "duplicate_of": duplicate_of,
        "hard_negative_of": hard_negative_of,
    }


def entry(category, **fields):
    return {
        "category": category, "category_confidence": 0.9 if category else 0.0,
        "fields": {k: {"value": v, "confidence": 0.9 if v is not None else 0.0} for k, v in fields.items()},
        "model": "fake-model",
    }


TICKETS = [
    ticket("O-1", "background", -100),
    ticket("D-1", "duplicate", -80, duplicate_of="O-1"),
    ticket("N-1", "hard_negative_near_miss", -60, hard_negative_of="O-1"),
    ticket("B-1", "background", -40, subject="Slow wifi", message="It is slow."),
]
CACHE = {
    # Two fields each, so the queries pass Agent 4's minimum-field gate.
    "O-1": entry("wifi_internet", symptom_type="drops", device_type="laptop"),
    "D-1": entry("wifi_internet", symptom_type="drops every few minutes", device_type="laptop"),
    "N-1": entry("wifi_internet", symptom_type="drops on my phone", device_type="phone"),
    "B-1": entry("wifi_internet"),  # no fields extracted
}


class FakeEmbed:
    """'drops' texts share one direction, 'slow' another, anything else a third."""

    def __init__(self):
        self.calls = 0

    def __call__(self, texts):
        self.calls += 1
        return [
            [1.0, 0.0, 0.0] if "drops" in t else [0.0, 1.0, 0.0] if "slow" in t else [0.0, 0.0, 1.0]
            for t in texts
        ]


def test_query_and_pool_texts_use_cached_category_and_fields():
    raw = ticket("X-1", "background", -10, category="Printer Support")  # dataset label differs
    cache = {"X-1": entry("wifi_internet", symptom_type="drops", ssid=None)}

    state = cached_state("X-1", cache["X-1"])
    [record] = build_pool_records([raw], cache, REF_NOW)

    expected = "symptom_type: drops\nCategory: Wifi/Internet Support"
    assert query_text(state) == expected
    assert record.correlation_text == expected
    assert record.category == "printer_support"  # stored category still from seed_to_record


def test_ticket_with_no_fields_gets_no_correlation_text():
    records = {r.ticket_id: r for r in build_pool_records(TICKETS, CACHE, REF_NOW)}

    assert records["B-1"].correlation_text is None
    assert candidate_text(records["B-1"]) == "Slow wifi\nIt is slow."


async def test_near_miss_expected_no_match_and_duplicate_call_is_hard_negative_fp():
    assert expected_outcomes(TICKETS)["N-1"] == "NO_MATCH"

    out = await run_sweep(TICKETS, CACHE, FakeEmbed(), thresholds=[0.9])
    [row] = out.rows

    assert out.predictions[0.9]["N-1"].outcome == "DUPLICATE"
    assert row["hn_near_miss_dup_fp"] == 1 and row["hn_dup_fp_total"] == 1
    assert row["dup_tp"] == 1 and row["dup_fn"] == 0
    assert out.diagnostics["D-1"]["original_is_top"]


async def test_embeddings_computed_once_per_run():
    embed = FakeEmbed()

    out = await run_sweep(TICKETS, CACHE, embed, thresholds=[0.5, 0.9, 0.99])

    assert embed.calls == 1
    assert len(out.rows) == 3


def row(t, recall, hn_fp=0, bg_rate=0.0):
    return {"dup_threshold": t, "dup_recall": recall, "hn_dup_fp_total": hn_fp,
            "background_dup_fp_rate": bg_rate}


def test_recommend_applies_rules_in_order():
    rows = [
        row(0.80, 1.00, hn_fp=2),          # fails rule 1
        row(0.85, 0.95, bg_rate=0.05),     # fails rule 2
        row(0.90, 0.90, bg_rate=0.01),     # passes, best recall
        row(0.95, 0.70),
    ]

    chosen, working = recommend(rows)

    assert chosen["dup_threshold"] == 0.90
    assert len(working) == 3


def test_recommend_prefers_higher_threshold_on_tie():
    chosen, _ = recommend([row(0.90, 0.8), row(0.92, 0.8), row(0.95, 0.6)])

    assert chosen["dup_threshold"] == 0.92


@pytest.mark.parametrize("rows", [
    [row(0.90, 1.0, hn_fp=1), row(0.95, 0.9, hn_fp=1)],
    [row(0.90, 1.0, bg_rate=0.02), row(0.95, 0.9, bg_rate=0.011)],
])
def test_recommend_returns_none_when_rules_1_2_fail(rows):
    chosen, working = recommend(rows)

    assert chosen is None
    assert "no threshold is recommended" in working[-1]


def srow(t, recall, fp=0):
    return {"dup_threshold": t, "dup_recall": recall, "dup_tp": round(recall * 10),
            "dup_fn": 10 - round(recall * 10), "dup_fp": fp}


def test_suggestion_rule_picks_highest_threshold_with_recall_at_least_090():
    chosen, working = suggestion_rule(
        [srow(0.80, 1.0, fp=30), srow(0.85, 0.9, fp=12), srow(0.90, 0.8, fp=5)], n_non_duplicates=200,
    )
    assert chosen["dup_threshold"] == 0.85
    assert "6.0 per 100 non-duplicate tickets" in working[-1]


def test_suggestion_rule_none_when_recall_never_reaches_090():
    chosen, working = suggestion_rule([srow(0.90, 0.8), srow(0.95, 0.5)], n_non_duplicates=100)
    assert chosen is None and "No threshold" in working[0]
