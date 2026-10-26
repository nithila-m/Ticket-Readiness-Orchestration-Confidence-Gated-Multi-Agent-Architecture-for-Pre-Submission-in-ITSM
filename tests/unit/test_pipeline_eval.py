"""scripts/run_pipeline_eval.py: fakes only - no Groq, no Gemini, no Chroma,
no model download. Only the pure parts and the replay provider are used."""

import json

import pytest

from app.providers.exceptions import LLMProviderError
from app.schemas.conversation import ConversationState
from scripts import run_pipeline_eval as pe

CACHE = json.loads(pe.CACHE_PATH.read_text(encoding="utf-8"))
TICKETS = pe.load_tickets()
TICKETS_BY_ID = {t["ticket_id"]: t for t in TICKETS}


# --- Replay provider + message builder ---------------------------------------------


async def test_replay_exact_hit_returns_cached_fields():
    entry = CACHE["WIFI-021"]
    result = await pe.ReplayExtractionProvider(CACHE).extract_information(entry["input_text"])
    assert result.category.value == entry["category"]
    assert result.category.confidence == entry["category_confidence"]
    assert {k: (f.value, f.confidence) for k, f in result.extracted_fields.items()} == {
        k: (f["value"], f["confidence"]) for k, f in entry["fields"].items()
    }


async def test_replay_miss_raises_and_never_falls_back():
    provider = pe.ReplayExtractionProvider(CACHE)
    with pytest.raises(pe.ReplayCacheMiss, match="cache miss"):
        await provider.extract_information(CACHE["WIFI-021"]["input_text"] + " extra")


@pytest.mark.parametrize("ticket_id", ["WIFI-021", "TEAMS-025", "PRN-001"])
def test_message_builder_reproduces_cache_input_text(ticket_id):
    by_text = pe.cache_by_input_text(CACHE)
    message = pe.build_user_message(TICKETS_BY_ID[ticket_id], by_text)
    assert pe.cache_key_for(message) == CACHE[ticket_id]["input_text"]


def test_message_builder_adds_ocr_only_when_the_cache_has_it():
    ticket = {"subject": "S", "message": "M", "ocr_text": "ERR 42"}
    with_ocr = pe.cache_key_for("S. M [Screenshot text: ERR 42]")
    assert pe.build_user_message(ticket, set()) == "S. M"
    assert pe.build_user_message(ticket, {with_ocr}) == "S. M [Screenshot text: ERR 42]"


def test_every_original_ticket_has_a_cache_hit():
    by_text = pe.cache_by_input_text(CACHE)
    assert len(TICKETS) == 150
    assert all(pe.cache_key_for(pe.build_user_message(t, by_text)) in by_text for t in TICKETS)


# --- Retry wrapper -------------------------------------------------------------------


async def test_retry_waits_backoff_and_counts():
    calls, waits = [], []

    async def fn():
        calls.append(1)
        if len(calls) < 3:
            raise LLMProviderError("Groq API call failed: Error code: 429 rate limit")
        return "ok"

    async def sleep(s):
        waits.append(s)

    counter = pe.RetryCounter()
    assert await pe.call_with_retry(fn, counter, "x", sleep=sleep, log=lambda _: None) == "ok"
    assert waits == [15.0, 30.0] and counter.retries == 2


async def test_retry_does_not_retry_other_errors():
    async def fn():
        raise LLMProviderError("schema mismatch")

    with pytest.raises(LLMProviderError):
        await pe.call_with_retry(fn, pe.RetryCounter(), "x", sleep=None, log=lambda _: None)


# --- Scripted user -------------------------------------------------------------------


def state(**kw):
    return ConversationState(conversation_id="c", **kw)


def test_deflected_and_deflectable_sends_resolved():
    assert pe.scripted_step(state(), "DEFLECTED", {"is_deflectable": True}) == ("deflection_feedback", True)


def test_deflected_not_deflectable_sends_not_resolved():
    assert pe.scripted_step(state(), "DEFLECTED", {"is_deflectable": False}) == ("deflection_feedback", False)


def test_deflection_feedback_is_sent_once():
    assert pe.scripted_step(state(kb_feedback="NOT_RESOLVED"), "DEFLECTED",
                            {"is_deflectable": False}) == ("stop", None)


def test_duplicate_question_is_answered_not_a_duplicate():
    s = state(awaiting_duplicate_confirmation=True)
    assert pe.scripted_step(s, "READY", {"is_deflectable": False}) == ("duplicate_feedback", False)


def test_ask_stops():
    assert pe.scripted_step(state(), "ASK_CLARIFICATION", {"is_deflectable": False}) == ("stop", None)


def test_terminal_outcome_stops():
    s = state(terminal_outcome="SELF_RESOLVED", kb_feedback="RESOLVED")
    assert pe.scripted_step(s, "DEFLECTED", {"is_deflectable": True}) == ("stop", None)


# --- Slices + files --------------------------------------------------------------------


def test_slice_selection_boundaries():
    items = [{"ticket_id": f"T-{i:03d}"} for i in range(1, 151)]
    sel = pe.select_slice(items, 1, 60)
    assert len(sel) == 60 and sel[0] == (1, items[0]) and sel[-1] == (60, items[59])
    sel = pe.select_slice(items, 61, 120)
    assert sel[0][0] == 61 and sel[-1][0] == 120 and len(sel) == 60
    assert pe.resolve_slice(150, None, None) == (1, 150)
    assert pe.resolve_slice(150, 150, None) == (150, 150)
    for bad in [(0, 10), (10, 5), (1, 151)]:
        with pytest.raises(ValueError):
            pe.resolve_slice(150, *bad)


def test_per_slice_file_names(tmp_path):
    assert pe.slice_csv_path(1, 60, tmp_path) == tmp_path / "pipeline_eval_results_1-60.csv"
    assert pe.slice_md_path(61, 120, tmp_path) == tmp_path / "pipeline_eval_61-120.md"
    assert pe.slice_db_path(1, 60, tmp_path) == tmp_path / "tro_pipeline_eval_1-60.db"
    assert pe.slice_label(61, 120) == "61-120"


# --- Metric fixtures -------------------------------------------------------------------


def row(tid, slice_="1-60", pos=1, **kw):
    base = {k: None for k in pe.CSV_FIELDS}
    base.update({
        "ticket_id": tid, "slice": slice_, "position": pos, "gt_category": "wifi_internet",
        "gt_team": "Network Operations", "gt_priority": "P3", "is_deflectable": False,
        "completeness_level": "complete", "gate_passed": False, "deflected": False,
        "reached_agent4": False, "reached_agent5": False, "duplicate_suggested": False,
        "duplicate_same_cluster": None, "rule_fallback": None, "groq_retries": 0,
        "final_outcome": "STOPPED_AT_ASK", "first_action": "ASK_CLARIFICATION",
    })
    base.update(kw)
    return base


def done(tid, pos, priority, gt="P3", slice_="1-60", **kw):
    return row(tid, slice_, pos, gate_passed=True, kb_outcome="NO_MATCH", first_action="READY",
               reached_agent4=True, reached_agent5=True, correlation_outcome="NO_MATCH",
               pred_category="wifi_internet", pred_team="Network Operations", pred_priority=priority,
               gt_priority=gt, classify_source="llm", rule_fallback=False, gate_decision="DISPATCH",
               final_outcome="DISPATCHED", **kw)


ROWS = [
    row("A", pos=1, gate_passed=True, kb_outcome="STRONG_MATCH", first_action="DEFLECTED",
        deflected=True, is_deflectable=True, deflection_feedback="RESOLVED", final_outcome="SELF_RESOLVED"),
    row("B", pos=2, gate_passed=True, kb_outcome="STRONG_MATCH", first_action="DEFLECTED",
        deflected=True, is_deflectable=False, deflection_feedback="NOT_RESOLVED",
        reached_agent4=True, reached_agent5=True, pred_category="wifi_internet",
        pred_team="Network Operations", pred_priority="P3", classify_source="rule",
        rule_fallback=True, gate_decision="REVIEW", final_outcome="PENDING_REVIEW"),
    row("C", pos=3, is_deflectable=True, completeness_level="vague"),   # FN for KB
    done("D", 4, "P2", gt="P2", lat_kb_ms=10.0, lat_classify_ms=100.0),
    done("E", 5, "P4", gt="P3", lat_kb_ms=30.0, lat_classify_ms=300.0,
         groq_retries=2, lat_clarify_ms=99999.0),
    row("F", pos=6, error="ReplayCacheMiss: boom", final_outcome="ERROR", first_action=None),
]


def test_funnel_counts():
    f = pe.funnel(ROWS)
    assert (f["n"], f["n_ok"], f["n_error"]) == (6, 5, 1)
    assert f["gate_passed"] == 4
    assert f["kb"] == {"STRONG_MATCH": 2, "WEAK_MATCH": 0, "NO_MATCH": 2}
    assert f["n_agent2"] == 3
    assert f["agent2_actions"]["ASK_CLARIFICATION"] == 1 and f["agent2_actions"]["READY"] == 2
    assert (f["deflected"], f["deflected_resolved"], f["deflected_not_resolved"]) == (2, 1, 1)
    assert (f["reached_agent4"], f["reached_agent5"]) == (3, 3)
    assert f["outcomes"]["DISPATCHED"] == 2 and f["outcomes"]["ERROR"] == 1


def test_kb_precision_recall_f1():
    kb = pe.kb_prf(ROWS)
    assert (kb["tp"], kb["fp"], kb["fn"], kb["tn"]) == (1, 1, 1, 2)
    assert kb["precision"] == pytest.approx(0.5)
    assert kb["recall"] == pytest.approx(0.5)
    assert kb["f1"] == pytest.approx(0.5)


def test_macro_f1_and_confusion_with_p1_column():
    rows = [done("1", 1, "P2", gt="P2"), done("2", 2, "P3", gt="P3"),
            done("3", 3, "P1", gt="P4"), done("4", 4, "P4", gt="P4")]
    c = pe.classification_metrics(rows)
    assert c["matrix"]["P4"]["P1"] == 1 and c["n_p1"] == 1
    assert c["priority_accuracy"] == pytest.approx(0.75)
    # P2: f1 1, P3: f1 1, P4: precision 1 recall 0.5 -> 2/3
    assert c["macro_f1"] == pytest.approx((1 + 1 + 2 / 3) / 3)


def test_classification_rates():
    c = pe.classification_metrics(ROWS)
    assert c["n"] == 3
    assert c["rule_fallback_rate"] == pytest.approx(1 / 3)
    assert c["review_rate"] == pytest.approx(1 / 3)
    assert c["routing_accuracy"] == 1.0


def test_latency_excludes_retried_tickets():
    stats = pe.latency_stats(ROWS)
    assert stats["excluded_retried"] == 1
    assert stats["stages"]["kb"]["n"] == 1 and stats["stages"]["kb"]["max"] == 10.0
    assert stats["stages"]["clarify"]["n"] == 0
    assert stats["stages"]["classify"]["median"] == 100.0


def test_crosstab_counts_first_action_by_level():
    cols, table = pe.agent2_crosstab(ROWS)
    assert table["complete"]["DEFLECTED"] == 2 and table["complete"]["READY"] == 2
    assert table["vague"]["ASK_CLARIFICATION"] == 1


# --- CSV, resume, merge -----------------------------------------------------------------


def test_resume_skips_already_recorded_ids(tmp_path):
    path = tmp_path / "s.csv"
    pe.append_row(path, row("T-1", pos=1))
    pe.append_row(path, row("T-2", pos=2, error="x", final_outcome="ERROR"))
    selected = [(1, {"ticket_id": "T-1"}), (2, {"ticket_id": "T-2"}), (3, {"ticket_id": "T-3"})]
    assert pe.pending_tickets(selected, pe.load_done_ids(path)) == [(3, {"ticket_id": "T-3"})]


def test_csv_round_trip_restores_types(tmp_path):
    path = tmp_path / "s.csv"
    original = done("D", 4, "P2", gt="P2", lat_kb_ms=12.5)
    pe.append_row(path, original)
    back = pe.read_rows(path)[0]
    assert back["reached_agent5"] is True and back["rule_fallback"] is False
    assert back["lat_kb_ms"] == 12.5 and back["position"] == 4 and back["error"] is None


def _write(path, rows):
    for r in rows:
        pe.append_row(path, r)
    return path


def test_merge_two_csvs(tmp_path):
    a = _write(tmp_path / "a.csv", [done("D", 4, "P2", gt="P2"), row("A", pos=1)])
    b = _write(tmp_path / "b.csv", [done("X", 61, "P3", slice_="61-120")])
    merged = pe.merge_csvs([a, b])
    assert [r["ticket_id"] for r in merged] == ["A", "D", "X"]
    assert {r["slice"] for r in merged} == {"1-60", "61-120"}


def test_merge_fails_on_duplicate_ticket_id(tmp_path):
    a = _write(tmp_path / "a.csv", [row("A", pos=1)])
    b = _write(tmp_path / "b.csv", [row("A", slice_="61-120", pos=61)])
    with pytest.raises(pe.DuplicateTicketError, match="A"):
        pe.merge_csvs([a, b])
    assert pe.report_only([str(a), str(b)], out=tmp_path / "m.md") == 1
    assert not (tmp_path / "m.md").exists()


def test_merged_report_lists_slices_and_within_slice_note(tmp_path):
    a = _write(tmp_path / "a.csv", ROWS)
    b = _write(tmp_path / "b.csv", [done("X", 61, "P3", slice_="61-120", lat_kb_ms=5.0),
                                    done("Y", 62, "P3", slice_="61-120", lat_kb_ms=7.0)])
    out = tmp_path / "pipeline_eval.md"
    assert pe.report_only([str(a), str(b)], out=out) == 0
    md = out.read_text(encoding="utf-8")
    assert "| 1-60 | 1-6 | A | F | 6 |" in md
    assert "| 61-120 | 61-62 | X | Y | 2 |" in md
    assert "Duplicates were only compared within a slice" in md
    assert "### Slice 1-60" in md and "### Slice 61-120" in md
    assert "mixed-machine" in md


def test_slice_report_has_no_merge_note():
    md = pe.render_markdown(ROWS, merged=False)
    assert "## Limitations" in md and "Duplicates were only compared within a slice" not in md
    assert md.index("## Limitations") < md.index("## 1. Funnel")
