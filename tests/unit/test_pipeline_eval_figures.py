"""Figure scripts: hand-built rows in a temp dir; no API calls, no model download."""

import csv
import json

import pytest

from scripts import plot_agent4_calibration as a4
from scripts import plot_pipeline_eval as pp
from scripts import run_pipeline_eval as pe


def row(tid, pos, slice_="1-3", **kw):
    r = {k: None for k in pe.CSV_FIELDS}
    r.update({"ticket_id": tid, "slice": slice_, "position": pos, "gt_category": "printer_support",
              "gt_team": "Desktop Support", "gt_priority": "P3", "is_deflectable": False,
              "completeness_level": "complete", "gate_passed": False, "deflected": False,
              "reached_agent4": False, "reached_agent5": False, "duplicate_suggested": False,
              "groq_retries": 0, "first_action": "ASK_CLARIFICATION", "final_outcome": "STOPPED_AT_ASK",
              "lat_clarify_ms": 1500.0})
    r.update(kw)
    return r


def ready(tid, pos, pred, gt, slice_="1-3", **kw):
    return row(tid, pos, slice_, gate_passed=True, kb_outcome="NO_MATCH", kb_similarity=0.4,
               first_action="READY", reached_agent4=True, reached_agent5=True,
               correlation_outcome="NO_MATCH", pred_category="printer_support",
               pred_team="Desktop Support", pred_priority=pred, gt_priority=gt,
               classify_source="llm", rule_fallback=False, gate_decision="DISPATCH",
               final_outcome="DISPATCHED", lat_kb_ms=40.0, lat_correlate_ms=60.0,
               lat_classify_ms=1200.0, lat_dispatch_ms=15.0, **kw)


ROWS = [
    row("PRN-1", 1, gate_passed=True, kb_outcome="STRONG_MATCH", first_action="DEFLECTED",
        deflected=True, is_deflectable=True, deflection_feedback="RESOLVED",
        final_outcome="SELF_RESOLVED", lat_clarify_ms=None, lat_kb_ms=30.0),
    row("PRN-2", 2, is_deflectable=True, completeness_level="vague"),
    ready("PRN-3", 3, "P2", "P2"),
    ready("WIFI-1", 4, "P3", "P4", slice_="4-6", gt_category="wifi_internet",
          gt_team="Network Operations", groq_retries=1),
    ready("WIFI-2", 5, "P3", "P3", slice_="4-6", gt_category="wifi_internet",
          gt_team="Network Operations"),
    row("WIFI-3", 6, slice_="4-6", gt_category="wifi_internet", first_action="RECHECK",
        final_outcome="STOPPED_AT_RECHECK", completeness_level="partial"),
]


@pytest.fixture
def merged(tmp_path):
    path = tmp_path / pe.MERGED_CSV_NAME
    pe.write_rows_csv(path, ROWS)
    return path


def test_every_pipeline_png_is_written(tmp_path, merged):
    out, readme = tmp_path / "pipeline", tmp_path / "README.md"
    assert pp.main(["--input", str(merged), "--out-dir", str(out), "--readme", str(readme)]) == 0
    for name in pp.FIG_NAMES:
        assert (out / name).stat().st_size > 10_000, name
    text = readme.read_text(encoding="utf-8")
    assert all(f"### {name}" in text for name in pp.FIG_NAMES)

    # rerunning replaces the section instead of appending a second copy
    pp.main(["--input", str(merged), "--out-dir", str(out), "--readme", str(readme)])
    assert readme.read_text(encoding="utf-8").count("<!-- pipeline:start -->") == 1


def test_figure_numbers_match_metric_functions(tmp_path, merged):
    rows = pe.read_rows(merged)          # same path the script takes: CSV round trip
    out = tmp_path / "figs"

    assert pp.fig_outcomes(rows, out)["counts"] == pe.funnel(rows)["outcomes"]

    kb = pp.fig_kb_confusion(rows, out)
    expected = pe.kb_prf(rows)
    assert {k: kb[k] for k in expected} == expected
    assert (kb["tp"], kb["fp"], kb["fn"], kb["tn"]) == (1, 0, 1, 4)

    c = pp.fig_classification(rows, out)
    m = pe.classification_metrics(rows)
    for key in ("n", "category_accuracy", "routing_accuracy", "priority_accuracy", "macro_f1"):
        assert c[key] == m[key]
    assert c["n"] == 3 and c["priority_accuracy"] == pytest.approx(2 / 3)
    assert c["always_p3_baseline"] == pytest.approx(1 / 3)


def test_stage_bars_are_not_forced_to_shrink():
    stages = dict(pp.stage_counts(ROWS))
    assert stages["All tickets"] == 6
    assert stages["Passed completeness gate"] == 4
    assert stages["Agent 2 ran (first turn)"] == 5      # more than passed the gate
    assert stages["Dispatched or merged"] == 3


def test_latency_figure_excludes_retried_tickets(tmp_path):
    lat = pp.fig_latency(ROWS, tmp_path)
    assert lat["excluded_retried"] == 1 and lat["n_slices"] == 2
    assert lat["stages"]["classify"]["n"] == 2


# --- Agent 4 --------------------------------------------------------------------------


def _sweep_csv(path):
    fields = ["dup_threshold", "dup_tp", "dup_fp", "dup_fn", "dup_precision", "dup_recall", "dup_f1",
              "background_n", "background_dup_fp_rate"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, t in enumerate([0.85, 0.86, 0.87, 0.95]):
            w.writerow({"dup_threshold": t, "dup_tp": 9 - i, "dup_fp": 5 - i, "dup_fn": 1 + i,
                        "dup_precision": 0.5 + 0.1 * i, "dup_recall": 0.9 - 0.1 * i, "dup_f1": 0.6,
                        "background_n": 20, "background_dup_fp_rate": 0.1 - 0.02 * i})


def test_agent4_pngs_from_saved_scores_without_model(tmp_path, monkeypatch):
    def no_model():
        raise AssertionError("must not recompute when the score lists exist")

    monkeypatch.setattr(a4, "recompute_score_lists", no_model)
    sweep, scores = tmp_path / "sweep.csv", tmp_path / "scores.json"
    _sweep_csv(sweep)
    scores.write_text(json.dumps({"duplicates_vs_true_original": [0.86, 0.9, 0.97, 1.0],
                                  "non_duplicates_vs_top_candidate": [0.42, 0.7, 0.88],
                                  "duplicates_gated": ["D-9"], "non_duplicates_gated": 2}))
    out, readme = tmp_path / "agent4", tmp_path / "README.md"
    assert a4.main(["--sweep", str(sweep), "--scores", str(scores), "--out-dir", str(out),
                    "--readme", str(readme)]) == 0
    for name in ("threshold_sweep.png", "score_distributions.png"):
        assert (out / name).stat().st_size > 10_000
    text = readme.read_text(encoding="utf-8")
    assert "at 0.87: precision 0.700" in text and "3 of 4 duplicates, 1 of 3 non-duplicates" in text


def test_agent4_bins_match_calibration_histogram_rules(tmp_path):
    dups, non = [0.86, 0.9, 1.0], [0.42, 0.88]
    lo = a4.histogram_start(dups, non)
    assert lo == pytest.approx(0.40)
    bins = dict(a4.bin_counts(dups, lo))
    assert bins[0.85] == 1 and bins[0.9] == 1 and bins[0.95] == 1   # 1.0 lands in the last bin
    md = tmp_path / "cal.md"
    md.write_text("| [0.85, 0.90) | 1 | 1 |\n| [0.95, 1.00] | 1 | 0 |\n")
    assert a4.parse_md_histogram(md) == [(0.85, 1, 1), (0.95, 1, 0)]
