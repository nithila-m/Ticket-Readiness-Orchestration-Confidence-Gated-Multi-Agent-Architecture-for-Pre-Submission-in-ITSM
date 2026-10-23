"""
Leave-one-out evaluation of Agent 4 (CorrelationEngine) on
data/eval/tickets_augmented.json, with a threshold sweep.

Run from the repo root (loads all-MiniLM-L6-v2 once):
    python scripts/run_agent4_eval.py

Protocol (temporal leave-one-out):
  - Every non-background ticket is a query; background tickets are also
    queried, only to measure false positives on ordinary traffic.
  - The pool is every other ticket (the query ticket is excluded), and the
    engine's clock is set to the query ticket's created_at, so it sees only
    tickets that already existed when the query arrived.
  - Query (eval option A): Agent 1's extracted fields are not stored for
    this dataset, so the request is simulated as ONE extracted field
    holding subject + message, with the dataset's true_category as the
    category. Building mode "known" puts the ticket's building in the
    extracted fields (as if Agent 1 extracted it); "unknown" leaves it out
    (what Agent 1 does today).

Expected outcome per query:
  duplicate        DUPLICATE of its duplicate_of ticket
  cluster_member   INCIDENT_CLUSTER if at least (corr_cluster_min_count - 1)
                   earlier members of the same gold cluster were created
                   within corr_cluster_window_hours before it; earlier
                   members are "neutral" (too few prior reports to detect)
                   and excluded from cluster scoring
  everything else  NO_MATCH

Metrics per (building mode, dup threshold, cluster threshold):
  DUPLICATE        TP = DUPLICATE with the right duplicate_of on a
                   duplicate query; a wrong target counts as FP and FN;
                   any other DUPLICATE prediction is a FP
  INCIDENT_CLUSTER TP = INCIDENT_CLUSTER on a positive cluster query;
                   FP = INCIDENT_CLUSTER on any non-positive, non-neutral query
  hard-negative FP rate = share of the 15 hard negatives predicted
                   DUPLICATE or INCIDENT_CLUSTER (also per type)
  background FP rate    = same, over background tickets

LIMITATION: every label comes from scripts/augment_tickets.py, a synthetic
generator designed for this project, not from real ITSM history. The query
is a free-text proxy for Agent 1's structured fields; live structured
queries embed differently (a real duplicate scored ~0.66 in the Phase 4
smoke test), so these thresholds should not be copied to the live path
without re-checking.

Outputs: results/agent4_threshold_sweep.csv, results/agent4_eval.md
"""

import asyncio
import csv
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import json  # noqa: E402

from app.agents.correlation_engine import CorrelationEngine, _LazySentenceTransformer  # noqa: E402
from app.agents.seed_ticket_pool import DEFAULT_SEED_PATH, JsonOpenTicketPool  # noqa: E402
from app.config.category_profiles import normalize_category_label  # noqa: E402
from app.config.settings import settings  # noqa: E402
from app.schemas.conversation import ConversationState  # noqa: E402
from app.schemas.extraction import CategoryPrediction, ExtractedField  # noqa: E402

CSV_PATH = REPO_ROOT / "results" / "agent4_threshold_sweep.csv"
MD_PATH = REPO_ROOT / "results" / "agent4_eval.md"

REF_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)  # any fixed instant; only offsets matter
DUP_THRESHOLDS = [0.75, 0.80, 0.85, 0.90, 0.95]
CLUSTER_THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
BUILDING_MODES = ["known", "unknown"]
HARD_NEGATIVE_TYPES = ["hard_negative_building", "hard_negative_time", "hard_negative_category"]


class _ExcludingPool:
    """The shared seed pool minus one ticket (the query)."""

    def __init__(self, pool: JsonOpenTicketPool, exclude_id: str):
        self._pool = pool
        self._exclude_id = exclude_id

    async def list_open(self, since, category=None):
        rows = await self._pool.list_open(since, category)
        return [r for r in rows if r.ticket_id != self._exclude_id]


def query_state(raw: dict, building_mode: str) -> ConversationState:
    text = f"{raw.get('subject', '')}\n{raw.get('message', '')}".strip()
    fields = {"description": ExtractedField(value=text, confidence=1.0)}
    if building_mode == "known" and raw.get("building"):
        fields["building"] = ExtractedField(value=raw["building"], confidence=1.0)
    category = normalize_category_label(raw.get("true_category"))
    return ConversationState(
        conversation_id=f"eval-{raw['ticket_id']}",
        extracted_fields=fields,
        detected_category=CategoryPrediction(value=category, confidence=1.0) if category else None,
    )


def expected_outcomes(raw_tickets: list[dict]) -> dict[str, str]:
    """ticket_id -> DUPLICATE / INCIDENT_CLUSTER / NEUTRAL / NO_MATCH."""
    window = settings.corr_cluster_window_hours * 60
    need_prior = settings.corr_cluster_min_count - 1
    by_cluster: dict[str, list[dict]] = {}
    for t in raw_tickets:
        if t["label_type"] == "cluster_member":
            by_cluster.setdefault(t["gold_cluster_id"], []).append(t)

    expected = {}
    for t in raw_tickets:
        if t["label_type"] == "duplicate":
            expected[t["ticket_id"]] = "DUPLICATE"
        elif t["label_type"] == "cluster_member":
            me = t["created_at_offset_minutes"]
            prior = [
                m for m in by_cluster[t["gold_cluster_id"]]
                if m is not t and me - window <= m["created_at_offset_minutes"] < me
            ]
            expected[t["ticket_id"]] = "INCIDENT_CLUSTER" if len(prior) >= need_prior else "NEUTRAL"
        else:
            expected[t["ticket_id"]] = "NO_MATCH"
    return expected


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


def score(raw_tickets, expected, predictions) -> dict:
    by_id = {t["ticket_id"]: t for t in raw_tickets}
    dup_tp = dup_fp = dup_fn = 0
    cl_tp = cl_fp = cl_fn = 0
    purity = []
    hn_hits = {k: 0 for k in HARD_NEGATIVE_TYPES}
    bg_hits = bg_n = 0

    for tid, result in predictions.items():
        raw, exp, out = by_id[tid], expected[tid], result.outcome
        # duplicates
        if exp == "DUPLICATE":
            if out == "DUPLICATE" and result.duplicate_of == raw["duplicate_of"]:
                dup_tp += 1
            else:
                dup_fn += 1
                dup_fp += out == "DUPLICATE"
        elif out == "DUPLICATE":
            dup_fp += 1
        # clusters (neutral queries are not scored)
        if exp == "INCIDENT_CLUSTER":
            if out == "INCIDENT_CLUSTER":
                cl_tp += 1
                same = [s for s in result.similar_tickets if by_id[s.ticket_id].get("gold_cluster_id") == raw["gold_cluster_id"]]
                purity.append(len(same) / len(result.similar_tickets))
            else:
                cl_fn += 1
        elif exp != "NEUTRAL" and out == "INCIDENT_CLUSTER":
            cl_fp += 1
        # false-positive rates
        flagged = out in ("DUPLICATE", "INCIDENT_CLUSTER")
        if raw["label_type"] in hn_hits:
            hn_hits[raw["label_type"]] += flagged
        elif raw["label_type"] == "background":
            bg_n += 1
            bg_hits += flagged

    hn_n = {k: sum(1 for t in raw_tickets if t["label_type"] == k) for k in HARD_NEGATIVE_TYPES}
    dp, dr, df = _prf(dup_tp, dup_fp, dup_fn)
    cp, cr, cf = _prf(cl_tp, cl_fp, cl_fn)
    row = {
        "dup_tp": dup_tp, "dup_fp": dup_fp, "dup_fn": dup_fn,
        "dup_precision": dp, "dup_recall": dr, "dup_f1": df,
        "cluster_tp": cl_tp, "cluster_fp": cl_fp, "cluster_fn": cl_fn,
        "cluster_precision": cp, "cluster_recall": cr, "cluster_f1": cf,
        "cluster_member_purity": round(sum(purity) / len(purity), 4) if purity else "",
        "hard_negative_fp_rate": round(sum(hn_hits.values()) / sum(hn_n.values()), 4),
    }
    for k in HARD_NEGATIVE_TYPES:
        row[f"{k}_fp_rate"] = round(hn_hits[k] / hn_n[k], 4) if hn_n[k] else ""
    row["background_fp_rate"] = round(bg_hits / bg_n, 4) if bg_n else ""
    return row


async def run_config(raw_tickets, pool, embed_fn, building_mode, dup_t, cl_t) -> dict:
    predictions = {}
    for raw in raw_tickets:
        created_at = REF_NOW + timedelta(minutes=raw["created_at_offset_minutes"])
        engine = CorrelationEngine(
            embed_fn=embed_fn, clock=lambda c=created_at: c,
            dup_threshold=dup_t, cluster_threshold=cl_t,
        )
        predictions[raw["ticket_id"]] = await engine.check(
            query_state(raw, building_mode), _ExcludingPool(pool, raw["ticket_id"])
        )
    return predictions


def duplicate_error_sources(raw_tickets, expected, predictions) -> dict[str, int]:
    """Why each wrong DUPLICATE call happened, bucketed by what was matched."""
    by_id = {t["ticket_id"]: t for t in raw_tickets}
    buckets: dict[str, int] = {}
    for tid, result in predictions.items():
        raw = by_id[tid]
        if result.outcome != "DUPLICATE":
            continue
        if expected[tid] == "DUPLICATE" and result.duplicate_of == raw["duplicate_of"]:
            continue
        match = by_id[result.duplicate_of]
        if raw.get("gold_cluster_id") and match.get("gold_cluster_id") == raw["gold_cluster_id"]:
            reason = "cluster member matched a sibling in its own cluster"
        elif raw["label_type"] == "cluster_member" and match["label_type"] == "hard_negative_time":
            reason = "cluster member matched its cluster's outside-window hard negative"
        elif match["label_type"] == "cluster_member" and raw["label_type"].startswith("hard_negative"):
            reason = f"{raw['label_type']} matched a cluster member"
        else:
            reason = f"{raw['label_type']} matched a {match['label_type']} ticket"
        buckets[reason] = buckets.get(reason, 0) + 1
    return dict(sorted(buckets.items(), key=lambda kv: -kv[1]))


def best(rows, mode, key):
    """Highest key; ties broken by lower hard-negative FP rate, then higher thresholds."""
    cands = [r for r in rows if r["building_mode"] == mode]
    return max(cands, key=lambda r: (r[key], -r["hard_negative_fp_rate"], r["dup_threshold"], r["cluster_threshold"]))


def write_markdown(rows, raw_tickets, expected, error_sources):
    counts = {k: sum(1 for v in expected.values() if v == k) for k in ("DUPLICATE", "INCIDENT_CLUSTER", "NEUTRAL")}
    n_hn = sum(1 for t in raw_tickets if t["label_type"] in HARD_NEGATIVE_TYPES)
    n_bg = sum(1 for t in raw_tickets if t["label_type"] == "background")
    d0, c0 = settings.corr_dup_threshold, settings.corr_cluster_threshold

    def at(mode, d, c):
        return next(r for r in rows if r["building_mode"] == mode and r["dup_threshold"] == d and r["cluster_threshold"] == c)

    def fmt(r):
        return (f"| {r['building_mode']} | {r['dup_threshold']:.2f} | {r['cluster_threshold']:.2f} | "
                f"{r['dup_precision']:.2f} / {r['dup_recall']:.2f} / {r['dup_f1']:.2f} | "
                f"{r['cluster_precision']:.2f} / {r['cluster_recall']:.2f} / {r['cluster_f1']:.2f} | "
                f"{r['hard_negative_fp_rate']:.2f} | {r['background_fp_rate']:.2f} |")

    header = [
        "| building | dup t | cluster t | DUPLICATE P / R / F1 | INCIDENT_CLUSTER P / R / F1 | hard-neg FP rate | background FP rate |",
        "|---|---|---|---|---|---|---|",
    ]
    lines = [
        "# Agent 4 evaluation (leave-one-out threshold sweep)",
        "",
        "Generated by `scripts/run_agent4_eval.py`. Do not edit by hand.",
        "",
        "## Limitations (read first)",
        "",
        "- **Synthetic labels.** Every label comes from `scripts/augment_tickets.py`, a generator designed",
        "  for this project, not from real ITSM history. Scores show the rules behave as designed on",
        "  data built to exercise them; they are not evidence of real-world accuracy.",
        "- **Proxy query.** Agent 1's structured fields are not stored for this dataset, so each request is",
        "  simulated as one field holding subject + message (eval option A) with the true category. Live",
        "  queries built from Agent 1's structured fields embed differently and score lower (a real",
        "  duplicate scored ~0.66 in the Phase 4 smoke test), so these thresholds may not transfer to the live path.",
        "- **Small positive sets:** "
        f"{counts['DUPLICATE']} duplicate queries and {counts['INCIDENT_CLUSTER']} detectable cluster queries; "
        "one ticket moves recall by several points.",
        "",
        "## Protocol",
        "",
        "Temporal leave-one-out: each ticket queries the pool minus itself, with the engine clock at the",
        "ticket's own created_at, so only earlier tickets are visible. A cluster member is a positive only",
        f"when at least {settings.corr_cluster_min_count - 1} earlier members of its gold cluster arrived within",
        f"{settings.corr_cluster_window_hours:g}h before it ({counts['NEUTRAL']} earlier members are neutral and not scored).",
        f"Hard negatives: {n_hn}; background tickets (FP check only): {n_bg}.",
        "`known` = the ticket's building is passed as an extracted field (as if Agent 1 extracted it);",
        "`unknown` = no building, which is what Agent 1 produces today.",
        "",
        "## Current defaults",
        "",
        *header,
        *(fmt(at(m, d0, c0)) for m in BUILDING_MODES),
        "",
        "## Best configurations",
        "",
        "Best by duplicate F1, by cluster F1, and by their sum (ties: lower hard-negative FP rate).",
        "",
        "| criterion | " + header[0][2:],
        "|---|" + header[1][1:],
    ]
    for mode in BUILDING_MODES:
        for label, key in (("dup F1", "dup_f1"), ("cluster F1", "cluster_f1"), ("dup F1 + cluster F1", "combined_f1")):
            lines.append(f"| {label} " + fmt(best(rows, mode, key)))
    lines += [
        "",
        "## Hard negatives at current defaults (FP rate by type)",
        "",
        "| building | other building | outside window | other category |",
        "|---|---|---|---|",
    ]
    for mode in BUILDING_MODES:
        r = at(mode, d0, c0)
        lines.append(
            f"| {mode} | {r['hard_negative_building_fp_rate']:.2f} | "
            f"{r['hard_negative_time_fp_rate']:.2f} | {r['hard_negative_category_fp_rate']:.2f} |"
        )
    lines += [
        "",
        "## Where the wrong DUPLICATE calls come from (current defaults)",
        "",
        "DUPLICATE is checked before INCIDENT_CLUSTER, and duplicates have no time limit beyond the",
        "lookback, so near-identical reports of one outage are called duplicates of each other.",
        "",
        "| building | reason | count |",
        "|---|---|---|",
    ]
    for mode in BUILDING_MODES:
        for reason, n in error_sources[mode].items():
            lines.append(f"| {mode} | {reason} | {n} |")
    lines += ["", "## Full sweep", "", *header]
    lines += [fmt(r) for r in rows]
    lines.append("")
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")


async def main():
    raw_tickets = json.loads(Path(DEFAULT_SEED_PATH).read_text(encoding="utf-8"))
    expected = expected_outcomes(raw_tickets)
    pool = JsonOpenTicketPool(clock=lambda: REF_NOW)
    embed_fn = _LazySentenceTransformer()  # shared, so every text is embedded once

    rows = []
    for mode in BUILDING_MODES:
        for dup_t in DUP_THRESHOLDS:
            for cl_t in CLUSTER_THRESHOLDS:
                predictions = await run_config(raw_tickets, pool, embed_fn, mode, dup_t, cl_t)
                row = {"building_mode": mode, "dup_threshold": dup_t, "cluster_threshold": cl_t}
                row.update(score(raw_tickets, expected, predictions))
                row["combined_f1"] = round(row["dup_f1"] + row["cluster_f1"], 4)
                rows.append(row)
        print(f"building={mode}: done")

    error_sources = {}
    for mode in BUILDING_MODES:
        defaults = await run_config(
            raw_tickets, pool, embed_fn, mode, settings.corr_dup_threshold, settings.corr_cluster_threshold
        )
        error_sources[mode] = duplicate_error_sources(raw_tickets, expected, defaults)

    CSV_PATH.parent.mkdir(exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows, raw_tickets, expected, error_sources)
    print(f"Wrote {CSV_PATH.relative_to(REPO_ROOT)} and {MD_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())
