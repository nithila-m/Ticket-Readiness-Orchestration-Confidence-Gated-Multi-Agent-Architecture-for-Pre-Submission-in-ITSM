"""
Live-format duplicate-threshold calibration for Agent 4 (CorrelationEngine).

Run from the repo root (no API calls; loads all-MiniLM-L6-v2 once):
    python scripts/run_agent4_live_calibration.py

Why a separate script: scripts/run_agent4_eval.py ("option A") embeds
subject + message on both sides. The live app compares
build_kb_query(Agent 1 category, Agent 1 fields) against each ticket's
correlation_text built the same way, so its thresholds do not transfer.
This script measures the live format from cached REAL Agent 1 output.

Data (all read-only):
  tickets   data/eval/tickets_augmented.json (190)
            + data/eval/agent4_calibration_extra.json (40)
  Agent 1   data/eval/agent1_fields_cache.json; every ticket must be present
            (the script exits listing any that are missing)

Live-format texts, exactly as the app builds them:
  query     ConversationState with detected_category = the CACHED Agent 1
            category and confidence (not the dataset label) and
            extracted_fields = the cached fields; no building (building mode
            "unknown", what Agent 1 produces today). The engine turns it into
            build_kb_query(category, fields) itself.
  pool      seed_to_record(raw, now) as the live seed does, then
            correlation_text = ticket_builder.build_correlation_text() on the
            same cached state (the real function, not a copy): None when no
            field has a value, and the engine then falls back to
            title + description, same as live.

Protocol (same as run_agent4_eval.py): temporal leave-one-out - every
ticket queries the pool minus itself with the engine clock at its own
created_at. The real CorrelationEngine runs with the category-agreement
bonus, cross-category rule, lookback and cluster settings from
app.config.settings; only dup_threshold is overridden per sweep step
(0.60-0.98, step 0.01). Every text is embedded once up front and reused by
all steps; only the decisions change.

Expected outcomes come from run_agent4_eval.expected_outcomes (duplicate ->
DUPLICATE of duplicate_of; detectable cluster member -> INCIDENT_CLUSTER;
early cluster member -> NEUTRAL; everything else, hard_negative_near_miss
included, -> NO_MATCH).

The engine compares RAW cosine with dup_threshold (the category bonus only
ranks candidates and counts for clusters) and applies the minimum-field
gate (corr_dup_min_fields) to the query and to each candidate's
correlation_text, so the diagnostics below use raw cosine over
DUPLICATE-eligible candidates; a guard checks they reproduce every
DUPLICATE decision at every threshold.

Strict rule (applied mechanically by recommend()):
  1. 0 DUPLICATE-caused FP on all four hard-negative types
  2. background DUPLICATE-caused FP rate <= 1%
  3. highest duplicate recall; ties -> the HIGHER threshold
  4. report the margin at the chosen value
INCIDENT_CLUSTER FPs are reported separately: they come from the cluster
threshold, which is not swept. Cluster members called DUPLICATE are
reported but kept out of rules 1-2 (same incident merged - a lesser error
than merging a different issue).

Suggestion rule (suggestion_rule()): a DUPLICATE is now a suggestion the
user confirms, so also report the HIGHEST threshold with duplicate recall
>= 0.90 and its wrong suggestions (all wrong DUPLICATE calls) per 100
non-duplicate tickets.

Score diagnostics (true-original score per duplicate, top DUPLICATE-
eligible raw score per non-duplicate) recompute the engine's scores with
its own _cosine, bonus, field gate and cross-category rule over the same
candidate set; guards assert they match the engine's max_similarity and
its DUPLICATE decisions.

Outputs: results/agent4_live_threshold_sweep.csv, results/agent4_live_calibration.md
"""

import asyncio
import csv
import json
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.agents.correlation_engine import (  # noqa: E402
    CorrelationEngine,
    _cosine,
    _LazySentenceTransformer,
    candidate_text,
    correlation_field_lines,
    query_field_count,
)
from app.agents.request_text import build_kb_query  # noqa: E402
from app.agents.seed_ticket_pool import JsonOpenTicketPool, seed_to_record  # noqa: E402
from app.agents.ticket_builder import build_correlation_text  # noqa: E402
from app.config.buildings import normalize_building  # noqa: E402
from app.config.settings import settings  # noqa: E402
from app.schemas.conversation import ConversationState  # noqa: E402
from app.schemas.extraction import CategoryPrediction, ExtractedField  # noqa: E402
from app.schemas.pipeline import TicketRecord  # noqa: E402
from scripts.run_agent4_eval import (  # noqa: E402
    REF_NOW,
    _ExcludingPool,
    _prf,
    duplicate_error_sources,
    expected_outcomes,
)

DATA_DIR = REPO_ROOT / "data" / "eval"
TICKET_FILES = (DATA_DIR / "tickets_augmented.json", DATA_DIR / "agent4_calibration_extra.json")
CACHE_PATH = DATA_DIR / "agent1_fields_cache.json"
CSV_PATH = REPO_ROOT / "results" / "agent4_live_threshold_sweep.csv"
MD_PATH = REPO_ROOT / "results" / "agent4_live_calibration.md"

DUP_THRESHOLDS = [round(0.60 + 0.01 * i, 2) for i in range(39)]  # 0.60 .. 0.98
HARD_NEGATIVE_TYPES = {
    "hard_negative_building": "building",
    "hard_negative_time": "time",
    "hard_negative_category": "category",
    "hard_negative_near_miss": "near_miss",
}
MAX_BACKGROUND_DUP_FP_RATE = 0.01
SUGGESTION_MIN_RECALL = 0.90
THIN_FIELD_COUNT = 1  # "thin text" = at most this many non-null fields


# ---------------------------------------------------------------- data


def load_tickets(paths=TICKET_FILES) -> list[dict]:
    tickets = []
    for path in paths:
        tickets.extend(json.loads(Path(path).read_text(encoding="utf-8")))
    return tickets


def load_cache(tickets: list[dict], path: Path = CACHE_PATH) -> dict[str, dict]:
    cache = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [t["ticket_id"] for t in tickets if t["ticket_id"] not in cache]
    if missing:
        raise SystemExit(
            f"{len(missing)} ticket(s) missing from {path.name}; run "
            f"scripts/cache_agent1_fields.py first: {', '.join(missing)}"
        )
    return cache


def non_null_field_count(entry: dict) -> int:
    return sum(f["value"] is not None for f in entry["fields"].values())


def cached_state(ticket_id: str, entry: dict) -> ConversationState:
    """Agent 1's cached output as the live ConversationState (no building)."""
    category = entry.get("category")
    return ConversationState(
        conversation_id=f"live-cal-{ticket_id}",
        extracted_fields={name: ExtractedField(**f) for name, f in entry["fields"].items()},
        detected_category=(
            CategoryPrediction(value=category, confidence=entry["category_confidence"])
            if category else None
        ),
    )


def build_pool_records(tickets: list[dict], cache: dict[str, dict], now: datetime) -> list[TicketRecord]:
    records = []
    for raw in tickets:
        state = cached_state(raw["ticket_id"], cache[raw["ticket_id"]])
        record = seed_to_record(raw, now)
        records.append(record.model_copy(update={"correlation_text": build_correlation_text(state)}))
    return records


class LivePool(JsonOpenTicketPool):
    """JsonOpenTicketPool's list_open() filtering over prebuilt records."""

    def __init__(self, records: list[TicketRecord]):
        super().__init__()
        self._records = records

    def records(self) -> list[TicketRecord]:
        return self._records


def query_text(state: ConversationState) -> str:
    category = state.detected_category.value if state.detected_category else None
    return build_kb_query(category, state.extracted_fields)


# ---------------------------------------------------------------- embeddings


class PrecomputedEmbedder:
    """
    The engine's embed_fn: looks up vectors computed in ONE call to the base
    embedder. An unseen text is recorded (the engine swallows exceptions and
    would silently return NO_MATCH), and the sweep fails on it.
    """

    def __init__(self, texts: list[str], base_embed_fn):
        unique = list(dict.fromkeys(t for t in texts if t))
        vectors = base_embed_fn(unique)
        self._vectors = {t: [float(x) for x in v] for t, v in zip(unique, vectors)}
        self.misses: list[str] = []

    def __call__(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            if t not in self._vectors:
                self.misses.append(t)
                raise KeyError("text was not precomputed")
            out.append(self._vectors[t])
        return out

    def vector(self, text: str) -> list[float]:
        return self._vectors[text]


# ---------------------------------------------------------------- running


def created_at(raw: dict) -> datetime:
    return REF_NOW + timedelta(minutes=raw["created_at_offset_minutes"])


async def run_threshold(tickets, states, pool, embedder, dup_threshold) -> dict:
    predictions = {}
    for raw in tickets:
        tid = raw["ticket_id"]
        engine = CorrelationEngine(
            embed_fn=embedder, clock=lambda c=created_at(raw): c, dup_threshold=dup_threshold,
        )
        predictions[tid] = await engine.check(states[tid], _ExcludingPool(pool, tid))
    if embedder.misses:
        raise RuntimeError(f"{len(embedder.misses)} text(s) were not precomputed")
    return predictions


@dataclass
class SweepOutput:
    tickets: list[dict]
    cache: dict[str, dict]
    expected: dict[str, str]
    rows: list[dict]
    predictions: dict[float, dict]
    diagnostics: dict[str, dict] = field(default_factory=dict)


async def run_sweep(tickets, cache, base_embed_fn, thresholds=DUP_THRESHOLDS) -> SweepOutput:
    expected = expected_outcomes(tickets)
    states = {t["ticket_id"]: cached_state(t["ticket_id"], cache[t["ticket_id"]]) for t in tickets}
    records = build_pool_records(tickets, cache, REF_NOW)
    pool = LivePool(records)
    embedder = PrecomputedEmbedder(
        [query_text(s) for s in states.values()] + [candidate_text(r) for r in records], base_embed_fn
    )

    rows, predictions = [], {}
    for t in thresholds:
        predictions[t] = await run_threshold(tickets, states, pool, embedder, t)
        row = {"dup_threshold": t}
        row.update(live_score(tickets, expected, predictions[t]))
        rows.append(row)

    diagnostics = await score_diagnostics(tickets, states, pool, embedder, predictions[thresholds[0]])
    check_decisions_match(diagnostics, predictions)
    return SweepOutput(tickets, cache, expected, rows, predictions, diagnostics)


def check_decisions_match(diagnostics: dict[str, dict], predictions: dict[float, dict]) -> None:
    """The engine says DUPLICATE exactly when the top eligible raw score reaches the threshold."""
    for t, preds in predictions.items():
        for tid, result in preds.items():
            top = diagnostics[tid]["top_dup_raw"]
            if (result.outcome == "DUPLICATE") != (top is not None and top >= t):
                raise AssertionError(f"{tid} at {t}: engine {result.outcome}, diagnostic top raw {top}")


# ---------------------------------------------------------------- scoring


def live_score(tickets, expected, predictions) -> dict:
    """
    Same counting rules as run_agent4_eval.score(), with the near-miss type
    added and every false positive split by cause (DUPLICATE vs
    INCIDENT_CLUSTER).
    """
    by_id = {t["ticket_id"]: t for t in tickets}
    dup_tp = dup_fp = dup_fn = 0
    cl_tp = cl_fp = cl_fn = 0
    hn_dup = dict.fromkeys(HARD_NEGATIVE_TYPES, 0)
    hn_cl = dict.fromkeys(HARD_NEGATIVE_TYPES, 0)
    bg_n = bg_dup = bg_cl = 0
    members_as_dup = 0

    for tid, result in predictions.items():
        raw, exp, out = by_id[tid], expected[tid], result.outcome
        label = raw["label_type"]
        if exp == "DUPLICATE":
            if out == "DUPLICATE" and result.duplicate_of == raw["duplicate_of"]:
                dup_tp += 1
            else:
                dup_fn += 1
                dup_fp += out == "DUPLICATE"
        elif out == "DUPLICATE":
            dup_fp += 1
        if exp == "INCIDENT_CLUSTER":
            if out == "INCIDENT_CLUSTER":
                cl_tp += 1
            else:
                cl_fn += 1
        elif exp != "NEUTRAL" and out == "INCIDENT_CLUSTER":
            cl_fp += 1

        if label in HARD_NEGATIVE_TYPES:
            hn_dup[label] += out == "DUPLICATE"
            hn_cl[label] += out == "INCIDENT_CLUSTER"
        elif label == "background":
            bg_n += 1
            bg_dup += out == "DUPLICATE"
            bg_cl += out == "INCIDENT_CLUSTER"
        elif label == "cluster_member":
            members_as_dup += out == "DUPLICATE"

    hn_n = {k: sum(t["label_type"] == k for t in tickets) for k in HARD_NEGATIVE_TYPES}
    dp, dr, df = _prf(dup_tp, dup_fp, dup_fn)
    cp, cr, cf = _prf(cl_tp, cl_fp, cl_fn)
    row = {
        "dup_tp": dup_tp, "dup_fp": dup_fp, "dup_fn": dup_fn,
        "dup_precision": dp, "dup_recall": dr, "dup_f1": df,
    }
    for k, short in HARD_NEGATIVE_TYPES.items():
        row[f"hn_{short}_n"] = hn_n[k]
        row[f"hn_{short}_dup_fp"] = hn_dup[k]
        row[f"hn_{short}_dup_fp_rate"] = round(hn_dup[k] / hn_n[k], 4) if hn_n[k] else 0.0
    row["hn_dup_fp_total"] = sum(hn_dup.values())
    row["hn_cluster_fp_total"] = sum(hn_cl.values())
    for k, short in HARD_NEGATIVE_TYPES.items():
        row[f"hn_{short}_cluster_fp"] = hn_cl[k]
    row.update({
        "background_n": bg_n,
        "background_dup_fp": bg_dup,
        "background_dup_fp_rate": round(bg_dup / bg_n, 4) if bg_n else 0.0,
        "background_cluster_fp": bg_cl,
        "background_cluster_fp_rate": round(bg_cl / bg_n, 4) if bg_n else 0.0,
        "cluster_members_called_dup": members_as_dup,
        "cluster_tp": cl_tp, "cluster_fp": cl_fp, "cluster_fn": cl_fn,
        "cluster_precision": cp, "cluster_recall": cr, "cluster_f1": cf,
    })
    return row


async def score_diagnostics(tickets, states, pool, embedder, reference_predictions) -> dict[str, dict]:
    """
    Per query, recomputed as the engine does: every eligible candidate's raw
    cosine and bonus-adjusted ranking score, and which candidates the
    DUPLICATE step may pick (minimum-field gate + cross-category rule).
    top_dup_raw - the decision value - is None when the query itself is
    below the field gate.
    """
    bonus = settings.corr_category_agreement_bonus
    cross = settings.corr_cross_category_dup_threshold
    min_fields = settings.corr_dup_min_fields
    lookback = timedelta(hours=settings.corr_lookback_hours)
    out = {}
    for raw in tickets:
        tid = raw["ticket_id"]
        state = states[tid]
        text = query_text(state)
        query_gated = query_field_count(state) < min_fields
        empty = {"top_id": None, "top_score": None, "top_raw": None, "query_gated": query_gated,
                 "top_dup_id": None, "top_dup_raw": None}
        if not text:
            out[tid] = empty
            continue
        now = created_at(raw)
        category = state.detected_category.value if state.detected_category else None
        candidates = [
            c for c in await _ExcludingPool(pool, tid).list_open(since=now - lookback)
            if c.created_at <= now
        ]
        qv = embedder.vector(text)
        scored = []
        for c in candidates:
            r = max(0.0, min(1.0, _cosine(qv, embedder.vector(candidate_text(c)))))
            agrees = category is not None and c.category == category
            lines = correlation_field_lines(c.correlation_text)
            dup_ok = (lines is None or lines >= min_fields) and (agrees or category is None or r >= cross)
            scored.append((min(1.0, r + bonus) if agrees else r, r, agrees, c.ticket_id, dup_ok))
        scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
        if not scored:
            out[tid] = empty
            continue

        engine_max = reference_predictions[tid].max_similarity
        mine = round(max(s[1] for s in scored), 4)
        if engine_max != mine:
            raise AssertionError(f"{tid}: diagnostic max raw {mine} != engine {engine_max}")

        eligible = [] if query_gated else [s for s in scored if s[4]]
        top_dup = max(eligible, key=lambda s: s[1]) if eligible else None
        d = {
            "top_id": scored[0][3], "top_score": scored[0][0], "top_raw": scored[0][1],
            "query_gated": query_gated,
            "top_dup_id": top_dup[3] if top_dup else None,
            "top_dup_raw": top_dup[1] if top_dup else None,
        }
        if raw["label_type"] == "duplicate":
            orig = next((s for s in scored if s[3] == raw["duplicate_of"]), None)
            d.update({
                "original_in_pool": orig is not None,
                "original_score": orig[0] if orig else None,
                "original_raw": orig[1] if orig else None,
                "original_category_agrees": orig[2] if orig else None,
                "original_dup_eligible": bool(orig and orig[4]) and not query_gated,
                "original_is_top": scored[0][3] == raw["duplicate_of"],
            })
        out[tid] = d
    return out


def thin_text_breakdown(tickets, cache, expected, predictions) -> dict:
    """Thin = at most THIN_FIELD_COUNT non-null fields in Agent 1's output."""
    by_id = {t["ticket_id"]: t for t in tickets}

    def thin(tid):
        return non_null_field_count(cache[tid]) <= THIN_FIELD_COUNT

    wrong_dups = [
        tid for tid, r in predictions.items()
        if r.outcome == "DUPLICATE"
        and not (expected[tid] == "DUPLICATE" and r.duplicate_of == by_id[tid]["duplicate_of"])
    ]
    return {
        "thin_queries": sum(thin(t["ticket_id"]) for t in tickets),
        "thin_positives": sum(thin(t["ticket_id"]) for t in tickets if t["label_type"] == "duplicate"),
        "dup_fp_total": len(wrong_dups),
        "dup_fp_thin_query": sum(thin(tid) for tid in wrong_dups),
        "dup_fp_thin_match": sum(thin(predictions[tid].duplicate_of) for tid in wrong_dups),
        "dup_fp_thin_either": sum(thin(tid) or thin(predictions[tid].duplicate_of) for tid in wrong_dups),
        "missed_positives_thin": sum(
            thin(t["ticket_id"]) or thin(t["duplicate_of"])
            for t in tickets
            if t["label_type"] == "duplicate"
            and not (predictions[t["ticket_id"]].outcome == "DUPLICATE"
                     and predictions[t["ticket_id"]].duplicate_of == t["duplicate_of"])
        ),
    }


# ---------------------------------------------------------------- recommendation


def recommend(rows: list[dict]) -> tuple[dict | None, list[str]]:
    """Applies rules 1-3; returns (chosen row or None, the working as markdown lines)."""
    def ts(rs):
        return ", ".join(f"{r['dup_threshold']:.2f}" for r in rs) or "none"

    lines = []
    rule1 = [r for r in rows if r["hn_dup_fp_total"] == 0]
    lines.append(f"1. **0 DUPLICATE-caused FP on all hard negatives** (building, time, category, "
                 f"near-miss): {len(rule1)}/{len(rows)} thresholds pass - {ts(rule1)}.")
    rule2 = [r for r in rule1 if r["background_dup_fp_rate"] <= MAX_BACKGROUND_DUP_FP_RATE]
    lines.append(f"2. **Background DUPLICATE-caused FP rate <= {MAX_BACKGROUND_DUP_FP_RATE:.0%}**: "
                 f"{len(rule2)} of those pass - {ts(rule2)}.")
    if not rule2:
        lines.append("3. No threshold satisfies rules 1-2, so **no threshold is recommended**.")
        return None, lines
    best_recall = max(r["dup_recall"] for r in rule2)
    tied = [r for r in rule2 if r["dup_recall"] == best_recall]
    chosen = max(tied, key=lambda r: r["dup_threshold"])
    lines.append(f"3. **Highest duplicate recall** among them is {best_recall:.4f}, reached at "
                 f"{ts(tied)}; the tie goes to the higher (more conservative) threshold: "
                 f"**{chosen['dup_threshold']:.2f}**.")
    return chosen, lines


def suggestion_rule(rows: list[dict], n_non_duplicates: int) -> tuple[dict | None, list[str]]:
    """Highest threshold with duplicate recall >= SUGGESTION_MIN_RECALL; returns (row, working)."""
    passing = [r for r in rows if r["dup_recall"] >= SUGGESTION_MIN_RECALL]
    if not passing:
        return None, [f"No threshold reaches duplicate recall >= {SUGGESTION_MIN_RECALL:.2f}."]
    chosen = max(passing, key=lambda r: r["dup_threshold"])
    per_100 = 100 * chosen["dup_fp"] / n_non_duplicates if n_non_duplicates else 0.0
    thresholds = ", ".join(f"{r['dup_threshold']:.2f}" for r in passing)
    return chosen, [
        f"Thresholds with duplicate recall >= {SUGGESTION_MIN_RECALL:.2f}: {thresholds}.",
        f"The highest is **{chosen['dup_threshold']:.2f}**: recall {chosen['dup_recall']:.4f} "
        f"({chosen['dup_tp']}/{chosen['dup_tp'] + chosen['dup_fn']}), {chosen['dup_fp']} wrong suggestions = "
        f"**{per_100:.1f} per 100 non-duplicate tickets** ({n_non_duplicates} non-duplicate queries; "
        "a wrong suggestion is any wrong DUPLICATE call, which the user is now asked to confirm).",
    ]


def margin_at(threshold, out: SweepOutput) -> dict:
    preds = out.predictions[threshold]
    diag = out.diagnostics
    caught = [
        diag[t["ticket_id"]]["original_raw"] for t in out.tickets
        if t["label_type"] == "duplicate"
        and preds[t["ticket_id"]].outcome == "DUPLICATE"
        and preds[t["ticket_id"]].duplicate_of == t["duplicate_of"]
    ]

    def top(labels):
        vals = [(diag[t["ticket_id"]]["top_dup_raw"], t["ticket_id"]) for t in out.tickets
                if t["label_type"] in labels and diag[t["ticket_id"]]["top_dup_raw"] is not None]
        return max(vals) if vals else (None, None)

    return {
        "lowest_caught": min(caught) if caught else None,
        "highest_non_dup": top({k for k in _all_labels(out.tickets) if k != "duplicate"}),
        "highest_rule_set": top(set(HARD_NEGATIVE_TYPES) | {"background"}),
    }


def _all_labels(tickets) -> set[str]:
    return {t["label_type"] for t in tickets}


# ---------------------------------------------------------------- report


def histogram_lines(dups: list[float], non_dups: list[float], width: float = 0.05) -> list[str]:
    lo = min(dups + non_dups) if dups + non_dups else 0.0
    start = int(lo / width) * width
    bins = []
    b = start
    while b < 1.0 - 1e-9:
        bins.append(round(b, 2))
        b += width
    lines = ["| raw cosine bin | duplicates: true original | non-duplicates: top DUPLICATE-eligible candidate |",
             "|---|---|---|"]
    for i, b in enumerate(bins):
        last = i == len(bins) - 1
        hi = round(b + width, 2)

        def count(vals):
            return sum(b <= v < hi or (last and v == 1.0) for v in vals)

        label = f"[{b:.2f}, {hi:.2f}{']' if last else ')'}"
        lines.append(f"| {label} | {count(dups)} | {count(non_dups)} |")
    return lines


def _stats(vals: list[float]) -> str:
    if not vals:
        return "n/a"
    return f"min {min(vals):.4f} / median {statistics.median(vals):.4f} / max {max(vals):.4f} (n={len(vals)})"


def _fmt_row(r: dict, tag: str = "") -> str:
    return (
        f"| {r['dup_threshold']:.2f}{tag} | {r['dup_tp']} / {r['dup_fp']} / {r['dup_fn']} "
        f"| {r['dup_precision']:.2f} / {r['dup_recall']:.2f} / {r['dup_f1']:.2f} "
        f"| {r['hn_building_dup_fp']} / {r['hn_time_dup_fp']} / {r['hn_category_dup_fp']} / {r['hn_near_miss_dup_fp']} "
        f"| {r['background_dup_fp']} ({r['background_dup_fp_rate']:.1%}) "
        f"| {r['cluster_members_called_dup']} "
        f"| {r['hn_cluster_fp_total']} | {r['background_cluster_fp']} ({r['background_cluster_fp_rate']:.1%}) "
        f"| {r['cluster_precision']:.2f} / {r['cluster_recall']:.2f} / {r['cluster_f1']:.2f} |"
    )


SWEEP_HEADER = [
    "| dup t | DUP TP / FP / FN | DUP P / R / F1 | hard-neg DUP FP (bld / time / cat / near-miss) "
    "| background DUP FP | cluster members -> DUP | hard-neg CLUSTER FP | background CLUSTER FP "
    "| INCIDENT_CLUSTER P / R / F1 |",
    "|---|---|---|---|---|---|---|---|---|",
]


def blocking_lines(out: SweepOutput, threshold: float) -> list[str]:
    """The DUPLICATE calls on hard negatives / background that still fail rules 1-2 at `threshold`."""
    by_id = {t["ticket_id"]: t for t in out.tickets}
    lines = [
        f"### What still blocks rules 1-2 at the highest swept threshold ({threshold:.2f})",
        "",
        "| query | label | matched | raw | same building | fields (query / matched) |",
        "|---|---|---|---|---|---|",
    ]
    for tid, r in out.predictions[threshold].items():
        label = by_id[tid]["label_type"]
        if r.outcome != "DUPLICATE" or not (label in HARD_NEGATIVE_TYPES or label == "background"):
            continue
        match = r.duplicate_of
        raw = next(s.similarity for s in r.similar_tickets if s.ticket_id == match)
        same = normalize_building(by_id[tid]["building"]) == normalize_building(by_id[match]["building"])
        lines.append(
            f"| {tid} | {label} | {match} | {raw:.4f} | {'yes' if same else 'no'} "
            f"| {non_null_field_count(out.cache[tid])} / {non_null_field_count(out.cache[match])} |"
        )
    return lines


def write_csv(rows: list[dict], path: Path = CSV_PATH) -> None:
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(out: SweepOutput, chosen: dict | None, working: list[str],
                   suggestion: tuple[dict | None, list[str]], path: Path = MD_PATH) -> None:
    tickets, cache, diag = out.tickets, out.cache, out.diagnostics
    by_t = {r["dup_threshold"]: r for r in out.rows}
    current = settings.corr_dup_threshold
    n_pos = sum(t["label_type"] == "duplicate" for t in tickets)
    n_aug_dup = sum(t["label_type"] == "duplicate" and t["ticket_id"].startswith("AUG-") for t in tickets)
    n_neutral = sum(v == "NEUTRAL" for v in out.expected.values())
    n_cl_pos = sum(v == "INCIDENT_CLUSTER" for v in out.expected.values())
    label_counts = {}
    for t in tickets:
        label_counts[t["label_type"]] = label_counts.get(t["label_type"], 0) + 1

    dups = [t for t in tickets if t["label_type"] == "duplicate"]
    non_dups = [t for t in tickets if t["label_type"] != "duplicate"]
    dup_raw = [diag[t["ticket_id"]]["original_raw"] for t in dups
               if diag[t["ticket_id"]].get("original_dup_eligible")]
    dup_gated = [t["ticket_id"] for t in dups if not diag[t["ticket_id"]].get("original_dup_eligible")]
    non_dup = [diag[t["ticket_id"]]["top_dup_raw"] for t in non_dups
               if diag[t["ticket_id"]]["top_dup_raw"] is not None]
    non_dup_gated = sum(diag[t["ticket_id"]]["top_dup_raw"] is None for t in non_dups)
    rule_set = [diag[t["ticket_id"]]["top_dup_raw"] for t in tickets
                if t["label_type"] in set(HARD_NEGATIVE_TYPES) | {"background"}
                and diag[t["ticket_id"]]["top_dup_raw"] is not None]
    suggested, suggestion_working = suggestion
    model = next(iter(cache.values()))["model"]

    lines = [
        "# Agent 4 live-format duplicate threshold calibration",
        "",
        "Generated by `scripts/run_agent4_live_calibration.py`. Do not edit by hand. "
        "Recommendation only: settings are unchanged.",
        "",
        "## 1. Limitations (read first)",
        "",
        "- **Synthetic labels.** Every label comes from `scripts/augment_tickets.py` or the hand-written "
        "calibration set (`scripts/build_agent4_calibration_set.py`), both written for this project, not "
        "real ITSM history.",
        f"- **{n_pos} positives** ({n_aug_dup} AUG-DUP + {n_pos - n_aug_dup} CAL-DUP), so one ticket "
        "moves duplicate recall by about "
        f"{100 / n_pos if n_pos else 0:.1f} points.",
        f"- **Single Agent 1 run per ticket** (`{model}`, temperature 0). Run-to-run extraction variation "
        "is not included yet; the variation check (`scripts/check_agent1_variation.py`) is pending.",
        "- **Building unknown.** Agent 1 does not extract a building today, so building-based separation "
        "never applies to duplicates.",
        "- **Specific to all-MiniLM-L6-v2 and this field format** (`build_kb_query` lines). A prompt, field "
        "or embedder change invalidates these numbers.",
        "",
        "## 2. Protocol",
        "",
        f"- Tickets: {len(tickets)} ({', '.join(f'{k} {v}' for k, v in sorted(label_counts.items()))}).",
        "- Query: `build_kb_query(cached Agent 1 category, cached fields)`; no building. Pool tickets: "
        "`seed_to_record()` plus `correlation_text = build_correlation_text()` from the same cache "
        "(None when no field has a value -> engine falls back to title + description).",
        "- Temporal leave-one-out: each ticket queries the pool minus itself, engine clock at its own "
        "created_at.",
        f"- Real `CorrelationEngine`: category-agreement bonus {settings.corr_category_agreement_bonus}, "
        f"cross-category duplicate needs raw >= {settings.corr_cross_category_dup_threshold}, lookback "
        f"{settings.corr_lookback_hours:g}h, cluster threshold {settings.corr_cluster_threshold} / "
        f"min count {settings.corr_cluster_min_count} / window {settings.corr_cluster_window_hours:g}h "
        "(settings defaults). Only `dup_threshold` is swept, 0.60-0.98 in 0.01 steps; texts are embedded once.",
        f"- Expected: duplicate -> DUPLICATE of `duplicate_of`; {n_cl_pos} detectable cluster members -> "
        f"INCIDENT_CLUSTER ({n_neutral} early members neutral); everything else, near-misses included, -> NO_MATCH.",
        f"- DUPLICATE decision: RAW cosine >= dup_threshold, among candidates allowed by the minimum-field "
        f"gate (query and candidate correlation_text need >= {settings.corr_dup_min_fields} field lines; "
        "candidates without correlation_text stay eligible) and the cross-category rule. The ranking "
        f"score is raw + {settings.corr_category_agreement_bonus} when the cached Agent 1 category equals "
        "the ticket's stored category (capped at 1.0); it orders candidates and counts for clusters only.",
        "",
        "## 3. Score distributions",
        "",
        "All values are raw cosine, the value the DUPLICATE decision now uses.",
        "",
        f"- Duplicates, raw cosine of the TRUE original (when the gate allows it): {_stats(dup_raw)}. "
        f"Blocked by the minimum-field gate, so never caught: {len(dup_gated)}"
        f"{' (' + ', '.join(dup_gated) + ')' if dup_gated else ''}.",
        f"- Non-duplicates, top raw cosine among DUPLICATE-eligible candidates: {_stats(non_dup)}. "
        f"Queries the gate removes from the DUPLICATE step entirely: {non_dup_gated}.",
        f"- Of those, hard negatives + background only (the set rules 1-2 protect): {_stats(rule_set)}.",
        f"- True original ranked first for "
        f"{sum(bool(diag[t['ticket_id']].get('original_is_top')) for t in tickets if t['label_type'] == 'duplicate')}"
        f"/{n_pos} duplicates.",
        "",
        *histogram_lines(dup_raw, non_dup),
        "",
        "### Per duplicate",
        "",
        "| query | original | raw (decision) | ranking score | gate allows | category agrees | "
        f"original ranked first | top candidate | caught at {current:.2f} |"
        + (f" caught at {suggested['dup_threshold']:.2f} (suggestion rule) |" if suggested else ""),
        "|---|---|---|---|---|---|---|---|---|" + ("---|" if suggested else ""),
    ]

    def caught(tid, t):
        r = out.predictions[t][tid]
        return "yes" if r.outcome == "DUPLICATE" and r.duplicate_of == by_id[tid]["duplicate_of"] else (
            f"no ({r.outcome}{' ' + r.duplicate_of if r.duplicate_of else ''})")

    by_id = {t["ticket_id"]: t for t in tickets}
    for t in sorted(dups, key=lambda t: diag[t["ticket_id"]].get("original_raw") or 0):
        d = diag[t["ticket_id"]]
        lines.append(
            f"| {t['ticket_id']} | {t['duplicate_of']} | {d['original_raw']:.4f} | {d['original_score']:.4f} "
            f"| {'yes' if d['original_dup_eligible'] else 'no'} "
            f"| {'yes' if d['original_category_agrees'] else 'no'} | {'yes' if d['original_is_top'] else 'no'} "
            f"| {d['top_id']} ({d['top_score']:.4f}) | {caught(t['ticket_id'], current)} |"
            + (f" {caught(t['ticket_id'], suggested['dup_threshold'])} |" if suggested else "")
        )

    lines += [
        "",
        "### Highest-scoring non-duplicates (DUPLICATE-eligible candidates, raw cosine)",
        "",
        "| query | label | top eligible candidate | candidate label | raw |",
        "|---|---|---|---|---|",
    ]
    top_non = sorted(
        (t for t in non_dups if diag[t["ticket_id"]]["top_dup_raw"] is not None),
        key=lambda t: diag[t["ticket_id"]]["top_dup_raw"], reverse=True,
    )[:15]
    for t in top_non:
        d = diag[t["ticket_id"]]
        lines.append(f"| {t['ticket_id']} | {t['label_type']} | {d['top_dup_id']} "
                     f"| {by_id[d['top_dup_id']]['label_type']} | {d['top_dup_raw']:.4f} |")

    lines += [
        "",
        f"### Thin text (<= {THIN_FIELD_COUNT} non-null Agent 1 field)",
        "",
        "| dup t | thin queries | thin positives | DUP FP total | FP with thin query | FP with thin "
        "matched ticket | FP with either thin | missed positives with thin text |",
        "|---|---|---|---|---|---|---|---|",
    ]
    extra = [r["dup_threshold"] for r in (chosen, suggested) if r]
    for t in sorted({current, *extra}):
        tb = thin_text_breakdown(tickets, cache, out.expected, out.predictions[t])
        lines.append(f"| {t:.2f} | {tb['thin_queries']} | {tb['thin_positives']} | {tb['dup_fp_total']} "
                     f"| {tb['dup_fp_thin_query']} | {tb['dup_fp_thin_match']} | {tb['dup_fp_thin_either']} "
                     f"| {tb['missed_positives_thin']} |")

    shown = sorted({round(0.60 + 0.05 * i, 2) for i in range(8)} | {current} | set(extra))
    lines += [
        "",
        "## 4. Sweep",
        "",
        "Every 0.05 plus the current default and the chosen value; the full 0.01 sweep is in "
        "`results/agent4_live_threshold_sweep.csv`. DUP FP counts every wrong DUPLICATE call (wrong target, "
        "hard negatives, background, cluster members). CLUSTER columns come from the cluster threshold "
        f"({settings.corr_cluster_threshold}, not swept).",
        "",
        *SWEEP_HEADER,
    ]
    for t in shown:
        tags = []
        if chosen and t == chosen["dup_threshold"]:
            tags.append("chosen")
        if suggested and t == suggested["dup_threshold"]:
            tags.append("suggestion rule")
        if t == current:
            tags.append("current")
        lines.append(_fmt_row(by_t[t], f" ({', '.join(tags)})" if tags else ""))

    lines += ["", "## 5. Recommendation", "", "### Strict rule", "",
              "Rules applied mechanically to the 0.01 sweep:", "", *working, ""]
    lines += [
        "Judgements behind the rules:",
        "",
        "- Rules 1-2 count **DUPLICATE-caused** FPs only. INCIDENT_CLUSTER FPs depend on the cluster "
        f"threshold ({settings.corr_cluster_threshold}), which this sweep does not change; they are a "
        "cluster-threshold issue, shown in their own columns and called out below.",
        "- **Cluster members called DUPLICATE** are reported but kept out of rules 1-2: a cluster member "
        "merged into a sibling's ticket is the same incident, a lesser error than merging a different issue.",
        "",
    ]
    if chosen:
        m = margin_at(chosen["dup_threshold"], out)
        hn_val, hn_id = m["highest_rule_set"]
        nd_val, nd_id = m["highest_non_dup"]
        lines += [
            f"4. **Margin at {chosen['dup_threshold']:.2f}:** lowest true-original score still caught = "
            f"{m['lowest_caught']:.4f}; highest top score among hard negatives + background = "
            f"{hn_val:.4f} ({hn_id}); highest among all non-duplicates (cluster members included) = "
            f"{nd_val:.4f} ({nd_id}).",
            "",
            f"**Recommended dup_threshold: {chosen['dup_threshold']:.2f}** (recommendation only; "
            "`corr_dup_threshold` is unchanged).",
        ]
    else:
        lines.append("**No threshold is recommended.**")
        lines += ["", *blocking_lines(out, max(by_t))]

    lines += [
        "",
        "### Suggestion rule",
        "",
        "A DUPLICATE is now a suggestion the user confirms before anything is merged, so a wrong "
        "suggestion costs one extra question rather than a wrong merge. Rule: the HIGHEST threshold "
        f"with duplicate recall >= {SUGGESTION_MIN_RECALL:.2f}.",
        "",
        *suggestion_working,
    ]
    if suggested:
        m = margin_at(suggested["dup_threshold"], out)
        hn_val, hn_id = m["highest_rule_set"]
        lines += [
            "",
            f"Margin at {suggested['dup_threshold']:.2f}: lowest true-original raw score still caught = "
            f"{m['lowest_caught']:.4f}; highest eligible raw score among hard negatives + background = "
            f"{hn_val:.4f} ({hn_id}).",
            "",
            f"**Suggestion-rule threshold: {suggested['dup_threshold']:.2f}** (recommendation only; "
            "`corr_dup_threshold` is unchanged).",
        ]

    cur = by_t[current]
    lines += [
        "",
        "### Cluster-threshold issue (not affected by the duplicate threshold)",
        "",
        f"At the current default, INCIDENT_CLUSTER is called on {cur['hn_cluster_fp_total']} hard negative(s) "
        f"(building {cur['hn_building_cluster_fp']}, time {cur['hn_time_cluster_fp']}, category "
        f"{cur['hn_category_cluster_fp']}, near-miss {cur['hn_near_miss_cluster_fp']}) and "
        f"{cur['background_cluster_fp']} background ticket(s). These need a cluster-threshold sweep on the "
        "live format, which is out of scope here.",
        "",
        "## 6. Comparison with the current default",
        "",
        *SWEEP_HEADER,
        _fmt_row(cur, " (current)"),
    ]
    if chosen:
        lines.append(_fmt_row(chosen, " (chosen)"))
    if suggested and suggested["dup_threshold"] != current:
        lines.append(_fmt_row(suggested, " (suggestion rule)"))
    lines += ["", "Where the wrong DUPLICATE calls come from:", "", "| dup t | reason | count |", "|---|---|---|"]
    for t in sorted({current, *extra}):
        sources = duplicate_error_sources(tickets, out.expected, out.predictions[t])
        for reason, n in sources.items():
            lines.append(f"| {t:.2f} | {reason} | {n} |")
        if not sources:
            lines.append(f"| {t:.2f} | none | 0 |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


async def main():
    if settings.corr_dup_threshold not in DUP_THRESHOLDS:
        raise SystemExit(f"current corr_dup_threshold {settings.corr_dup_threshold} is not on the sweep grid")
    tickets = load_tickets()
    cache = load_cache(tickets)
    out = await run_sweep(tickets, cache, _LazySentenceTransformer())
    chosen, working = recommend(out.rows)
    n_non_dup = sum(t["label_type"] != "duplicate" for t in tickets)
    suggestion = suggestion_rule(out.rows, n_non_dup)
    write_csv(out.rows)
    write_markdown(out, chosen, working, suggestion)
    print(f"Strict rule: {chosen['dup_threshold']:.2f}" if chosen else "Strict rule: no threshold")
    print(f"Suggestion rule: {suggestion[0]['dup_threshold']:.2f}" if suggestion[0] else "Suggestion rule: none")
    print(f"Wrote {CSV_PATH.relative_to(REPO_ROOT)} and {MD_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())
