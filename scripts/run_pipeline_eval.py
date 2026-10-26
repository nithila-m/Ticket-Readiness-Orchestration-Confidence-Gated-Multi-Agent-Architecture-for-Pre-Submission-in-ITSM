"""
Full-pipeline evaluation of TRO on the 150 original tickets, through the
REAL orchestrator (settings.orchestrator, LangGraph by default).

Costs real Groq calls (Agents 2 and 5) - not part of the pytest suite.
Run from the repo root:
    python scripts/run_pipeline_eval.py --dry-run                  # no API calls
    python scripts/run_pipeline_eval.py                            # all 150 tickets
    python scripts/run_pipeline_eval.py --start 1 --end 60         # one slice
    python scripts/run_pipeline_eval.py --start 1 --end 60 --resume
    python scripts/run_pipeline_eval.py --start 1 --end 60 --resume --retry-errors
    python scripts/run_pipeline_eval.py --report-only --inputs a.csv b.csv
    python scripts/run_pipeline_eval.py --report-only --auto   # every slice CSV in results/

What is real and what is not:
  - Agent 1 is REPLAYED from data/eval/agent1_fields_cache.json
    (ReplayExtractionProvider): no Gemini call, and a cache miss is an error.
  - Agents 2 and 5: real Groq, behind an eval-only 429 retry wrapper
    (15 s, 30 s, 60 s, or retry-after when the error carries one).
    Agent 5's timeout is raised to 150 s here (eval only) so a retry wait
    is not turned into a rule fallback.
  - Agents 3 and 4: real (local Chroma + all-MiniLM-L6-v2).
  - Agent 6: local only (NullAdapter, never Jira), in a temporary SQLite
    file that starts EMPTY - the seed pool holds these same tickets and
    would match each one against its own copy.
  - auto_dispatch_on_ready is forced on in this process only.

Per ticket (fixed order, sorted by ticket_id; one new conversation each):
one user message "{subject}. {message}", then a scripted user:
DEFLECTED -> deflection feedback resolved=is_deflectable; a suspected
duplicate -> recorded, answered "not a duplicate"; anything else stops.

Slices: --start/--end (1-based, inclusive) pick part of the fixed order,
so teammates can split the run across Groq accounts. Each slice writes
results/pipeline_eval_results_{start}-{end}.csv (one row per ticket,
appended as soon as it finishes), results/pipeline_eval_{start}-{end}.md,
and uses %TEMP%/tro_pipeline_eval_{start}-{end}.db. The temp DB is kept
when a run is interrupted (so --resume keeps Agent 4's pool) and deleted
once every ticket of the slice is recorded. If Groq's quota runs
out (a 429 after every retry, in Agent 2 or behind Agent 5's rule
fallback), the run stops without recording that ticket and exits 2;
rerun with --resume once the quota resets. --report-only merges slice
CSVs into results/pipeline_eval.md without any API call or model load.
"""

import argparse
import asyncio
import csv
import json
import math
import re
import statistics
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Light imports only: tests, --dry-run and --report-only must not load
# Chroma or sentence-transformers. The agents are imported in build_service().
from app.config.category_profiles import normalize_category_label  # noqa: E402
from app.config.routing import DATASET_PRIORITY_TO_P, team_for_category  # noqa: E402
from app.config.settings import settings  # noqa: E402
from app.providers.base import LLMProvider  # noqa: E402
from app.providers.clarification_base import ClarificationProvider  # noqa: E402
from app.providers.classification_base import ClassificationProvider  # noqa: E402
from app.providers.exceptions import LLMProviderError  # noqa: E402
from app.schemas.conversation import Message  # noqa: E402
from app.schemas.extraction import CategoryPrediction, ExtractedField, ExtractionResult  # noqa: E402
from app.services.transcript_formatter import format_transcript_for_extraction  # noqa: E402
from scripts.agent1_cache_lib import is_quota_error, parse_retry_after, prompt_sha256  # noqa: E402

DATA_DIR = ROOT / "data" / "eval"
RESULTS_DIR = ROOT / "results"
CACHE_PATH = DATA_DIR / "agent1_fields_cache.json"
MERGED_MD = RESULTS_DIR / "pipeline_eval.md"
MERGED_CSV_NAME = "pipeline_eval_results_merged.csv"
EXPECTED_TOTAL = 150

DATASET_FILES = [
    "wifi_internet.json",
    "ms_teams.json",
    "vit_email.json",
    "ad_account_creation.json",
    "printer_support.json",
]

# Same labeling-drift fixes as run_agent5_eval.py.
LABEL_FIX = {
    "AD Account Support": "AD Account Creation",
    "VIT Email": "VIT Email Support",
}

PRIORITY_LABELS = ["P2", "P3", "P4"]          # ground-truth classes
PREDICTED_LABELS = ["P1", "P2", "P3", "P4"]   # P1 possible, never correct here

RETRY_BACKOFF_SECONDS = (15.0, 30.0, 60.0)
AGENT5_TIMEOUT_SECONDS = 150.0   # > 15 + 30 + 60 s of backoff plus the calls
DEFAULT_DELAY_SECONDS = 10.0
EST_GROQ_LATENCY_SECONDS = 2.0   # rough, for the dry-run estimate only
EST_STARTUP_SECONDS = 30.0       # Chroma + sentence-transformers load
LATENCY_STAGES = ["kb", "clarify", "correlate", "classify", "dispatch"]
OUTCOMES = ["SELF_RESOLVED", "DISPATCHED", "MERGED", "PENDING_REVIEW", "ESCALATED",
            "STOPPED_AT_ASK", "STOPPED_AT_RECHECK", "ERROR"]

CSV_FIELDS = [
    "slice", "position", "ticket_id",
    # ground truth
    "gt_category", "gt_priority_label", "gt_priority", "gt_team",
    "is_deflectable", "completeness_level", "duplicate_cluster_id",
    # Agent 1 (replayed) + completeness gate
    "agent1_category", "agent1_category_confidence", "completeness_score", "gate_passed",
    # Agent 3
    "kb_outcome", "kb_similarity", "kb_matched_id",
    # Agent 2 / deflection
    "first_action", "first_action_confidence", "deflected", "deflection_feedback",
    # Agent 4
    "reached_agent4", "correlation_outcome", "correlation_max_similarity",
    "duplicate_suggested", "duplicate_candidate_store_id", "duplicate_candidate_ticket_id",
    "duplicate_similarity", "duplicate_same_cluster", "cluster_id",
    # Agent 5 + confidence gate
    "reached_agent5", "pred_category", "pred_priority", "pred_team",
    "classify_confidence", "classify_source", "rule_fallback", "gate_decision",
    # outcome
    "final_outcome", "store_ticket_id", "review_reason",
    # latency (audit latency_ms)
    "lat_kb_ms", "lat_clarify_ms", "lat_correlate_ms", "lat_classify_ms", "lat_dispatch_ms",
    # run info
    "groq_retries", "groq_model", "classification_model", "error",
]

BOOL_FIELDS = {"is_deflectable", "gate_passed", "deflected", "reached_agent4",
               "duplicate_suggested", "duplicate_same_cluster", "reached_agent5", "rule_fallback"}
FLOAT_FIELDS = {"agent1_category_confidence", "completeness_score", "kb_similarity",
                "first_action_confidence", "correlation_max_similarity", "duplicate_similarity",
                "classify_confidence"} | {f"lat_{s}_ms" for s in LATENCY_STAGES}
INT_FIELDS = {"position", "groq_retries"}


# --- Tickets, slices, ground truth ---------------------------------------------


def load_tickets() -> list[dict]:
    """The 150 original tickets in the fixed order (sorted by ticket_id)."""
    tickets = []
    for filename in DATASET_FILES:
        tickets.extend(json.loads((DATA_DIR / filename).read_text(encoding="utf-8")))
    return sorted(tickets, key=lambda t: t["ticket_id"])


def resolve_slice(total: int, start: int | None, end: int | None) -> tuple[int, int]:
    """1-based inclusive bounds; defaults to the whole list."""
    start = 1 if start is None else start
    end = total if end is None else end
    if not 1 <= start <= end <= total:
        raise ValueError(f"invalid slice --start {start} --end {end} (need 1 <= start <= end <= {total})")
    return start, end


def select_slice(tickets: list[dict], start: int, end: int) -> list[tuple[int, dict]]:
    """(1-based position in the fixed order, ticket) for tickets start..end."""
    return [(pos, t) for pos, t in enumerate(tickets, start=1) if start <= pos <= end]


def slice_label(start: int, end: int) -> str:
    return f"{start}-{end}"


def slice_csv_path(start: int, end: int, results_dir: Path = RESULTS_DIR) -> Path:
    return results_dir / f"pipeline_eval_results_{slice_label(start, end)}.csv"


def slice_md_path(start: int, end: int, results_dir: Path = RESULTS_DIR) -> Path:
    return results_dir / f"pipeline_eval_{slice_label(start, end)}.md"


def slice_db_path(start: int, end: int, temp_dir: Path | None = None) -> Path:
    base = Path(temp_dir) if temp_dir is not None else Path(tempfile.gettempdir())
    return base / f"tro_pipeline_eval_{slice_label(start, end)}.db"


def delete_db(path: Path) -> None:
    for p in (path, path.with_name(path.name + "-wal"), path.with_name(path.name + "-shm")):
        p.unlink(missing_ok=True)


def ground_truth_category(ticket: dict) -> str | None:
    raw = ticket["true_category"]
    return normalize_category_label(LABEL_FIX.get(raw, raw))


def ground_truth(ticket: dict) -> dict:
    category = ground_truth_category(ticket)
    return {
        "ticket_id": ticket["ticket_id"],
        "gt_category": category,
        "gt_priority_label": ticket["priority"],
        "gt_priority": DATASET_PRIORITY_TO_P.get(ticket["priority"]),
        "gt_team": team_for_category(category),
        "is_deflectable": bool(ticket["is_deflectable"]),
        "completeness_level": ticket["completeness_level"],
        "duplicate_cluster_id": ticket.get("duplicate_cluster_id"),
    }


# --- Agent 1 replay ----------------------------------------------------------------


def load_cache(path: Path = CACHE_PATH) -> dict[str, dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def cache_by_input_text(cache: dict[str, dict]) -> dict[str, dict]:
    return {entry["input_text"]: entry for entry in cache.values()}


def cache_key_for(user_message: str) -> str:
    """What Agent 1 receives for a single user turn (the live transcript format)."""
    return format_transcript_for_extraction([Message(role="user", content=user_message)])


def build_user_message(ticket: dict, cached_texts) -> str:
    """
    f"{subject}. {message}", plus the OCR suffix only when the cache was
    built with it - the message must reproduce the cache key exactly.
    """
    base = f"{ticket.get('subject') or ''}. {ticket.get('message') or ''}"
    if ticket.get("ocr_text"):
        with_ocr = f"{base} [Screenshot text: {ticket['ocr_text']}]"
        if cache_key_for(with_ocr) in cached_texts:
            return with_ocr
    return base


class ReplayCacheMiss(LLMProviderError):
    """Agent 1 was asked about a text that is not in the cache."""


class ReplayExtractionProvider(LLMProvider):
    """Agent 1's LLMProvider, answered from the cache by exact input_text."""

    def __init__(self, cache: dict[str, dict]):
        self._by_text = cache_by_input_text(cache)

    async def extract_information(self, issue_text: str) -> ExtractionResult:
        entry = self._by_text.get(issue_text)
        if entry is None:
            raise ReplayCacheMiss(
                f"Agent 1 cache miss (no Gemini fallback): {issue_text[:120]!r}"
            )
        category = entry.get("category")
        return ExtractionResult(
            category=CategoryPrediction(
                value=category,
                confidence=entry.get("category_confidence", 0.0) if category else 0.0,
            ),
            extracted_fields={
                name: ExtractedField(value=f["value"], confidence=f["confidence"])
                for name, f in entry.get("fields", {}).items()
            },
        )


def cache_prompt_mismatches(cache: dict[str, dict], current_hash: str | None = None) -> int:
    current_hash = current_hash or prompt_sha256()
    return sum(1 for e in cache.values() if e.get("prompt_sha256") != current_hash)


# --- Groq 429 retry (eval only) ----------------------------------------------------


@dataclass
class RetryCounter:
    retries: int = 0
    quota_exhausted: bool = False   # a rate limit outlasted every retry

    def reset(self) -> None:
        self.retries = 0
        self.quota_exhausted = False


def _rate_limited(exc: Exception) -> bool:
    cause = getattr(exc, "cause", None)
    return getattr(cause, "status_code", None) == 429 or is_quota_error(exc)


def _retry_after(exc: Exception) -> float | None:
    response = getattr(getattr(exc, "cause", None), "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        value = headers.get("retry-after")
        try:
            return float(value) if value is not None else None
        except ValueError:
            pass
    return parse_retry_after(exc)


async def call_with_retry(fn, counter: RetryCounter, label: str, sleep=asyncio.sleep, log=print):
    for attempt in range(len(RETRY_BACKOFF_SECONDS) + 1):
        try:
            return await fn()
        except LLMProviderError as exc:
            if not _rate_limited(exc):
                raise
            if attempt == len(RETRY_BACKOFF_SECONDS):
                # Agent 5 swallows this into a rule fallback, so the run
                # loop checks the flag instead of relying on the exception.
                counter.quota_exhausted = True
                raise
            wait = max(RETRY_BACKOFF_SECONDS[attempt], _retry_after(exc) or 0.0)
            counter.retries += 1
            log(f"    -> Groq 429 ({label}), retry {attempt + 1}/{len(RETRY_BACKOFF_SECONDS)} in {wait:.0f}s")
            await sleep(wait)
    raise AssertionError("unreachable")


class RetryingClarificationProvider(ClarificationProvider):
    def __init__(self, inner: ClarificationProvider, counter: RetryCounter, sleep=asyncio.sleep):
        self._inner, self._counter, self._sleep = inner, counter, sleep

    async def decide_clarification(self, state):
        return await call_with_retry(
            lambda: self._inner.decide_clarification(state), self._counter, "Agent 2", self._sleep
        )


class RetryingClassificationProvider(ClassificationProvider):
    def __init__(self, inner: ClassificationProvider, counter: RetryCounter, sleep=asyncio.sleep):
        self._inner, self._counter, self._sleep = inner, counter, sleep

    async def classify_ticket(self, state):
        return await call_with_retry(
            lambda: self._inner.classify_ticket(state), self._counter, "Agent 5", self._sleep
        )


# --- Service (built like app/dependencies.py, with injected parts) ----------------


@dataclass
class EvalService:
    service: object
    store: object
    repository: object
    counter: RetryCounter


def build_service(db_path: Path, cache: dict[str, dict]) -> EvalService:
    # Heavy imports here so tests / --dry-run / --report-only never load them.
    from app.agents.adaptive_clarifier import AdaptiveClarifier
    from app.agents.classification_agent import ClassificationAgent
    from app.agents.correlation_engine import CorrelationEngine
    from app.agents.dispatcher import Dispatcher
    from app.agents.information_extractor import InformationExtractor
    from app.agents.kb_deflection_agent import KBDeflectionAgent
    from app.integrations.base import NullAdapter
    from app.orchestration import GraphConversationService
    from app.providers.groq_classification_provider import GroqClassificationProvider
    from app.providers.groq_clarification_provider import GroqClarificationProvider
    from app.repositories.in_memory_conversation_repository import InMemoryConversationRepository
    from app.services.clarification_service import ClarificationService
    from app.services.ticket_analysis_service import TicketAnalysisService
    from app.services.ticket_pipeline import TicketPipeline
    from app.storage.sqlite_store import SqliteStore

    # This run only; .env and the settings default are untouched.
    settings.auto_dispatch_on_ready = True

    counter = RetryCounter()
    store = SqliteStore(db_path)
    store.init_db()   # empty: no seed pool

    extraction_service = TicketAnalysisService(InformationExtractor(ReplayExtractionProvider(cache)))
    clarifier = AdaptiveClarifier(
        provider=RetryingClarificationProvider(
            GroqClarificationProvider(api_key=settings.groq_api_key, model=settings.groq_model), counter
        ),
        max_turns=settings.max_clarification_turns,
    )
    classifier = ClassificationAgent(
        RetryingClassificationProvider(
            GroqClassificationProvider(
                api_key=settings.groq_api_key, model=settings.classification_groq_model
            ),
            counter,
        ),
        timeout_seconds=AGENT5_TIMEOUT_SECONDS,
    )
    pipeline = TicketPipeline(
        correlation_agent=CorrelationEngine(),
        pool=store,
        classifier=classifier,
        dispatcher=Dispatcher(store, NullAdapter()),
        store=store,
        confidence_threshold=settings.classification_confidence_threshold,
    )
    repository = InMemoryConversationRepository()
    service_cls = (
        GraphConversationService if settings.orchestrator == "langgraph" else ClarificationService
    )
    service = service_cls(
        extraction_service, clarifier, repository, KBDeflectionAgent(),
        ticket_pipeline=pipeline, store=store,
    )
    return EvalService(service, store, repository, counter)


# --- Scripted user -----------------------------------------------------------------


def scripted_step(state, first_action: str | None, ticket: dict) -> tuple[str, bool | None]:
    """
    The scripted user's next move, applied mechanically:
      ("duplicate_feedback", False)  a suspected duplicate: answer "no"
      ("deflection_feedback", bool)  a KB deflection: resolved = is_deflectable
      ("stop", None)                 terminal outcome, a question, or anything else
    """
    if state is None or state.terminal_outcome is not None:
        return "stop", None
    if state.awaiting_duplicate_confirmation:
        return "duplicate_feedback", False
    if first_action == "DEFLECTED" and state.kb_feedback is None:
        return "deflection_feedback", bool(ticket["is_deflectable"])
    return "stop", None


def _final_outcome(state, first_action: str | None) -> str:
    if state is not None and state.terminal_outcome is not None:
        return state.terminal_outcome
    if first_action == "ASK_CLARIFICATION":
        return "STOPPED_AT_ASK"
    return f"STOPPED_AT_{first_action}" if first_action else "ERROR"


def _first(events, stage):
    return next((e for e in events if e.stage == stage), None)


def build_row(ticket: dict, position: int, slice_name: str, state, events,
              first_action: str | None, dup: dict, store_to_ticket: dict[str, str],
              cluster_by_ticket: dict[str, str | None], retries: int, error: str | None) -> dict:
    row = {k: None for k in CSV_FIELDS}
    row.update(ground_truth(ticket))
    row.update({"slice": slice_name, "position": position, "groq_retries": retries,
                "groq_model": settings.groq_model,
                "classification_model": settings.classification_groq_model, "error": error})

    extract, kb, clarify = _first(events, "extract"), _first(events, "kb"), _first(events, "clarify")
    correlate, classify = _first(events, "correlate"), _first(events, "classify")
    gate, dispatch = _first(events, "confidence_gate"), _first(events, "dispatch")

    if state is not None and state.detected_category is not None:
        row["agent1_category"] = state.detected_category.value
        row["agent1_category_confidence"] = state.detected_category.confidence
    if extract is not None:
        row["completeness_score"] = extract.confidence
    row["gate_passed"] = kb is not None
    if kb is not None:
        row["kb_outcome"] = kb.decision
        row["kb_similarity"] = kb.confidence
        row["kb_matched_id"] = kb.payload.get("matched_kb_id")
        row["lat_kb_ms"] = kb.latency_ms

    row["first_action"] = first_action
    if clarify is not None:
        row["first_action_confidence"] = clarify.confidence
        # A DEFLECTED decision is audited under "clarify" too, but Agent 2 did not run.
        if clarify.decision != "DEFLECTED":
            row["lat_clarify_ms"] = clarify.latency_ms
    row["deflected"] = first_action == "DEFLECTED"
    if state is not None:
        row["deflection_feedback"] = state.kb_feedback

    row["reached_agent4"] = correlate is not None
    if correlate is not None:
        row["correlation_outcome"] = correlate.decision
        row["correlation_max_similarity"] = correlate.confidence
        row["cluster_id"] = correlate.payload.get("cluster_id")
        row["lat_correlate_ms"] = correlate.latency_ms
    row["duplicate_suggested"] = bool(dup)
    if dup:
        store_id = dup.get("store_id")
        candidate = store_to_ticket.get(store_id) if store_id else None
        row["duplicate_candidate_store_id"] = store_id
        row["duplicate_candidate_ticket_id"] = candidate
        row["duplicate_similarity"] = dup.get("similarity")
        own, other = ticket.get("duplicate_cluster_id"), cluster_by_ticket.get(candidate)
        row["duplicate_same_cluster"] = bool(own and other and own == other)

    row["reached_agent5"] = classify is not None
    if classify is not None:
        row["pred_category"] = classify.decision
        row["pred_priority"] = classify.payload.get("priority")
        row["pred_team"] = classify.payload.get("assignee_group")
        row["classify_confidence"] = classify.confidence
        row["classify_source"] = classify.payload.get("source")
        row["rule_fallback"] = classify.payload.get("source") == "rule"
        row["lat_classify_ms"] = classify.latency_ms
    if gate is not None:
        row["gate_decision"] = gate.decision
    if dispatch is not None:
        row["lat_dispatch_ms"] = dispatch.latency_ms

    if state is not None:
        row["store_ticket_id"] = state.ticket.ticket_id if state.ticket else None
        row["review_reason"] = state.review_reason
    row["final_outcome"] = "ERROR" if error else _final_outcome(state, first_action)
    return row


def _candidate_similarity(correlation) -> float | None:
    if correlation is None:
        return None
    for similar in correlation.similar_tickets:
        if similar.ticket_id == correlation.duplicate_of:
            return similar.similarity
    return correlation.max_similarity


async def run_ticket(ev: EvalService, ticket: dict, position: int, slice_name: str,
                     cached_texts, store_to_ticket: dict[str, str],
                     cluster_by_ticket: dict[str, str | None]) -> dict:
    # Unique per attempt: a resumed or retried ticket must not read the audit
    # rows an earlier, unrecorded attempt left in the kept temp DB.
    cid = f"eval-{ticket['ticket_id']}-{uuid.uuid4().hex[:8]}"
    ev.counter.reset()
    state, first_action, dup, error = None, None, {}, None
    try:
        message = build_user_message(ticket, cached_texts)
        state, decision = await ev.service.handle_message(cid, message)
        first_action = decision.action
        for _ in range(4):  # at most: deflection feedback, then a duplicate answer
            step, value = scripted_step(state, first_action, ticket)
            if step == "stop":
                break
            if step == "deflection_feedback":
                state = await ev.service.handle_deflection_feedback(cid, value)
            else:
                dup = {
                    "store_id": state.duplicate_candidate.ticket_id if state.duplicate_candidate else None,
                    "similarity": _candidate_similarity(state.correlation),
                }
                state = await ev.service.handle_duplicate_feedback(cid, value)
    except Exception as exc:  # noqa: BLE001 - record and continue with the next ticket
        error = f"{type(exc).__name__}: {exc}"
        state = await ev.repository.get(cid) or state
    events = await ev.store.fetch_audit(cid)
    return build_row(ticket, position, slice_name, state, events, first_action, dup,
                     store_to_ticket, cluster_by_ticket, ev.counter.retries, error)


# --- CSV ---------------------------------------------------------------------------


def _to_csv(value):
    return "" if value is None else value


def append_row(path: Path, row: dict) -> None:
    new = not path.exists() or path.stat().st_size == 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({k: _to_csv(row.get(k)) for k in CSV_FIELDS})


def normalize_row(raw: dict) -> dict:
    row = {}
    for key in CSV_FIELDS:
        value = raw.get(key)
        if value == "" or value is None:
            row[key] = None
        elif key in BOOL_FIELDS:
            row[key] = value if isinstance(value, bool) else value == "True"
        elif key in FLOAT_FIELDS:
            row[key] = float(value)
        elif key in INT_FIELDS:
            row[key] = int(float(value))
        else:
            row[key] = value
    return row


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [normalize_row(r) for r in csv.DictReader(f)]


def load_done_ids(path: Path) -> set[str]:
    return {r["ticket_id"] for r in read_rows(path)}


def drop_error_rows(path: Path) -> list[str]:
    """Rewrite the CSV without its ERROR rows; returns the dropped ticket_ids."""
    rows = read_rows(path)
    dropped = [r["ticket_id"] for r in rows if r["final_outcome"] == "ERROR"]
    if dropped:
        path.unlink()
        for r in rows:
            if r["final_outcome"] != "ERROR":
                append_row(path, r)
    return dropped


def pending_tickets(selected: list[tuple[int, dict]], done: set[str]) -> list[tuple[int, dict]]:
    return [(pos, t) for pos, t in selected if t["ticket_id"] not in done]


class DuplicateTicketError(ValueError):
    pass


def merge_csvs(paths: list[Path]) -> list[dict]:
    """All rows of all inputs; a ticket_id in two inputs (or twice in one) is an error."""
    rows, seen = [], {}
    for path in paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"input CSV not found: {path}")
        for row in read_rows(path):
            tid = row["ticket_id"]
            if tid in seen:
                raise DuplicateTicketError(
                    f"ticket_id {tid} appears in both {seen[tid]} and {path.name}; "
                    "slices must not overlap"
                )
            seen[tid] = path.name
            rows.append(row)
    return sorted(rows, key=lambda r: (r["position"] or 0, r["ticket_id"]))


# --- Metrics -----------------------------------------------------------------------


def _pct(n: int, d: int) -> str:
    return f"{n} ({n / d:.1%})" if d else f"{n} (n/a)"


def _ok(rows):
    return [r for r in rows if not r["error"]]


def funnel(rows: list[dict]) -> dict:
    ok = _ok(rows)
    gate = [r for r in ok if r["gate_passed"]]
    agent2 = [r for r in ok if r["first_action"] and r["first_action"] != "DEFLECTED"]
    actions = {a: sum(1 for r in agent2 if r["first_action"] == a)
               for a in ("ASK_CLARIFICATION", "READY", "ESCALATE")}
    actions["other"] = len(agent2) - sum(actions.values())
    deflected = [r for r in ok if r["deflected"]]
    return {
        "n": len(rows),
        "n_ok": len(ok),
        "n_error": len(rows) - len(ok),
        "gate_passed": len(gate),
        "kb": {o: sum(1 for r in gate if r["kb_outcome"] == o)
               for o in ("STRONG_MATCH", "WEAK_MATCH", "NO_MATCH")},
        "n_agent2": len(agent2),
        "agent2_actions": actions,
        "deflected": len(deflected),
        "deflected_resolved": sum(1 for r in deflected if r["deflection_feedback"] == "RESOLVED"),
        "deflected_not_resolved": sum(1 for r in deflected if r["deflection_feedback"] == "NOT_RESOLVED"),
        "reached_agent4": sum(1 for r in ok if r["reached_agent4"]),
        "reached_agent5": sum(1 for r in ok if r["reached_agent5"]),
        "outcomes": {o: sum(1 for r in rows if r["final_outcome"] == o) for o in OUTCOMES},
    }


def agent2_crosstab(rows: list[dict]) -> tuple[list[str], dict[str, dict[str, int]]]:
    cols = ["ASK_CLARIFICATION", "READY", "ESCALATE", "RECHECK", "DEFLECTED"]
    table = {lvl: {c: 0 for c in cols} for lvl in ("complete", "partial", "vague")}
    for r in _ok(rows):
        if r["completeness_level"] in table and r["first_action"] in cols:
            table[r["completeness_level"]][r["first_action"]] += 1
    return cols, table


def kb_prf(rows: list[dict]) -> dict:
    """Deflected = STRONG_MATCH; positive label = is_deflectable. All non-error tickets."""
    tp = fp = fn = tn = 0
    for r in _ok(rows):
        pred, gold = r["kb_outcome"] == "STRONG_MATCH", bool(r["is_deflectable"])
        if pred and gold:
            tp += 1
        elif pred:
            fp += 1
        elif gold:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1}


def priority_confusion(rows: list[dict]) -> dict[str, dict[str, int]]:
    matrix = {gt: {p: 0 for p in PREDICTED_LABELS} for gt in PRIORITY_LABELS}
    for r in rows:
        if r["gt_priority"] in matrix and r["pred_priority"] in PREDICTED_LABELS:
            matrix[r["gt_priority"]][r["pred_priority"]] += 1
    return matrix


def macro_f1(matrix: dict[str, dict[str, int]]) -> float:
    f1s = []
    for label in PRIORITY_LABELS:
        tp = matrix[label][label]
        fn = sum(matrix[label].values()) - tp
        fp = sum(matrix[gt][label] for gt in PRIORITY_LABELS) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(f1s) / len(f1s)


def classification_metrics(rows: list[dict]) -> dict:
    scored = [r for r in _ok(rows) if r["reached_agent5"]]
    n = len(scored)

    def rate(pred):
        return sum(1 for r in scored if pred(r)) / n if n else 0.0

    matrix = priority_confusion(scored)
    return {
        "n": n,
        "category_accuracy": rate(lambda r: r["pred_category"] == r["gt_category"]),
        "routing_accuracy": rate(lambda r: r["pred_team"] == r["gt_team"]),
        "priority_accuracy": rate(lambda r: r["pred_priority"] == r["gt_priority"]),
        "macro_f1": macro_f1(matrix),
        "matrix": matrix,
        "rule_fallback_rate": rate(lambda r: r["rule_fallback"]),
        "review_rate": rate(lambda r: r["gate_decision"] == "REVIEW"),
        "n_p1": sum(1 for r in scored if r["pred_priority"] == "P1"),
    }


def duplicate_report(rows: list[dict]) -> dict:
    ok = _ok(rows)
    suggested = [r for r in ok if r["duplicate_suggested"]]
    clusters = [r for r in ok if r["correlation_outcome"] == "INCIDENT_CLUSTER"]
    return {
        "suggested": suggested,
        "n_same_cluster": sum(1 for r in suggested if r["duplicate_same_cluster"]),
        "n_incident_cluster": len(clusters),
        "n_forced_p1": sum(1 for r in clusters if r["pred_priority"] == "P1"),
        "n_with_gt_cluster": sum(1 for r in ok if r["duplicate_cluster_id"]),
    }


def _nearest_rank(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[max(1, math.ceil(q * len(ordered))) - 1]


def latency_stats(rows: list[dict]) -> dict:
    """Median / p95 / max per stage; tickets with Groq retries are excluded."""
    ok = _ok(rows)
    clean = [r for r in ok if not r["groq_retries"]]
    stages = {}
    for stage in LATENCY_STAGES:
        values = [r[f"lat_{stage}_ms"] for r in clean if r[f"lat_{stage}_ms"] is not None]
        stages[stage] = (
            {"n": len(values), "median": statistics.median(values),
             "p95": _nearest_rank(values, 0.95), "max": max(values)}
            if values else {"n": 0, "median": None, "p95": None, "max": None}
        )
    return {"excluded_retried": len(ok) - len(clean), "stages": stages}


def slice_summary(rows: list[dict]) -> list[dict]:
    out = {}
    for r in rows:
        s = out.setdefault(r["slice"], {"slice": r["slice"], "n": 0, "positions": [], "ids": []})
        s["n"] += 1
        s["positions"].append(r["position"] or 0)
        s["ids"].append(r["ticket_id"])

    def key(s):
        m = re.match(r"(\d+)-(\d+)", s["slice"] or "")
        return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

    return [
        {"slice": s["slice"], "n": s["n"], "first_id": min(s["ids"]), "last_id": max(s["ids"]),
         "min_pos": min(s["positions"]), "max_pos": max(s["positions"])}
        for s in sorted(out.values(), key=key)
    ]


# --- Markdown ----------------------------------------------------------------------

WITHIN_SLICE_NOTE = (
    "**Duplicates were only compared within a slice.** Each slice started with "
    "its own empty pool, so a duplicate whose earlier copy was processed in a "
    "different slice is not detected; cross-slice duplicates are not counted."
)


def _fmt_ms(v):
    return "-" if v is None else f"{v:.0f}"


def _latency_table(stats: dict) -> list[str]:
    lines = ["| Stage | n | Median ms | p95 ms | Max ms |", "|---|---|---|---|---|"]
    for stage, s in stats["stages"].items():
        lines.append(f"| {stage} | {s['n']} | {_fmt_ms(s['median'])} | {_fmt_ms(s['p95'])} | {_fmt_ms(s['max'])} |")
    return lines


def render_markdown(rows: list[dict], merged: bool) -> str:
    slices = slice_summary(rows)
    f = funnel(rows)
    n_total = f["n"]
    models = sorted({r["groq_model"] for r in rows if r["groq_model"]})
    cls_models = sorted({r["classification_model"] for r in rows if r["classification_model"]})

    lines = [
        "# Full-Pipeline Evaluation" + (" (merged)" if merged else f" (slice {slices[0]['slice']})" if slices else ""),
        "",
        f"Orchestrator: `{settings.orchestrator}`. Agent 2 model: {', '.join(f'`{m}`' for m in models) or '-'}; "
        f"Agent 5 model: {', '.join(f'`{m}`' for m in cls_models) or '-'} (Groq). "
        "Agent 1: replayed from `data/eval/agent1_fields_cache.json`.",
        "",
        "## Limitations",
        "",
        "- **Synthetic labels.** Categories, priorities, completeness levels, deflectability "
        "and duplicate clusters were set by the data generator, not independent annotators.",
        "- **Single run.** LLM output (Agents 2 and 5) is non-deterministic; one sample, not a stable estimate.",
        "- **Agent 1 replayed** from one cached Gemini run, so its latency is not measured here "
        "(see the Agent 1 eval, `results/agent1_eval_results.csv`).",
        "- **Single user turn.** A ticket that gets a clarifying question is stopped there "
        "(STOPPED_AT_ASK); the follow-up turns are not simulated.",
        "- **Empty starting pool.** Agent 4 only sees tickets dispatched earlier in the same run, "
        "so duplicates can only come from earlier tickets in the fixed order.",
        "- **Scripted deflection feedback** uses the dataset label (`is_deflectable`), not a real user.",
        f"- **Agent 5 timeout raised to {AGENT5_TIMEOUT_SECONDS:g} s** (eval only; production default 20 s) so "
        "the 429 retry wait (15/30/60 s) is not turned into a rule fallback.",
        "- **auto_dispatch_on_ready forced on** for this run; Jira is never called (NullAdapter).",
        "- **No P1 ground truth.** The dataset has no P1 label; any P1 prediction counts as wrong.",
    ]
    if merged:
        lines.append(f"- {WITHIN_SLICE_NOTE}")
    lines += [
        f"- Tickets that errored ({f['n_error']}) are excluded from every metric except the funnel "
        "outcome counts and the Errors section.",
        "",
        "## Slices",
        "",
        "| Slice | Ticket range (fixed order) | First id | Last id | n |",
        "|---|---|---|---|---|",
    ]
    for s in slices:
        lines.append(f"| {s['slice']} | {s['min_pos']}-{s['max_pos']} | {s['first_id']} | {s['last_id']} | {s['n']} |")
    lines += ["", f"Total tickets: {n_total}", ""]

    ok_n = f["n_ok"]
    a2 = f["agent2_actions"]
    lines += [
        "## 1. Funnel",
        "",
        f"Percentages are of the {ok_n} tickets without errors unless stated.",
        "",
        "| Stage | Count |",
        "|---|---|",
        f"| Completeness gate passed (reached Agent 3) | {_pct(f['gate_passed'], ok_n)} |",
        f"| KB STRONG_MATCH (% of gate passed) | {_pct(f['kb']['STRONG_MATCH'], f['gate_passed'])} |",
        f"| KB WEAK_MATCH (% of gate passed) | {_pct(f['kb']['WEAK_MATCH'], f['gate_passed'])} |",
        f"| KB NO_MATCH (% of gate passed) | {_pct(f['kb']['NO_MATCH'], f['gate_passed'])} |",
        f"| Agent 2 ran on the first turn | {_pct(f['n_agent2'], ok_n)} |",
        f"| - ASK_CLARIFICATION (% of Agent 2) | {_pct(a2['ASK_CLARIFICATION'], f['n_agent2'])} |",
        f"| - READY (% of Agent 2) | {_pct(a2['READY'], f['n_agent2'])} |",
        f"| - ESCALATE (% of Agent 2) | {_pct(a2['ESCALATE'], f['n_agent2'])} |",
        f"| - other, e.g. RECHECK (% of Agent 2) | {_pct(a2['other'], f['n_agent2'])} |",
        f"| DEFLECTED | {_pct(f['deflected'], ok_n)} |",
        f"| - feedback resolved / not resolved | {f['deflected_resolved']} / {f['deflected_not_resolved']} |",
        f"| Reached Agent 4 | {_pct(f['reached_agent4'], ok_n)} |",
        f"| Reached Agent 5 | {_pct(f['reached_agent5'], ok_n)} |",
        "",
        f"Final outcomes (% of all {n_total} tickets):",
        "",
        "| Outcome | Count |",
        "|---|---|",
    ]
    for o, c in f["outcomes"].items():
        lines.append(f"| {o} | {_pct(c, n_total)} |")

    cols, table = agent2_crosstab(rows)
    lines += [
        "",
        "## 2. First-turn action vs completeness label",
        "",
        "Rows = dataset completeness_level, columns = first-turn decision (DEFLECTED means "
        "Agent 3 short-circuited Agent 2). ASK on `complete` tickets is over-asking.",
        "",
        "| Level | " + " | ".join(cols) + " | Total |",
        "|---|" + "---|" * (len(cols) + 1),
    ]
    for lvl, counts in table.items():
        lines.append(f"| {lvl} | " + " | ".join(str(counts[c]) for c in cols) + f" | {sum(counts.values())} |")

    kb = kb_prf(rows)
    lines += [
        "",
        "## 3. KB deflection vs is_deflectable",
        "",
        "Deflected = KB STRONG_MATCH. Tickets below the completeness gate never reach Agent 3 "
        "and count as not deflected.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Precision | {kb['precision']:.3f} |",
        f"| Recall | {kb['recall']:.3f} |",
        f"| F1 | {kb['f1']:.3f} |",
        f"| TP / FP / FN / TN | {kb['tp']} / {kb['fp']} / {kb['fn']} / {kb['tn']} |",
    ]

    c = classification_metrics(rows)
    lines += [
        "",
        f"## 4. Classification (tickets that reached Agent 5, n={c['n']})",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Category accuracy | {c['category_accuracy']:.1%} |",
        f"| Routing (team) accuracy | {c['routing_accuracy']:.1%} |",
        f"| Priority accuracy | {c['priority_accuracy']:.1%} |",
        f"| Priority macro-F1 (P2/P3/P4) | {c['macro_f1']:.3f} |",
        f"| Rule-fallback rate | {c['rule_fallback_rate']:.1%} |",
        f"| Confidence-gate review rate (REVIEW) | {c['review_rate']:.1%} |",
        f"| P1 predictions (counted wrong) | {c['n_p1']} |",
        "",
        "Priority confusion matrix (rows = ground truth, columns = predicted):",
        "",
        "| GT \\ Pred | " + " | ".join(PREDICTED_LABELS) + " |",
        "|---|" + "---|" * len(PREDICTED_LABELS),
    ]
    for gt in PRIORITY_LABELS:
        lines.append(f"| {gt} | " + " | ".join(str(c["matrix"][gt][p]) for p in PREDICTED_LABELS) + " |")

    d = duplicate_report(rows)
    lines += [
        "",
        "## 5. Agent 4",
        "",
        f"- DUPLICATE_SUGGESTED: {len(d['suggested'])} (answered \"not a duplicate\" by the scripted user); "
        f"{d['n_same_cluster']} share a dataset duplicate_cluster_id with the candidate.",
        f"- Tickets with a dataset duplicate_cluster_id: {d['n_with_gt_cluster']}.",
        f"- INCIDENT_CLUSTER results: {d['n_incident_cluster']}, of which forced P1: {d['n_forced_p1']}. "
        "(A cluster needs a known building on the stored tickets, which comes only from an extracted "
        "building/location field.)",
    ]
    if merged:
        lines.append(f"- {WITHIN_SLICE_NOTE}")
    if d["suggested"]:
        lines += ["", "| Slice | Ticket | Candidate | Similarity | Same cluster |", "|---|---|---|---|---|"]
        for r in d["suggested"]:
            sim = "-" if r["duplicate_similarity"] is None else f"{r['duplicate_similarity']:.3f}"
            cand = r["duplicate_candidate_ticket_id"] or r["duplicate_candidate_store_id"] or "-"
            lines.append(f"| {r['slice']} | {r['ticket_id']} | {cand} | {sim} | "
                         f"{'yes' if r['duplicate_same_cluster'] else 'no'} |")

    lines += ["", "## 6. Latency per stage (audit latency_ms)", ""]
    if merged and len(slices) > 1:
        for s in slices:
            stats = latency_stats([r for r in rows if r["slice"] == s["slice"]])
            lines += [f"### Slice {s['slice']}", "",
                      f"Excluded (Groq retries): {stats['excluded_retried']}.", ""]
            lines += _latency_table(stats) + [""]
        stats = latency_stats(rows)
        lines += ["### Combined (mixed-machine: slices ran on different machines and Groq accounts)", "",
                  f"Excluded (Groq retries): {stats['excluded_retried']}.", ""]
        lines += _latency_table(stats)
    else:
        stats = latency_stats(rows)
        lines += [f"Excluded (Groq retries): {stats['excluded_retried']}. The clarify stage counts only "
                  "tickets where Agent 2 actually ran (not DEFLECTED). Groq SDK-internal retries, if "
                  "any, are inside these numbers.", ""]
        lines += _latency_table(stats)

    errors = [r for r in rows if r["error"]]
    lines += ["", f"## 7. Errors ({len(errors)})", ""]
    if errors:
        lines += ["| Slice | Ticket | Error |", "|---|---|---|"]
        for r in errors:
            msg = str(r["error"]).replace("|", "\\|").replace("\n", " ")[:300]
            lines.append(f"| {r['slice']} | {r['ticket_id']} | {msg} |")
    else:
        lines.append("None.")
    lines.append("")
    return "\n".join(lines)


def write_markdown(rows: list[dict], path: Path, merged: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(rows, merged), encoding="utf-8")


# --- Dry run -----------------------------------------------------------------------


def dry_run(selected: list[tuple[int, dict]], cache: dict[str, dict],
            delay: float, start: int, end: int, total: int) -> int:
    by_text = cache_by_input_text(cache)
    misses = []
    gate_passed = 0
    for _, t in selected:
        key = cache_key_for(build_user_message(t, by_text))
        entry = by_text.get(key)
        if entry is None:
            misses.append(t["ticket_id"])
        elif (entry.get("completeness_score") or 0.0) >= settings.kb_gate_completeness_threshold:
            gate_passed += 1
    n = len(selected)
    mismatched = cache_prompt_mismatches(cache)
    print(f"Slice {slice_label(start, end)}: {n} tickets "
          f"({selected[0][1]['ticket_id']} .. {selected[-1][1]['ticket_id']})" if n else "Slice is empty.")
    print(f"Cache hits: {n - len(misses)}/{n}" + (f"  MISSES: {', '.join(misses)}" if misses else ""))
    if mismatched:
        print(f"WARNING: {mismatched} cache entries have a prompt_sha256 that differs from the "
              "current EXTRACTION_SYSTEM_PROMPT.")
    ocr = sum(1 for _, t in selected if t.get("ocr_text"))
    print(f"Tickets with ocr_text: {ocr} (the cache keys include OCR text for "
          f"{sum(1 for _, t in selected if t.get('ocr_text') and '[Screenshot text:' in build_user_message(t, by_text))})")
    # Agent 2 runs on every ticket that is not deflected; Agent 5 only after
    # READY or an unresolved deflection - both at most once per ticket here.
    expected = n + gate_passed
    upper = 2 * n
    print(f"Completeness gate passed (cached scores): {gate_passed}/{n}")
    print(f"Estimated Groq calls: ~{expected} (Agent 2 <= {n}; Agent 5 ~{gate_passed}, one per ticket "
          f"past the completeness gate), "
          f"upper bound {upper}, plus any 429 retries")
    seconds = max(n - 1, 0) * delay + expected * EST_GROQ_LATENCY_SECONDS + EST_STARTUP_SECONDS
    print(f"Estimated run time: ~{seconds / 60:.0f} min (delay {delay:g}s between tickets)")
    print(f"CSV: {slice_csv_path(start, end)}\nTemp DB: {slice_db_path(start, end)}")

    py = "python scripts/run_pipeline_eval.py"
    print("\nTwo-person split:")
    print(f"  Person 1: {py} --start 1 --end 60")
    print(f"  Person 2: {py} --start 61 --end 120")
    if total > 120:
        print(f"  NOTE: 1-60 and 61-120 cover 120 of {total} tickets; the rest needs "
              f"{py} --start 121 --end {total}")
    print("  (add --resume to continue an interrupted slice)")
    inputs = [slice_csv_path(1, 60), slice_csv_path(61, 120)]
    if total > 120:
        inputs.append(slice_csv_path(121, total))
    rel = " ".join(str(p.relative_to(ROOT)).replace("\\", "/") for p in inputs)
    print(f"  Merge:    {py} --report-only --inputs {rel}")
    return 1 if misses else 0


# --- Main --------------------------------------------------------------------------


def _progress(pos: int, total: int, row: dict) -> str:
    first = row["first_action"] or "-"
    first = "ASK" if first == "ASK_CLARIFICATION" else first
    detail = ""
    if row["pred_priority"]:
        detail = f" ({row['pred_priority']}, {row['pred_team']})"
    if row["duplicate_suggested"]:
        detail += f" [dup? {row['duplicate_candidate_ticket_id'] or row['duplicate_candidate_store_id']}]"
    if row["groq_retries"]:
        detail += f" [{row['groq_retries']} retries]"
    if row["error"]:
        detail += f" ERROR: {row['error'][:120]}"
    return f"[{pos}/{total}] {row['ticket_id']} {first} -> {row['final_outcome']}{detail}"


QUOTA_EXIT_CODE = 2


async def run(args, *, build=build_service, tickets: list[dict] | None = None,
              cache: dict[str, dict] | None = None, results_dir: Path = RESULTS_DIR,
              temp_dir: Path | None = None) -> int:
    tickets = load_tickets() if tickets is None else tickets
    total = len(tickets)
    start, end = resolve_slice(total, args.start, args.end)
    selected = select_slice(tickets, start, end)
    cache = load_cache() if cache is None else cache

    if args.dry_run:
        return dry_run(selected if not args.limit else selected[: args.limit],
                       cache, args.delay, start, end, total)
    if args.retry_errors and not args.resume:
        print("ERROR: --retry-errors needs --resume")
        return 2

    mismatched = cache_prompt_mismatches(cache)
    if mismatched:
        print(f"WARNING: {mismatched} Agent 1 cache entries were built with a different "
              "EXTRACTION_SYSTEM_PROMPT (prompt_sha256 mismatch). Replaying anyway.")

    name = slice_label(start, end)
    csv_path = slice_csv_path(start, end, results_dir)
    md_path = slice_md_path(start, end, results_dir)
    db_path = slice_db_path(start, end, temp_dir)
    if args.resume:
        if args.retry_errors:
            dropped = drop_error_rows(csv_path)
            print(f"--retry-errors: re-running {len(dropped)} ERROR ticket(s)"
                  + (f": {', '.join(dropped)}" if dropped else ""))
        done = load_done_ids(csv_path)
        if done and not db_path.exists():
            print(f"WARNING: {db_path} is missing; Agent 4's pool restarts empty, so tickets "
                  "already recorded cannot be found as duplicates.")
    else:
        csv_path.unlink(missing_ok=True)
        delete_db(db_path)
        done = set()

    todo = pending_tickets(selected, done)
    if args.limit:
        todo = todo[: args.limit]
    print(f"Slice {name}: {len(selected)} tickets, {len(done)} already recorded, {len(todo)} to run. "
          f"Delay {args.delay:g}s between tickets.")

    previous = read_rows(csv_path)
    store_to_ticket = {r["store_ticket_id"]: r["ticket_id"] for r in previous if r["store_ticket_id"]}
    cluster_by_ticket = {t["ticket_id"]: t.get("duplicate_cluster_id") for t in tickets}

    if todo:
        ev = build(db_path, cache)
        cached_texts = cache_by_input_text(cache)
        for i, (pos, ticket) in enumerate(todo):
            if i:
                await asyncio.sleep(args.delay)
            row = await run_ticket(ev, ticket, pos, name, cached_texts, store_to_ticket, cluster_by_ticket)
            if ev.counter.quota_exhausted:
                # Agent 2 raised, or Agent 5 silently fell back to its rule:
                # either way the row is not a real result. Not recorded, so
                # --resume runs this ticket again.
                rows = read_rows(csv_path)
                if rows:
                    write_markdown(rows, md_path, merged=False)
                print(f"Groq quota exhausted after {ticket['ticket_id']}. "
                      f"Recorded {len(rows)}/{len(selected)}.\n"
                      "Rerun the same command with --resume later (quota resets ~24h).")
                print(f"Temp DB kept for --resume: {db_path}")
                return QUOTA_EXIT_CODE
            append_row(csv_path, row)
            if row["store_ticket_id"]:
                store_to_ticket[row["store_ticket_id"]] = row["ticket_id"]
            print(_progress(pos, total, row), flush=True)

    rows = read_rows(csv_path)
    if rows:
        write_markdown(rows, md_path, merged=False)
    recorded = {r["ticket_id"] for r in rows}
    complete = all(t["ticket_id"] in recorded for _, t in selected)
    if complete:
        delete_db(db_path)
    print(f"\nCSV: {csv_path}\nReport: {md_path}")
    print("Slice complete; temp DB deleted." if complete else
          f"Slice not complete; temp DB kept for --resume: {db_path}")
    return 0


_SLICE_CSV_RE = re.compile(r"^pipeline_eval_results_(\d+)-(\d+)\.csv$")


def discover_slice_csvs(results_dir: Path = RESULTS_DIR) -> list[Path]:
    """Every results/pipeline_eval_results_<start>-<end>.csv (never the merged file), by start."""
    found = []
    for path in Path(results_dir).glob("pipeline_eval_results_*-*.csv"):
        m = _SLICE_CSV_RE.match(path.name)
        if m:
            found.append((int(m.group(1)), int(m.group(2)), path))
    return [p for _, _, p in sorted(found)]


def validate_merge(rows: list[dict], expected_total: int = EXPECTED_TOTAL) -> dict:
    """Counts for the merge printout. Duplicate ids already failed in merge_csvs()."""
    positions = {r["position"] for r in rows if r["position"] is not None}
    return {
        "slices": slice_summary(rows),
        "n_rows": len(rows),
        "n_unique_ids": len({r["ticket_id"] for r in rows}),
        "missing_positions": [p for p in range(1, expected_total + 1) if p not in positions],
        "n_error": sum(1 for r in rows if r["final_outcome"] == "ERROR"),
    }


def write_rows_csv(path: Path, rows: list[dict]) -> None:
    """All rows in one go (same CSV_FIELDS and encoding as append_row)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _to_csv(row.get(k)) for k in CSV_FIELDS})


def _compact_ranges(values: list[int]) -> str:
    out, i = [], 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[j + 1] == values[j] + 1:
            j += 1
        out.append(str(values[i]) if i == j else f"{values[i]}-{values[j]}")
        i = j + 1
    return ", ".join(out)


def report_only(inputs: list[str], out: Path = MERGED_MD, *, auto: bool = False,
                results_dir: Path = RESULTS_DIR, merged_csv: Path | None = None,
                expected_total: int = EXPECTED_TOTAL) -> int:
    """
    Merge slice CSVs into the markdown report and one merged CSV (next to
    the report unless merged_csv is given). Duplicate ticket_ids fail;
    missing positions are reported but do not.
    """
    paths = [Path(p) for p in inputs]
    if auto:
        paths += [p for p in discover_slice_csvs(results_dir) if p not in paths]
    if not paths:
        print("--report-only needs --inputs a.csv [b.csv ...] or --auto")
        return 2
    try:
        rows = merge_csvs(paths)
    except (DuplicateTicketError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1

    v = validate_merge(rows, expected_total)
    print(f"Slices: {len(v['slices'])} ({', '.join(p.name for p in paths)})")
    for s in v["slices"]:
        print(f"  slice {s['slice']}: positions {s['min_pos']}-{s['max_pos']}, rows {s['n']}")
    print(f"Unique ticket_ids: {v['n_unique_ids']} (expected {expected_total})")
    print("Missing positions: " + (_compact_ranges(v["missing_positions"]) or "none"))
    print("Duplicate ticket_ids: none")
    print(f"ERROR rows: {v['n_error']}")

    merged_csv = Path(merged_csv) if merged_csv is not None else Path(out).parent / MERGED_CSV_NAME
    write_rows_csv(merged_csv, rows)
    write_markdown(rows, out, merged=True)
    print(f"Merged {len(rows)} rows from {len(paths)} file(s) -> {out}, {merged_csv}")
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", type=int, default=None, help="first ticket (1-based, fixed order)")
    p.add_argument("--end", type=int, default=None, help="last ticket (inclusive)")
    p.add_argument("--limit", type=int, default=None, help="at most N tickets of the slice")
    p.add_argument("--delay", type=float, default=DEFAULT_DELAY_SECONDS, help="seconds between tickets")
    p.add_argument("--resume", action="store_true", help="skip tickets already in the slice CSV")
    p.add_argument("--retry-errors", action="store_true",
                   help="with --resume: also re-run tickets recorded as ERROR")
    p.add_argument("--dry-run", action="store_true", help="check cache hits and estimate; no API calls")
    p.add_argument("--report-only", action="store_true", help="merge --inputs CSVs into the report")
    p.add_argument("--inputs", nargs="*", default=[], help="slice CSVs for --report-only")
    p.add_argument("--auto", action="store_true",
                   help="with --report-only: merge every results/pipeline_eval_results_<n>-<m>.csv")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.report_only:
        return report_only(args.inputs, auto=args.auto)
    try:
        return asyncio.run(run(args))
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
