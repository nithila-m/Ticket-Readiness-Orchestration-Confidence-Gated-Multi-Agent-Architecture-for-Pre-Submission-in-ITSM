"""
Shared plumbing for the Agent 1 caching scripts (cache_agent1_fields.py,
check_agent1_variation.py): input text, prompt hash, rate limiting,
quota backoff, resume and atomic cache writes.

No API client lives here - callers inject an async `call(input_text)`
that returns a cache entry, plus `sleep`, so tests run with fakes, no
network and no real waiting.

Rate limiting:
  - after every API call, success or failure, wait `delay` seconds;
  - on a quota error (429 / RESOURCE_EXHAUSTED / "quota") wait 30, 60,
    120, 240 s between retries (4 retries max) instead - each wait is
    max(backoff step, delay, retry-after from the error if it has one);
  - every attempt, retries included, counts towards `max_calls`.
After the final quota retry the cache is saved and QuotaExhausted ends
the run; the caller exits non-zero so a rerun later resumes.
"""

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from app.providers.gemini_provider import EXTRACTION_SYSTEM_PROMPT
from app.schemas.conversation import Message
from app.services.transcript_formatter import format_transcript_for_extraction

DEFAULT_DELAY_SECONDS = 6.0
DEFAULT_MAX_CALLS = 300
QUOTA_BACKOFF_SECONDS = (30.0, 60.0, 120.0, 240.0)
EST_LATENCY_SECONDS = 2.0  # rough Gemini response time, used for estimates only

_QUOTA_PATTERN = re.compile(r"\b429\b|RESOURCE_EXHAUSTED|quota", re.IGNORECASE)
# Gemini puts e.g. "'retryDelay': '37s'" in the error; HTTP-style "Retry-After: 37" also matches.
_RETRY_AFTER_PATTERN = re.compile(
    r"retry[-_ ]?(?:after|delay)['\"]?\s*[:=]?\s*['\"]?(\d+(?:\.\d+)?)", re.IGNORECASE
)

Sleep = Callable[[float], Awaitable[None]]
Log = Callable[[str], None]


class QuotaExhausted(Exception):
    """Quota errors persisted through every backoff retry."""


class _MaxCallsReached(Exception):
    """The per-run --max-calls budget is spent."""


def build_input_text(ticket: dict) -> str:
    """One user turn, formatted exactly as the live chat formats a transcript."""
    content = f"{ticket.get('subject') or ''}. {ticket.get('message') or ''}"
    return format_transcript_for_extraction([Message(role="user", content=content)])


def prompt_sha256(prompt: str = EXTRACTION_SYSTEM_PROMPT) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _exception_chain(exc: BaseException):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = getattr(exc, "cause", None) or exc.__cause__


def is_quota_error(exc: BaseException) -> bool:
    for e in _exception_chain(exc):
        if getattr(e, "code", None) == 429 or getattr(e, "status", None) == "RESOURCE_EXHAUSTED":
            return True
        if _QUOTA_PATTERN.search(str(e)):
            return True
    return False


def parse_retry_after(exc: BaseException) -> float | None:
    for e in _exception_chain(exc):
        match = _RETRY_AFTER_PATTERN.search(str(e))
        if match:
            return float(match.group(1))
    return None


def load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, data: Any) -> None:
    """Temp file in the same directory, then os.replace - never a half-written cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def is_cached(cache: dict[str, dict], key: str, prompt_hash: str) -> bool:
    return cache.get(key, {}).get("prompt_sha256") == prompt_hash


def estimate_seconds(n_calls: int, delay: float) -> float:
    return n_calls * (delay + EST_LATENCY_SECONDS)


@dataclass
class CallItem:
    key: str  # cache key, e.g. "PRN-012" or "PRN-012#run2"
    ticket_id: str
    input_text: str


@dataclass
class RunResult:
    calls_made: int = 0
    succeeded: int = 0
    skipped: int = 0
    failed: list[str] = field(default_factory=list)
    stopped_reason: str | None = None  # None | "max_calls" | "quota"
    cached: int = 0
    remaining: int = 0


async def run_cached_calls(
    items: list[CallItem],
    cache: dict[str, dict],
    cache_path: Path,
    call: Callable[[str], Awaitable[dict]],
    *,
    prompt_hash: str,
    delay: float = DEFAULT_DELAY_SECONDS,
    max_calls: int = DEFAULT_MAX_CALLS,
    sleep: Sleep,
    log: Log = print,
    describe: Callable[[dict], str] = lambda entry: "",
) -> RunResult:
    """
    Calls `call(item.input_text)` for every item not already cached with
    `prompt_hash`, storing the returned entry under item.key and writing
    the cache to disk after each success. Non-quota failures are logged
    and skipped (a rerun retries them). Mutates `cache` in place.
    """
    result = RunResult()
    total = len(items)

    async def attempt(item: CallItem) -> dict:
        for retry in range(len(QUOTA_BACKOFF_SECONDS) + 1):
            if result.calls_made >= max_calls:
                raise _MaxCallsReached
            result.calls_made += 1
            try:
                entry = await call(item.input_text)
            except Exception as exc:
                if not is_quota_error(exc):
                    await sleep(delay)
                    raise
                if retry == len(QUOTA_BACKOFF_SECONDS):
                    raise QuotaExhausted(str(exc)) from exc
                wait = max(QUOTA_BACKOFF_SECONDS[retry], delay, parse_retry_after(exc) or 0.0)
                log(f"    quota error on {item.ticket_id}, waiting {wait:.0f}s "
                    f"(retry {retry + 1}/{len(QUOTA_BACKOFF_SECONDS)})")
                await sleep(wait)
                continue
            await sleep(delay)
            return entry
        raise AssertionError("unreachable")

    try:
        for pos, item in enumerate(items, start=1):
            if is_cached(cache, item.key, prompt_hash):
                result.skipped += 1
                continue
            try:
                entry = await attempt(item)
            except (QuotaExhausted, _MaxCallsReached):
                raise
            except Exception as exc:
                result.failed.append(item.key)
                log(f"[{pos}/{total}] {item.key} FAILED: {exc}")
                continue
            cache[item.key] = entry
            atomic_write_json(cache_path, cache)
            result.succeeded += 1
            suffix = describe(entry)
            log(f"[{pos}/{total}] {item.key} ok{f' ({suffix})' if suffix else ''}")
    except _MaxCallsReached:
        result.stopped_reason = "max_calls"
    except QuotaExhausted:
        await sleep(delay)
        result.stopped_reason = "quota"
    finally:
        atomic_write_json(cache_path, cache)
        result.cached = sum(is_cached(cache, i.key, prompt_hash) for i in items)
        result.remaining = total - result.cached

    return result


def report_and_exit_code(result: RunResult, rerun_hint: str, log: Log = print) -> int:
    """Prints the run summary; returns the process exit code."""
    log(f"\nCalls made: {result.calls_made}, ok: {result.succeeded}, "
        f"skipped (cached): {result.skipped}, failed: {len(result.failed)}")
    log(f"Cached: {result.cached}, remaining: {result.remaining}")
    if result.failed:
        log(f"Failed (non-quota, retried on next run): {', '.join(result.failed)}")
    if result.stopped_reason == "quota":
        log(f"\nGemini quota exhausted after {len(QUOTA_BACKOFF_SECONDS)} retries. Cache saved. "
            f"Rerun later to resume:\n    {rerun_hint}")
        return 2
    if result.stopped_reason == "max_calls":
        log(f"\nStopped at --max-calls. Cache saved. Rerun to continue:\n    {rerun_hint}")
    elif result.remaining:
        log(f"\n{result.remaining} item(s) still missing. Rerun to retry:\n    {rerun_hint}")
    return 0
