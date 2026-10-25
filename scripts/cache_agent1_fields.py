"""
Caches REAL Agent 1 output for every ticket Agent 4 is calibrated on.

Run from the repo root:
    python scripts/cache_agent1_fields.py --dry-run     # no API calls
    python scripts/cache_agent1_fields.py               # Gemini, ~10 req/min

Tickets: data/eval/tickets_augmented.json (190) + data/eval/
agent4_calibration_extra.json (40). Output: data/eval/agent1_fields_cache.json,
keyed by ticket_id:
    {input_text, category, category_confidence,
     fields: {name: {value, confidence}}, completeness_score,
     model, prompt_sha256, extracted_at}

Same code path as the live app: GeminiProvider -> InformationExtractor ->
TicketAnalysisService, built like app.dependencies.get_ticket_analysis_service
from settings (model, API key), so the prompt and temperature=0 match.

Input text choice: format_transcript_for_extraction([Message(role="user",
content=f"{subject}. {message}")]), i.e. "User: <subject>. <message>" -
a single user turn in the same transcript format the live chat sends to
Agent 1. Subject and message are joined because a ticket's opening chat
message is the only place a user states both.

An entry is reused only while its prompt_sha256 equals the hash of the
current EXTRACTION_SYSTEM_PROMPT, so a prompt change invalidates the cache.
Gemini only - Groq is never called. Rate limiting, quota backoff and
resume are in scripts/agent1_cache_lib.py. Exit code 2 = quota exhausted.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.agent1_cache_lib import (  # noqa: E402
    DEFAULT_DELAY_SECONDS,
    DEFAULT_MAX_CALLS,
    CallItem,
    build_input_text,
    estimate_seconds,
    is_cached,
    load_cache,
    prompt_sha256,
    report_and_exit_code,
    run_cached_calls,
    utc_now_iso,
)

DATA_DIR = REPO_ROOT / "data" / "eval"
TICKET_FILES = (DATA_DIR / "tickets_augmented.json", DATA_DIR / "agent4_calibration_extra.json")
CACHE_PATH = DATA_DIR / "agent1_fields_cache.json"


def load_tickets(paths=TICKET_FILES) -> list[dict]:
    tickets = []
    for path in paths:
        tickets.extend(json.loads(Path(path).read_text(encoding="utf-8")))
    return tickets


def build_items(tickets: list[dict]) -> list[CallItem]:
    return [CallItem(t["ticket_id"], t["ticket_id"], build_input_text(t)) for t in tickets]


def analysis_to_entry(input_text: str, analysis, model: str, prompt_hash: str,
                      extracted_at: str) -> dict:
    return {
        "input_text": input_text,
        "category": analysis.category.value,
        "category_confidence": analysis.category.confidence,
        "fields": {
            name: {"value": f.value, "confidence": f.confidence}
            for name, f in analysis.extracted_fields.items()
        },
        "completeness_score": analysis.completeness_score,
        "model": model,
        "prompt_sha256": prompt_hash,
        "extracted_at": extracted_at,
    }


def describe_entry(entry: dict) -> str:
    return f"{entry['category']}, {len(entry['fields'])} fields"


def make_analysis_call(model: str, prompt_hash: str):
    """Real Agent 1, constructed as app.dependencies does (Gemini only)."""
    from app.agents.information_extractor import InformationExtractor
    from app.config.settings import settings
    from app.providers.gemini_provider import GeminiProvider
    from app.services.ticket_analysis_service import TicketAnalysisService

    provider = GeminiProvider(api_key=settings.gemini_api_key or "", model=model)
    service = TicketAnalysisService(InformationExtractor(provider))

    async def call(input_text: str) -> dict:
        analysis = await service.analyze(input_text)
        return analysis_to_entry(input_text, analysis, model, prompt_hash, utc_now_iso())

    return call


def print_dry_run(pending: list[CallItem], total: int, delay: float, max_calls: int) -> None:
    n = min(len(pending), max_calls)
    print(f"Dry run: {len(pending)} of {total} tickets need an API call "
          f"({total - len(pending)} cached with the current prompt hash).")
    if pending:
        print("Would call: " + ", ".join(i.key for i in pending[:n]))
    if len(pending) > max_calls:
        print(f"--max-calls {max_calls} stops this run after {max_calls}; "
              f"{len(pending) - max_calls} left for the next run.")
    lower = n * delay
    est = estimate_seconds(n, delay)
    print(f"Estimated time for {n} calls at --delay {delay:g}: "
          f"~{est / 60:.0f} min (at least {lower / 60:.0f} min of delay alone, "
          f"plus Gemini latency and any quota backoff).")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--delay", type=float, default=DEFAULT_DELAY_SECONDS,
                   help="seconds to wait after every API call (default 6.0, ~10 req/min)")
    p.add_argument("--max-calls", type=int, default=DEFAULT_MAX_CALLS,
                   help="API calls allowed this run, retries included (default 300)")
    p.add_argument("--limit", type=int, default=None, help="only consider the first N tickets")
    p.add_argument("--dry-run", action="store_true", help="list what would be called; no API calls")
    p.add_argument("--cache", type=Path, default=CACHE_PATH, help=argparse.SUPPRESS)
    return p.parse_args(argv)


async def main_async(args: argparse.Namespace) -> int:
    from app.config.settings import settings

    tickets = load_tickets()
    if args.limit is not None:
        tickets = tickets[: args.limit]
    items = build_items(tickets)
    prompt_hash = prompt_sha256()
    cache = load_cache(args.cache)
    pending = [i for i in items if not is_cached(cache, i.key, prompt_hash)]

    if args.dry_run:
        print(f"Model: {settings.gemini_model}  prompt_sha256: {prompt_hash[:12]}...")
        print_dry_run(pending, len(items), args.delay, args.max_calls)
        return 0

    from app.providers.exceptions import LLMProviderError

    try:
        call = make_analysis_call(settings.gemini_model, prompt_hash)
    except LLMProviderError as exc:
        print(f"Cannot start: {exc}")
        return 1

    print(f"{len(pending)} of {len(items)} tickets to extract with {settings.gemini_model}, "
          f"--delay {args.delay:g}, --max-calls {args.max_calls}")
    result = await run_cached_calls(
        items, cache, args.cache, call,
        prompt_hash=prompt_hash, delay=args.delay, max_calls=args.max_calls,
        sleep=asyncio.sleep, describe=describe_entry,
    )
    return report_and_exit_code(result, "python scripts/cache_agent1_fields.py")


def main(argv=None) -> int:
    return asyncio.run(main_async(parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
