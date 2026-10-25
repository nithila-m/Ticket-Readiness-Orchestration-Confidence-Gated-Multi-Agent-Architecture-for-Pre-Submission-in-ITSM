"""
Measures how much REAL Agent 1 output varies between identical calls.

Run from the repo root, AFTER scripts/cache_agent1_fields.py has cached
the selected tickets:
    python scripts/check_agent1_variation.py --dry-run
    python scripts/check_agent1_variation.py
    python scripts/check_agent1_variation.py --report-only   # no API calls

Picks 20 tickets with a fixed seed - the originals of the 5 AUG-DUP
duplicates plus 3 more per category - and re-runs Agent 1 on each one 2
more times (40 Gemini calls) with the same input_text the main cache used.
Runs 2 and 3 go to data/eval/agent1_variation_runs.json under keys
"<ticket_id>#run2" / "#run3", with the same rate limiting, quota backoff
and resume logic (scripts/agent1_cache_lib.py). Run 1 is the entry in
data/eval/agent1_fields_cache.json.

For each ticket the 3 runs are compared on: same category, same set of
field names, exact field-value agreement (share of field names, over the
union of all runs, that every run returned with the identical value) and
the pairwise cosine between their build_kb_query() texts, embedded locally
with all-MiniLM-L6-v2 (the Agent 4 embedder; no API). The report goes to
results/agent1_extraction_variation.md. Exit code 2 = quota exhausted.
"""

import argparse
import asyncio
import json
import random
import sys
from itertools import combinations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.agents.request_text import build_kb_query  # noqa: E402
from app.schemas.extraction import ExtractedField  # noqa: E402
from scripts.agent1_cache_lib import (  # noqa: E402
    DEFAULT_DELAY_SECONDS,
    DEFAULT_MAX_CALLS,
    CallItem,
    estimate_seconds,
    is_cached,
    load_cache,
    prompt_sha256,
    report_and_exit_code,
    run_cached_calls,
)
from scripts.cache_agent1_fields import (  # noqa: E402
    CACHE_PATH,
    describe_entry,
    make_analysis_call,
)

SEED = 20261025
DATA_DIR = REPO_ROOT / "data" / "eval"
SOURCE_PATH = DATA_DIR / "tickets_augmented.json"
VARIATION_PATH = DATA_DIR / "agent1_variation_runs.json"
REPORT_PATH = REPO_ROOT / "results" / "agent1_extraction_variation.md"
N_TICKETS = 20
EXTRA_RUNS = (2, 3)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def select_tickets(source: list[dict], n: int = N_TICKETS, seed: int = SEED) -> list[str]:
    """AUG-DUP originals first, then a fixed-seed draw spread over the categories."""
    from app.config.category_profiles import normalize_category_label

    forced = sorted({t["duplicate_of"] for t in source if t["label_type"] == "duplicate"})
    by_category: dict[str, list[str]] = {}
    for t in source:
        if t["ticket_id"] in forced:
            continue
        cat = normalize_category_label(t["true_category"]) or normalize_category_label(t["category_selected"])
        by_category.setdefault(cat or "general", []).append(t["ticket_id"])

    rng = random.Random(seed)
    cats = sorted(by_category)
    pools = {c: sorted(by_category[c]) for c in cats}
    for c in cats:
        rng.shuffle(pools[c])
    chosen = list(forced)
    i = 0
    while len(chosen) < n and any(pools.values()):
        c = cats[i % len(cats)]
        if pools[c]:
            chosen.append(pools[c].pop())
        i += 1
    return chosen


def variation_key(ticket_id: str, run: int) -> str:
    return f"{ticket_id}#run{run}"


def kb_text(entry: dict) -> str:
    fields = {name: ExtractedField(**f) for name, f in entry["fields"].items()}
    return build_kb_query(entry["category"], fields)


def compare_runs(entries: list[dict], embed_fn) -> dict:
    """
    embed_fn(list[str]) -> list of L2-normalised vectors (so dot = cosine).
    """
    categories = [e["category"] for e in entries]
    name_sets = [set(e["fields"]) for e in entries]
    union = set().union(*name_sets)
    agreeing = [
        n for n in union
        if all(n in e["fields"] for e in entries)
        and len({json.dumps(e["fields"][n]["value"], sort_keys=True) for e in entries}) == 1
    ]
    texts = [kb_text(e) for e in entries]
    vectors = embed_fn(texts)
    cosines = [
        float(sum(a * b for a, b in zip(vectors[i], vectors[j])))
        for i, j in combinations(range(len(entries)), 2)
    ]
    return {
        "categories": categories,
        "same_category": len(set(categories)) == 1,
        "same_field_names": all(s == name_sets[0] for s in name_sets),
        "value_agreement": len(agreeing) / len(union) if union else 1.0,
        "identical_text": len(set(texts)) == 1,
        "cosines": cosines,
    }


def sentence_transformer_embed_fn():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def embed(texts: list[str]):
        return [list(v) for v in model.encode(texts, normalize_embeddings=True)]

    return embed


def render_report(rows: list[tuple[str, dict]], incomplete: list[str], model: str,
                  prompt_hash: str) -> str:
    all_cos = [c for _, r in rows for c in r["cosines"]]
    lines = [
        "# Agent 1 extraction variation",
        "",
        f"Generated by `scripts/check_agent1_variation.py`. Model `{model}`, temperature 0, "
        f"prompt_sha256 `{prompt_hash[:12]}`. Each ticket was extracted 3 times from the same "
        "input text (run 1 from `data/eval/agent1_fields_cache.json`, runs 2-3 from "
        "`data/eval/agent1_variation_runs.json`). Cosine = all-MiniLM-L6-v2 similarity between "
        "the `build_kb_query()` texts of two runs (3 pairs per ticket).",
        "",
    ]
    if all_cos:
        lines += [
            "## Summary",
            "",
            f"- Tickets compared: {len(rows)}",
            f"- Same category in all 3 runs: {sum(r['same_category'] for _, r in rows)}/{len(rows)}",
            f"- Same field names in all 3 runs: {sum(r['same_field_names'] for _, r in rows)}/{len(rows)}",
            f"- Identical build_kb_query() text in all 3 runs: "
            f"{sum(r['identical_text'] for _, r in rows)}/{len(rows)}",
            f"- Mean exact field-value agreement: "
            f"{100 * sum(r['value_agreement'] for _, r in rows) / len(rows):.1f}%",
            f"- Cosine between runs: min {min(all_cos):.4f}, mean {sum(all_cos) / len(all_cos):.4f} "
            f"({len(all_cos)} pairs)",
            "",
            "**How to read the cosine.** It shows how much one request's text varies between "
            "runs, i.e. how far an exact repeat of the same request can fall below 1.0 before "
            "Agent 4 ever compares it with anything else. It is an upper bound on what a "
            "duplicate can score: a real duplicate is worded differently on top of this run-to-run "
            "variation, so it should not be expected to score higher than these values.",
            "",
        ]
    lines += [
        "## Per ticket",
        "",
        "| Ticket | Categories (run 1 / 2 / 3) | Same category | Same field names "
        "| Value agreement | Min cosine | Mean cosine |",
        "|---|---|---|---|---|---|---|",
    ]
    for tid, r in rows:
        cos = r["cosines"]
        lines.append(
            f"| {tid} | {' / '.join(str(c) for c in r['categories'])} "
            f"| {'yes' if r['same_category'] else 'no'} | {'yes' if r['same_field_names'] else 'no'} "
            f"| {100 * r['value_agreement']:.0f}% | {min(cos):.4f} | {sum(cos) / len(cos):.4f} |"
        )
    if incomplete:
        lines += ["", f"Not compared (missing runs, rerun the script): {', '.join(incomplete)}"]
    return "\n".join(lines) + "\n"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--delay", type=float, default=DEFAULT_DELAY_SECONDS,
                   help="seconds to wait after every API call (default 6.0, ~10 req/min)")
    p.add_argument("--max-calls", type=int, default=DEFAULT_MAX_CALLS,
                   help="API calls allowed this run, retries included (default 300)")
    p.add_argument("--dry-run", action="store_true", help="list what would be called; no API calls")
    p.add_argument("--report-only", action="store_true", help="write the report from cached runs only")
    return p.parse_args(argv)


async def main_async(args: argparse.Namespace) -> int:
    from app.config.settings import settings

    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    ticket_ids = select_tickets(source)
    prompt_hash = prompt_sha256()
    base_cache = load_cache(CACHE_PATH)
    missing_base = [t for t in ticket_ids if not is_cached(base_cache, t, prompt_hash)]
    if missing_base:
        print("Run scripts/cache_agent1_fields.py first; these tickets have no cached run 1 "
              f"with the current prompt: {', '.join(missing_base)}")
        return 1

    items = [
        CallItem(variation_key(t, run), t, base_cache[t]["input_text"])
        for t in ticket_ids for run in EXTRA_RUNS
    ]
    runs = load_cache(VARIATION_PATH)
    pending = [i for i in items if not is_cached(runs, i.key, prompt_hash)]

    if args.dry_run:
        n = min(len(pending), args.max_calls)
        print(f"Selected tickets: {', '.join(ticket_ids)}")
        print(f"{len(pending)} of {len(items)} runs need an API call. Estimated "
              f"~{estimate_seconds(n, args.delay) / 60:.0f} min at --delay {args.delay:g}.")
        return 0

    exit_code = 0
    if pending and not args.report_only:
        from app.providers.exceptions import LLMProviderError

        try:
            call = make_analysis_call(settings.gemini_model, prompt_hash)
        except LLMProviderError as exc:
            print(f"Cannot start: {exc}")
            return 1
        result = await run_cached_calls(
            items, runs, VARIATION_PATH, call,
            prompt_hash=prompt_hash, delay=args.delay, max_calls=args.max_calls,
            sleep=asyncio.sleep, describe=describe_entry,
        )
        exit_code = report_and_exit_code(result, "python scripts/check_agent1_variation.py")
        if result.stopped_reason == "quota":
            return exit_code

    complete = [t for t in ticket_ids
                if all(is_cached(runs, variation_key(t, r), prompt_hash) for r in EXTRA_RUNS)]
    incomplete = [t for t in ticket_ids if t not in complete]
    if not complete:
        print("No ticket has all 3 runs yet; report not written.")
        return exit_code

    embed = sentence_transformer_embed_fn()
    rows = [
        (t, compare_runs([base_cache[t]] + [runs[variation_key(t, r)] for r in EXTRA_RUNS], embed))
        for t in complete
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        render_report(rows, incomplete, base_cache[complete[0]]["model"], prompt_hash),
        encoding="utf-8",
    )
    print(f"Wrote {REPORT_PATH.relative_to(REPO_ROOT)} ({len(rows)} tickets compared)")
    return exit_code


def main(argv=None) -> int:
    return asyncio.run(main_async(parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
