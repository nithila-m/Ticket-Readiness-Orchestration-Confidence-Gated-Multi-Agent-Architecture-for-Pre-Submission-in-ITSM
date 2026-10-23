"""
Proposes a correct-article label (kb_id) for every is_deflectable=true
ticket and writes them to data/eval/agent3_gold_kb.json for HUMAN REVIEW.

Run from the repo root:
    python scripts/propose_agent3_gold.py

How a label is proposed (deterministic):
  1. Take the unfiltered top-1 KB article for the ticket text.
  2. Same-category check: if that article's category does not normalize
     to the ticket's true_category, also record the top-1 article within
     the true_category ("same_category_top1") and flag the entry. (Some
     tickets' true_category differs from what the text describes, e.g.
     TEAMS-004 "Can't log into Teams" is labelled VIT Email Support, so
     forcing a same-category article would often pick a worse one.)
  3. Flag "low_similarity" when top-1 similarity < 0.50: the KB may have
     no fitting article at all.

A reviewer may set "kb_id": null to record "no correct article exists in
the KB" for a ticket.

Every entry written here has "source": "proposed". These are NOT ground
truth: they come from the same retriever being evaluated, so scoring the
retriever against them is circular. To confirm an entry, a human sets
"source" to "confirmed" (and fixes kb_id if wrong). Re-running this script
never overwrites entries whose source is not "proposed".
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "TRO_Codes"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from agents.kb_retrieval_agent import search_kb  # noqa: E402
from app.config.category_profiles import normalize_category_label  # noqa: E402
from run_agent3_eval import build_ticket_text, load_all_tickets  # noqa: E402

GOLD_PATH = REPO_ROOT / "data" / "eval" / "agent3_gold_kb.json"


LOW_SIMILARITY = 0.50


def _top1(text: str, category: str | None) -> tuple[str, str, str | None, float]:
    results = search_kb(text, top_k=1, category=category)
    meta = results["metadatas"][0][0]
    return (
        results["ids"][0][0],
        meta.get("title", ""),
        normalize_category_label(meta.get("category")),
        1 - results["distances"][0][0],
    )


def main():
    existing = json.loads(GOLD_PATH.read_text(encoding="utf-8")) if GOLD_PATH.exists() else {}
    tickets = [t for t in load_all_tickets() if t.get("is_deflectable")]

    gold: dict[str, dict] = {}
    rows = []
    for ticket in tickets:
        tid = ticket["ticket_id"]
        if tid in existing and existing[tid].get("source") != "proposed":
            gold[tid] = existing[tid]  # human-reviewed: keep as-is
            rows.append((ticket["subject"], existing[tid].get("kb_title", ""), None, existing[tid]["source"]))
            continue

        text = build_ticket_text(ticket)
        category = normalize_category_label(ticket.get("true_category"))
        kb_id, title, kb_category, sim = _top1(text, None)

        entry = {
            "kb_id": kb_id,
            "kb_title": title,
            "similarity": round(sim, 4),
            "true_category": ticket.get("true_category"),
            "subject": ticket["subject"],
            "source": "proposed",
            "flags": [],
        }
        if kb_category != category:
            alt_id, alt_title, _, alt_sim = _top1(text, category)
            entry["flags"].append("category_mismatch")
            entry["same_category_top1"] = {
                "kb_id": alt_id, "kb_title": alt_title, "similarity": round(alt_sim, 4)
            }
        if sim < LOW_SIMILARITY:
            entry["flags"].append("low_similarity")
        gold[tid] = entry
        marks = ("C" if "category_mismatch" in entry["flags"] else "") + (
            "L" if "low_similarity" in entry["flags"] else "")
        rows.append((ticket["subject"], title, sim, "proposed" + (f" [{marks}]" if marks else "")))

    GOLD_PATH.write_text(json.dumps(gold, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote {len(gold)} entries to {GOLD_PATH.relative_to(REPO_ROOT)}\n")
    print(f"{'ticket subject':<58} | {'proposed article title':<58} | sim  | source")
    print("-" * 140)
    for (subject, title, sim, source), tid in zip(rows, gold):
        sim_s = f"{sim:.2f}" if sim is not None else "  - "
        print(f"{(tid + ' ' + subject)[:58]:<58} | {title[:58]:<58} | {sim_s} | {source}")
    n_c = sum(1 for e in gold.values() if "category_mismatch" in e.get("flags", []))
    n_l = sum(1 for e in gold.values() if "low_similarity" in e.get("flags", []))
    print(f"\n[C] article category != ticket true_category ({n_c}); same-category alternative stored in the JSON")
    print(f"[L] similarity < {LOW_SIMILARITY:.2f} ({n_l}); the KB may have no fitting article")
    print('To confirm an entry, set "source": "confirmed" (fix kb_id first; null = no correct article).')


if __name__ == "__main__":
    main()
