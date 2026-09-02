"""
agents/retrieval_agent.py
===========================

Agent 3 — KB Retrieval & Deflection Gate.

Given a user's ticket text, checks it against the already-ingested
kb_articles collection and decides whether the issue can be deflected
(self-resolved via a KB article) or should proceed to Agent 4 for
correlation/duplicate checks.

This file does NOT re-embed or re-insert any KB articles — that already
happened once via insert_to_chromadb.py. This module only opens the
existing collection (via shared/chroma_client.py) and queries it.

--------------------------------------------------------------------------
DEPENDENCIES
--------------------------------------------------------------------------
    pip install chromadb sentence-transformers

--------------------------------------------------------------------------
PREREQUISITE
--------------------------------------------------------------------------
    insert_to_chromadb.py must have already been run at least once, so
    chroma_db/ contains a populated "kb_articles" collection with
    hnsw:space="cosine" set. If you're not sure, re-run it — upserts are
    safe and idempotent.
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chroma_client1 import get_kb_collection

# Loaded once at import time and reused across calls — avoids reopening
# the persisted client/collection on every single deflect() call.
kb_collection = get_kb_collection()


def deflect( ticket_text: str, top_k: int = 3, strong_threshold: float = 0.72, weak_threshold: float = 0.45,) -> dict:
    """
    Queries the KB collection for the closest matching article(s) to
    `ticket_text` and classifies the outcome into one of three states:

        STRONG_MATCH  - confidence >= strong_threshold -> safe to deflect
        WEAK_MATCH    - weak_threshold <= confidence < strong_threshold
                        -> a related article exists but isn't a confident
                           enough match to self-resolve on its own
        NO_MATCH      - confidence < weak_threshold, or the collection is
                        empty -> nothing usable in the KB for this ticket

    `deflect` stays as a boolean (True only for STRONG_MATCH) so existing
    callers built against the old binary contract don't break; `outcome`
    is the new field that actually distinguishes WEAK_MATCH from NO_MATCH.

    Args:
        ticket_text: the user's issue description (subject + message,
                      combined the same way tickets were embedded at
                      ingestion time — see prepare_ticket_records()).
        top_k: how many KB candidates to retrieve for logging/audit.
        strong_threshold: minimum cosine similarity required to offer
                           deflection outright. 0.72 is a starting point -
                           validate on your own eval set before trusting
                           it in a live demo.
        weak_threshold: minimum cosine similarity to surface a candidate
                         at all (e.g. "you might also check..."), below
                         which the match is treated as noise. 0.45 is a
                         starting point alongside strong_threshold - both
                         should be tuned together on the same eval set.

    Returns:
        dict with outcome, deflect decision, confidence score, KB ids
        checked, and (for STRONG_MATCH/WEAK_MATCH) the matched title +
        resolution text.
    """
    results = kb_collection.query(
        query_texts=[ticket_text],   # collection's own embedding_function handles encoding
        n_results=top_k,
    )

    # No KB articles in the collection at all — nothing to deflect against.
    if not results["ids"][0]:
        return {
            "outcome": "NO_MATCH",
            "deflect": False,
            "kb_articles_checked": [],
            "resolution_confidence": 0.0,
            "note": "kb_articles collection is empty — check ingestion ran successfully",
        }

    best_distance = results["distances"][0][0]   # cosine distance (0 = identical, 2 = opposite)
    confidence = 1 - best_distance                # valid similarity only because hnsw:space="cosine"
    best_metadata = results["metadatas"][0][0]    # resolution/title/category live in metadata, not documents

    base_response = {
        "kb_articles_checked": results["ids"][0],
        "resolution_confidence": round(confidence, 2),
    }

    if confidence >= strong_threshold:
        return {
            **base_response,
            "outcome": "STRONG_MATCH",
            "deflect": True,
            "matched_kb_id": results["ids"][0][0],
            "matched_title": best_metadata.get("title", ""),
            "offered_resolution": best_metadata.get("resolution", ""),
        }

    if confidence >= weak_threshold:
        return {
            **base_response,
            "outcome": "WEAK_MATCH",
            "deflect": False,
            "matched_kb_id": results["ids"][0][0],
            "matched_title": best_metadata.get("title", ""),
            "offered_resolution": best_metadata.get("resolution", ""),
        }

    return {
        **base_response,
        "outcome": "NO_MATCH",
        "deflect": False,
    }


if __name__ == "__main__":
    # Quick manual sanity check — run `python -m agents.retrieval_agent`
    # from TRO_Codes/ to try a sample query against your real KB data.

    sample_ticket_text = "VPN keeps disconnecting right after my laptop wakes up from sleep"
    #sample_ticket_text = "The classroom projector is not turning on"
    #sample_ticket_text = "It isn't working. Please fix it."
    result = deflect(sample_ticket_text)

    print(f"Query: {sample_ticket_text}")
    print(f"Result: {result}")