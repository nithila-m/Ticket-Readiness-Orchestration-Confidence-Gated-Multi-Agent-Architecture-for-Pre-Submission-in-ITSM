"""
scripts/smoke_test_kb_query_builder.py
=======================================

Manual, no-network smoke test that prints the OLD (transcript-based) query
Agent 3 used to receive, next to the NEW (structured-field) query it
receives after this fix - for the same sample conversation.

Does not touch ChromaDB or any LLM provider - this only exercises string
formatting (format_transcript_for_extraction and build_kb_query), so it's
safe to run without chromadb/sentence-transformers installed and without
any API keys configured.

Run from the repo root:
    python scripts/smoke_test_kb_query_builder.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.agents.kb_deflection_agent import build_kb_query  # noqa: E402
from app.schemas.conversation import Message  # noqa: E402
from app.schemas.extraction import CategoryPrediction, ExtractedField  # noqa: E402
from app.services.transcript_formatter import format_transcript_for_extraction  # noqa: E402


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# A 3-turn sample conversation, close to the "WiFi keeps disconnecting in
# my hostel room" example used in the original audit trace.
SAMPLE_MESSAGES = [
    Message(role="user", content="My wifi keeps disconnecting."),
    Message(role="assistant", content="Which building/room are you in, and on what device?"),
    Message(role="user", content="SJT-412, on my laptop, started this morning."),
]

# What Agent 1 would have extracted from that conversation by this point -
# hand-constructed here since this script doesn't call the real LLM.
SAMPLE_CATEGORY = CategoryPrediction(value="wifi_internet", confidence=0.92)
SAMPLE_EXTRACTED_FIELDS = {
    "symptom_type": ExtractedField(value="connection drops repeatedly", confidence=0.85),
    "device_type": ExtractedField(value="laptop", confidence=0.8),
    "when_started": ExtractedField(value="this morning", confidence=0.75),
    "ssid": ExtractedField(value=None, confidence=0.0),  # not mentioned - correctly omitted
}


def main() -> None:
    old_query = format_transcript_for_extraction(SAMPLE_MESSAGES)
    new_query = build_kb_query(SAMPLE_CATEGORY, SAMPLE_EXTRACTED_FIELDS)

    print_section("OLD: what clarification_service.py used to send to Agent 3")
    print(old_query)
    print(f"\n(length: {len(old_query)} chars - grows every clarification turn)")

    print_section("NEW: what clarification_service.py sends to Agent 3 now")
    print(new_query)
    print(f"\n(length: {len(new_query)} chars - stable regardless of turn count)")

    print_section("Reference: how a KB article is embedded (title + symptoms)")
    print("Wi-Fi disconnects repeatedly after device wakes from sleep. "
          "Connection drops or shows 'limited connectivity' right after "
          "laptop/phone wakes from sleep or standby")

    print(
        "\nNotice the NEW query reads like the KB text above (short, "
        "symptom-focused, no dialogue). The OLD query has speaker labels "
        "and a clarifying question embedded in it, which is what the "
        "audit flagged as diluting the match."
    )


if __name__ == "__main__":
    main()