"""
Runs the 8 behavioral scenarios (digest.txt Section 20) against the REAL,
locked Groq-backed AdaptiveClarifier. Not part of pytest - live-model,
non-deterministic, and deliberately excluded from CI (see M5.8 design note).

Run from the project root: python -m scripts.run_agent2_scenarios

Prints a full transcript per scenario plus the lightweight automated check
result, and saves the same to a markdown file for the report (M5.11).
"""

import asyncio
from datetime import datetime, timezone

from app.agents.adaptive_clarifier import AdaptiveClarifier
from app.config.settings import settings
from app.providers.groq_clarification_provider import GroqClarificationProvider
from tests.eval.agent2_scenarios import ALL_SCENARIOS

OUTPUT_PATH = "tests/eval/results_latest.md"


async def main():
    provider = GroqClarificationProvider(api_key=settings.groq_api_key, model=settings.groq_model)
    agent = AdaptiveClarifier(provider=provider, max_turns=settings.max_clarification_turns)

    report_lines = [
        f"# Agent 2 Behavioral Scenario Results",
        f"Run: {datetime.now(timezone.utc).isoformat()}",
        f"Model: {settings.groq_model}",
        "",
    ]

    pass_count = 0
    manual_review_count = 0
    fail_count = 0

    for scenario in ALL_SCENARIOS:
        print(f"\n{'=' * 70}\n{scenario.id}: {scenario.description}\n{'=' * 70}")
        decision = await agent.decide(scenario.state)
        passed, note = scenario.check(decision)

        if passed is True:
            status = "PASS"
            pass_count += 1
        elif passed is False:
            status = "FAIL"
            fail_count += 1
        else:
            status = "MANUAL REVIEW"
            manual_review_count += 1

        print(f"Action: {decision.action}")
        if decision.question:
            print(f"Question: {decision.question}")
        print(f"Reasoning: {decision.reasoning}")
        print(f"Affected fields: {decision.affected_fields}")
        print(f"Expected info gain: {decision.expected_information_gain} | Confidence: {decision.confidence}")
        print(f"--> {status}: {note}")

        report_lines += [
            f"## {scenario.id}",
            f"**Description:** {scenario.description}",
            f"**Status:** {status}",
            f"**Note:** {note}",
            "",
            f"- Action: `{decision.action}`",
            f"- Question: {decision.question or '(none)'}",
            f"- Reasoning: {decision.reasoning}",
            f"- Affected fields: {decision.affected_fields}",
            f"- Expected information gain: {decision.expected_information_gain}",
            f"- Confidence: {decision.confidence}",
            "",
        ]

    summary = (
        f"\n{'=' * 70}\nSUMMARY: {pass_count} passed, "
        f"{manual_review_count} need manual review, {fail_count} failed "
        f"(out of {len(ALL_SCENARIOS)})\n{'=' * 70}"
    )
    print(summary)
    report_lines.insert(3, summary.strip() + "\n")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\nFull report saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())