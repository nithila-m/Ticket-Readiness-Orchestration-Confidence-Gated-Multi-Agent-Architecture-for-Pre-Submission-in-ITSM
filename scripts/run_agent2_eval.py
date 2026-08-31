"""
M5.10 - scores Agent 2 against the 8 evaluation dimensions from
digest.txt Section 25. Reruns the live scenarios (fresh model calls,
same as run_agent2_scenarios.py) and combines each scenario's automated
check with any recorded human verdict, then aggregates per dimension.

Run from the project root: python -m scripts.run_agent2_eval

If a scenario's action has changed since its human verdict was recorded
(model non-determinism, or a prompt change), this is flagged explicitly
rather than silently reusing a stale judgment.
"""

import asyncio
from collections import defaultdict
from datetime import datetime, timezone

from app.agents.adaptive_clarifier import AdaptiveClarifier
from app.config.settings import settings
from app.providers.groq_clarification_provider import GroqClarificationProvider
from tests.eval.agent2_scenarios import ALL_SCENARIOS
from tests.eval.dimension_map import DIMENSION_MAP
from tests.eval.human_verdicts import HUMAN_VERDICTS

OUTPUT_PATH = "tests/eval/dimension_scores_latest.md"

VERDICT_WEIGHT = {"PASS": 1.0, "PARTIAL": 0.5, "FAIL": 0.0}


async def score_scenario(agent: AdaptiveClarifier, scenario) -> dict:
    decision = await agent.decide(scenario.state)
    auto_passed, auto_note = scenario.check(decision)

    human = HUMAN_VERDICTS.get(scenario.id)
    stale_warning = None

    if human is not None:
        verdict = human.verdict
        note = human.note
        if human.reviewed_against_action != decision.action:
            stale_warning = (
                f"Action changed since human review: was "
                f"'{human.reviewed_against_action}', now '{decision.action}'. "
                f"Verdict below may be stale - re-review recommended."
            )
    elif auto_passed is True:
        verdict, note = "PASS", auto_note
    elif auto_passed is False:
        verdict, note = "FAIL", auto_note
    else:
        verdict, note = "PARTIAL", f"No human verdict recorded yet. Automated note: {auto_note}"

    return {
        "scenario_id": scenario.id,
        "action": decision.action,
        "verdict": verdict,
        "note": note,
        "stale_warning": stale_warning,
    }


async def main():
    provider = GroqClarificationProvider(api_key=settings.groq_api_key, model=settings.groq_model)
    agent = AdaptiveClarifier(provider=provider, max_turns=settings.max_clarification_turns)

    results_by_id = {}
    for scenario in ALL_SCENARIOS:
        print(f"Scoring: {scenario.id} ...")
        results_by_id[scenario.id] = await score_scenario(agent, scenario)

    report_lines = [
        "# Agent 2 Dimension Scores (M5.10)",
        f"Run: {datetime.now(timezone.utc).isoformat()}",
        f"Model: {settings.groq_model}",
        "",
    ]

    dimension_summary = {}

    for dimension, scenario_ids in DIMENSION_MAP.items():
        report_lines.append(f"## {dimension.replace('_', ' ').title()}")

        if not scenario_ids:
            report_lines.append("**NOT YET MEASURABLE** - no fixture coverage. "
                                 "Requires a multi-turn oracle comparison, not "
                                 "a single-shot scenario. See M5.6 end-to-end "
                                 "smoke test for a qualitative 3-turn example "
                                 "in the meantime.\n")
            dimension_summary[dimension] = "NOT MEASURABLE"
            continue

        scores = []
        for sid in scenario_ids:
            r = results_by_id[sid]
            scores.append(VERDICT_WEIGHT[r["verdict"]])
            report_lines.append(f"- `{sid}`: **{r['verdict']}** — {r['note']}")
            if r["stale_warning"]:
                report_lines.append(f"  - ⚠️ {r['stale_warning']}")

        avg = sum(scores) / len(scores)
        dimension_summary[dimension] = f"{avg:.2f} / 1.00"
        report_lines.append(f"\n**Score: {avg:.2f} / 1.00** (avg over {len(scores)} scenario(s))\n")

    # --- Summary table ---
    report_lines.insert(3, "## Summary\n")
    summary_table = ["| Dimension | Score |", "|---|---|"]
    for dim, val in dimension_summary.items():
        summary_table.append(f"| {dim.replace('_', ' ').title()} | {val} |")
    report_lines[4:4] = summary_table + [""]

    print("\n" + "\n".join(summary_table))

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\nFull report saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())