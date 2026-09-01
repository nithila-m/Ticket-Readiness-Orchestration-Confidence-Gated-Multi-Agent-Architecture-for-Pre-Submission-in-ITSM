"""
Ablation comparison: real Agent 2 (LLM, via Groq) vs
RuleBasedClarificationProvider (deterministic, free, instant) on the
SAME evaluation scenarios.

This is the concrete evidence behind Claim A ("Agent 2's behavior is
not a deterministic function of the completeness score"). Both
providers see the exact same completeness_score for each scenario. If
Agent 2 were secretly a threshold rule, its actions would agree with
the rule-based baseline every time. Divergences - and specifically
where they fall relative to score - are the actual data point, not an
assertion.

Costs real Groq API calls for the LLM half; the rule-based half is
free and instant. Run from the project root:
    python scripts/run_ablation_comparison.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.adaptive_clarifier import AdaptiveClarifier
from app.config.settings import settings
from app.providers.groq_clarification_provider import GroqClarificationProvider
from app.providers.rule_based_clarification_provider import RuleBasedClarificationProvider
from tests.eval.agent2_scenarios import ALL_SCENARIOS

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "results" / "ablation_comparison.md"


async def main():
    groq_provider = GroqClarificationProvider(api_key=settings.groq_api_key, model=settings.groq_model)
    llm_agent = AdaptiveClarifier(provider=groq_provider, max_turns=settings.max_clarification_turns)
    rule_agent = AdaptiveClarifier(
        provider=RuleBasedClarificationProvider(), max_turns=settings.max_clarification_turns
    )

    rows = []
    for scenario in ALL_SCENARIOS:
        llm_decision = await llm_agent.decide(scenario.state)
        rule_decision = await rule_agent.decide(scenario.state)

        agree = llm_decision.action == rule_decision.action
        rows.append({
            "id": scenario.id,
            "completeness_score": scenario.state.completeness_score,
            "rule_action": rule_decision.action,
            "llm_action": llm_decision.action,
            "agree": agree,
            "llm_reasoning": llm_decision.reasoning,
        })
        print(
            f"{scenario.id:40s} score={scenario.state.completeness_score:.2f}  "
            f"rule={rule_decision.action:20s} llm={llm_decision.action:20s} "
            f"{'AGREE' if agree else 'DIVERGE'}"
        )

    rows_sorted = sorted(rows, key=lambda r: r["completeness_score"])
    n_diverge = sum(1 for r in rows if not r["agree"])

    lines = [
        "# Ablation comparison — rule-based baseline vs Agent 2 (LLM)",
        f"n = {len(rows)} scenarios, {n_diverge} divergences ({n_diverge / len(rows):.0%})",
        "",
        "If Agent 2's decisions were a deterministic function of "
        "completeness_score alone, this table would show 0 divergences - "
        "both providers see the exact same score for every row. Sorted "
        "by score so any threshold-like pattern would be visible.",
        "",
        "| Scenario | Score | Rule-based | Agent 2 (LLM) | Agree? |",
        "|---|---|---|---|---|",
    ]
    for r in rows_sorted:
        lines.append(
            f"| {r['id']} | {r['completeness_score']:.2f} | {r['rule_action']} "
            f"| {r['llm_action']} | {'✓' if r['agree'] else '✗'} |"
        )

    lines.append("")
    lines.append("## Divergence detail (Agent 2's own reasoning, where it disagreed)")
    for r in rows_sorted:
        if not r["agree"]:
            lines.append(
                f"\n**{r['id']}** (score={r['completeness_score']:.2f}, "
                f"rule said {r['rule_action']}, Agent 2 said {r['llm_action']}):"
            )
            lines.append(f"> {r['llm_reasoning']}")

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
