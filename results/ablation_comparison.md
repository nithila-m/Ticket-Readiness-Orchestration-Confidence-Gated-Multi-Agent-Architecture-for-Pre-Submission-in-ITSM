# Ablation comparison — rule-based baseline vs Agent 2 (LLM)
n = 8 scenarios, 3 divergences (38%)

If Agent 2's decisions were a deterministic function of completeness_score alone, this table would show 0 divergences - both providers see the exact same score for every row. Sorted by score so any threshold-like pattern would be visible.

| Scenario | Score | Rule-based | Agent 2 (LLM) | Agree? |
|---|---|---|---|---|
| ambiguous_category | 0.00 | ASK_CLARIFICATION | ASK_CLARIFICATION | ✓ |
| low_completeness_actionable | 0.40 | ASK_CLARIFICATION | ASK_CLARIFICATION | ✓ |
| ready_despite_incomplete_fields | 0.45 | ASK_CLARIFICATION | ASK_CLARIFICATION | ✓ |
| turn_budget_conservatism | 0.50 | ASK_CLARIFICATION | ASK_CLARIFICATION | ✓ |
| one_question_multiple_gaps | 0.55 | ASK_CLARIFICATION | ASK_CLARIFICATION | ✓ |
| avoid_repetition | 0.60 | READY | ASK_CLARIFICATION | ✗ |
| contradiction_detection | 0.70 | READY | ASK_CLARIFICATION | ✗ |
| high_completeness_critical_ambiguity | 0.85 | READY | ASK_CLARIFICATION | ✗ |

## Divergence detail (Agent 2's own reasoning, where it disagreed)

**avoid_repetition** (score=0.60, rule said READY, Agent 2 said ASK_CLARIFICATION):
> The ticket lacks scope (whether it is a local or widespread issue) and error signal (any error message or logs), both of which are important for diagnosis and routing. A single question that asks about both will resolve two uncertainties with minimal user effort and is likely to guide the next support step.

**contradiction_detection** (score=0.70, rule said READY, Agent 2 said ASK_CLARIFICATION):
> The conversation confirms that the issue affects multiple devices and started today, so it’s a network‑level problem. Knowing the SSID would let the team verify the network status and check for outages, which is the next logical step toward resolution. This question is specific, likely to be answered, and will reduce uncertainty about routing. No other fields remain unresolved that would change diagnosis more significantly.

**high_completeness_critical_ambiguity** (score=0.85, rule said READY, Agent 2 said ASK_CLARIFICATION):
> The user’s problem is clearly stated and all other key fields are resolved, but the system hasn’t recorded what troubleshooting they’ve performed. Knowing this will prevent redundant steps and help determine if the account is simply pending replication or if a different issue is at play. This question is specific and will materially improve routing and resolution.