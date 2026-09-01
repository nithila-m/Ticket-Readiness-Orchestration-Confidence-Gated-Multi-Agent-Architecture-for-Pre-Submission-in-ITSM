# Agent 2 Dimension Scores (M5.10)
Run: 2026-08-31T05:12:14.838907+00:00
Model: openai/gpt-oss-20b
## Summary

| Dimension | Score |
|---|---|
| Question Quality | 0.75 / 1.00 |
| Redundancy | 0.75 / 1.00 |
| Information Gain | 0.50 / 1.00 |
| Turn Efficiency | NOT MEASURABLE |
| Readiness Quality | 0.67 / 1.00 |
| Over Questioning | 0.50 / 1.00 |
| Adaptation | 0.50 / 1.00 |
| Contradiction Handling | 0.00 / 1.00 |


## Question Quality
- `one_question_multiple_gaps`: **PARTIAL** — No human verdict recorded yet. Automated note: Only targeted ['device_platform'] - didn't combine multiple gaps into one question. Not automatically wrong, but check manually whether a combined question was actually possible here.
- `ambiguous_category`: **PASS** — Asked a clarifying question with category undetermined

**Score: 0.75 / 1.00** (avg over 2 scenario(s))

## Redundancy
- `one_question_multiple_gaps`: **PARTIAL** — No human verdict recorded yet. Automated note: Only targeted ['device_platform'] - didn't combine multiple gaps into one question. Not automatically wrong, but check manually whether a combined question was actually possible here.
- `avoid_repetition`: **PASS** — Did not re-ask device_platform; moved to ['error_signal']

**Score: 0.75 / 1.00** (avg over 2 scenario(s))

## Information Gain
- `one_question_multiple_gaps`: **PARTIAL** — No human verdict recorded yet. Automated note: Only targeted ['device_platform'] - didn't combine multiple gaps into one question. Not automatically wrong, but check manually whether a combined question was actually possible here.
- `turn_budget_conservatism`: **PARTIAL** — Action itself is reasonable (one combined question, not wasteful). But reasoning never references the turn budget or the approaching limit at all - reads identically to a turn-1 decision. The prompt's explicit 'weigh the cost of another turn more heavily near the budget' instruction is not demonstrably applied here, even though the outcome happens to be fine.

**Score: 0.50 / 1.00** (avg over 2 scenario(s))

## Turn Efficiency
**NOT YET MEASURABLE** - no fixture coverage. Requires a multi-turn oracle comparison, not a single-shot scenario. See M5.6 end-to-end smoke test for a qualitative 3-turn example in the meantime.

## Readiness Quality
- `ready_despite_incomplete_fields`: **PASS** — Returned ASK_CLARIFICATION rather than the spec's illustrative READY. Reasoning explicitly ties device_platform to differing troubleshooting paths (a real diagnostic claim, not a filler justification) - defensible disagreement with digest.txt's Example D, not a bug. digest.txt's example is illustrative, not labeled ground truth.
- `high_completeness_critical_ambiguity`: **PASS** — Correctly asked despite high completeness score
- `low_completeness_actionable`: **FAIL** — Asked for printer_model to fix a paper jam at a located, specific printer. Justification ('essential for precise troubleshooting instructions') is thinner than comparable scenario 2's reasoning - reads as field-weight-driven rather than a genuine diagnostic argument, since clearing a jam doesn't typically require the model number. Assessed as genuine over-questioning, not a defensible judgment call.

**Score: 0.67 / 1.00** (avg over 3 scenario(s))

## Over Questioning
- `high_completeness_critical_ambiguity`: **PASS** — Correctly asked despite high completeness score
- `low_completeness_actionable`: **FAIL** — Asked for printer_model to fix a paper jam at a located, specific printer. Justification ('essential for precise troubleshooting instructions') is thinner than comparable scenario 2's reasoning - reads as field-weight-driven rather than a genuine diagnostic argument, since clearing a jam doesn't typically require the model number. Assessed as genuine over-questioning, not a defensible judgment call.

**Score: 0.50 / 1.00** (avg over 2 scenario(s))

## Adaptation
- `avoid_repetition`: **PASS** — Did not re-ask device_platform; moved to ['error_signal']
- `contradiction_detection`: **FAIL** — Reasoning never references the single->multiple device contradiction at all; treats 'multiple' as settled fact and moves to an unrelated question (SSID). Root cause is architectural, not a prompt wording issue: Agent 1 re-extracts fresh from the full transcript every turn, so Agent 2 only ever sees the latest, already-reconciled snapshot - it never observes the conflict as a conflict. RECHECK may be largely unreachable under the current re-extraction design. Documented as a known architectural limitation, not something to prompt-tune around.

**Score: 0.50 / 1.00** (avg over 2 scenario(s))

## Contradiction Handling
- `contradiction_detection`: **FAIL** — Reasoning never references the single->multiple device contradiction at all; treats 'multiple' as settled fact and moves to an unrelated question (SSID). Root cause is architectural, not a prompt wording issue: Agent 1 re-extracts fresh from the full transcript every turn, so Agent 2 only ever sees the latest, already-reconciled snapshot - it never observes the conflict as a conflict. RECHECK may be largely unreachable under the current re-extraction design. Documented as a known architectural limitation, not something to prompt-tune around.

**Score: 0.00 / 1.00** (avg over 1 scenario(s))
