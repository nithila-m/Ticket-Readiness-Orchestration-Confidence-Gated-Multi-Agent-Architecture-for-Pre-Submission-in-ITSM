"""
Human-reviewed verdicts for scenarios where automated checks alone can't
judge quality (digest.txt Section 20's "manual review" cases - e.g. "was
this a good question?" isn't a keyword match). Each entry records the
verdict, the reasoning behind it, and the model action it was reviewed
against - so a future re-run can detect if the model's behavior has
drifted and the verdict may be stale.

Update this file after each fresh run_agent2_scenarios.py execution if
manual review is needed again - do not treat these as permanent labels.
"""

from dataclasses import dataclass
from typing import Literal

Verdict = Literal["PASS", "PARTIAL", "FAIL"]


@dataclass
class HumanVerdict:
    verdict: Verdict
    note: str
    reviewed_against_action: str  # the decision.action this verdict was based on


HUMAN_VERDICTS: dict[str, HumanVerdict] = {
    "ready_despite_incomplete_fields": HumanVerdict(
        verdict="PASS",
        note=(
            "Returned ASK_CLARIFICATION rather than the spec's illustrative "
            "READY. Reasoning explicitly ties device_platform to differing "
            "troubleshooting paths (a real diagnostic claim, not a filler "
            "justification) - defensible disagreement with digest.txt's "
            "Example D, not a bug. digest.txt's example is illustrative, "
            "not labeled ground truth."
        ),
        reviewed_against_action="ASK_CLARIFICATION",
    ),
    "contradiction_detection": HumanVerdict(
        verdict="FAIL",
        note=(
            "Reasoning never references the single->multiple device "
            "contradiction at all; treats 'multiple' as settled fact and "
            "moves to an unrelated question (SSID). Root cause is "
            "architectural, not a prompt wording issue: Agent 1 re-extracts "
            "fresh from the full transcript every turn, so Agent 2 only "
            "ever sees the latest, already-reconciled snapshot - it never "
            "observes the conflict as a conflict. RECHECK may be largely "
            "unreachable under the current re-extraction design. Documented "
            "as a known architectural limitation, not something to prompt-"
            "tune around."
        ),
        reviewed_against_action="ASK_CLARIFICATION",
    ),
    "low_completeness_actionable": HumanVerdict(
        verdict="FAIL",
        note=(
            "Asked for printer_model to fix a paper jam at a located, "
            "specific printer. Justification ('essential for precise "
            "troubleshooting instructions') is thinner than comparable "
            "scenario 2's reasoning - reads as field-weight-driven rather "
            "than a genuine diagnostic argument, since clearing a jam "
            "doesn't typically require the model number. Assessed as "
            "genuine over-questioning, not a defensible judgment call."
        ),
        reviewed_against_action="ASK_CLARIFICATION",
    ),
    "turn_budget_conservatism": HumanVerdict(
        verdict="PARTIAL",
        note=(
            "Action itself is reasonable (one combined question, not "
            "wasteful). But reasoning never references the turn budget or "
            "the approaching limit at all - reads identically to a turn-1 "
            "decision. The prompt's explicit 'weigh the cost of another "
            "turn more heavily near the budget' instruction is not "
            "demonstrably applied here, even though the outcome happens "
            "to be fine."
        ),
        reviewed_against_action="ASK_CLARIFICATION",
    ),
}