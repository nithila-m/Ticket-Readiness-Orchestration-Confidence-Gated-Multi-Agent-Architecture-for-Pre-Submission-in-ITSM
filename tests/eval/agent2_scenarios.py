"""
The 8 required behavioral test scenarios for Agent 2 (digest.txt Section 20).

Each scenario is a hand-built ConversationState (as if Agent 1 already ran)
paired with a check function. Checks return (passed, note):
  - passed=True/False: an automated, mechanical check succeeded/failed
  - passed=None: no reliable automated check exists: this scenario needs
    a human to read the decision and judge it (e.g. "is this a good
    question?" isn't something a keyword match can verify)

Run via: python -m scripts.run_agent2_scenarios
"""

from dataclasses import dataclass
from typing import Callable

from app.schemas.clarification import ClarificationDecision, ClarificationLogEntry
from app.schemas.conversation import ConversationState, Message
from app.schemas.extraction import CategoryPrediction, ExtractedField

CheckResult = tuple[bool | None, str]
CheckFn = Callable[[ClarificationDecision], CheckResult]


@dataclass
class Scenario:
    id: str
    description: str
    state: ConversationState
    check: CheckFn


# --- Scenario 1: one question resolves multiple gaps ---

def _check_scenario_1(d: ClarificationDecision) -> CheckResult:
    if d.action != "ASK_CLARIFICATION":
        return False, f"Expected ASK_CLARIFICATION, got {d.action}"
    if "scope" in d.affected_fields:
        return False, "Re-asked about scope, which was already resolved"
    if len(d.affected_fields) < 2:
        return None, (
            f"Only targeted {d.affected_fields} - didn't combine multiple gaps "
            "into one question. Not automatically wrong, but check manually "
            "whether a combined question was actually possible here."
        )
    return True, f"Asked one question covering {d.affected_fields}, did not re-ask scope"


scenario_1 = Scenario(
    id="one_question_multiple_gaps",
    description="Teams audio failure affecting multiple users - scope already known",
    state=ConversationState(
        conversation_id="eval-1",
        turn_count=1,
        raw_messages=[
            Message(
                role="user",
                content="Teams audio stopped during my class and two classmates have the same problem.",
            ),
        ],
        extracted_fields={
            "failure_type": ExtractedField(value="audio", confidence=0.85),
            "scope": ExtractedField(value="multiple users", confidence=0.8),
        },
        detected_category=CategoryPrediction(value="ms_teams", confidence=0.9),
        completeness_score=0.55,
        missing_or_uncertain_fields=["error_signal", "device_platform"],
    ),
    check=_check_scenario_1,
)


# --- Scenario 2: READY despite incomplete profile fields ---

def _check_scenario_2(d: ClarificationDecision) -> CheckResult:
    if d.action == "READY":
        return True, "Correctly judged ticket actionable despite incomplete fields"
    return None, (
        f"Returned {d.action} instead of READY. digest.txt's own Example D "
        "asserts READY here, but this is an illustrative example, not a "
        "labeled ground truth - read decision.reasoning and judge whether "
        "the remaining gap is genuinely load-bearing before calling this wrong."
    )


scenario_2 = Scenario(
    id="ready_despite_incomplete_fields",
    description="Teams camera fails, isolated to Teams (works in Zoom) - device_platform/scope unresolved",
    state=ConversationState(
        conversation_id="eval-2",
        turn_count=1,
        raw_messages=[
            Message(
                role="user",
                content="My Teams camera isn't working. It works in Zoom but not Teams.",
            ),
        ],
        extracted_fields={
            "failure_type": ExtractedField(value="camera", confidence=0.9),
            "error_signal": ExtractedField(
                value="isolated to Teams app, works fine in Zoom", confidence=0.85
            ),
        },
        detected_category=CategoryPrediction(value="ms_teams", confidence=0.9),
        completeness_score=0.45,
        missing_or_uncertain_fields=["scope", "device_platform"],
    ),
    check=_check_scenario_2,
)


# --- Scenario 3: avoid repetition ---

def _check_scenario_3(d: ClarificationDecision) -> CheckResult:
    if "device_platform" in d.affected_fields:
        return False, "Re-asked about device_platform despite it being answered last turn"
    return True, f"Did not re-ask device_platform; moved to {d.affected_fields}"


scenario_3 = Scenario(
    id="avoid_repetition",
    description="Device platform already answered in prior turn's Q&A, in different wording",
    state=ConversationState(
        conversation_id="eval-3",
        turn_count=2,
        raw_messages=[
            Message(role="user", content="Teams keeps crashing."),
            Message(
                role="assistant",
                content="Are you using the Teams desktop app or the browser version?",
            ),
            Message(role="user", content="Desktop app on Windows."),
        ],
        extracted_fields={
            "failure_type": ExtractedField(value="crashing", confidence=0.85),
            "device_platform": ExtractedField(value="Windows desktop app", confidence=0.9),
        },
        detected_category=CategoryPrediction(value="ms_teams", confidence=0.9),
        completeness_score=0.6,
        missing_or_uncertain_fields=["scope", "error_signal"],
        clarification_log=[
            ClarificationLogEntry(
                turn=0,
                user_message="Teams keeps crashing.",
                decision=ClarificationDecision(
                    action="ASK_CLARIFICATION",
                    reasoning="Device platform unknown.",
                    information_gap="device_platform",
                    question="Are you using the Teams desktop app or the browser version?",
                    expected_information_gain=0.7,
                    affected_fields=["device_platform"],
                    priority="high",
                    confidence=0.8,
                ),
            ),
        ],
    ),
    check=_check_scenario_3,
)


# --- Scenario 4: contradiction detection ---

def _check_scenario_4(d: ClarificationDecision) -> CheckResult:
    if d.action == "RECHECK":
        return True, "Correctly flagged contradiction as RECHECK"
    return None, (
        f"Returned {d.action} instead of RECHECK. The transcript contains a "
        "clear contradiction (single device -> multiple devices) - read "
        "reasoning to check whether the model at least noticed the "
        "conflict, even if it chose a different action label for it."
    )


scenario_4 = Scenario(
    id="contradiction_detection",
    description="User says 'only my laptop', then later 'my phone too' - state now shows multiple",
    state=ConversationState(
        conversation_id="eval-4",
        turn_count=2,
        raw_messages=[
            Message(role="user", content="Only my laptop has the WiFi problem."),
            Message(role="assistant", content="When did this start happening?"),
            Message(role="user", content="Actually my phone can't connect either, started today."),
        ],
        extracted_fields={
            "symptom_type": ExtractedField(value="can't connect", confidence=0.85),
            "single_or_multiple_devices": ExtractedField(value="multiple", confidence=0.8),
            "when_started": ExtractedField(value="today", confidence=0.9),
        },
        detected_category=CategoryPrediction(value="wifi_internet", confidence=0.9),
        completeness_score=0.7,
        missing_or_uncertain_fields=["device_type", "ssid"],
        clarification_log=[
            ClarificationLogEntry(
                turn=0,
                user_message="Only my laptop has the WiFi problem.",
                decision=ClarificationDecision(
                    action="ASK_CLARIFICATION",
                    reasoning="Need to know when this started.",
                    information_gap="when_started",
                    question="When did this start happening?",
                    expected_information_gain=0.6,
                    affected_fields=["when_started"],
                    priority="medium",
                    confidence=0.75,
                ),
            ),
        ],
    ),
    check=_check_scenario_4,
)


# --- Scenario 5: ambiguous category ---

def _check_scenario_5(d: ClarificationDecision) -> CheckResult:
    if d.action != "ASK_CLARIFICATION":
        return None, f"Returned {d.action} - check whether escalating/proceeding without a category is reasonable here"
    if not d.question:
        return False, "ASK_CLARIFICATION but no question provided"
    return True, "Asked a clarifying question with category undetermined"


scenario_5 = Scenario(
    id="ambiguous_category",
    description="Category could not be determined - vague message with no category-defining signal",
    state=ConversationState(
        conversation_id="eval-5",
        turn_count=1,
        raw_messages=[Message(role="user", content="Nothing is working today, please help.")],
        extracted_fields={},
        detected_category=CategoryPrediction(value=None, confidence=0.0),
        completeness_score=0.0,
        missing_or_uncertain_fields=["symptom_or_error", "when_started", "scope"],
    ),
    check=_check_scenario_5,
)


# --- Scenario 6: high completeness but critical ambiguity remains ---

def _check_scenario_6(d: ClarificationDecision) -> CheckResult:
    if d.action == "ASK_CLARIFICATION":
        return True, "Correctly asked despite high completeness score"
    return None, (
        f"Returned {d.action} with completeness=0.85 - check reasoning to see "
        "if it explicitly engaged with the ambiguity (conflicting symptom "
        "descriptions) or just deferred to the high score."
    )


scenario_6 = Scenario(
    id="high_completeness_critical_ambiguity",
    description="AD account ticket: fields resolved, but symptom description is internally contradictory",
    state=ConversationState(
        conversation_id="eval-6",
        turn_count=1,
        raw_messages=[
            Message(
                role="user",
                content=(
                    "My AD account was created but I can log in with the old "
                    "password, though it says my account doesn't exist yet."
                ),
            ),
        ],
        extracted_fields={
            "error_or_symptom": ExtractedField(
                value="can log in with old password but account doesn't exist", confidence=0.7
            ),
            "username_domain": ExtractedField(value="provided", confidence=0.85),
            "when_started": ExtractedField(value="today", confidence=0.9),
            "device_context": ExtractedField(value="campus PC", confidence=0.8),
        },
        detected_category=CategoryPrediction(value="ad_account_creation", confidence=0.85),
        completeness_score=0.85,
        missing_or_uncertain_fields=["troubleshooting_done"],
    ),
    check=_check_scenario_6,
)


# --- Scenario 7: low completeness but actionable ---

def _check_scenario_7(d: ClarificationDecision) -> CheckResult:
    if d.action == "READY":
        return True, "Correctly judged actionable despite low completeness score"
    return None, (
        f"Returned {d.action} with completeness=0.4. Read reasoning: does it "
        "engage with why the specific detail given (paper jam, one printer) "
        "is or isn't enough to route, or does it default to the low score?"
    )


scenario_7 = Scenario(
    id="low_completeness_actionable",
    description="Printer jam, single printer, specific and unambiguous despite few fields resolved",
    state=ConversationState(
        conversation_id="eval-7",
        turn_count=1,
        raw_messages=[
            Message(
                role="user",
                content="The printer on the 3rd floor near the library entrance has a paper jam.",
            ),
        ],
        extracted_fields={
            "symptom": ExtractedField(value="paper jam", confidence=0.95),
            "scope": ExtractedField(value="one printer, 3rd floor", confidence=0.85),
        },
        detected_category=CategoryPrediction(value="printer_support", confidence=0.9),
        completeness_score=0.4,
        missing_or_uncertain_fields=[
            "error_message", "connection_type", "printer_model", "when_started", "troubleshooting_done"
        ],
    ),
    check=_check_scenario_7,
)


# --- Scenario 8: clarification budget - conservatism near the limit ---

def _check_scenario_8(d: ClarificationDecision) -> CheckResult:
    if d.action in ("READY", "ESCALATE"):
        return True, f"Chose {d.action} at the last available turn rather than asking again"
    return None, (
        f"Chose ASK_CLARIFICATION at turn_count=max_turns-1 (last turn before "
        "the safeguard would force ESCALATE). Check reasoning: does it argue "
        "this specific question is high-value enough to justify the last "
        "turn, per the prompt's 'weigh the cost of another turn more "
        "heavily' instruction? Note: if it does ask, AdaptiveClarifier's "
        "safeguard will force ESCALATE next turn regardless - this scenario "
        "tests the model's own judgment approaching the limit, not the "
        "safeguard itself (that's already covered in M5.8 unit tests)."
    )


scenario_8 = Scenario(
    id="turn_budget_conservatism",
    description="One turn remaining before budget-forced escalation - still one field unresolved",
    state=ConversationState(
        conversation_id="eval-8",
        turn_count=2,  # max_turns=3 -> this is the last turn before forced escalation
        raw_messages=[
            Message(role="user", content="My VIT email won't send attachments."),
            Message(role="assistant", content="What's the error message you see?"),
            Message(role="user", content="It just says 'failed to send', no other details."),
        ],
        extracted_fields={
            "failure_type": ExtractedField(value="can't send attachments", confidence=0.85),
            "error_signal": ExtractedField(value="failed to send", confidence=0.6),
        },
        detected_category=CategoryPrediction(value="vit_email", confidence=0.9),
        completeness_score=0.5,
        missing_or_uncertain_fields=["scope", "device_platform"],
        clarification_log=[
            ClarificationLogEntry(
                turn=0,
                user_message="My VIT email won't send attachments.",
                decision=ClarificationDecision(
                    action="ASK_CLARIFICATION",
                    reasoning="Need error detail.",
                    information_gap="error_signal",
                    question="What's the error message you see?",
                    expected_information_gain=0.7,
                    affected_fields=["error_signal"],
                    priority="high",
                    confidence=0.8,
                ),
            ),
        ],
    ),
    check=_check_scenario_8,
)


ALL_SCENARIOS: list[Scenario] = [
    scenario_1, scenario_2, scenario_3, scenario_4,
    scenario_5, scenario_6, scenario_7, scenario_8,
]