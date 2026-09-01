# Agent 2 Behavioral Scenario Results
Run: 2026-08-31T05:02:36.857019+00:00
Model: openai/gpt-oss-20b
======================================================================
SUMMARY: 4 passed, 4 need manual review, 0 failed (out of 8)
======================================================================


## one_question_multiple_gaps
**Description:** Teams audio failure affecting multiple users - scope already known
**Status:** PASS
**Note:** Asked one question covering ['error_signal', 'device_platform'], did not re-ask scope

- Action: `ASK_CLARIFICATION`
- Question: Which Teams client are you using (desktop app, web, or mobile), and did you see any error message or notification when the audio stopped?
- Reasoning: The ticket lacks key information on the client platform and the error signal. Knowing the platform allows us to apply the correct troubleshooting steps, while any error message can pinpoint a specific issue. A single question that addresses both fields will efficiently gather the missing data and make the ticket actionable.
- Affected fields: ['error_signal', 'device_platform']
- Expected information gain: 0.7
- Confidence: 0.9

## ready_despite_incomplete_fields
**Description:** Teams camera fails, isolated to Teams (works in Zoom) - device_platform/scope unresolved
**Status:** MANUAL REVIEW
**Note:** Returned ASK_CLARIFICATION instead of READY. digest.txt's own Example D asserts READY here, but this is an illustrative example, not a labeled ground truth - read decision.reasoning and judge whether the remaining gap is genuinely load-bearing before calling this wrong.

- Action: `ASK_CLARIFICATION`
- Question: Could you let me know which device and operating system you’re using Teams on (e.g., Windows PC, MacBook, iPhone, etc.)?
- Reasoning: The user has indicated the camera works in Zoom but not in Teams, so the issue is isolated to Teams. To determine the appropriate troubleshooting steps and support routing, we need to know the device platform. This information will help identify platform‑specific solutions. No prior clarification has covered this, and the question is concise and likely to be answered readily. The expected information gain is moderate, and the turn budget allows for one clarification turn.
- Affected fields: ['device_platform']
- Expected information gain: 0.6
- Confidence: 0.85

## avoid_repetition
**Description:** Device platform already answered in prior turn's Q&A, in different wording
**Status:** PASS
**Note:** Did not re-ask device_platform; moved to ['scope']

- Action: `ASK_CLARIFICATION`
- Question: Are the crashes happening for all users in your organization, or just for you?
- Reasoning: The user has confirmed Teams crashes on the Windows desktop app. The only remaining high‑weight fields are scope and error signal. Determining whether the issue is user‑specific or organization‑wide directly influences routing (global update vs. user‑level troubleshooting) and is the most valuable clarification we can obtain in the remaining turn.
- Affected fields: ['scope']
- Expected information gain: 0.8
- Confidence: 0.9

## contradiction_detection
**Description:** User says 'only my laptop', then later 'my phone too' - state now shows multiple
**Status:** MANUAL REVIEW
**Note:** Returned ASK_CLARIFICATION instead of RECHECK. The transcript contains a clear contradiction (single device -> multiple devices) - read reasoning to check whether the model at least noticed the conflict, even if it chose a different action label for it.

- Action: `ASK_CLARIFICATION`
- Question: Which WiFi network (SSID) are you trying to connect to?
- Reasoning: The user reported connectivity issues on multiple devices but did not specify the network name. Knowing the SSID would help determine if the problem is with the router, the specific network, or the devices.
- Affected fields: ['ssid']
- Expected information gain: 0.7
- Confidence: 0.85

## ambiguous_category
**Description:** Category could not be determined - vague message with no category-defining signal
**Status:** PASS
**Note:** Asked a clarifying question with category undetermined

- Action: `ASK_CLARIFICATION`
- Question: Could you tell me what specific issue you're experiencing today?
- Reasoning: The user’s message is very vague, with no symptom, timing, or scope identified.  The symptom_or_error field is the most valuable for diagnosis and routing (weight 1.0).  Asking for a clear description of the issue will provide the critical missing data and allow the ticket to be triaged appropriately.
- Affected fields: ['symptom_or_error']
- Expected information gain: 0.9
- Confidence: 0.95

## high_completeness_critical_ambiguity
**Description:** AD account ticket: fields resolved, but symptom description is internally contradictory
**Status:** PASS
**Note:** Correctly asked despite high completeness score

- Action: `ASK_CLARIFICATION`
- Question: Have you tried resetting your password or logging in from a different device to see if the issue persists?
- Reasoning: The user reports being able to log in with the old password but the system says the account doesn't exist, which suggests a sync or activation issue. Knowing whether standard troubleshooting steps like password reset or using a different device have been attempted will help determine if the problem is with AD provisioning or local caching, and therefore guide the appropriate resolution path.
- Affected fields: ['troubleshooting_done']
- Expected information gain: 0.8
- Confidence: 0.9

## low_completeness_actionable
**Description:** Printer jam, single printer, specific and unambiguous despite few fields resolved
**Status:** MANUAL REVIEW
**Note:** Returned ASK_CLARIFICATION with completeness=0.4. Read reasoning: does it engage with why the specific detail given (paper jam, one printer) is or isn't enough to route, or does it default to the low score?

- Action: `ASK_CLARIFICATION`
- Question: Could you let me know the model number of the printer on the 3rd floor near the library entrance?
- Reasoning: The user has reported a paper jam and identified the location, but we lack the printer model, which is essential for precise troubleshooting instructions and proper routing. Asking for this information will help determine the correct service path and avoid generic guidance.
- Affected fields: ['printer_model']
- Expected information gain: 0.6
- Confidence: 0.8

## turn_budget_conservatism
**Description:** One turn remaining before budget-forced escalation - still one field unresolved
**Status:** MANUAL REVIEW
**Note:** Chose ASK_CLARIFICATION at turn_count=max_turns-1 (last turn before the safeguard would force ESCALATE). Check reasoning: does it argue this specific question is high-value enough to justify the last turn, per the prompt's 'weigh the cost of another turn more heavily' instruction? Note: if it does ask, AdaptiveClarifier's safeguard will force ESCALATE next turn regardless - this scenario tests the model's own judgment approaching the limit, not the safeguard itself (that's already covered in M5.8 unit tests).

- Action: `ASK_CLARIFICATION`
- Question: Could you let me know what email client or device you’re using, and the type or size of the attachments you’re trying to send?
- Reasoning: The failure type and error signal are already resolved, but we lack device platform and scope details, which are crucial for diagnosing attachment sending issues. Providing both in one question maximizes information gain and stays within the turn budget.
- Affected fields: ['scope', 'device_platform']
- Expected information gain: 0.8
- Confidence: 0.9
