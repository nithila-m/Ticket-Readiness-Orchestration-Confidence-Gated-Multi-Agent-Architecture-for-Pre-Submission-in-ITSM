# Handles the initial triage step by extracting a basic issue type and flagging missing info.
def intake_clarify(state: dict) -> dict:
    state["extracted_fields"] = {"issue_type": "stub_issue"}
    state["missing_fields"] = ["urgency", "location"]
    state["completeness_score"] = 0.5
    return state