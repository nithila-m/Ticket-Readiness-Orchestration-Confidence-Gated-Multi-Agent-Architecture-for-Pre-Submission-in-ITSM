# Checks for similar past tickets so the workflow can detect duplicates or near-duplicates early.
def duplicate_check(state: dict) -> dict:
    state["similar_tickets"] = [
        {"ticket_id": "STUB_TICKET_001", "similarity": 0.0}
    ]
    return state