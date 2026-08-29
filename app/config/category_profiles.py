"""
Category-specific field weight profiles for completeness scoring.

Each profile maps field_name -> weight (0.0-1.0), representing how much
that field contributes to the completeness score for that category.
This is a DATA REQUIREMENTS mapping for scoring — not clarification
questions. Agent 2 decides what to ask; this only decides how "complete"
a set of extracted fields is once they exist.
"""

CATEGORY_FIELD_PROFILES: dict[str, dict[str, float]] = {
    "wifi_internet": {
        "affected_system": 1.0,   # wifi vs ethernet vs vpn-over-wifi
        "location": 0.9,          # hostel block, academic building, library, etc.
        "trigger": 0.7,           # e.g. "after connecting", "during class hours"
        "frequency": 0.7,
        "error_message": 0.6,
        "device": 0.5,
    },
    "vit_email": {
        "affected_system": 1.0,   # webmail vs Outlook client vs mobile app
        "action_attempted": 1.0,  # login, send, receive, password reset
        "error_message": 0.9,
        "device": 0.5,
        "frequency": 0.5,
    },
    "ms_teams": {
        "affected_system": 1.0,   # desktop app vs web vs mobile
        "trigger": 0.9,           # joining a meeting, during a call, screen share
        "error_message": 0.8,
        "device": 0.6,
        "frequency": 0.5,
    },
    "ad_account_creation": {
        "requester_role": 1.0,    # student, faculty, staff
        "department": 0.9,
        "account_type": 0.8,      # what kind of AD account is needed
        "required_by_date": 0.6,
        "approval_reference": 0.5,  # e.g. HOD/faculty approval reference
    },
    "printer_support": {
        "affected_system": 1.0,   # printer/device identifier or model
        "location": 0.9,          # which lab/floor/department
        "error_message": 0.9,
        "network_context": 0.5,   # shared network printer vs local/USB
        "frequency": 0.5,
    },
    # Fallback for messages that don't cleanly match any category above.
    # Kept deliberately minimal — this is a safety net, not a real category.
    "general": {
        "affected_system": 0.8,
        "error_message": 0.7,
        "location": 0.5,
    },
}

VALID_CATEGORIES: list[str] = list(CATEGORY_FIELD_PROFILES.keys())


def get_profile(category: str | None) -> dict[str, float]:
    """
    Return the field-weight profile for a category.
    Falls back to 'general' if category is None or unrecognized.
    """
    if category is None or category not in CATEGORY_FIELD_PROFILES:
        return CATEGORY_FIELD_PROFILES["general"]
    return CATEGORY_FIELD_PROFILES[category]