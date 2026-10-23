"""
Routing tables owned by Agent 5 (Classify, Prioritise, Explain).

assignee_group is ALWAYS looked up here, never taken from the LLM - the
model has no way to know the real team roster, and letting it name a
team would let a hallucinated team name reach the dispatcher.

Keys are the internal category keys from category_profiles.py. If a
category is added there, add its team here too (DEFAULT_TEAM catches
anything missing, but silently - tests/unit/test_classification_agent.py
checks the two stay in sync).
"""

CATEGORY_TO_TEAM: dict[str, str] = {
    "wifi_internet": "Network Operations",
    "ms_teams": "Collaboration Tools Support",
    "vit_email": "Messaging Support",
    "ad_account_creation": "Identity & Access Management",
    "printer_support": "Desktop Support",
    "general": "Service Desk L1",
}
DEFAULT_TEAM = "Service Desk L1"

# Synthetic dataset priority labels -> TRO priority. Eval-only: the
# dataset has no P1 label (P1 is only ever set by the incident-cluster
# override), so P1 predictions have no ground truth to score against.
DATASET_PRIORITY_TO_P: dict[str, str] = {
    "HIGH": "P2",
    "MEDIUM": "P3",
    "LOW": "P4",
}


def team_for_category(category: str | None) -> str:
    """Look up the support team for an internal category key."""
    if category is None:
        return DEFAULT_TEAM
    return CATEGORY_TO_TEAM.get(category, DEFAULT_TEAM)
