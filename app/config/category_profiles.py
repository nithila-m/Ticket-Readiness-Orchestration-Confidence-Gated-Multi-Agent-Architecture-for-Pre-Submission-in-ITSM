"""
Category-specific field weight profiles for completeness scoring.

Field vocabulary here is sourced directly from the team's synthetic
evaluation dataset generator (the actual `missing_fields` checklists
used to create the 30-ticket sets per category) - NOT invented
independently. If the dataset's checklist changes, this file must
change with it.

Weights are still a judgment call (no labeled ground-truth score exists
to fit them against - see prior discussion on why backprop doesn't
apply here). They reflect how much each field matters to a human
agent actually acting on the ticket, not measured importance.
"""

CATEGORY_FIELD_PROFILES: dict[str, dict[str, float]] = {
    "wifi_internet": {
        "symptom_type": 1.0,               # won't connect / drops / slow, etc.
        "single_or_multiple_devices": 0.8,  # scopes the problem: device vs. infra
        "when_started": 0.7,
        "device_type": 0.6,
        "ssid": 0.5,
    },
    "ms_teams": {
        "failure_type": 1.0,
        "scope": 0.8,                      # one user vs. everyone in a call/team
        "error_signal": 0.7,
        "device_platform": 0.6,
    },
    "vit_email": {
        "failure_type": 1.0,
        "scope": 0.8,
        "error_signal": 0.7,
        "device_platform": 0.6,
    },
    "ad_account_creation": {
        "error_or_symptom": 1.0,
        "username_domain": 0.7,
        "when_started": 0.6,
        "device_context": 0.6,
        "troubleshooting_done": 0.5,
    },
    "printer_support": {
        "symptom": 1.0,
        "scope": 0.8,                      # one printer vs. all printers on a floor
        "error_message": 0.7,
        "connection_type": 0.6,
        "printer_model": 0.6,
        "when_started": 0.6,
        "troubleshooting_done": 0.5,
    },
    # Fallback for messages that don't cleanly match any category above.
    # Deliberately generic - not tied to any one category's real checklist.
    "general": {
        "symptom_or_error": 1.0,
        "when_started": 0.6,
        "scope": 0.5,
    },
}

VALID_CATEGORIES: list[str] = list(CATEGORY_FIELD_PROFILES.keys())

# Maps the human-readable category labels used in the synthetic dataset
# (category_selected / true_category) to this codebase's internal keys.
# The first label listed for each key is its canonical display label;
# entries after the canonical five are aliases for label drift found in
# the data ("AD Account Support" in 5 tickets' true_category, "VIT Email"
# in 3 tickets' true_category and in every email KB article's category).
CATEGORY_DISPLAY_TO_KEY: dict[str, str] = {
    "Wifi/Internet Support": "wifi_internet",
    "Microsoft Teams Support": "ms_teams",
    "VIT Email Support": "vit_email",
    "AD Account Creation": "ad_account_creation",
    "Printer Support": "printer_support",
    # aliases
    "AD Account Support": "ad_account_creation",
    "VIT Email": "vit_email",
}

# Reverse lookup, canonical labels only (first label wins, so aliases
# above never leak back out as a display label).
CATEGORY_KEY_TO_DISPLAY: dict[str, str] = {}
for _label, _key in CATEGORY_DISPLAY_TO_KEY.items():
    CATEGORY_KEY_TO_DISPLAY.setdefault(_key, _label)


def normalize_category_label(display_name: str | None) -> str | None:
    """
    Convert a human-readable dataset category label to this codebase's
    internal snake_case key. Returns None if unrecognized (does NOT
    fall back to 'general' here - that's the scorer's job, not this
    function's; conflating the two would hide a real labeling bug
    behind a silent fallback).
    """
    if display_name is None:
        return None
    return CATEGORY_DISPLAY_TO_KEY.get(display_name)


def category_key_to_display(category: str | None) -> str | None:
    """Canonical display label for an internal key, or None if unknown."""
    if category is None:
        return None
    return CATEGORY_KEY_TO_DISPLAY.get(category)


def display_labels_for_key(category: str | None) -> list[str]:
    """
    Every display label (canonical first, then aliases) that normalizes to
    `category`. Used for matching stored data that may carry an alias
    (e.g. KB metadata says "VIT Email"); never use it to pick a label to
    show - use category_key_to_display() for that.
    """
    return [label for label, key in CATEGORY_DISPLAY_TO_KEY.items() if key == category]


def get_profile(category: str | None) -> dict[str, float]:
    """
    Return the field-weight profile for an internal category key.
    Falls back to 'general' if category is None or unrecognized.
    """
    if category is None or category not in CATEGORY_FIELD_PROFILES:
        return CATEGORY_FIELD_PROFILES["general"]
    return CATEGORY_FIELD_PROFILES[category]