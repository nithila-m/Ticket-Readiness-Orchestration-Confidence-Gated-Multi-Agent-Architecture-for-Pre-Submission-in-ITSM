import json
from pathlib import Path

import pytest

from app.config.category_profiles import (
    CATEGORY_DISPLAY_TO_KEY,
    CATEGORY_FIELD_PROFILES,
    VALID_CATEGORIES,
    category_key_to_display,
    display_labels_for_key,
    get_profile,
    normalize_category_label,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TICKET_DATA_DIRS = [_REPO_ROOT / "data" / "eval", _REPO_ROOT / "chroma_db" / "data" / "tickets"]


def test_all_expected_categories_present():
    expected = {
        "wifi_internet",
        "ms_teams",
        "vit_email",
        "ad_account_creation",
        "printer_support",
        "general",
    }
    assert expected.issubset(set(VALID_CATEGORIES))


def test_all_weights_in_valid_range():
    for category, fields in CATEGORY_FIELD_PROFILES.items():
        for field_name, weight in fields.items():
            assert 0.0 < weight <= 1.0, f"{category}.{field_name} weight out of range"


def test_each_category_has_at_least_one_field():
    for category, fields in CATEGORY_FIELD_PROFILES.items():
        assert len(fields) > 0, f"{category} has no fields defined"


def test_teams_and_email_share_identical_vocabulary_by_design():
    # Not a bug - both categories fail in the same shape (SSO/app-layer
    # issues). This test documents that the overlap is intentional,
    # so a future "fix" doesn't accidentally diverge them without reason.
    assert set(CATEGORY_FIELD_PROFILES["ms_teams"].keys()) == set(
        CATEGORY_FIELD_PROFILES["vit_email"].keys()
    )


def test_get_profile_returns_correct_profile():
    profile = get_profile("printer_support")
    assert profile == CATEGORY_FIELD_PROFILES["printer_support"]


def test_get_profile_falls_back_on_none():
    assert get_profile(None) == CATEGORY_FIELD_PROFILES["general"]


def test_get_profile_falls_back_on_unknown_category():
    assert get_profile("some_made_up_category") == CATEGORY_FIELD_PROFILES["general"]


def test_ad_account_creation_has_expected_fields():
    profile = get_profile("ad_account_creation")
    assert "error_or_symptom" in profile
    assert "username_domain" in profile


def test_ms_teams_has_expected_fields():
    profile = get_profile("ms_teams")
    assert "failure_type" in profile
    assert "error_signal" in profile


def test_normalize_category_label_maps_known_display_names():
    assert normalize_category_label("Wifi/Internet Support") == "wifi_internet"
    assert normalize_category_label("AD Account Creation") == "ad_account_creation"
    assert normalize_category_label("Microsoft Teams Support") == "ms_teams"
    assert normalize_category_label("VIT Email Support") == "vit_email"
    assert normalize_category_label("Printer Support") == "printer_support"


def test_normalize_category_label_returns_none_for_unrecognized():
    assert normalize_category_label("Some Random Category") is None


def test_normalize_category_label_handles_none():
    assert normalize_category_label(None) is None


def test_normalize_category_label_maps_drift_aliases():
    assert normalize_category_label("AD Account Support") == "ad_account_creation"
    assert normalize_category_label("VIT Email") == "vit_email"


def test_key_to_display_returns_canonical_label_not_alias():
    assert category_key_to_display("ad_account_creation") == "AD Account Creation"
    assert category_key_to_display("vit_email") == "VIT Email Support"
    assert category_key_to_display("general") is None
    assert category_key_to_display(None) is None


def test_every_mapped_key_has_a_canonical_label_that_round_trips():
    for key in set(CATEGORY_DISPLAY_TO_KEY.values()):
        assert key in VALID_CATEGORIES
        label = category_key_to_display(key)
        assert normalize_category_label(label) == key


def test_display_labels_for_key_lists_canonical_first_then_aliases():
    assert display_labels_for_key("vit_email") == ["VIT Email Support", "VIT Email"]
    assert display_labels_for_key("printer_support") == ["Printer Support"]
    assert display_labels_for_key("general") == []


def _dataset_category_labels() -> set[str]:
    labels: set[str] = set()
    for data_dir in _TICKET_DATA_DIRS:
        for path in sorted(data_dir.glob("*.json")):
            for ticket in json.loads(path.read_text(encoding="utf-8")):
                for field in ("true_category", "category_selected"):
                    if ticket.get(field):
                        labels.add(ticket[field])
    return labels


def test_every_dataset_category_label_normalizes_to_a_valid_key():
    labels = _dataset_category_labels()
    if not labels:
        pytest.skip("no ticket data files found")
    unmapped = sorted(label for label in labels if normalize_category_label(label) not in VALID_CATEGORIES)
    assert unmapped == []
