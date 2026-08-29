from app.config.category_profiles import (
    CATEGORY_FIELD_PROFILES,
    VALID_CATEGORIES,
    get_profile,
)


def test_all_expected_categories_present():
    expected = {
        "wifi_internet",
        "vit_email",
        "ms_teams",
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


def test_categories_have_distinct_field_sets():
    # Sanity check that categories aren't accidentally identical copies
    field_sets = [frozenset(fields.keys()) for fields in CATEGORY_FIELD_PROFILES.values()]
    assert len(set(field_sets)) == len(field_sets), "Two categories have identical field sets"


def test_get_profile_returns_correct_profile():
    profile = get_profile("printer_support")
    assert profile == CATEGORY_FIELD_PROFILES["printer_support"]


def test_get_profile_falls_back_on_none():
    profile = get_profile(None)
    assert profile == CATEGORY_FIELD_PROFILES["general"]


def test_get_profile_falls_back_on_unknown_category():
    profile = get_profile("some_made_up_category")
    assert profile == CATEGORY_FIELD_PROFILES["general"]


def test_ad_account_creation_has_expected_fields():
    profile = get_profile("ad_account_creation")
    assert "requester_role" in profile
    assert "department" in profile


def test_ms_teams_has_expected_fields():
    profile = get_profile("ms_teams")
    assert "trigger" in profile
    assert "affected_system" in profile