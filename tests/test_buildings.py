from app.config.buildings import normalize_building


def test_block_suffix_variants_normalize_to_same_key():
    assert normalize_building("Men's Hostel A") == normalize_building("Men's Hostel A-Block")
    assert normalize_building("Ladies Hostel D") == normalize_building("Ladies Hostel D-Block")
    assert normalize_building("Men's Hostel B Block") == "men's hostel b"


def test_case_and_whitespace_ignored():
    assert normalize_building("  sjt ") == normalize_building("SJT") == "sjt"


def test_distinct_buildings_stay_distinct():
    assert normalize_building("Ladies Hostel C-Block") != normalize_building("Ladies Hostel D-Block")
    assert normalize_building("TT") != normalize_building("SJT")


def test_empty_values_are_none():
    assert normalize_building(None) is None
    assert normalize_building("   ") is None
