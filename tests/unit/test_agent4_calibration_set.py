"""data/eval/agent4_calibration_extra.json and its generator."""

import json
from collections import Counter

import pytest

from scripts.build_agent4_calibration_set import (
    CATEGORIES,
    OUT_PATH,
    build_records,
    category_of,
    load_source,
    render,
    validate,
)


@pytest.fixture(scope="module")
def source():
    return load_source()


@pytest.fixture(scope="module")
def committed():
    return json.loads(OUT_PATH.read_text(encoding="utf-8"))


def test_committed_file_matches_generator_output(source):
    assert OUT_PATH.read_text(encoding="utf-8") == render(build_records(source))


def test_committed_file_validates(committed, source):
    validate(committed, source)


def test_ids_unique_and_well_formed(committed):
    ids = [r["ticket_id"] for r in committed]
    assert len(ids) == len(set(ids))
    assert sorted(i for i in ids if i.startswith("CAL-DUP-")) == [f"CAL-DUP-{n:02d}" for n in range(1, 26)]
    assert sorted(i for i in ids if i.startswith("CAL-NEG-")) == [f"CAL-NEG-{n:02d}" for n in range(1, 16)]


def test_references_exist_and_labels_valid(committed, source):
    by_id = {t["ticket_id"]: t for t in source}
    assert Counter(r["label_type"] for r in committed) == {
        "duplicate": 25, "hard_negative_near_miss": 15,
    }
    for r in committed:
        ref = r["duplicate_of"] or r["hard_negative_of"]
        assert ref in by_id
        assert by_id[ref]["label_type"] == "background"
        assert category_of(r) in CATEGORIES


def test_same_schema_as_tickets_augmented(committed, source):
    keys = list(source[0])
    assert all(list(r) == keys for r in committed)


def test_validate_rejects_bad_reference(committed, source):
    broken = [dict(r) for r in committed]
    broken[0]["duplicate_of"] = "NOPE-999"
    with pytest.raises(ValueError, match="unknown ticket"):
        validate(broken, source)


def test_validate_rejects_duplicate_ids(committed, source):
    broken = [dict(r) for r in committed]
    broken[1]["ticket_id"] = broken[0]["ticket_id"]
    with pytest.raises(ValueError, match="duplicate ticket_id"):
        validate(broken, source)


def test_validate_rejects_invalid_label(committed, source):
    broken = [dict(r) for r in committed]
    broken[0]["label_type"] = "maybe_duplicate"
    with pytest.raises(ValueError, match="invalid label_type"):
        validate(broken, source)
