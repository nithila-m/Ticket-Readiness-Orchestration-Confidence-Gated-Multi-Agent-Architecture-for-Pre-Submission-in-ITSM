"""scripts/agent1_cache_lib.py and the cache/variation scripts: fakes only,
no API calls, no model download, no real sleep."""

import json

import pytest

from app.providers.exceptions import LLMProviderError
from app.schemas.analysis import AnalysisResult
from app.schemas.extraction import CategoryPrediction, ExtractedField
from scripts.agent1_cache_lib import (
    QUOTA_BACKOFF_SECONDS,
    CallItem,
    build_input_text,
    is_quota_error,
    parse_retry_after,
    prompt_sha256,
    report_and_exit_code,
    run_cached_calls,
)
from scripts.cache_agent1_fields import analysis_to_entry, main as cache_main
from scripts.check_agent1_variation import compare_runs, select_tickets

HASH = prompt_sha256()


class FakeSleep:
    def __init__(self):
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


class FakeCall:
    """Raises the queued exceptions in order, then returns entries."""

    def __init__(self, errors=()):
        self.errors = list(errors)
        self.inputs: list[str] = []

    async def __call__(self, input_text: str) -> dict:
        self.inputs.append(input_text)
        if self.errors:
            raise self.errors.pop(0)
        return {"input_text": input_text, "category": "wifi_internet", "fields": {},
                "prompt_sha256": HASH}


def quota_error(text="429 RESOURCE_EXHAUSTED. You exceeded your current quota."):
    return LLMProviderError(f"Gemini API call failed: {text}", cause=RuntimeError(text))


def items(n):
    return [CallItem(f"T-{i}", f"T-{i}", f"User: ticket {i}.") for i in range(1, n + 1)]


async def run(item_list, cache, path, call, sleep, **kw):
    return await run_cached_calls(item_list, cache, path, call, prompt_hash=HASH,
                                  sleep=sleep, log=lambda _: None, **kw)


async def test_quota_error_twice_then_success_backs_off_then_delays(tmp_path):
    sleep, call = FakeSleep(), FakeCall([quota_error(), quota_error()])
    cache = {}

    result = await run(items(1), cache, tmp_path / "c.json", call, sleep, delay=6.0)

    assert sleep.calls == [30.0, 60.0, 6.0]
    assert result.calls_made == 3 and result.succeeded == 1 and result.stopped_reason is None
    assert "T-1" in json.loads((tmp_path / "c.json").read_text(encoding="utf-8"))


async def test_delay_after_every_call_including_non_quota_failures(tmp_path):
    sleep, call = FakeSleep(), FakeCall([LLMProviderError("schema validation failed")])

    result = await run(items(3), {}, tmp_path / "c.json", call, sleep, delay=6.0)

    assert sleep.calls == [6.0, 6.0, 6.0]
    assert result.failed == ["T-1"] and result.succeeded == 2


async def test_retry_after_in_error_is_honoured(tmp_path):
    sleep = FakeSleep()
    call = FakeCall([quota_error("429 RESOURCE_EXHAUSTED {'retryDelay': '75s'}")])

    await run(items(1), {}, tmp_path / "c.json", call, sleep, delay=6.0)

    assert sleep.calls == [75.0, 6.0]


async def test_quota_exhausted_after_four_retries_saves_and_exits_nonzero(tmp_path):
    path = tmp_path / "c.json"
    sleep = FakeSleep()
    call = FakeCall([quota_error()] * 5)
    cache = {"T-1": {"prompt_sha256": HASH, "category": "wifi_internet", "fields": {}}}

    result = await run(items(3), cache, path, call, sleep, delay=6.0)

    assert sleep.calls == [*QUOTA_BACKOFF_SECONDS, 6.0]
    assert result.stopped_reason == "quota" and result.calls_made == 5
    assert result.cached == 1 and result.remaining == 2
    assert list(json.loads(path.read_text(encoding="utf-8"))) == ["T-1"]
    lines = []
    assert report_and_exit_code(result, "rerun", log=lines.append) != 0
    assert any("Cached: 1, remaining: 2" in line for line in lines)


async def test_resume_skips_matching_hash_and_reruns_mismatched(tmp_path):
    cache = {
        "T-1": {"prompt_sha256": HASH, "category": "kept"},
        "T-2": {"prompt_sha256": "stale-prompt-hash", "category": "old"},
    }
    call = FakeCall()

    result = await run(items(2), cache, tmp_path / "c.json", call, FakeSleep())

    assert call.inputs == ["User: ticket 2."]
    assert result.skipped == 1 and result.succeeded == 1
    assert cache["T-1"]["category"] == "kept"
    assert cache["T-2"]["prompt_sha256"] == HASH


async def test_max_calls_stops_at_n_and_saves(tmp_path):
    path = tmp_path / "c.json"
    call = FakeCall()

    result = await run(items(5), {}, path, call, FakeSleep(), max_calls=2)

    assert len(call.inputs) == 2
    assert result.stopped_reason == "max_calls" and result.remaining == 3
    assert list(json.loads(path.read_text(encoding="utf-8"))) == ["T-1", "T-2"]


async def test_max_calls_counts_quota_retries(tmp_path):
    call = FakeCall([quota_error(), quota_error()])

    result = await run(items(1), {}, tmp_path / "c.json", call, FakeSleep(), max_calls=2)

    assert result.calls_made == 2 and result.stopped_reason == "max_calls"


def test_quota_detection_and_retry_after_parsing():
    assert is_quota_error(quota_error())
    assert is_quota_error(LLMProviderError("failed", cause=RuntimeError("Quota exceeded")))
    assert not is_quota_error(LLMProviderError("schema validation failed"))
    assert parse_retry_after(RuntimeError("Retry-After: 12")) == 12.0
    assert parse_retry_after(quota_error()) is None


def test_input_text_is_single_user_turn_transcript():
    assert build_input_text({"subject": "No wifi", "message": "Laptop drops."}) == (
        "User: No wifi. Laptop drops."
    )


def test_analysis_to_entry_shape():
    analysis = AnalysisResult(
        category=CategoryPrediction(value="printer_support", confidence=0.9),
        extracted_fields={"printer_model": ExtractedField(value="HP P1108", confidence=0.95)},
        completeness_score=0.4,
    )

    entry = analysis_to_entry("User: x.", analysis, "gemini-x", HASH, "2026-10-25T00:00:00+00:00")

    assert entry == {
        "input_text": "User: x.", "category": "printer_support", "category_confidence": 0.9,
        "fields": {"printer_model": {"value": "HP P1108", "confidence": 0.95}},
        "completeness_score": 0.4, "model": "gemini-x", "prompt_sha256": HASH,
        "extracted_at": "2026-10-25T00:00:00+00:00",
    }


def test_dry_run_makes_no_api_calls(tmp_path, monkeypatch, capsys):
    import scripts.cache_agent1_fields as mod

    def boom(*a, **k):
        raise AssertionError("dry run must not build the Gemini client")

    monkeypatch.setattr(mod, "make_analysis_call", boom)

    assert cache_main(["--dry-run", "--limit", "3", "--cache", str(tmp_path / "c.json")]) == 0
    out = capsys.readouterr().out
    assert "3 of 3 tickets need an API call" in out
    assert not (tmp_path / "c.json").exists()


def test_compare_runs_with_fake_embedder():
    def entry(cat, fields):
        return {"category": cat, "fields": {k: {"value": v, "confidence": 0.9} for k, v in fields.items()}}

    def embed(texts):
        # identical text -> same unit vector; different -> orthogonal-ish
        return [[1.0, 0.0] if t == texts[0] else [0.6, 0.8] for t in texts]

    runs = [
        entry("wifi_internet", {"device_type": "laptop", "symptom_type": "drops"}),
        entry("wifi_internet", {"device_type": "laptop", "symptom_type": "drops"}),
        entry("wifi_internet", {"device_type": "laptop", "symptom_type": "disconnects"}),
    ]

    r = compare_runs(runs, embed)

    assert r["same_category"] and r["same_field_names"] and not r["identical_text"]
    assert r["value_agreement"] == 0.5
    assert r["cosines"] == pytest.approx([1.0, 0.6, 0.6])


def test_variation_selection_includes_aug_dup_originals():
    source = json.loads(open("data/eval/tickets_augmented.json", encoding="utf-8").read())
    originals = {t["duplicate_of"] for t in source if t["label_type"] == "duplicate"}

    chosen = select_tickets(source)

    assert len(chosen) == len(set(chosen)) == 20
    assert originals <= set(chosen)
    assert chosen == select_tickets(source)
