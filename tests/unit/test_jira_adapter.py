"""
Unit tests for app/integrations/jira_adapter.py. Every request goes to an
httpx.MockTransport - no network, no real Jira site or token.
"""

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest

from app.agents.dispatcher import Dispatcher
from app.config.routing import team_for_category
from app.integrations import jira_adapter
from app.integrations.base import ExternalRef, ExternalTicketAdapter
from app.integrations.jira_adapter import JiraAdapter, slug, to_adf
from app.schemas.conversation import ConversationState, Message
from app.schemas.extraction import CategoryPrediction, ExtractedField
from app.schemas.pipeline import ClassificationResult, TicketRecord
from app.storage.sqlite_store import SqliteStore

BASE_URL = "https://tro-demo.atlassian.net"
PRIORITY_MAP = {"P1": "Highest", "P2": "High", "P3": "Medium", "P4": "Low"}


def make_adapter(handler, issue_type: str = "Service Request") -> JiraAdapter:
    return JiraAdapter(
        base_url=BASE_URL + "/",  # trailing slash must not produce "//browse"
        email="bot@example.com",
        api_token="not-a-real-token",
        project_key="TRO",
        issue_type=issue_type,
        priority_map=PRIORITY_MAP,
        transport=httpx.MockTransport(handler),
    )


def make_ticket(**overrides) -> TicketRecord:
    data = dict(
        ticket_id="TRO-2026-00001",
        conversation_id="conv-1",
        title="[Printer Support] paper jam",
        description="Reported by user:\nprinter jams\n\nCompleteness: 0.80",
        category="printer_support",
        priority="P2",
        assignee_group="Desktop Support",
        routing_rationale="Printer jam on a shared printer.",
        confidence_score=0.87,
        created_at=datetime(2026, 10, 24, 8, 0, tzinfo=timezone.utc),
    )
    data.update(overrides)
    return TicketRecord(**data)


class Recorder:
    """MockTransport handler that records requests and replays canned responses."""

    def __init__(self, *responses: httpx.Response):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.responses.pop(0)

    def body(self, i: int) -> dict:
        return json.loads(self.requests[i].content)


def created(key: str = "TRO-12") -> httpx.Response:
    return httpx.Response(201, json={"id": "10001", "key": key})


def comment_ok() -> httpx.Response:
    return httpx.Response(201, json={"id": "20001"})


@pytest.fixture
def instant_sleep(monkeypatch):
    calls: list[float] = []

    async def fake_sleep(seconds):
        calls.append(seconds)

    monkeypatch.setattr(jira_adapter, "_sleep", fake_sleep)
    return calls


# --- helpers ---------------------------------------------------------------

def test_to_adf_one_paragraph_per_non_empty_line():
    doc = to_adf("first line\n\n  second line  \n")
    assert doc["type"] == "doc" and doc["version"] == 1
    assert [p["content"][0]["text"] for p in doc["content"]] == ["first line", "second line"]


def test_to_adf_empty_text_is_still_a_valid_doc():
    assert to_adf("") == {"type": "doc", "version": 1, "content": []}


def test_slug_has_no_spaces():
    assert slug("Identity & Access Management") == "identity_access_management"
    assert slug("Desktop Support") == "desktop_support"
    assert slug("wifi_internet") == "wifi_internet"


def test_adapter_satisfies_protocol():
    assert isinstance(make_adapter(Recorder()), ExternalTicketAdapter)


# --- create ------------------------------------------------------------------

async def test_create_request_body():
    rec = Recorder(created(), comment_ok())
    adapter = make_adapter(rec, issue_type="Service Request")
    await adapter.create(make_ticket())

    req = rec.requests[0]
    assert req.method == "POST" and req.url.path == "/rest/api/3/issue"
    assert req.headers["Accept"] == "application/json"
    assert req.headers["Authorization"].startswith("Basic ")

    fields = rec.body(0)["fields"]
    assert fields["project"] == {"key": "TRO"}
    assert fields["issuetype"] == {"name": "Service Request"}
    assert fields["priority"] == {"name": "High"}  # P2
    assert fields["description"]["type"] == "doc"
    assert fields["description"]["content"][0]["content"][0]["text"] == "Reported by user:"
    assert fields["labels"] == ["tro", "printer_support", "team_desktop_support"]
    assert all(" " not in label for label in fields["labels"])


async def test_create_truncates_summary_to_255_chars():
    rec = Recorder(created(), comment_ok())
    # TicketRecord itself caps title at 255, so bypass validation to prove
    # the adapter enforces Jira's limit on its own.
    ticket = make_ticket().model_copy(update={"title": "y" * 300})
    await make_adapter(rec).create(ticket)
    assert len(rec.body(0)["fields"]["summary"]) == 255


async def test_create_returns_key_and_browse_url_and_posts_rationale_comment():
    rec = Recorder(created("TRO-12"), comment_ok())
    ref = await make_adapter(rec).create(make_ticket())

    assert ref == ExternalRef("TRO-12", f"{BASE_URL}/browse/TRO-12")
    assert rec.requests[1].url.path == "/rest/api/3/issue/TRO-12/comment"
    text = rec.body(1)["body"]["content"][0]["content"][0]["text"]
    assert text == (
        "Routing rationale: Printer jam on a shared printer. (confidence 0.87). "
        "TRO ticket TRO-2026-00001."
    )


async def test_create_retries_after_429(instant_sleep):
    rec = Recorder(httpx.Response(429, headers={"Retry-After": "0"}), created("TRO-13"), comment_ok())
    ref = await make_adapter(rec).create(make_ticket())
    assert ref.key == "TRO-13"
    assert instant_sleep == [0.0]
    assert len(rec.requests) == 3


async def test_retry_after_is_capped_and_defaulted(instant_sleep):
    rec = Recorder(
        httpx.Response(429, headers={"Retry-After": "120"}),
        httpx.Response(429),  # no header -> default
        created(), comment_ok(),
    )
    await make_adapter(rec).create(make_ticket())
    assert instant_sleep == [10.0, 2.0]


async def test_gives_up_after_two_retries(instant_sleep):
    rec = Recorder(*(httpx.Response(429, headers={"Retry-After": "0"}) for _ in range(3)))
    with pytest.raises(httpx.HTTPStatusError):
        await make_adapter(rec).create(make_ticket())
    assert len(rec.requests) == 3  # 1 attempt + 2 retries


async def test_401_raises_without_retry(instant_sleep):
    rec = Recorder(httpx.Response(401, json={"errorMessages": ["unauthorized"]}))
    with pytest.raises(httpx.HTTPStatusError):
        await make_adapter(rec).create(make_ticket())
    assert len(rec.requests) == 1
    assert instant_sleep == []


# --- link_duplicate --------------------------------------------------------

async def test_link_duplicate_posts_link_and_comment():
    rec = Recorder(httpx.Response(201), comment_ok())
    new = make_ticket(ticket_id="TRO-2026-00002", external_key="TRO-20")
    existing = make_ticket(external_key="TRO-12")
    await make_adapter(rec).link_duplicate(new, existing)

    assert rec.requests[0].url.path == "/rest/api/3/issueLink"
    assert rec.body(0) == {
        "type": {"name": "Duplicate"},
        "inwardIssue": {"key": "TRO-20"},
        "outwardIssue": {"key": "TRO-12"},
    }
    assert rec.requests[1].url.path == "/rest/api/3/issue/TRO-12/comment"
    assert "Duplicate report merged: TRO-2026-00002" in json.dumps(rec.body(1))


async def test_link_duplicate_without_new_jira_key_only_comments():
    rec = Recorder(comment_ok())
    new = make_ticket(ticket_id="TRO-2026-00002", status="MERGED")
    await make_adapter(rec).link_duplicate(new, make_ticket(external_key="TRO-12"))
    assert [r.url.path for r in rec.requests] == ["/rest/api/3/issue/TRO-12/comment"]


async def test_link_duplicate_is_noop_when_existing_has_no_jira_key():
    rec = Recorder()
    await make_adapter(rec).link_duplicate(make_ticket(ticket_id="TRO-2026-00002"), make_ticket())
    assert rec.requests == []


# --- from_settings / lifecycle -------------------------------------------

def _settings(**overrides):
    data = dict(
        jira_base_url=BASE_URL, jira_email="bot@example.com", jira_api_token="t",
        jira_project_key="TRO", jira_issue_type="Task", jira_priority_map=PRIORITY_MAP,
        jira_timeout_seconds=5.0,
    )
    data.update(overrides)
    return SimpleNamespace(**data)


async def test_from_settings_builds_working_adapter():
    rec = Recorder(created("TRO-1"), comment_ok())
    adapter = JiraAdapter.from_settings(_settings(), transport=httpx.MockTransport(rec))
    assert (await adapter.create(make_ticket())).url == f"{BASE_URL}/browse/TRO-1"
    assert rec.body(0)["fields"]["issuetype"] == {"name": "Task"}
    await adapter.aclose()


def test_from_settings_rejects_missing_credentials():
    with pytest.raises(ValueError, match="JIRA_API_TOKEN"):
        JiraAdapter.from_settings(_settings(jira_api_token=""))


# --- Dispatcher + Jira failure ----------------------------------------------

async def test_dispatcher_with_jira_401_still_saves_ticket(tmp_path):
    store = SqliteStore(tmp_path / "jira.db")
    store.init_db()
    adapter = make_adapter(Recorder(httpx.Response(401)))
    state = ConversationState(
        conversation_id="conv-1",
        raw_messages=[Message(role="user", content="printer jams")],
        extracted_fields={"symptom": ExtractedField(value="paper jam", confidence=0.9)},
        detected_category=CategoryPrediction(value="printer_support", confidence=0.9),
        classification=ClassificationResult(
            category="printer_support", priority="P3",
            assignee_group=team_for_category("printer_support"),
            routing_rationale="Printer jam.", confidence=0.8, source="stub",
        ),
    )

    result = await Dispatcher(store, adapter).dispatch(state)

    assert result.outcome == "DISPATCHED"
    assert result.external_error.startswith("HTTPStatusError")
    stored = await store.get_ticket(result.ticket.ticket_id)
    assert stored is not None and stored.status == "OPEN" and stored.external_key is None
    audit = await store.fetch_audit("conv-1")
    assert audit[0].payload["adapter"] == "jira"
    assert "401" in audit[0].payload["external_error"]
    await adapter.aclose()


def test_lifespan_closes_jira_client(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    import app.dependencies as deps
    from app.main import app

    monkeypatch.setattr(deps.settings, "jira_enabled", True)
    monkeypatch.setattr(deps.settings, "jira_base_url", BASE_URL)
    monkeypatch.setattr(deps.settings, "jira_email", "bot@example.com")
    monkeypatch.setattr(deps.settings, "jira_api_token", "t")
    deps.get_dispatcher.cache_clear()
    app.dependency_overrides[deps.get_store] = lambda: SqliteStore(tmp_path / "l.db")
    try:
        adapter = deps.get_dispatcher().adapter
        assert isinstance(adapter, JiraAdapter)
        with TestClient(app):
            pass
        assert adapter._client.is_closed
    finally:
        app.dependency_overrides.pop(deps.get_store, None)
        deps.get_dispatcher.cache_clear()
