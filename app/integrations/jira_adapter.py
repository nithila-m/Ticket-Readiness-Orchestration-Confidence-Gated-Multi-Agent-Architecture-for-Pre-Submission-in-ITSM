"""
Agent 6 - Jira Service Management adapter (REST API v3).

Implements ExternalTicketAdapter (app/integrations/base.py). The
Dispatcher calls it only AFTER the ticket is in SQLite and catches
anything raised here, so this module simply raises on failure
(raise_for_status) instead of swallowing errors - the error text ends up
in the dispatch audit row as external_error.

Rate limiting: HTTP 429 is retried after the Retry-After delay (default
2 s, capped at 10 s), at most MAX_RETRIES times. Every other non-2xx
status raises immediately.
"""

import asyncio
import re

import httpx

from app.integrations.base import ExternalRef
from app.schemas.pipeline import TicketRecord

MAX_RETRIES = 2
DEFAULT_RETRY_AFTER_SECONDS = 2.0
MAX_RETRY_AFTER_SECONDS = 10.0
SUMMARY_MAX_CHARS = 255

# Indirection so tests can make retries instant without patching the
# asyncio module for the whole event loop.
_sleep = asyncio.sleep


def to_adf(text: str) -> dict:
    """
    Plain text -> Atlassian Document Format. API v3 rejects plain-string
    descriptions/comments. One paragraph per non-empty line; an empty
    input still yields a valid (empty) document.
    """
    paragraphs = [
        {"type": "paragraph", "content": [{"type": "text", "text": line}]}
        for line in (ln.strip() for ln in text.splitlines())
        if line
    ]
    return {"type": "doc", "version": 1, "content": paragraphs}


def slug(text: str) -> str:
    """Jira labels cannot contain spaces: 'Desktop Support' -> 'desktop_support'."""
    return re.sub(r"[^a-z0-9_]+", "_", text.lower()).strip("_")


def _retry_after_seconds(response: httpx.Response) -> float:
    try:
        seconds = float(response.headers.get("Retry-After", DEFAULT_RETRY_AFTER_SECONDS))
    except ValueError:  # an HTTP-date instead of seconds
        seconds = DEFAULT_RETRY_AFTER_SECONDS
    return max(0.0, min(seconds, MAX_RETRY_AFTER_SECONDS))


class JiraAdapter:
    name = "jira"

    def __init__(
        self,
        base_url: str,
        email: str,
        api_token: str,
        project_key: str,
        issue_type: str,
        priority_map: dict[str, str],
        timeout_seconds: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._project_key = project_key
        self._issue_type = issue_type
        self._priority_map = priority_map
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            auth=(email, api_token),
            timeout=timeout_seconds,
            headers={"Accept": "application/json"},
            transport=transport,
        )

    @classmethod
    def from_settings(cls, settings, transport: httpx.AsyncBaseTransport | None = None) -> "JiraAdapter":
        missing = [
            name for name in ("jira_base_url", "jira_email", "jira_api_token")
            if not getattr(settings, name)
        ]
        if missing:
            raise ValueError(
                f"JIRA_ENABLED is true but {', '.join(m.upper() for m in missing)} "
                "is not set in .env"
            )
        return cls(
            base_url=settings.jira_base_url,
            email=settings.jira_email,
            api_token=settings.jira_api_token,
            project_key=settings.jira_project_key,
            issue_type=settings.jira_issue_type,
            priority_map=settings.jira_priority_map,
            timeout_seconds=settings.jira_timeout_seconds,
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _post(self, path: str, body: dict) -> httpx.Response:
        for attempt in range(MAX_RETRIES + 1):
            response = await self._client.post(path, json=body)
            if response.status_code != 429 or attempt == MAX_RETRIES:
                break
            await _sleep(_retry_after_seconds(response))
        response.raise_for_status()
        return response

    async def _comment(self, issue_key: str, text: str) -> None:
        await self._post(f"/rest/api/3/issue/{issue_key}/comment", {"body": to_adf(text)})

    def build_issue_fields(self, ticket: TicketRecord) -> dict:
        return {
            "project": {"key": self._project_key},
            "issuetype": {"name": self._issue_type},
            "summary": ticket.title[:SUMMARY_MAX_CHARS],
            "description": to_adf(ticket.description),
            "priority": {"name": self._priority_map[ticket.priority]},
            "labels": ["tro", slug(ticket.category), "team_" + slug(ticket.assignee_group)],
        }

    async def create(self, ticket: TicketRecord) -> ExternalRef:
        response = await self._post("/rest/api/3/issue", {"fields": self.build_issue_fields(ticket)})
        key = response.json()["key"]
        await self._comment(
            key,
            f"Routing rationale: {ticket.routing_rationale} "
            f"(confidence {ticket.confidence_score:.2f}). TRO ticket {ticket.ticket_id}.",
        )
        return ExternalRef(key, f"{self._base_url}/browse/{key}")

    async def link_duplicate(self, ticket: TicketRecord, existing: TicketRecord) -> None:
        # Nothing to link against if the original never reached Jira.
        if not existing.external_key:
            return
        # A merged ticket is normally never created in Jira, so it usually
        # has no key of its own; then only the comment is posted.
        if ticket.external_key:
            await self._post("/rest/api/3/issueLink", {
                "type": {"name": "Duplicate"},
                "inwardIssue": {"key": ticket.external_key},
                "outwardIssue": {"key": existing.external_key},
            })
        await self._comment(existing.external_key, f"Duplicate report merged: {ticket.ticket_id}")
