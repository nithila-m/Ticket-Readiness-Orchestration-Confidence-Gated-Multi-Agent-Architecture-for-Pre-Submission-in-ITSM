# TRO — Ticket Readiness Orchestration

A confidence-gated multi-agent prototype that works **before** an IT support
ticket is created. It checks whether a request is complete enough to act on,
whether a knowledge-base article already solves it, and whether it duplicates
an open ticket — and only then classifies, routes and dispatches it, with a
plain-language reason for the routing decision. Whenever the system is not
confident, it asks the user a question or hands the request to a human
reviewer instead of guessing.

BCSE497J Project I · VIT · Team: Trisha Annambhotla, Nithila, Medhini ·
Guide: Dr. Ushus Elizebeth Zachariah

---

## Architecture

Six agents, connected by deterministic gates. The order is orchestrated by a
LangGraph `StateGraph` (`app/orchestration/`); the same steps are also
available as a hand-written service (`ClarificationService`), selectable with
`ORCHESTRATOR`.

```
user message
  │
  ▼
Agent 1  Information Extractor (Gemini) ── structured fields + confidences,
  │                                         weighted completeness score
  ├─ completeness gate ── below KB_GATE_COMPLETENESS_THRESHOLD ──► Agent 2
  ▼
Agent 3  KB Retrieval / Deflection (ChromaDB + sentence-transformers)
  ├─ STRONG_MATCH ──► DEFLECTED: show the article, ask "did this fix it?"
  │                     ├─ yes ──► SELF_RESOLVED (no ticket)
  │                     └─ no  ──► ticket pipeline ▼
  ▼  WEAK_MATCH / NO_MATCH
Agent 2  Adaptive Clarifier (Groq) ── ASK_CLARIFICATION / RECHECK / READY / ESCALATE
  │                                    (hard turn budget: forced ESCALATE)
  ├─ ESCALATE ──► human review queue (ESCALATED)
  ▼  READY (with AUTO_DISPATCH_ON_READY=true)
Agent 4  Correlation Engine ── DUPLICATE / INCIDENT_CLUSTER / NO_MATCH
  ▼                             against open tickets in SQLite
Agent 5  Classify, Prioritise, Explain (Groq; rule fallback) ── category,
  │        priority (incident cluster ⇒ P1), team from the routing table, rationale
  ├─ confidence gate:
  │     DUPLICATE                         ──► MERGED (linked to the existing ticket)
  │     confidence < CLASSIFICATION_CONFIDENCE_THRESHOLD ──► review queue (PENDING_REVIEW)
  ▼     otherwise
Agent 6  Ticket builder + Dispatcher ── SQLite first (source of truth), then
                                         Jira Service Management (optional, best-effort)
                                         ──► DISPATCHED
```

Every stage writes an audit row (`audit_events` in SQLite) with its decision,
confidence and latency. A reviewer can approve (optionally overriding priority
or team) or reject queued requests.

## Setup

Requires Python 3.11.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only torch first
python -m pip install -r requirements.txt
```

(Any Python 3.11 environment works, e.g. a conda env — activate it instead of
`.venv`.)

Copy `.env.example` to `.env` and fill in your own values. Never commit `.env`.

| Key | Purpose |
|---|---|
| `GEMINI_API_KEY`, `GEMINI_MODEL`, `LLM_PROVIDER` | Agent 1 (extraction) |
| `GROQ_API_KEY` | Agents 2 and 5 |
| `APP_ENV` | shown by `/health` |
| `AUTO_DISPATCH_ON_READY` | `true` = run Agents 4–6 when clarification reaches READY |
| `ORCHESTRATOR` | `langgraph` (default) or `service` |
| `JIRA_ENABLED`, `JIRA_BASE_URL`, `JIRA_EMAIL`, `JIRA_API_TOKEN`, `JIRA_PROJECT_KEY`, `JIRA_ISSUE_TYPE` | optional Jira dispatch |

Other thresholds (`KB_*`, `CORR_*`, `CLASSIFICATION_CONFIDENCE_THRESHOLD`,
`MAX_CLARIFICATION_TURNS`, `SQLITE_PATH`) have defaults in
`app/config/settings.py` and can be overridden the same way.

## Run

Seed the open-ticket pool that Agent 4 correlates against (asks before
deleting existing tickets):

```powershell
python -m scripts.seed_ticket_store --reset
```

Start the server **from the repository root** (the frontend is mounted with a
relative path):

```powershell
python -m uvicorn app.main:app --reload
```

| URL | What |
|---|---|
| http://127.0.0.1:8000/chat/ | user chat (ticket card, KB feedback, audit trail) |
| http://127.0.0.1:8000/chat/reviewer.html | human review queue |
| http://127.0.0.1:8000/docs | interactive API docs |
| http://127.0.0.1:8000/metrics | outcome rates, ticket counts, latency per stage |

The SQLite database (`tro.db` by default) is created on startup.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/conversations/{id}/messages` | one conversational turn |
| POST | `/conversations/{id}/deflection-feedback` | `{"resolved": bool}` after a KB deflection |
| GET | `/conversations/{id}/audit` | audit trail, in order |
| GET | `/tickets?status=&limit=` · `/tickets/{ticket_id}` | ticket store |
| GET | `/review?status=PENDING` | review queue (`PENDING`/`APPROVED`/`REJECTED`/`ALL`) |
| POST | `/review/{id}/approve` · `/review/{id}/reject` | reviewer decision (approve accepts optional `priority`, `assignee_group`) |
| GET | `/metrics` | aggregate metrics |
| POST | `/analyze` | Agent 1 only (single-shot evaluation endpoint) |
| GET | `/health` | liveness |

A message to a conversation that already has a final outcome returns `409`.
LLM provider failures return `503`.

## Jira Service Management (optional)

1. Create an API token at id.atlassian.com → Security → API tokens.
2. Set `JIRA_BASE_URL`, `JIRA_EMAIL`, `JIRA_API_TOKEN` and `JIRA_PROJECT_KEY` in `.env`.
3. Check the connection (read-only, prints names only):
   ```powershell
   python -m scripts.smoke_test_jira
   ```
   Set `JIRA_ISSUE_TYPE` to an issue type it lists.
4. Set `JIRA_ENABLED=true` and restart.

Tickets are always written to SQLite first; a Jira failure is recorded in the
audit trail and never fails the user's conversation.

## Tests

```powershell
python -m pytest -q
```

The suite uses fakes, `httpx.MockTransport` and temporary SQLite files — it
never calls Gemini, Groq or Jira, never downloads a model, and never touches
`tro.db`.

## Layout

```
app/agents/          Agents 1-6, gates and decision rules
app/orchestration/   LangGraph state, pure routers, nodes, graph, service
app/services/        ClarificationService (turn steps), TicketPipeline (Agents 4-6)
app/storage/         SQLite store: tickets, audit_events, review_queue
app/integrations/    external ticket adapters (Jira, local-only)
app/api/             FastAPI routers
frontend/            chat page and reviewer page (vanilla JS)
scripts/             seeding, smoke tests, evaluation scripts
tests/               unit and API tests
```

## Known limitations

- Conversation state is held in memory: tickets, audit rows and reviews
  survive a restart (SQLite), open conversations do not.
- All evaluation uses synthetic tickets; results are directional, not
  production performance claims.
