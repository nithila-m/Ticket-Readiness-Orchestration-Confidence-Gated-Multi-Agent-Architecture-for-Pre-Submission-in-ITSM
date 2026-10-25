# Review 2 report and deck — statements that no longer match the code

The Review 2 report and slides (22-08-2026) describe the project as it was
planned at that point. The items below have since changed in the
implementation. This list is for updating the next report; the Review 2
documents themselves are left unchanged.

| # | Topic | Review 2 says | The code now |
|---|---|---|---|
| 1 | LLM providers | Azure OpenAI | Google **Gemini** for Agent 1 (extraction) and **Groq** for Agents 2 (clarification) and 5 (classification). No Azure OpenAI dependency. |
| 2 | Audit / persistence database | PostgreSQL for audit logging of gate decisions (deck slide 9, objective 4) | **SQLite** (standard-library `sqlite3`, WAL mode) in `tro.db`: tables `tickets`, `audit_events`, `review_queue`. Conversation state itself is kept in memory. |
| 3 | Orchestration framework | LangGraph pipeline | Implemented in `app/orchestration/` as a LangGraph `StateGraph` (nodes: extract, kb, deflect, clarify, apply_feedback, close_resolved, correlate, classify, enqueue_review, dispatch) and used by the API by default (`ORCHESTRATOR=langgraph`). At Review 2 time the flow was a hand-written service; that service still exists as a fallback (`ORCHESTRATOR=service`) and both run the same step code and tests. No LangGraph checkpointer is used. |
| 4 | Number of agents | "a three-agent system" (deck slide 5, abstract) | **Six agents**: 1 Information Extractor, 2 Adaptive Clarifier, 3 KB Retrieval/Deflection, 4 Correlation Engine (duplicates + incident clusters), 5 Classify/Prioritise/Explain, 6 Ticket builder + Dispatcher. |
| 5 | Endpoints | `/analyze` and `/conversations/{id}/messages` (deck slide 11) | Also: `POST /conversations/{id}/deflection-feedback`, `GET /conversations/{id}/audit`, `GET /tickets`, `GET /tickets/{ticket_id}`, `GET /review`, `POST /review/{id}/approve`, `POST /review/{id}/reject`, `GET /metrics`, `GET /health`. |
| 6 | Clarification turn cap | — | After `MAX_CLARIFICATION_TURNS` (default 3), a further question is overridden to **ESCALATE**, and an escalation now goes to the human **review queue** (outcome `ESCALATED`) — no ticket is auto-created. (Deck slide 11 describes the forced ESCALATE correctly; check the report text for any wording that says a ticket is created at the cap.) |
| 7 | Decision actions | ASK_CLARIFICATION / READY / RECHECK / ESCALATE (deck slides 11, 16) | Plus **DEFLECTED** (strong KB match, Agent 2 skipped). Final outcomes: `SELF_RESOLVED`, `DISPATCHED`, `MERGED`, `PENDING_REVIEW`, `ESCALATED`. |
| 8 | Ticket dispatch | Integrated with Jira Service Management (deck slide 9) | Tickets are always written to SQLite first; Jira is an **optional**, best-effort sink (`JIRA_ENABLED`, off by default). A Jira failure is audited and never fails the conversation. |
| 9 | Agent 5 model | HGB classifier as the planned Agent-5 upgrade (deck slide 7, Sravani et al.) | Agent 5 is an LLM (Groq) with a deterministic rule fallback (confidence 0.3, always sent to review). The HGB option is future work and is not part of this build. |
| 10 | Status / next steps | Agents 3–5 still to build; "All 18 unit tests passed" (deck slides 13, 16) | Agents 3–6, the confidence gate, review queue, reviewer page, metrics and LangGraph orchestration are implemented; the automated test suite has grown accordingly (run `python -m pytest -q` for the current count). |
