# Full-Pipeline Evaluation (slice 121-150)

Orchestrator: `langgraph`. Agent 2 model: `openai/gpt-oss-20b`; Agent 5 model: `openai/gpt-oss-20b` (Groq). Agent 1: replayed from `data/eval/agent1_fields_cache.json`.

## Limitations

- **Synthetic labels.** Categories, priorities, completeness levels, deflectability and duplicate clusters were set by the data generator, not independent annotators.
- **Single run.** LLM output (Agents 2 and 5) is non-deterministic; one sample, not a stable estimate.
- **Agent 1 replayed** from one cached Gemini run, so its latency is not measured here (see the Agent 1 eval, `results/agent1_eval_results.csv`).
- **Single user turn.** A ticket that gets a clarifying question is stopped there (STOPPED_AT_ASK); the follow-up turns are not simulated.
- **Empty starting pool.** Agent 4 only sees tickets dispatched earlier in the same run, so duplicates can only come from earlier tickets in the fixed order.
- **Scripted deflection feedback** uses the dataset label (`is_deflectable`), not a real user.
- **Agent 5 timeout raised to 150 s** (eval only; production default 20 s) so the 429 retry wait (15/30/60 s) is not turned into a rule fallback.
- **auto_dispatch_on_ready forced on** for this run; Jira is never called (NullAdapter).
- **No P1 ground truth.** The dataset has no P1 label; any P1 prediction counts as wrong.
- Tickets that errored (0) are excluded from every metric except the funnel outcome counts and the Errors section.

## Slices

| Slice | Ticket range (fixed order) | First id | Last id | n |
|---|---|---|---|---|
| 121-150 | 121-150 | WIFI-001 | WIFI-030 | 30 |

Total tickets: 30

## 1. Funnel

Percentages are of the 30 tickets without errors unless stated.

| Stage | Count |
|---|---|
| Completeness gate passed (reached Agent 3) | 18 (60.0%) |
| KB STRONG_MATCH (% of gate passed) | 2 (11.1%) |
| KB WEAK_MATCH (% of gate passed) | 16 (88.9%) |
| KB NO_MATCH (% of gate passed) | 0 (0.0%) |
| Agent 2 ran on the first turn | 28 (93.3%) |
| - ASK_CLARIFICATION (% of Agent 2) | 18 (64.3%) |
| - READY (% of Agent 2) | 10 (35.7%) |
| - ESCALATE (% of Agent 2) | 0 (0.0%) |
| - other, e.g. RECHECK (% of Agent 2) | 0 (0.0%) |
| DEFLECTED | 2 (6.7%) |
| - feedback resolved / not resolved | 1 / 1 |
| Reached Agent 4 | 11 (36.7%) |
| Reached Agent 5 | 11 (36.7%) |

Final outcomes (% of all 30 tickets):

| Outcome | Count |
|---|---|
| SELF_RESOLVED | 1 (3.3%) |
| DISPATCHED | 11 (36.7%) |
| MERGED | 0 (0.0%) |
| PENDING_REVIEW | 0 (0.0%) |
| ESCALATED | 0 (0.0%) |
| STOPPED_AT_ASK | 18 (60.0%) |
| STOPPED_AT_RECHECK | 0 (0.0%) |
| ERROR | 0 (0.0%) |

## 2. First-turn action vs completeness label

Rows = dataset completeness_level, columns = first-turn decision (DEFLECTED means Agent 3 short-circuited Agent 2). ASK on `complete` tickets is over-asking.

| Level | ASK_CLARIFICATION | READY | ESCALATE | RECHECK | DEFLECTED | Total |
|---|---|---|---|---|---|---|
| complete | 3 | 5 | 0 | 0 | 2 | 10 |
| partial | 6 | 4 | 0 | 0 | 0 | 10 |
| vague | 9 | 1 | 0 | 0 | 0 | 10 |

## 3. KB deflection vs is_deflectable

Deflected = KB STRONG_MATCH. Tickets below the completeness gate never reach Agent 3 and count as not deflected.

| Metric | Value |
|---|---|
| Precision | 0.500 |
| Recall | 0.067 |
| F1 | 0.118 |
| TP / FP / FN / TN | 1 / 1 / 14 / 14 |

## 4. Classification (tickets that reached Agent 5, n=11)

| Metric | Value |
|---|---|
| Category accuracy | 90.9% |
| Routing (team) accuracy | 90.9% |
| Priority accuracy | 45.5% |
| Priority macro-F1 (P2/P3/P4) | 0.350 |
| Rule-fallback rate | 0.0% |
| Confidence-gate review rate (REVIEW) | 0.0% |
| P1 predictions (counted wrong) | 0 |

Priority confusion matrix (rows = ground truth, columns = predicted):

| GT \ Pred | P1 | P2 | P3 | P4 |
|---|---|---|---|---|
| P2 | 0 | 4 | 0 | 0 |
| P3 | 0 | 2 | 1 | 0 |
| P4 | 0 | 0 | 4 | 0 |

## 5. Agent 4

- DUPLICATE_SUGGESTED: 0 (answered "not a duplicate" by the scripted user); 0 share a dataset duplicate_cluster_id with the candidate.
- Tickets with a dataset duplicate_cluster_id: 6.
- INCIDENT_CLUSTER results: 0, of which forced P1: 0. (A cluster needs a known building on the stored tickets, which comes only from an extracted building/location field.)

## 6. Latency per stage (audit latency_ms)

Excluded (Groq retries): 0. The clarify stage counts only tickets where Agent 2 actually ran (not DEFLECTED). Groq SDK-internal retries, if any, are inside these numbers.

| Stage | n | Median ms | p95 ms | Max ms |
|---|---|---|---|---|
| kb | 18 | 58 | 25962 | 25962 |
| clarify | 28 | 1498 | 6843 | 7790 |
| correlate | 11 | 61 | 5481 | 5481 |
| classify | 11 | 1016 | 11891 | 11891 |
| dispatch | 11 | 17 | 20 | 20 |

## 7. Errors (0)

None.
