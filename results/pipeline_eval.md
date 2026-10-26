# Full-Pipeline Evaluation (merged)

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
- **Duplicates were only compared within a slice.** Each slice started with its own empty pool, so a duplicate whose earlier copy was processed in a different slice is not detected; cross-slice duplicates are not counted.
- Tickets that errored (0) are excluded from every metric except the funnel outcome counts and the Errors section.

## Slices

| Slice | Ticket range (fixed order) | First id | Last id | n |
|---|---|---|---|---|
| 1-60 | 1-60 | ADA-001 | EMAIL-030 | 60 |
| 61-120 | 61-120 | PRN-001 | TEAMS-030 | 60 |
| 121-150 | 121-150 | WIFI-001 | WIFI-030 | 30 |

Total tickets: 150

## 1. Funnel

Percentages are of the 150 tickets without errors unless stated.

| Stage | Count |
|---|---|
| Completeness gate passed (reached Agent 3) | 78 (52.0%) |
| KB STRONG_MATCH (% of gate passed) | 19 (24.4%) |
| KB WEAK_MATCH (% of gate passed) | 56 (71.8%) |
| KB NO_MATCH (% of gate passed) | 3 (3.8%) |
| Agent 2 ran on the first turn | 131 (87.3%) |
| - ASK_CLARIFICATION (% of Agent 2) | 106 (80.9%) |
| - READY (% of Agent 2) | 24 (18.3%) |
| - ESCALATE (% of Agent 2) | 0 (0.0%) |
| - other, e.g. RECHECK (% of Agent 2) | 1 (0.8%) |
| DEFLECTED | 19 (12.7%) |
| - feedback resolved / not resolved | 13 / 6 |
| Reached Agent 4 | 30 (20.0%) |
| Reached Agent 5 | 30 (20.0%) |

Final outcomes (% of all 150 tickets):

| Outcome | Count |
|---|---|
| SELF_RESOLVED | 13 (8.7%) |
| DISPATCHED | 30 (20.0%) |
| MERGED | 0 (0.0%) |
| PENDING_REVIEW | 0 (0.0%) |
| ESCALATED | 0 (0.0%) |
| STOPPED_AT_ASK | 106 (70.7%) |
| STOPPED_AT_RECHECK | 1 (0.7%) |
| ERROR | 0 (0.0%) |

## 2. First-turn action vs completeness label

Rows = dataset completeness_level, columns = first-turn decision (DEFLECTED means Agent 3 short-circuited Agent 2). ASK on `complete` tickets is over-asking.

| Level | ASK_CLARIFICATION | READY | ESCALATE | RECHECK | DEFLECTED | Total |
|---|---|---|---|---|---|---|
| complete | 31 | 13 | 0 | 1 | 12 | 57 |
| partial | 35 | 8 | 0 | 0 | 4 | 47 |
| vague | 40 | 3 | 0 | 0 | 3 | 46 |

## 3. KB deflection vs is_deflectable

Deflected = KB STRONG_MATCH. Tickets below the completeness gate never reach Agent 3 and count as not deflected.

| Metric | Value |
|---|---|
| Precision | 0.684 |
| Recall | 0.188 |
| F1 | 0.295 |
| TP / FP / FN / TN | 13 / 6 / 56 / 75 |

## 4. Classification (tickets that reached Agent 5, n=30)

| Metric | Value |
|---|---|
| Category accuracy | 86.7% |
| Routing (team) accuracy | 86.7% |
| Priority accuracy | 53.3% |
| Priority macro-F1 (P2/P3/P4) | 0.459 |
| Rule-fallback rate | 0.0% |
| Confidence-gate review rate (REVIEW) | 0.0% |
| P1 predictions (counted wrong) | 0 |

Priority confusion matrix (rows = ground truth, columns = predicted):

| GT \ Pred | P1 | P2 | P3 | P4 |
|---|---|---|---|---|
| P2 | 0 | 10 | 2 | 0 |
| P3 | 0 | 4 | 5 | 0 |
| P4 | 0 | 1 | 7 | 1 |

## 5. Agent 4

- DUPLICATE_SUGGESTED: 1 (answered "not a duplicate" by the scripted user); 1 share a dataset duplicate_cluster_id with the candidate.
- Tickets with a dataset duplicate_cluster_id: 28.
- INCIDENT_CLUSTER results: 0, of which forced P1: 0. (A cluster needs a known building on the stored tickets, which comes only from an extracted building/location field.)
- **Duplicates were only compared within a slice.** Each slice started with its own empty pool, so a duplicate whose earlier copy was processed in a different slice is not detected; cross-slice duplicates are not counted.

| Slice | Ticket | Candidate | Similarity | Same cluster |
|---|---|---|---|---|
| 61-120 | TEAMS-018 | TEAMS-017 | 0.888 | yes |

## 6. Latency per stage (audit latency_ms)

### Slice 1-60

Excluded (Groq retries): 3.

| Stage | n | Median ms | p95 ms | Max ms |
|---|---|---|---|---|
| kb | 27 | 25 | 45 | 11613 |
| clarify | 54 | 1874 | 11296 | 14993 |
| correlate | 9 | 279 | 8295 | 8295 |
| classify | 9 | 2480 | 13231 | 13231 |
| dispatch | 9 | 2 | 4 | 4 |

### Slice 61-120

Excluded (Groq retries): 0.

| Stage | n | Median ms | p95 ms | Max ms |
|---|---|---|---|---|
| kb | 31 | 53 | 75 | 82806 |
| clarify | 46 | 1671 | 11297 | 17053 |
| correlate | 9 | 56 | 5336 | 5336 |
| classify | 9 | 1142 | 16487 | 16487 |
| dispatch | 9 | 16 | 22 | 22 |

### Slice 121-150

Excluded (Groq retries): 0.

| Stage | n | Median ms | p95 ms | Max ms |
|---|---|---|---|---|
| kb | 18 | 58 | 25962 | 25962 |
| clarify | 28 | 1498 | 6843 | 7790 |
| correlate | 11 | 61 | 5481 | 5481 |
| classify | 11 | 1016 | 11891 | 11891 |
| dispatch | 11 | 17 | 20 | 20 |

### Combined (mixed-machine: slices ran on different machines and Groq accounts)

Excluded (Groq retries): 3.

| Stage | n | Median ms | p95 ms | Max ms |
|---|---|---|---|---|
| kb | 76 | 50 | 75 | 82806 |
| clarify | 128 | 1607 | 11047 | 17053 |
| correlate | 29 | 62 | 5481 | 8295 |
| classify | 29 | 1159 | 14210 | 16487 |
| dispatch | 29 | 16 | 20 | 22 |

## 7. Errors (0)

None.
