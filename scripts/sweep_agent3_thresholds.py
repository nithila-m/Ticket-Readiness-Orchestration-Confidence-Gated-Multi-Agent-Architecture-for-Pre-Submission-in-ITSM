"""
Threshold sweep for Agent 3 (KB Retrieval / Deflection Gate).

Run from the repo root (needs the populated chroma_db/):
    python scripts/sweep_agent3_thresholds.py

Sweeps strong_threshold from 0.50 to 0.80 in 0.02 steps. A ticket is
"deflected" when its top-1 KB similarity >= strong_threshold (STRONG_MATCH);
weak_threshold does not affect deflection, so it is not swept.

Category filter modes (each ticket is retrieved once per mode, then every
threshold is applied offline to the same unrounded similarities):
  none    - unfiltered search (old behaviour)
  agent1  - filtered by Agent 1's real predicted category, read from
            results/agent1_eval_results.csv (no filter where Agent 1
            predicted none) - what the live pipeline would do
  oracle  - filtered by the dataset's true_category; an upper bound for
            the filter, not achievable live

Gate modes:
  off - pure retrieval quality: every ticket is checked
  on  - pipeline view: tickets whose Agent 1 completeness_score is below
        settings.kb_gate_completeness_threshold never reach Agent 3 and
        count as "not deflected"

LIMITATIONS (repeat these wherever the numbers are quoted):
  - Deflect/no-deflect labels are the data generator's `is_deflectable`,
    a proxy for "the KB resolves this", not a check against this KB.
  - Queries are subject + message, as in run_agent3_eval.py. The live
    app queries with build_kb_query() over Agent 1's extracted fields,
    which are not stored for this dataset, so these numbers are a proxy
    for live retrieval.
  - Correct-article rate is computed ONLY on gold entries marked
    "source": "confirmed" in data/eval/agent3_gold_kb.json; until some are
    confirmed the column is empty. A confirmed label that still equals the
    unfiltered top-1 article (i.e. was confirmed without correction) makes
    the filter=none rate circular for that ticket; the report counts them.

Outputs: results/agent3_threshold_sweep.csv, results/agent3_eval.md
"""

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "TRO_Codes"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from agents.kb_retrieval_agent import search_kb  # noqa: E402
from app.config.category_profiles import normalize_category_label  # noqa: E402
from app.config.settings import settings  # noqa: E402
from run_agent3_eval import build_ticket_text, load_all_tickets  # noqa: E402

AGENT1_RESULTS = REPO_ROOT / "results" / "agent1_eval_results.csv"
GOLD_PATH = REPO_ROOT / "data" / "eval" / "agent3_gold_kb.json"
CSV_PATH = REPO_ROOT / "results" / "agent3_threshold_sweep.csv"
MD_PATH = REPO_ROOT / "results" / "agent3_eval.md"

THRESHOLDS = [round(0.50 + 0.02 * i, 2) for i in range(16)]  # 0.50 .. 0.80
FILTERS = ["none", "agent1", "oracle"]
GATES = ["off", "on"]


def load_agent1_predictions() -> dict[str, dict]:
    with open(AGENT1_RESULTS, encoding="utf-8") as f:
        return {
            row["ticket_id"]: {
                "category": row["predicted_category"] or None,
                "completeness": float(row["completeness_score_predicted"] or 0.0),
            }
            for row in csv.DictReader(f)
        }


def load_confirmed_gold() -> dict[str, str | None]:
    if not GOLD_PATH.exists():
        return {}
    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    return {tid: e.get("kb_id") for tid, e in gold.items() if e.get("source") == "confirmed"}


def retrieve_all(tickets: list[dict], agent1: dict[str, dict]) -> list[dict]:
    rows = []
    for ticket in tickets:
        tid = ticket["ticket_id"]
        text = build_ticket_text(ticket)
        pred = agent1.get(tid, {"category": None, "completeness": 0.0})
        categories = {
            "none": None,
            "agent1": pred["category"],
            "oracle": normalize_category_label(ticket.get("true_category")),
        }
        row = {
            "ticket_id": tid,
            "expected": bool(ticket.get("is_deflectable")),
            "completeness": pred["completeness"],
        }
        for mode, category in categories.items():
            results = search_kb(text, top_k=1, category=category)
            row[f"{mode}_kb_id"] = results["ids"][0][0]
            row[f"{mode}_sim"] = 1 - results["distances"][0][0]
        rows.append(row)
    return rows


def score(rows: list[dict], gold: dict[str, str | None], mode: str, gate: str, t: float) -> dict:
    tp = fp = fn = tn = 0
    gold_with_article = correct_article = wrong_article = 0
    for r in rows:
        reaches_agent3 = gate == "off" or r["completeness"] >= settings.kb_gate_completeness_threshold
        predicted = reaches_agent3 and r[f"{mode}_sim"] >= t
        if predicted and r["expected"]:
            tp += 1
        elif predicted:
            fp += 1
        elif r["expected"]:
            fn += 1
        else:
            tn += 1
        if r["ticket_id"] in gold and gold[r["ticket_id"]] is not None:
            gold_with_article += 1
            if predicted:
                if r[f"{mode}_kb_id"] == gold[r["ticket_id"]]:
                    correct_article += 1
                else:
                    wrong_article += 1

    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "filter": mode,
        "gate": gate,
        "strong_threshold": t,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "deflect_rate": round((tp + fp) / total, 4),
        "accuracy": round((tp + tn) / total, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confirmed_gold_n": gold_with_article,
        # correct article deflected / confirmed gold tickets that have an article
        "correct_article_rate": round(correct_article / gold_with_article, 4) if gold_with_article else "",
        "wrong_article_deflections": wrong_article if gold_with_article else "",
    }


def best(results: list[dict], mode: str, gate: str, metric: str = "accuracy") -> dict:
    """Highest `metric`; ties go to higher precision, then the higher threshold."""
    candidates = [r for r in results if r["filter"] == mode and r["gate"] == gate]
    return max(candidates, key=lambda r: (r[metric], r["precision"], r["strong_threshold"]))


def roc_auc(rows: list[dict], mode: str) -> float:
    """
    Threshold-free separability of top-1 similarity for deflectable vs not:
    probability a random deflectable ticket scores higher than a random
    non-deflectable one (0.5 = no signal). Ties count half.
    """
    pos = [r[f"{mode}_sim"] for r in rows if r["expected"]]
    neg = [r[f"{mode}_sim"] for r in rows if not r["expected"]]
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def _table(results: list[dict], gate: str) -> list[str]:
    lines = [
        "| threshold | filter | deflect rate | accuracy | precision | recall | F1 | correct-article rate* |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for t in THRESHOLDS:
        for mode in ("none", "agent1"):
            r = next(x for x in results if x["filter"] == mode and x["gate"] == gate and x["strong_threshold"] == t)
            car = f"{r['correct_article_rate']:.2f}" if r["correct_article_rate"] != "" else "n/a"
            lines.append(
                f"| {t:.2f} | {mode} | {r['deflect_rate']:.2f} | {r['accuracy']:.2f} | "
                f"{r['precision']:.2f} | {r['recall']:.2f} | {r['f1']:.2f} | {car} |"
            )
    return lines


def write_markdown(results: list[dict], rows: list[dict], gold: dict[str, str | None]):
    n_confirmed = len(gold)
    # Confirmed labels identical to this run's unfiltered top-1: the retriever's
    # own pick, so filter=none is scored against itself for these tickets.
    n_same_as_top1 = sum(1 for r in rows if gold.get(r["ticket_id"], "") == r["none_kb_id"])
    n = len(rows)
    n_pos = sum(r["expected"] for r in rows)
    current = settings.kb_strong_threshold

    def row_at(mode, gate, t):
        return next(x for x in results if x["filter"] == mode and x["gate"] == gate and x["strong_threshold"] == t)

    lines = [
        "# Agent 3 threshold sweep",
        "",
        "Generated by `scripts/sweep_agent3_thresholds.py`. Do not edit by hand.",
        "",
        "## What is measured, and the limits",
        "",
        f"- {n} synthetic tickets ({n_pos} labelled `is_deflectable=true`). The label is the data",
        "  generator's judgement, a **proxy** for \"this KB resolves the ticket\", not a check against this KB.",
        "- A ticket counts as deflected when its top-1 KB similarity >= `strong_threshold` (STRONG_MATCH).",
        "- Query text is subject + message. The live app queries with `build_kb_query()` over Agent 1's",
        "  extracted fields, which are not stored for this dataset, so these numbers are a **proxy** for live retrieval.",
        "- `agent1` filter = Agent 1's real predicted category (from `results/agent1_eval_results.csv`).",
        "  `oracle` (true_category) is in the CSV as an upper bound only.",
        f"- \\*Correct-article rate uses only gold entries marked `confirmed` ({n_confirmed} so far). "
        f"**{n_same_as_top1} of them equal the unfiltered top-1 article**, i.e. the retriever's own pick",
        "  confirmed without correction. For those tickets the `none` column is scored against itself",
        "  (circular: a deflection can never be \"wrong\"), so it measures agreement with the retriever,",
        "  not article accuracy. The filtered columns differ only where the category filter changes the",
        "  article offered.",
        "",
        "## Baselines and separability (gate off)",
        "",
        f"- Never deflect: accuracy {(n - n_pos) / n:.2f}, recall 0.00.",
        f"- Always deflect: accuracy {n_pos / n:.2f}, precision {n_pos / n:.2f}, recall 1.00, "
        f"F1 {2 * (n_pos / n) / (1 + n_pos / n):.2f}.",
        "- ROC AUC of top-1 similarity vs `is_deflectable` (0.5 = no signal): "
        + ", ".join(f"{mode} {roc_auc(rows, mode):.2f}" for mode in FILTERS) + ".",
        "",
        "Max-F1 is degenerate on this data: it picks the lowest threshold, which is about as good as",
        "always deflecting. The table below therefore ranks by accuracy (ties: higher precision); the",
        "max-F1 threshold is listed for reference only.",
        "",
        "## Best threshold per mode",
        "",
        "| filter | gate | best-accuracy threshold | accuracy | precision | recall | F1 | deflect rate | (max-F1 threshold) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for gate in GATES:
        for mode in FILTERS:
            b = best(results, mode, gate)
            bf = best(results, mode, gate, metric="f1")
            lines.append(
                f"| {mode} | {gate} | {b['strong_threshold']:.2f} | {b['accuracy']:.2f} | {b['precision']:.2f} | "
                f"{b['recall']:.2f} | {b['f1']:.2f} | {b['deflect_rate']:.2f} | ({bf['strong_threshold']:.2f}) |"
            )
    lines += ["", f"Current default (`kb_strong_threshold={current}`):", ""]
    for gate in GATES:
        for mode in ("none", "agent1"):
            r = row_at(mode, gate, round(current, 2))
            lines.append(
                f"- filter={mode}, gate={gate}: precision {r['precision']:.2f}, recall {r['recall']:.2f}, "
                f"F1 {r['f1']:.2f}, deflect rate {r['deflect_rate']:.2f}"
            )
    lines += [
        "",
        "## Sweep, completeness gate off (pure retrieval)",
        "",
        *_table(results, "off"),
        "",
        "## Sweep, completeness gate on (pipeline view)",
        "",
        f"Tickets with Agent 1 completeness < {settings.kb_gate_completeness_threshold} never reach Agent 3.",
        "",
        *_table(results, "on"),
        "",
    ]
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    tickets = load_all_tickets()
    agent1 = load_agent1_predictions()
    gold = load_confirmed_gold()
    print(f"{len(tickets)} tickets, {len(gold)} confirmed gold labels. Retrieving...")

    rows = retrieve_all(tickets, agent1)
    results = [score(rows, gold, mode, gate, t) for gate in GATES for mode in FILTERS for t in THRESHOLDS]

    CSV_PATH.parent.mkdir(exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    write_markdown(results, rows, gold)

    print(f"Wrote {CSV_PATH.relative_to(REPO_ROOT)} and {MD_PATH.relative_to(REPO_ROOT)}")
    for mode in FILTERS:
        print(f"  ROC AUC filter={mode:<6} {roc_auc(rows, mode):.3f}")
    for gate in GATES:
        for mode in FILTERS:
            b = best(results, mode, gate)
            print(f"  best accuracy  filter={mode:<6} gate={gate:<3} t={b['strong_threshold']:.2f}  "
                  f"Acc={b['accuracy']:.2f} P={b['precision']:.2f} R={b['recall']:.2f} F1={b['f1']:.2f}")


if __name__ == "__main__":
    main()
