"""
Figures for the full-pipeline evaluation.

Reads ONLY results/pipeline_eval_results_merged.csv (written by
`python scripts/run_pipeline_eval.py --report-only --auto`) and computes
every number with the metric functions of run_pipeline_eval.py, so the
figures match results/pipeline_eval.md exactly. No API calls, no model.

    python scripts/plot_pipeline_eval.py

Writes results/figures/pipeline/*.png (300 dpi) and the "pipeline"
section of results/figures/README.md.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import plot_style as st  # noqa: E402
from scripts import run_pipeline_eval as pe  # noqa: E402

MERGED_CSV = pe.RESULTS_DIR / pe.MERGED_CSV_NAME
OUT_DIR = ROOT / "results" / "figures" / "pipeline"
README = ROOT / "results" / "figures" / "README.md"

CATEGORY_LABELS = {
    "wifi_internet": "Wi-Fi / Internet",
    "ms_teams": "MS Teams",
    "vit_email": "VIT Email",
    "ad_account_creation": "AD Account",
    "printer_support": "Printer",
}
LEVELS = ["complete", "partial", "vague"]
FIG_NAMES = [
    "outcomes.png", "funnel.png", "agent2_vs_completeness.png", "kb_deflection_confusion.png",
    "classification_metrics.png", "priority_confusion.png", "outcomes_by_category.png",
    "latency_per_stage.png",
]


def subtitle(rows: list[dict], extra: str = "") -> str:
    base = f"Full-pipeline eval, {len(rows)} synthetic tickets, single run"
    return f"{base}; {extra}" if extra else base


def _pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else 0.0


# --- 1. Final outcomes ---------------------------------------------------------------


def fig_outcomes(rows: list[dict], out_dir: Path) -> dict:
    counts = pe.funnel(rows)["outcomes"]
    n = len(rows)
    order = st.OUTCOME_ORDER
    fig, ax = st.new_figure()
    y = np.arange(len(order))[::-1]
    ax.barh(y, [counts[o] for o in order], color=[st.OUTCOME_COLORS[o] for o in order], height=0.62)
    xmax = max(counts.values()) if counts else 1
    for yi, o in zip(y, order):
        ax.text(counts[o] + xmax * 0.01, yi, f"{counts[o]} ({_pct(counts[o], n):.1f}%)",
                va="center", ha="left", color=st.INK)
    ax.set_yticks(y, order)
    ax.set_xlim(0, xmax * 1.22)
    ax.set_xlabel("Tickets (count)")
    st.value_grid(ax, "x")
    st.finish(fig, f"Final outcome per ticket (n={n})", subtitle(rows), out_dir / "outcomes.png")
    return {"n": n, "counts": dict(counts)}


# --- 2. Stage reach (not a nested funnel) -------------------------------------------


def stage_counts(rows: list[dict]) -> list[tuple[str, int]]:
    f = pe.funnel(rows)
    return [
        ("All tickets", f["n"]),
        ("Passed completeness gate", f["gate_passed"]),
        ("Agent 2 ran (first turn)", f["n_agent2"]),
        ("Reached Agent 4", f["reached_agent4"]),
        ("Reached Agent 5", f["reached_agent5"]),
        ("Dispatched or merged", f["outcomes"]["DISPATCHED"] + f["outcomes"]["MERGED"]),
    ]


def fig_funnel(rows: list[dict], out_dir: Path) -> dict:
    stages = stage_counts(rows)
    n = len(rows)
    fig, ax = st.new_figure()
    y = np.arange(len(stages))[::-1]
    ax.barh(y, [c for _, c in stages], color=st.BLUE, height=0.62)
    for yi, (_, c) in zip(y, stages):
        ax.text(c + n * 0.01, yi, f"{c} ({_pct(c, n):.1f}%)", va="center", ha="left", color=st.INK)
    ax.set_yticks(y, [s for s, _ in stages])
    ax.set_xlim(0, n * 1.22)
    ax.set_xlabel("Tickets (count; % of all tickets)")
    st.value_grid(ax, "x")
    st.finish(fig, f"Tickets reaching each pipeline stage (n={n})",
              subtitle(rows, "stages are not nested: Agent 2 also runs below the gate"),
              out_dir / "funnel.png")
    return {"n": n, "stages": stages}


# --- 3. Agent 2 first-turn action vs completeness label -----------------------------


def fig_agent2_vs_completeness(rows: list[dict], out_dir: Path) -> dict:
    cols, table = pe.agent2_crosstab(rows)
    fig, ax = st.new_figure()
    x = np.arange(len(LEVELS))
    bottoms = np.zeros(len(LEVELS))
    totals = np.array([sum(table[lvl].values()) for lvl in LEVELS])
    for action in cols:
        counts = np.array([table[lvl][action] for lvl in LEVELS])
        shares = np.where(totals > 0, 100.0 * counts / np.maximum(totals, 1), 0.0)
        color = st.ACTION_COLORS[action]
        ax.bar(x, shares, bottom=bottoms, color=color, width=0.6, edgecolor=st.SURFACE,
               linewidth=1.5, label=st.ACTION_LABELS[action])
        for xi, s, c, b in zip(x, shares, counts, bottoms):
            if s >= 6:
                ax.text(xi, b + s / 2, f"{c} ({s:.0f}%)", ha="center", va="center", color=st.label_ink(color))
        bottoms += shares
    ax.set_xticks(x, [f"{lvl}\n(n={t})" for lvl, t in zip(LEVELS, totals)])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of tickets at this level (%)")
    ax.set_xlabel("Dataset completeness label")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), title="First-turn action")
    st.value_grid(ax, "y")
    st.finish(fig, f"Agent 2 first-turn action by completeness label (n={int(totals.sum())})",
              subtitle(rows, "ASK on complete tickets = over-asking"),
              out_dir / "agent2_vs_completeness.png")
    return {"table": table, "totals": dict(zip(LEVELS, totals.tolist()))}


# --- Confusion-matrix helper -------------------------------------------------------------


def _matrix(ax, values: np.ndarray, row_labels, col_labels, cell_text) -> None:
    vmax = values.max() if values.size and values.max() > 0 else 1
    ax.imshow(values, cmap=st.SEQ_BLUE, vmin=0, vmax=vmax, aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            fill = st.SEQ_BLUE(values[i, j] / vmax)
            hexfill = "#{:02x}{:02x}{:02x}".format(*(int(255 * c) for c in fill[:3]))
            ax.text(j, i, cell_text(i, j), ha="center", va="center", color=st.label_ink(hexfill))
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.tick_params(length=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)


# --- 4. KB deflection confusion -------------------------------------------------------


def fig_kb_confusion(rows: list[dict], out_dir: Path) -> dict:
    kb = pe.kb_prf(rows)
    m = np.array([[kb["tp"], kb["fn"]], [kb["fp"], kb["tn"]]])
    n = int(m.sum())
    names = [["TP", "FN"], ["FP", "TN"]]
    fig, ax = st.new_figure()
    _matrix(ax, m, ["Deflectable (label)", "Not deflectable"],
            ["Deflected\n(KB STRONG_MATCH)", "Not deflected"],
            lambda i, j: f"{names[i][j]}\n{m[i, j]}")
    ax.set_xlabel("Pipeline decision")
    ax.set_ylabel("Dataset label (is_deflectable)")
    st.finish(fig,
              f"KB deflection vs label (n={n}): precision {kb['precision']:.2f}, "
              f"recall {kb['recall']:.2f}, F1 {kb['f1']:.2f}",
              subtitle(rows, "below-gate tickets count as not deflected"),
              out_dir / "kb_deflection_confusion.png")
    return dict(kb, n=n)


# --- 5. Classification metrics ---------------------------------------------------------


def always_p3_baseline(rows: list[dict]) -> float:
    scored = [r for r in rows if not r["error"] and r["reached_agent5"]]
    return sum(1 for r in scored if r["gt_priority"] == "P3") / len(scored) if scored else 0.0


def fig_classification(rows: list[dict], out_dir: Path) -> dict:
    c = pe.classification_metrics(rows)
    baseline = always_p3_baseline(rows)
    bars = [("Category\naccuracy", c["category_accuracy"]), ("Routing (team)\naccuracy", c["routing_accuracy"]),
            ("Priority\naccuracy", c["priority_accuracy"]), ("Priority\nmacro-F1", c["macro_f1"])]
    fig, ax = st.new_figure()
    x = np.arange(len(bars))
    ax.bar(x, [v for _, v in bars], color=st.BLUE, width=0.58)
    for xi, (name, v) in zip(x, bars):
        label = f"{v:.2f}" if "F1" in name else f"{100 * v:.1f}%"
        ax.text(xi, v + 0.02, label, ha="center", va="bottom", color=st.INK)
    ax.axhline(baseline, color=st.INK_2, linestyle="--", linewidth=1.5,
               label=f"Always-P3 priority baseline ({100 * baseline:.1f}%)")
    ax.legend(loc="upper right")
    ax.set_xticks(x, [name for name, _ in bars])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score (0-1; accuracy as share correct)")
    st.value_grid(ax, "y")
    st.finish(fig, f"Agent 5 classification (n={c['n']})",
              subtitle(rows, f"n={c['n']} tickets reached Agent 5"),
              out_dir / "classification_metrics.png")
    return {"n": c["n"], "category_accuracy": c["category_accuracy"],
            "routing_accuracy": c["routing_accuracy"], "priority_accuracy": c["priority_accuracy"],
            "macro_f1": c["macro_f1"], "always_p3_baseline": baseline}


# --- 6. Priority confusion -----------------------------------------------------------------


def fig_priority_confusion(rows: list[dict], out_dir: Path) -> dict:
    c = pe.classification_metrics(rows)
    matrix = c["matrix"]
    m = np.array([[matrix[gt][p] for p in pe.PREDICTED_LABELS] for gt in pe.PRIORITY_LABELS])
    row_tot = m.sum(axis=1)
    fig, ax = st.new_figure()
    _matrix(ax, m, [f"{gt} (n={t})" for gt, t in zip(pe.PRIORITY_LABELS, row_tot)], pe.PREDICTED_LABELS,
            lambda i, j: f"{m[i, j]}\n{_pct(m[i, j], row_tot[i]):.0f}%")
    ax.set_xlabel("Predicted priority (P1 has no ground truth: counted wrong)")
    ax.set_ylabel("Ground-truth priority")
    st.finish(fig, f"Priority confusion matrix (n={c['n']})",
              subtitle(rows, f"n={c['n']} tickets reached Agent 5; cells: count, row %"),
              out_dir / "priority_confusion.png")
    return {"n": c["n"], "matrix": matrix}


# --- 7. Outcomes by ground-truth category ------------------------------------------------


def fig_outcomes_by_category(rows: list[dict], out_dir: Path) -> dict:
    cats = [k for k in CATEGORY_LABELS if any(r["gt_category"] == k for r in rows)]
    counts = {k: {o: sum(1 for r in rows if r["gt_category"] == k and r["final_outcome"] == o)
                  for o in st.OUTCOME_ORDER} for k in cats}
    present = [o for o in st.OUTCOME_ORDER if any(counts[k][o] for k in cats)]
    totals = {k: sum(counts[k].values()) for k in cats}
    fig, ax = st.new_figure()
    y = np.arange(len(cats))[::-1]
    left = np.zeros(len(cats))
    for o in present:
        shares = np.array([_pct(counts[k][o], totals[k]) for k in cats])
        color = st.OUTCOME_COLORS[o]
        ax.barh(y, shares, left=left, color=color, height=0.62, edgecolor=st.SURFACE,
                linewidth=1.5, label=o)
        for yi, s, k, lft in zip(y, shares, cats, left):
            if s >= 8:
                ax.text(lft + s / 2, yi, str(counts[k][o]), ha="center", va="center", color=st.label_ink(color))
        left += shares
    ax.set_yticks(y, [f"{CATEGORY_LABELS[k]} (n={totals[k]})" for k in cats])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of the category's tickets (%); labels are counts")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=min(3, len(present)))
    st.value_grid(ax, "x")
    st.finish(fig, f"Final outcome by ground-truth category (n={len(rows)})", subtitle(rows),
              out_dir / "outcomes_by_category.png")
    return {"counts": counts, "totals": totals}


# --- 8. Latency per stage ------------------------------------------------------------------


def fig_latency(rows: list[dict], out_dir: Path) -> dict:
    stats = pe.latency_stats(rows)
    clean = [r for r in rows if not r["error"] and not r["groq_retries"]]
    data = [[r[f"lat_{s}_ms"] for r in clean if r[f"lat_{s}_ms"] is not None] for s in pe.LATENCY_STAGES]
    keep = [i for i, d in enumerate(data) if d]
    fig, ax = st.new_figure()
    bp = ax.boxplot([data[i] for i in keep], patch_artist=True, widths=0.55, showfliers=True,
                    medianprops={"color": st.INK, "linewidth": 2},
                    whiskerprops={"color": st.INK_2}, capprops={"color": st.INK_2},
                    flierprops={"marker": "o", "markersize": 4, "markerfacecolor": st.MUTED,
                                "markeredgecolor": st.MUTED})
    for box in bp["boxes"]:
        box.set(facecolor="#cde2fb", edgecolor=st.BLUE, linewidth=1.2)
    ax.set_yscale("log")
    ax.set_xticks(range(1, len(keep) + 1),
                  [f"{pe.LATENCY_STAGES[i]}\n(n={stats['stages'][pe.LATENCY_STAGES[i]]['n']})" for i in keep])
    ax.set_ylabel("Latency per stage (ms, log scale)")
    st.value_grid(ax, "y")
    n_slices = len({r["slice"] for r in rows})
    extra = f"{stats['excluded_retried']} tickets with Groq retries excluded; outliers are cold starts"
    if n_slices > 1:
        extra += f"; mixed machines ({n_slices} slices)"
    st.finish(fig, f"Stage latency from the audit trail (n={len(clean)})", subtitle(rows, extra),
              out_dir / "latency_per_stage.png")
    return {"n": len(clean), "excluded_retried": stats["excluded_retried"], "stages": stats["stages"],
            "n_slices": n_slices}


# --- README ----------------------------------------------------------------------------


def _ms(v):
    return "-" if v is None else f"{v:.0f}"


def readme_section(results: dict, rows: list[dict]) -> str:
    n = len(rows)
    o, st2, a2 = results["outcomes"], results["funnel"], results["agent2"]
    kb, cl, pc = results["kb"], results["classification"], results["priority"]
    oc, lat = results["by_category"], results["latency"]
    src = f"`results/pipeline_eval_results_merged.csv` ({n} tickets, single run)"
    lines = ["## Full pipeline (`results/figures/pipeline/`)", "", f"Source: {src}.", ""]

    counts = ", ".join(f"{k} {v} ({_pct(v, n):.1f}%)" for k, v in o["counts"].items())
    lines += ["### outcomes.png", "",
              f"*Final outcome of each of the {n} synthetic tickets after one user turn and the scripted user.*",
              "", f"Numbers: {counts}.", ""]

    stages = ", ".join(f"{s} {c} ({_pct(c, n):.1f}%)" for s, c in st2["stages"])
    lines += ["### funnel.png", "",
              "*Number of tickets reaching each pipeline stage; stages are not nested, because Agent 2 "
              "also runs on requests below the completeness gate.*", "", f"Numbers: {stages}.", ""]

    level_txt = "; ".join(
        f"{lvl} (n={a2['totals'][lvl]}): " + ", ".join(
            f"{st.ACTION_LABELS[a]} {c}" for a, c in a2["table"][lvl].items())
        for lvl in LEVELS)
    lines += ["### agent2_vs_completeness.png", "",
              "*First-turn decision by dataset completeness label; ASK on complete tickets is over-asking.*",
              "", f"Numbers: {level_txt}.", ""]

    lines += ["### kb_deflection_confusion.png", "",
              f"*KB deflection (STRONG_MATCH) against the dataset's is_deflectable label (n={kb['n']}).*", "",
              f"Numbers: TP {kb['tp']}, FP {kb['fp']}, FN {kb['fn']}, TN {kb['tn']}; precision "
              f"{kb['precision']:.3f}, recall {kb['recall']:.3f}, F1 {kb['f1']:.3f}.", ""]

    lines += ["### classification_metrics.png", "",
              f"*Agent 5 classification on the n={cl['n']} tickets that reached Agent 5, with the always-P3 "
              "priority baseline on the same tickets.*", "",
              f"Numbers: category accuracy {100 * cl['category_accuracy']:.1f}%, routing accuracy "
              f"{100 * cl['routing_accuracy']:.1f}%, priority accuracy {100 * cl['priority_accuracy']:.1f}%, "
              f"priority macro-F1 {cl['macro_f1']:.3f}, always-P3 baseline {100 * cl['always_p3_baseline']:.1f}%.", ""]

    cells = "; ".join(f"{gt}: " + ", ".join(f"{p} {pc['matrix'][gt][p]}" for p in pe.PREDICTED_LABELS)
                      for gt in pe.PRIORITY_LABELS)
    lines += ["### priority_confusion.png", "",
              f"*Ground-truth vs predicted priority on the n={pc['n']} tickets that reached Agent 5; "
              "the dataset has no P1 label, so any P1 counts as wrong.*", "", f"Numbers (GT: predicted): {cells}.", ""]

    cat_txt = "; ".join(
        f"{CATEGORY_LABELS[k]} (n={oc['totals'][k]}): "
        + ", ".join(f"{o2} {c}" for o2, c in oc["counts"][k].items() if c)
        for k in oc["counts"])
    lines += ["### outcomes_by_category.png", "",
              "*Final outcome by ground-truth category, as a share of each category's tickets.*", "",
              f"Numbers: {cat_txt}.", ""]

    lat_txt = "; ".join(
        f"{s} n={v['n']}, median {_ms(v['median'])} ms, p95 {_ms(v['p95'])} ms, max {_ms(v['max'])} ms"
        for s, v in lat["stages"].items())
    mixed = f" Slices ran on {lat['n_slices']} different machines (mixed-machine figures)." if lat["n_slices"] > 1 else ""
    lines += ["### latency_per_stage.png", "",
              f"*Per-stage latency from the audit trail (log scale), excluding the {lat['excluded_retried']} "
              f"tickets with Groq 429 retries.{mixed} The kb outliers (11-83 s) are each slice's first KB "
              "query (cold start), and the three 5-8 s correlate outliers, one per slice, are consistent with "
              "Agent 4's embedding model loading on first use; they are inside the max and p95 figures.*", "",
              f"Numbers: {lat_txt}.", ""]
    return "\n".join(lines)


# --- Main ------------------------------------------------------------------------------


def make_all(rows: list[dict], out_dir: Path) -> dict:
    return {
        "outcomes": fig_outcomes(rows, out_dir),
        "funnel": fig_funnel(rows, out_dir),
        "agent2": fig_agent2_vs_completeness(rows, out_dir),
        "kb": fig_kb_confusion(rows, out_dir),
        "classification": fig_classification(rows, out_dir),
        "priority": fig_priority_confusion(rows, out_dir),
        "by_category": fig_outcomes_by_category(rows, out_dir),
        "latency": fig_latency(rows, out_dir),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=MERGED_CSV)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--readme", type=Path, default=README)
    args = p.parse_args(argv)
    if not args.input.exists():
        print(f"ERROR: {args.input} not found; run run_pipeline_eval.py --report-only --auto first")
        return 1
    rows = pe.read_rows(args.input)
    results = make_all(rows, args.out_dir)
    st.write_readme_section(args.readme, "pipeline", readme_section(results, rows))
    for name in FIG_NAMES:
        print(f"wrote {args.out_dir / name}")
    print(f"wrote {args.readme}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
