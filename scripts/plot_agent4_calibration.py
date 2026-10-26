"""
Figures for Agent 4's live-format duplicate-threshold calibration.

    python scripts/plot_agent4_calibration.py              # uses the saved score lists
    python scripts/plot_agent4_calibration.py --recompute  # rebuild them first

Inputs:
  results/agent4_live_threshold_sweep.csv   (scripts/run_agent4_live_calibration.py)
  results/agent4_live_score_lists.json      raw cosine score lists. The
      calibration run only kept 0.05-bin counts in its markdown, so when
      this file is missing (or with --recompute) the lists are rebuilt with
      that script's own helpers - run_sweep() at the adopted threshold and
      the same filters as its histogram. No API calls; it loads the local
      all-MiniLM-L6-v2 model. The rebuilt bins are checked against the
      histogram table in results/agent4_live_calibration.md.

Writes results/figures/agent4/*.png (300 dpi) and the "agent4" section of
results/figures/README.md.
"""

import argparse
import asyncio
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import plot_style as st  # noqa: E402

RESULTS = ROOT / "results"
SWEEP_CSV = RESULTS / "agent4_live_threshold_sweep.csv"
SCORES_JSON = RESULTS / "agent4_live_score_lists.json"
CALIBRATION_MD = RESULTS / "agent4_live_calibration.md"
OUT_DIR = RESULTS / "figures" / "agent4"
README = RESULTS / "figures" / "README.md"

ADOPTED = 0.87
PREVIOUS = 0.95
BIN_WIDTH = 0.05
SOURCE = "Agent 4 live-format calibration, synthetic tickets, leave-one-out, single run"


def read_sweep(path: Path = SWEEP_CSV) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


# --- Score lists ---------------------------------------------------------------------


def bin_counts(values: list[float], lo: float, width: float = BIN_WIDTH) -> list[tuple[float, int]]:
    """Same bins as run_agent4_live_calibration.histogram_lines: [b, b+w), last bin closed at 1.0."""
    edges = []
    b = lo
    while b < 1.0 - 1e-9:
        edges.append(round(b, 2))
        b += width
    out = []
    for i, e in enumerate(edges):
        hi = round(e + width, 2)
        last = i == len(edges) - 1
        out.append((e, sum(1 for v in values if e <= v < hi or (last and v == 1.0))))
    return out


def histogram_start(dups: list[float], non_dups: list[float], width: float = BIN_WIDTH) -> float:
    vals = dups + non_dups
    return int((min(vals) if vals else 0.0) / width) * width


def parse_md_histogram(path: Path = CALIBRATION_MD) -> list[tuple[float, int, int]]:
    """(bin start, duplicates, non-duplicates) rows of the markdown's raw-cosine histogram."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\|\s*\[(\d\.\d\d), \d\.\d\d[\)\]]\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|", line)
        if m:
            rows.append((float(m.group(1)), int(m.group(2)), int(m.group(3))))
    return rows


def recompute_score_lists() -> dict:
    """Rebuild the two raw-cosine lists exactly as write_markdown() does. Loads the local model."""
    from app.agents.correlation_engine import _LazySentenceTransformer
    from scripts import run_agent4_live_calibration as cal

    tickets = cal.load_tickets()
    cache = cal.load_cache(tickets)
    out = asyncio.run(cal.run_sweep(tickets, cache, _LazySentenceTransformer(), thresholds=[ADOPTED]))
    diag = out.diagnostics
    dups = [t for t in tickets if t["label_type"] == "duplicate"]
    non_dups = [t for t in tickets if t["label_type"] != "duplicate"]
    return {
        "source": "scripts/plot_agent4_calibration.py --recompute (run_agent4_live_calibration.run_sweep "
                  "diagnostics; same filters as its markdown histogram)",
        "model": next(iter(cache.values()))["model"],
        "n_tickets": len(tickets),
        "duplicates_vs_true_original": [
            diag[t["ticket_id"]]["original_raw"] for t in dups
            if diag[t["ticket_id"]].get("original_dup_eligible")],
        "duplicates_gated": [t["ticket_id"] for t in dups
                             if not diag[t["ticket_id"]].get("original_dup_eligible")],
        "non_duplicates_vs_top_candidate": [
            diag[t["ticket_id"]]["top_dup_raw"] for t in non_dups
            if diag[t["ticket_id"]]["top_dup_raw"] is not None],
        "non_duplicates_gated": sum(diag[t["ticket_id"]]["top_dup_raw"] is None for t in non_dups),
    }


def check_against_markdown(scores: dict, md_path: Path = CALIBRATION_MD) -> None:
    dups, non = scores["duplicates_vs_true_original"], scores["non_duplicates_vs_top_candidate"]
    lo = histogram_start(dups, non)
    mine = [(b, d, n) for (b, d), (_, n) in zip(bin_counts(dups, lo), bin_counts(non, lo))]
    theirs = parse_md_histogram(md_path)
    if mine != theirs:
        raise AssertionError(f"recomputed bins differ from {md_path.name}:\n{mine}\n{theirs}")


def load_scores(path: Path = SCORES_JSON, recompute: bool = False, md_path: Path = CALIBRATION_MD) -> dict:
    if recompute or not path.exists():
        scores = recompute_score_lists()
        check_against_markdown(scores, md_path)
        path.write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path} (bins match {md_path.name})")
        return scores
    return json.loads(path.read_text(encoding="utf-8"))


# --- 9. Threshold sweep -----------------------------------------------------------------


def _row_at(rows: list[dict], threshold: float) -> dict | None:
    return next((r for r in rows if math.isclose(r["dup_threshold"], threshold)), None)


def fig_threshold_sweep(rows: list[dict], out_dir: Path) -> dict:
    x = [r["dup_threshold"] for r in rows]
    series = [("Duplicate precision", "dup_precision", st.BLUE, "-"),
              ("Duplicate recall", "dup_recall", st.ORANGE, "-"),
              ("Duplicate F1", "dup_f1", st.AQUA, "-"),
              ("Background duplicate-FP rate", "background_dup_fp_rate", st.INK_2, "--")]
    fig, ax = st.new_figure()
    for label, key, color, ls in series:
        ax.plot(x, [r[key] for r in rows], color=color, linestyle=ls, linewidth=2, label=label)
    ax.axvline(PREVIOUS, color=st.AXIS, linestyle=":", linewidth=1.5)
    ax.text(PREVIOUS + 0.003, 1.03, f"previous {PREVIOUS:.2f}", color=st.MUTED, va="bottom", ha="left")
    ax.axvline(ADOPTED, color=st.INK, linestyle="--", linewidth=1.5)
    ax.text(ADOPTED - 0.003, 1.03, f"adopted {ADOPTED:.2f}", color=st.INK, va="bottom", ha="right")
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Duplicate threshold (raw cosine similarity)")
    ax.set_ylabel("Rate (0-1)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    st.value_grid(ax, "y")
    first = rows[0]
    n_dup = int(first["dup_tp"] + first["dup_fn"])
    n_bg = int(first["background_n"])
    st.finish(fig, f"Agent 4 duplicate threshold sweep (n={n_dup} duplicates, {n_bg} background)",
              f"{SOURCE}; sweep {min(x):.2f}-{max(x):.2f}",
              out_dir / "threshold_sweep.png")
    keys = ["dup_precision", "dup_recall", "dup_f1", "background_dup_fp_rate", "dup_tp", "dup_fp", "dup_fn"]
    return {"n_dup": n_dup, "n_background": n_bg,
            "adopted": {k: _row_at(rows, ADOPTED)[k] for k in keys} if _row_at(rows, ADOPTED) else None,
            "previous": {k: _row_at(rows, PREVIOUS)[k] for k in keys} if _row_at(rows, PREVIOUS) else None}


# --- 10. Score distributions -----------------------------------------------------------


def fig_score_distributions(scores: dict, out_dir: Path) -> dict:
    dups = scores["duplicates_vs_true_original"]
    non = scores["non_duplicates_vs_top_candidate"]
    lo = histogram_start(dups, non)
    edges = np.round(np.arange(lo, 1.0 + 1e-9, BIN_WIDTH), 2)
    fig, ax = st.new_figure()
    for vals, color, label in ((non, st.ORANGE, f"Non-duplicates vs their top candidate (n={len(non)})"),
                               (dups, st.BLUE, f"Duplicates vs their true original (n={len(dups)})")):
        ax.hist(vals, bins=edges, color=color, alpha=0.35, label=label)
        ax.hist(vals, bins=edges, histtype="step", color=color, linewidth=2)
    ax.axvline(ADOPTED, color=st.INK, linestyle="--", linewidth=1.5, label=f"Adopted threshold {ADOPTED:.2f}")
    ax.set_xlim(lo, 1.0)
    ax.set_xlabel(f"Raw cosine similarity (bins of {BIN_WIDTH:.2f})")
    ax.set_ylabel("Queries (count)")
    ax.legend(loc="upper left")
    st.value_grid(ax, "y")
    above_dup = sum(v >= ADOPTED for v in dups)
    above_non = sum(v >= ADOPTED for v in non)
    st.finish(fig, f"Raw cosine: duplicates vs non-duplicates (n={len(dups) + len(non)})",
              f"{SOURCE}; DUPLICATE-eligible candidates only; the overlap above {ADOPTED:.2f} "
              "is the false-duplicate risk",
              out_dir / "score_distributions.png")
    return {"n_dup": len(dups), "n_non": len(non), "dup_at_or_above": above_dup,
            "non_at_or_above": above_non, "bins": [
                (b, d, n) for (b, d), (_, n) in zip(bin_counts(dups, lo), bin_counts(non, lo))],
            "n_dup_gated": len(scores.get("duplicates_gated", [])),
            "n_non_gated": scores.get("non_duplicates_gated", 0)}


# --- README ------------------------------------------------------------------------------


def readme_section(sweep: dict, dist: dict) -> str:
    lines = ["## Agent 4 calibration (`results/figures/agent4/`)", "",
             "Source: `results/agent4_live_threshold_sweep.csv` and `results/agent4_live_score_lists.json` "
             "(live-format texts from cached Agent 1 output, temporal leave-one-out, single run).", ""]

    def at(label, r):
        if r is None:
            return f"{label}: not on the sweep grid"
        return (f"{label}: precision {r['dup_precision']:.3f}, recall {r['dup_recall']:.3f}, "
                f"F1 {r['dup_f1']:.3f}, background duplicate-FP rate {r['background_dup_fp_rate']:.4f} "
                f"(TP {int(r['dup_tp'])}, FP {int(r['dup_fp'])}, FN {int(r['dup_fn'])})")

    lines += ["### threshold_sweep.png", "",
              f"*Duplicate precision, recall, F1 and background duplicate-FP rate over the duplicate threshold "
              f"(n={sweep['n_dup']} duplicate queries, {sweep['n_background']} background tickets); dashed line = "
              f"adopted {ADOPTED:.2f}, dotted = previous {PREVIOUS:.2f}.*", "",
              f"Numbers: {at(f'at {ADOPTED:.2f}', sweep['adopted'])}; {at(f'at {PREVIOUS:.2f}', sweep['previous'])}.",
              ""]
    bins = "; ".join(f"[{b:.2f}, {b + BIN_WIDTH:.2f}) {d} / {n}" for b, d, n in dist["bins"])
    lines += ["### score_distributions.png", "",
              f"*Raw cosine similarity of each duplicate to its true original (n={dist['n_dup']}) and of each "
              f"non-duplicate to its top DUPLICATE-eligible candidate (n={dist['n_non']}); the overlap above "
              f"{ADOPTED:.2f} is where false duplicates come from.*", "",
              f"Numbers: at or above {ADOPTED:.2f}: {dist['dup_at_or_above']} of {dist['n_dup']} duplicates, "
              f"{dist['non_at_or_above']} of {dist['n_non']} non-duplicates. Excluded by the minimum-field gate: "
              f"{dist['n_dup_gated']} duplicate{'' if dist['n_dup_gated'] == 1 else 's'}, "
              f"{dist['n_non_gated']} non-duplicate{'' if dist['n_non_gated'] == 1 else 's'}. "
              f"Bins (duplicates / non-duplicates): {bins} (last bin includes 1.00).", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recompute", action="store_true", help="rebuild the score lists (local model, no API)")
    p.add_argument("--sweep", type=Path, default=SWEEP_CSV)
    p.add_argument("--scores", type=Path, default=SCORES_JSON)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--readme", type=Path, default=README)
    args = p.parse_args(argv)
    rows = read_sweep(args.sweep)
    scores = load_scores(args.scores, recompute=args.recompute)
    sweep = fig_threshold_sweep(rows, args.out_dir)
    dist = fig_score_distributions(scores, args.out_dir)
    st.write_readme_section(args.readme, "agent4", readme_section(sweep, dist))
    for name in ("threshold_sweep.png", "score_distributions.png"):
        print(f"wrote {args.out_dir / name}")
    print(f"wrote {args.readme}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
