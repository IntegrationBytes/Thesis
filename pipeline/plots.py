"""Thesis evaluation plots.

Produces three outputs to evaluation/outputs/plots/:

    conformance_pc.png     — SHACL conformance rate and Property Completeness
                             side-by-side for Systems A/B/C (RQ1 + RQ2 overview).
    violation_reduction.png — Violation count per SHACL cycle for System B,
                              showing how many violations remain after each
                              correction round (RQ2 / RQ5).
    appendix_trace.txt     — Human-readable pipeline trace for one vignette
                              showing extracted triples and SHACL feedback
                              across cycles (thesis appendix).

Usage::

    python pipeline/plots.py                    # defaults: schema/full, vignette_001
    python pipeline/plots.py --vignette vignette_030
    python pipeline/plots.py --track ontology --prompt ontology
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_PIPELINE_DIR))

PROJECT_ROOT = _PIPELINE_DIR.parent
OUTPUTS_ROOT = PROJECT_ROOT / "evaluation" / "outputs"
PLOTS_DIR = OUTPUTS_ROOT / "plots"
VIGNETTES_DIR = PROJECT_ROOT / "evaluation" / "corpus" / "vignettes"
CORPUS_GOLD_DIR = PROJECT_ROOT / "evaluation" / "corpus" / "abox_gold"

# Use the same bootstrap setup as stats_core so the CIs in plots and the
# JSON/Markdown report agree exactly.
from stats_core import _bootstrap_ci, _is_present  # noqa: E402

SYSTEM_LABELS = {"a": "System A\n(zero-shot)", "b": "System B\n(SHACL loop)", "c": "System C\n(self-review)"}
SYSTEM_COLORS = {"a": "#4C72B0", "b": "#DD8452", "c": "#55A868"}


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

def _load_eval(schema: str, track: str, prompt: str) -> dict | None:
    path = OUTPUTS_ROOT / schema / track / prompt / "eval.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _violation_count(results_summary: str) -> int:
    """Extract the N from 'Results (N):' in a SHACL results_summary string."""
    m = re.search(r"Results\s*\((\d+)\)", results_summary or "")
    return int(m.group(1)) if m else 0


def _load_cycle_summaries(schema: str, track: str, prompt: str, system: str) -> list[dict]:
    """Return list of {vignette, cycle, violations, conforms} rows."""
    base = OUTPUTS_ROOT / schema / track / prompt / system / "cycles"
    if not base.exists():
        return []
    rows = []
    for vig_dir in sorted(base.iterdir()):
        if not vig_dir.is_dir():
            continue
        summary_path = vig_dir / "summary.json"
        if not summary_path.exists():
            continue
        data = json.loads(summary_path.read_text())
        for entry in data.get("trace", []):
            rows.append({
                "vignette": vig_dir.name,
                "cycle": entry["cycle"],
                "violations": _violation_count(entry.get("results_summary", "")),
                "conforms": entry.get("conforms", False),
            })
    return rows


# ---------------------------------------------------------------------------
# Plot 1: Conformance + PC by system.
# ---------------------------------------------------------------------------

def plot_conformance_pc(schema: str, track: str, prompt: str, out_dir: Path) -> None:
    """Side-by-side bar chart: SHACL pass-rate and mean PC per system.

    Denominators are present-only (vignettes the system actually ran on),
    matching stats_core. The PC CI uses the shared bootstrap helper so the
    interval here equals the one in stats.json.
    """
    data = _load_eval(schema, track, prompt)
    if data is None:
        print(f"[SKIP] no eval.json for {schema}/{track}/{prompt}")
        return

    per_v = data["per_vignette"]
    systems = [s for s in ("a", "b", "c") if any(s in pv for pv in per_v)]

    shacl_rates, pc_means, pc_lows, pc_highs, present_counts = [], [], [], [], []
    for s in systems:
        present = [pv for pv in per_v if _is_present(pv, s)]
        n_present = len(present)
        present_counts.append(n_present)
        passes = sum(1 for pv in present if pv.get(s, {}).get("shacl_conforms") is True)
        shacl_rates.append(passes / n_present if n_present else 0.0)

        pcs = [pv[s]["pc"] for pv in present if pv[s].get("pc") is not None]
        if len(pcs) >= 2:
            mu = float(np.mean(pcs))
            lo, hi = _bootstrap_ci(pcs)
        elif pcs:
            mu = float(pcs[0])
            lo = hi = mu
        else:
            mu, lo, hi = 0.0, 0.0, 0.0
        pc_means.append(mu)
        pc_lows.append(max(0.0, mu - lo))
        pc_highs.append(max(0.0, hi - mu))

    x = np.arange(len(systems))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars_shacl = ax.bar(x - width / 2, shacl_rates, width, label="SHACL pass-rate",
                        color=[SYSTEM_COLORS[s] for s in systems], alpha=0.9)
    bars_pc = ax.bar(x + width / 2, pc_means, width,
                     yerr=[pc_lows, pc_highs], capsize=4,
                     label="Property Completeness (PC)", color=[SYSTEM_COLORS[s] for s in systems],
                     alpha=0.55, hatch="//")

    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score (0–1)", fontsize=11)
    n_label = "/".join(str(c) for c in present_counts) if present_counts else "?"
    ax.set_title(
        f"SHACL Conformance vs Property Completeness\n"
        f"schema={schema}, track={track}, prompt={prompt} "
        f"(n_present={n_label})",
        fontsize=11,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{SYSTEM_LABELS[s]}\nn={c}" for s, c in zip(systems, present_counts)],
        fontsize=10,
    )
    ax.axhline(1.0, color="grey", linewidth=0.5, linestyle="--")

    for bar in bars_shacl:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    for bar, mu in zip(bars_pc, pc_means):
        ax.text(bar.get_x() + bar.get_width() / 2, mu + 0.015,
                f"{mu:.2f}", ha="center", va="bottom", fontsize=8)

    solid_patch = mpatches.Patch(color="grey", alpha=0.9, label="SHACL pass-rate")
    hatch_patch = mpatches.Patch(facecolor="grey", alpha=0.55, hatch="//", label="Property Completeness (PC, ±95% CI)")
    ax.legend(handles=[solid_patch, hatch_patch], fontsize=9, loc="upper right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = out_dir / f"conformance_pc_{schema}_{track}_{prompt}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Plot 2: Violation reduction per SHACL cycle (System B).
# ---------------------------------------------------------------------------

def plot_violation_counts_by_cycle(schema: str, track: str, prompt: str, out_dir: Path) -> None:
    """Line + box plot: SHACL violation count by cycle for System B."""
    rows = _load_cycle_summaries(schema, track, prompt, "b")
    if not rows:
        print(f"[SKIP] no cycle summaries for {schema}/{track}/{prompt}/b")
        return

    # Group by vignette → ordered list of violation counts per cycle.
    from collections import defaultdict
    vig_series: dict[str, list[int]] = defaultdict(list)
    for r in sorted(rows, key=lambda x: (x["vignette"], x["cycle"])):
        # Only vignettes that had at least 1 violation at cycle 0 are interesting.
        vig_series[r["vignette"]].append(r["violations"])

    # Keep only vignettes that started with > 0 violations.
    series_with_violations = {v: vs for v, vs in vig_series.items() if vs[0] > 0}
    max_cycles = max(len(v) for v in series_with_violations.values()) if series_with_violations else 0

    if not series_with_violations:
        print(f"[SKIP] no vignettes with cycle-0 violations for {schema}/{track}/{prompt}/b")
        return

    # Pad shorter series to max_cycles with their last value.
    padded = {}
    for v, vs in series_with_violations.items():
        padded[v] = vs + [vs[-1]] * (max_cycles - len(vs))

    # Box plot per cycle + mean line.
    cycle_data = [
        [padded[v][c] for v in padded] for c in range(max_cycles)
    ]
    cycle_means = [np.mean(cd) for cd in cycle_data]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Individual vignette lines (light grey, low alpha)
    x_range = list(range(max_cycles))
    for v, vs in padded.items():
        ax.plot(x_range, vs, color="grey", alpha=0.15, linewidth=0.8)

    # Box plot per cycle
    bp = ax.boxplot(cycle_data, positions=x_range, widths=0.4,
                    patch_artist=True, showfliers=True,
                    boxprops=dict(facecolor="#DD8452", alpha=0.4),
                    medianprops=dict(color="#AA4400", linewidth=2),
                    flierprops=dict(marker="o", markersize=3, alpha=0.4))

    # Mean line
    ax.plot(x_range, cycle_means, "o-", color="#DD8452", linewidth=2,
            markersize=6, label="Mean violations", zorder=3)
    for xi, mu in zip(x_range, cycle_means):
        ax.text(xi, mu + 0.2, f"{mu:.1f}", ha="center", va="bottom", fontsize=8, color="#AA4400")

    ax.set_xlabel("SHACL Correction Cycle", fontsize=11)
    ax.set_ylabel("Violation Count", fontsize=11)
    ax.set_title(f"SHACL Violation Reduction — System B\n"
                 f"schema={schema}, track={track}, prompt={prompt} "
                 f"(n={len(series_with_violations)} vignettes with violations)", fontsize=10)
    ax.set_xticks(x_range)
    ax.set_xticklabels([f"Cycle {c}" for c in x_range])
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    mean_patch = mpatches.Patch(color="#DD8452", label=f"Mean (starts {cycle_means[0]:.1f}, ends {cycle_means[-1]:.1f})")
    ax.legend(handles=[mean_patch], fontsize=9)

    fig.tight_layout()
    out = out_dir / f"violation_counts_by_cycle_{schema}_{track}_{prompt}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Plot 3: RQ5 — Violation shape resolution rate (horizontal bar chart).
# ---------------------------------------------------------------------------
# For each named SHACL shape that appears in System B cycles, compute:
#   total_count  — total times that shape fired across all cycles
#   first_count  — times it fired at cycle 0 (baseline exposure)
#   final_count  — times it was still present in the LAST cycle of
#                  non-conformant vignettes (resistant violations)
#   resolved     = first_count - final_count
#   resolution_rate = resolved / first_count
#
# Shapes with high resolution rate are correctable by the SHACL loop.
# Shapes with low resolution rate resist correction — RQ5's main finding.
# ---------------------------------------------------------------------------

def _read_violations_from_txt(path: Path) -> list[str]:
    """Parse violation Source Shape names from a violations_NN.txt file.

    Named shapes (e.g. chr:HasLocatedInDomainShape) are returned as-is.
    Anonymous blank-node shapes (MinCount/MaxCount on a specific path) are
    labelled by their Result Path or Message so they're still identifiable.
    """
    shapes: list[str] = []
    text = path.read_text()
    for block in re.split(r"Constraint Violation", text):
        src_m = re.search(r"Source Shape:\s*(\S.*?)(?:\n|$)", block)
        if not src_m:
            continue
        src = src_m.group(1).strip()
        if src.startswith("["):
            # anonymous shape — use Result Path or Message as label
            path_m = re.search(r"Result Path:\s*(\S+)", block)
            msg_m = re.search(r"Message:\s*(.+?)(?:\n|$)", block)
            if path_m:
                src = f"anon:{path_m.group(1).strip()}"
            elif msg_m:
                src = f"anon:{msg_m.group(1).strip()[:40]}"
            else:
                src = "anon:unknown"
        shapes.append(src)
    return shapes


def plot_violation_resolution(schema: str, track: str, prompt: str, out_dir: Path) -> None:
    """Horizontal bar chart of violation shape resolution rate for System B."""
    base = OUTPUTS_ROOT / schema / track / prompt / "b" / "cycles"
    if not base.exists():
        print(f"[SKIP] no cycle data for {schema}/{track}/{prompt}/b")
        return

    from collections import defaultdict, Counter
    first_counts: Counter = Counter()    # violations at cycle 0
    final_counts: Counter = Counter()   # violations in last cycle of non-conformant runs

    for vig_dir in sorted(base.iterdir()):
        if not vig_dir.is_dir():
            continue
        summary_path = vig_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())

        # Cycle 0 violations
        v0 = vig_dir / "violations_00.txt"
        if v0.exists():
            for shape in _read_violations_from_txt(v0):
                first_counts[shape] += 1

        # Final-cycle violations for runs that didn't fully conform
        if not summary.get("final_conforms", True):
            n_cycles = len(summary.get("trace", []))
            vf = vig_dir / f"violations_{n_cycles - 1:02d}.txt"
            if vf.exists():
                for shape in _read_violations_from_txt(vf):
                    final_counts[shape] += 1

    # Only shapes seen at cycle 0
    shapes_seen = [s for s, c in first_counts.most_common() if c >= 2]
    if not shapes_seen:
        print(f"[SKIP] no cycle-0 violations for {schema}/{track}/{prompt}/b")
        return

    resolved_rates = []
    for shape in shapes_seen:
        first = first_counts[shape]
        final = final_counts.get(shape, 0)
        resolved_rates.append((shape, first, final, 1.0 - final / first))

    resolved_rates.sort(key=lambda x: x[3])  # ascending resolution = worst at top

    labels = [r[0].replace("chr:", "").replace("anon:", "↳ ") for r in resolved_rates]
    rates = [r[3] for r in resolved_rates]
    firsts = [r[1] for r in resolved_rates]

    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.4 * len(labels) + 1.0)))
    colors = ["#DD8452" if r < 0.5 else "#55A868" for r in rates]
    bars = ax.barh(labels, rates, color=colors, alpha=0.85)

    for bar, (_, first, final, rate) in zip(bars, resolved_rates):
        ax.text(
            min(rate + 0.02, 0.98), bar.get_y() + bar.get_height() / 2,
            f"{rate:.0%}  ({first - final}/{first})",
            va="center", ha="left", fontsize=8,
        )

    ax.set_xlim(0, 1.25)
    ax.axvline(0.5, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("Resolution Rate  (1 − final_count/first_count)", fontsize=10)
    ax.set_title(
        f"SHACL Violation Resolution Rate by Shape — System B\n"
        f"schema={schema}, track={track}, prompt={prompt}\n"
        f"(green ≥ 50% resolved, orange < 50%)",
        fontsize=10,
    )
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#55A868", alpha=0.85, label="Resolved ≥ 50%"),
        Patch(facecolor="#DD8452", alpha=0.85, label="Resistant < 50%"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="lower right")

    fig.tight_layout()
    out = out_dir / f"violation_resolution_{schema}_{track}_{prompt}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Plot 4 (text): Appendix trace for one vignette.
# ---------------------------------------------------------------------------

def _read_ttl_summary(path: Path) -> str:
    """Return a short summary of a Turtle file's typed triples."""
    if not path.exists():
        return "  (file not found)"
    try:
        from rdflib import Graph, RDF
        g = Graph()
        g.parse(path.as_posix(), format="turtle")
        type_counts: dict[str, int] = {}
        predicates: set[str] = set()
        for s, p, o in g:
            if p == RDF.type:
                cls = str(o).rsplit("/", 1)[-1].rsplit("#", 1)[-1]
                type_counts[cls] = type_counts.get(cls, 0) + 1
            else:
                pred = str(p).rsplit("/", 1)[-1].rsplit("#", 1)[-1]
                predicates.add(pred)
        total = sum(1 for _ in g) - sum(type_counts.values())
        class_summary = ", ".join(f"{v}× {k}" for k, v in sorted(type_counts.items()))
        return (f"  Typed instances: {class_summary or '(none)'}\n"
                f"  Relation triples: {total}  |  Predicates used: {', '.join(sorted(predicates)) or '(none)'}")
    except Exception as e:
        return f"  (parse error: {e})"


def _parse_violations(results_summary: str) -> list[str]:
    """Extract individual violation messages from a SHACL results_summary."""
    violations = []
    for block in re.split(r"Constraint Violation", results_summary):
        msg_m = re.search(r"Message:\s*(.+?)(?:\n|$)", block)
        shape_m = re.search(r"Source Shape:\s*(.+?)(?:\n|$)", block)
        focus_m = re.search(r"Focus Node:\s*(.+?)(?:\n|$)", block)
        if msg_m:
            msg = msg_m.group(1).strip()
            shape = shape_m.group(1).strip() if shape_m else "?"
            focus = focus_m.group(1).strip() if focus_m else "?"
            violations.append(f"    [{shape}] on {focus}: {msg}")
    return violations


def write_appendix_trace(
    schema: str, track: str, prompt: str, vignette: str, out_dir: Path
) -> None:
    """Write a human-readable pipeline trace for thesis appendix."""
    base = OUTPUTS_ROOT / schema / track / prompt / "b" / "cycles" / vignette
    summary_path = base / "summary.json"
    if not summary_path.exists():
        print(f"[SKIP] no cycle data at {base}")
        return

    summary = json.loads(summary_path.read_text())
    trace = summary.get("trace", [])

    # Vignette source text
    text_matches = list(VIGNETTES_DIR.glob(f"{vignette}_*.txt")) + list(VIGNETTES_DIR.glob(f"{vignette}.txt"))
    source_text = text_matches[0].read_text().strip() if text_matches else "(vignette text not found)"

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"APPENDIX — Pipeline Trace: {vignette}")
    lines.append(f"Schema: {schema}  Track: {track}  Prompt: {prompt}  System: B (SHACL loop)")
    lines.append("=" * 72)
    lines.append("")
    lines.append("INPUT CLINICAL TEXT")
    lines.append("-" * 40)
    for para in source_text.split("\n"):
        lines.append(textwrap.fill(para, width=70) if para.strip() else "")
    lines.append("")

    for entry in trace:
        c = entry["cycle"]
        conforms = entry.get("conforms", False)
        n_viol = _violation_count(entry.get("results_summary", ""))
        violations = _parse_violations(entry.get("results_summary", ""))

        ttl_path = base / f"cycle_{c:02d}.ttl"

        lines.append(f"CYCLE {c}  {'─' * 55}")
        if c == 0:
            lines.append("Source: zero-shot extraction (System A output)")
        else:
            lines.append("Source: SHACL-feedback correction (System B retry)")
        lines.append("")
        lines.append("Extracted graph:")
        lines.append(_read_ttl_summary(ttl_path))
        lines.append("")
        if conforms:
            lines.append("SHACL result: CONFORMS ✓  (0 violations)")
        else:
            lines.append(f"SHACL result: FAIL  ({n_viol} violation{'s' if n_viol != 1 else ''})")
            for v in violations:
                lines.append(v)
        lines.append("")

    # Summary
    final = summary.get("final_conforms", False)
    llm_calls = summary.get("llm_calls", len(trace) - 1)
    lines.append("─" * 72)
    lines.append(f"OUTCOME: {'Conformant ✓' if final else 'Non-conformant ✗  (retry budget exhausted)'}")
    lines.append(f"LLM correction calls: {llm_calls}")
    lines.append("")

    # Identify persistent violations (appear in last cycle)
    if trace:
        last_violations = _parse_violations(trace[-1].get("results_summary", ""))
        if last_violations and not final:
            lines.append("RESISTANT VIOLATIONS (present in final cycle):")
            for v in last_violations:
                lines.append(v)
            lines.append("")
            lines.append("Interpretation: These violations resisted correction across all")
            lines.append("retry cycles, indicating a gap between the ontological constraint")
            lines.append("and the LLM's prompt-level understanding of the distinction.")

    lines.append("=" * 72)

    out = out_dir / f"appendix_trace_{schema}_{track}_{prompt}_{vignette}.txt"
    out.write_text("\n".join(lines))
    print(f"Wrote {out.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schema", default="chr")
    ap.add_argument("--track", default="schema")
    ap.add_argument("--prompt", default="full")
    ap.add_argument("--vignette", default="vignette_001",
                    help="Vignette ID for appendix trace (default: vignette_001)")
    ap.add_argument("--all-tracks", action="store_true",
                    help="Generate plots for all available schema/track/prompt combos.")
    args = ap.parse_args()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.all_tracks:
        combos = [
            ("chr", "schema",   "full"),
            ("chr", "ontology", "ontology"),
        ]
    else:
        combos = [(args.schema, args.track, args.prompt)]

    for schema, track, prompt in combos:
        print(f"\n── {schema}/{track}/{prompt} ──")
        plot_conformance_pc(schema, track, prompt, PLOTS_DIR)
        plot_violation_counts_by_cycle(schema, track, prompt, PLOTS_DIR)
        plot_violation_resolution(schema, track, prompt, PLOTS_DIR)

    # Appendix trace — always for the primary combo.
    write_appendix_trace(args.schema, args.track, args.prompt, args.vignette, PLOTS_DIR)


if __name__ == "__main__":
    main()
