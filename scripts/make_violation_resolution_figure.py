"""Violation-resolution figure — per-shape A→B counts, schema and ontology
tracks side by side.

Panel (a): SHACL shape-level violations on the schema track.
Panel (b): OWL axiom-level violations on the ontology track (Disjointness,
           FunctionalProperty, Range).

Each row shows cycle-0 violations (light, hollow) and cycle-3 violations
(solid). Annotation shows raw counts and fix rate.

Output: report/figures/fig_violation_resolution.png
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report/figures/fig_violation_resolution.png"
DATA = ROOT / "evaluation/outputs/per_shape_violations.json"

GPTOSS_ORANGE = "#c44e2e"
ONTO_BLUE     = "#1f6fb4"


def short_shape(name: str) -> str:
    if name == "[":
        return "(unnamed)"
    return name.rsplit(":", 1)[-1].rsplit("/", 1)[-1].replace("Shape", "")[:34]


def model_data(d: dict, a_key: str, b_key: str) -> dict[str, tuple[int, int]]:
    a = d.get(a_key, {})
    b = d.get(b_key, {})
    out: dict[str, tuple[int, int]] = {}
    for sh in set(a) | set(b):
        if sh == "[":
            continue
        out[sh] = (a.get(sh, 0), b.get(sh, 0))
    return out


def draw_panel(ax, items, color: str, title: str, xlabel: str) -> None:
    if not items:
        ax.text(0.5, 0.5, "(no data yet)", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="#9ca3af")
        ax.set_title(title, fontsize=10.5, loc="left", pad=8, color=color,
                     fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ("top", "right", "left", "bottom"):
            ax.spines[s].set_visible(False)
        return

    labels = [n for n, _, _ in items]
    a_vals = [a for _, a, _ in items]
    b_vals = [b for _, _, b in items]

    y = np.arange(len(items))
    h = 0.66

    ax.barh(y, a_vals, h, facecolor=color, alpha=0.22,
            edgecolor=color, lw=1.0, label="cycle-0 (A)")
    ax.barh(y, b_vals, h, color=color, edgecolor="black", lw=0.4,
            label="cycle-3 (B)")

    xmax = max(a_vals + [1]) * 1.02
    for i, (a, b) in enumerate(zip(a_vals, b_vals)):
        fix = 100 * (1 - b / a) if a else 0
        txt = f"{a}→{b}  ({fix:.0f}%)"
        ax.text(a + xmax * 0.015, y[i], txt, va="center",
                fontsize=8.6, color="#374151", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, xmax * 1.18)
    ax.set_xlabel(xlabel, fontsize=9.3)
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.3, ls=":")
    ax.set_title(title, fontsize=10.5, loc="left", pad=8, color=color,
                 fontweight="bold")
    ax.legend(loc="lower right", fontsize=8.2, framealpha=0.92)


def main() -> None:
    d = json.load(open(DATA))

    # Panel (a): schema track. The historical "gemini" key now points to
    # gpt-oss-120b data (outputs/chr/...).
    schema_md = model_data(d.get("schema", {}), "gemini_a_all", "gemini_b_all")
    schema_items = sorted(
        [(short_shape(n), a, b) for n, (a, b) in schema_md.items() if a >= 1],
        key=lambda r: r[1], reverse=True,
    )

    # Panel (b): ontology track. 3 axiom-level violation types.
    onto_md = model_data(d.get("ontology", {}), "gptoss_a_all", "gptoss_b_all")
    # Preserve the canonical (Disjointness, Functional, Range) order
    order = ["DisjointnessClash", "FunctionalPropertyConflict", "RangeViolation"]
    onto_items = []
    for n in order:
        a, b = onto_md.get(n, (0, 0))
        # Shorten long names so they fit
        short = n.replace("DisjointnessClash", "Disjointness")\
                  .replace("FunctionalPropertyConflict", "FunctionalProperty")\
                  .replace("RangeViolation", "Range")
        if a >= 1 or b >= 1:
            onto_items.append((short, a, b))

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.8, 3.6),
                                       gridspec_kw={"width_ratios": [1.0, 0.85]})

    draw_panel(ax_l, schema_items, GPTOSS_ORANGE,
               "(a)  Schema track  ·  SHACL shape-level violations",
               "SHACL violations across n=200 vignettes")

    draw_panel(ax_r, onto_items, ONTO_BLUE,
               "(b)  Ontology track  ·  OWL axiom-level violations",
               "OWL violations across n=200 vignettes")

    fig.tight_layout(pad=1.2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
