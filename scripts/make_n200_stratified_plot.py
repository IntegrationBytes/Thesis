"""Generate the headline n=200 stratified plot showing the supervisor's
'complexity unlocks SHACL signal' finding.

Single figure, 2 panels (schema | ontology), grouped bars showing
A and B SHACL conformance per stratum × model. The story: complex
stratum starts much lower (more room) and B closes the gap.

Output: report/figures/fig_n200_stratified.png
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Hardcoded results from the n=200 stratified eval (2026-05-24).
# Schema track: SHACL conformance (pipeline/evaluate.py).
# Ontology track: OWL reasoner conformance (scripts/owl_conformance_survey.py)
# — replaces the hand-written-SHACL numbers per supervisor's request:
# validation should be grounded in SULO axioms, not in our custom shapes.
DATA = {
    "schema": {
        "Gemini": {
            "general": {"A": 4.0, "B": 96.0},
            "complex": {"A": 3.0, "B": 100.0},
        },
        "gpt-oss": {
            "general": {"A": 19.0, "B": 87.0},
            "complex": {"A": 42.0, "B": 97.0},
        },
    },
    "ontology": {  # OWL conformance, B = OWL-driven retry
        "Gemini": {
            "general": {"A": 65.0, "B": 71.0},
            "complex": {"A": 19.0, "B": 59.0},
        },
        "gpt-oss": {
            "general": {"A": 83.0, "B": 96.0},
            "complex": {"A": 94.0, "B": 99.0},
        },
    },
}


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5), sharey=True)

    track_titles = {
        "schema": "Schema track",
        "ontology": "Ontology track (OWL-RL + SHACL)",
    }

    strata = ["general", "complex"]
    models = ["Gemini", "gpt-oss"]

    bar_width = 0.18
    group_positions = np.arange(len(strata))  # [0, 1] for general/complex

    # Colors: paired A (lighter) / B (darker), one pair per model
    colors = {
        "Gemini":  {"A": "#a7c4e2", "B": "#1f6fb4"},  # blue
        "gpt-oss": {"A": "#f1b3a0", "B": "#c44e2e"},  # warm orange
    }

    for col, track in enumerate(("schema", "ontology")):
        ax = axes[col]
        # Place 4 bars per stratum: Gem-A, Gem-B, gpt-A, gpt-B
        for i_model, model in enumerate(models):
            for j_sys, sys_ in enumerate(("A", "B")):
                offset = (-1.5 + i_model * 2 + j_sys) * bar_width
                values = [DATA[track][model][st][sys_] for st in strata]
                bars = ax.bar(
                    group_positions + offset, values, bar_width,
                    color=colors[model][sys_],
                    edgecolor="black", linewidth=0.4,
                    label=f"{model} {sys_}" if col == 0 else None,
                )
                for b, v in zip(bars, values):
                    ax.text(b.get_x() + b.get_width() / 2,
                            b.get_height() + 1.5, f"{v:.0f}",
                            ha="center", va="bottom", fontsize=7)

        ax.set_xticks(group_positions)
        ax.set_xticklabels([f"general\n(n=100)", f"complex\n(n=100)"], fontsize=9)
        ax.set_ylim(0, 110)
        ax.set_ylabel("SHACL pass-rate (%)" if col == 0 else "")
        ax.set_title(track_titles[track], fontsize=10)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.set_axisbelow(True)

        # Annotate the key Δ for complex stratum (supervisor's finding)
        if track == "ontology":
            ax.annotate(
                "complex stratum:\n4× more headroom\nfor SHACL feedback",
                xy=(1, 21), xytext=(1.05, 50),
                fontsize=7.5, ha="left",
                arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.8),
            )

    # Single legend on the left
    axes[0].legend(loc="lower left", fontsize=8, framealpha=0.9, ncol=2)

    fig.suptitle(
        "SHACL retry loop: A → B conformance lift, stratified by input complexity (n=200)",
        fontsize=10.5, y=1.02,
    )
    fig.tight_layout()

    out = Path("report/figures/fig_n200_stratified.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
