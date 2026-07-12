"""Light worked-example figure for the OWL ontology track (slide 8).

A four-step, left-to-right walkthrough in the same light pastel style as the
SHACL worked example: one over-typed node -> OWL-RL closure -> disjointness
clash -> feedback fix. Output: report/figures/fig_owl_worked.png
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "report/figures/fig_owl_worked.png"

C_INPUT = "#dfe4f5"
C_ONTO  = "#fff2c4"
C_VIO   = "#f9d6d5"
C_OK    = "#cfe9cf"
EDGE    = "#1f2937"
MUTED   = "#6b7280"
RED     = "#b91c1c"
GREEN   = "#15803d"
BLUE    = "#1f3a8a"

fig, ax = plt.subplots(figsize=(12, 5.11))
ax.set_xlim(0, 240)
ax.set_ylim(0, 102)
ax.axis("off")
ax.set_facecolor("white")

ax.text(120, 97,
        "Ontology track:  OWL-RL closure turns one over-typed node into a disjointness clash",
        ha="center", fontsize=12, fontweight="bold", color=BLUE)

xs = [4, 63, 122, 181]
W = 51
BY, BH = 20, 58

def step(x, color, label, lines, accent="#374151", marker=None, marker_color=RED):
    box = FancyBboxPatch((x, BY), W, BH, boxstyle="round,pad=0.4",
                         facecolor=color, edgecolor=EDGE, lw=0.9)
    ax.add_patch(box)
    ax.text(x + W / 2, BY + BH - 3, label, ha="center", va="top",
            fontsize=8.4, fontweight="bold", color=accent)
    ax.text(x + 3.0, BY + BH - 12, lines, ha="left", va="top",
            fontsize=7.6, family="monospace", color="#111827", linespacing=1.4)
    if marker:
        ax.text(x + W / 2, BY + 6.5, marker, ha="center", va="center",
                fontsize=12, fontweight="bold", color=marker_color,
                linespacing=1.0)

step(xs[0], C_INPUT, "1 · Candidate ABox",
     "ex:careplan_2020\n"
     "   a chr:CarePlan ,\n"
     "     chr:TreatmentPlan .\n\n"
     "one node,\ntwo class types",
     accent=BLUE)

step(xs[1], C_ONTO, "2 · OWL-RL closure",
     "CarePlan      ⊑ Object\n"
     "TreatmentPlan ⊑ Process\n\n"
     "⇒ careplan inferred\n"
     "   as sulo:Object\n"
     "   AND sulo:Process",
     accent="#7c2d12")

step(xs[2], C_VIO, "3 · Axiom check (cax-dw)",
     "sulo:Object\n"
     "  owl:disjointWith\n"
     "  sulo:Process\n\n"
     "both hold on\none node  →",
     accent=RED, marker="Disjointness\nClash", marker_color=RED)

step(xs[3], C_OK, "4 · Feedback → fix",
     "report → LLM (k+1)\n"
     "drop TreatmentPlan\n\n"
     "⇒ graph consistent\n   (one retry)",
     accent=GREEN, marker="✓ conforms", marker_color=GREEN)

def arrow(x1, x2):
    a = FancyArrowPatch((x1, BY + BH / 2), (x2, BY + BH / 2),
                        arrowstyle="-|>", mutation_scale=16, lw=1.7, color=EDGE)
    ax.add_patch(a)

arrow(xs[0] + W, xs[1])
arrow(xs[1] + W, xs[2])
arrow(xs[2] + W, xs[3])

ax.text(120, 9,
        "validate_with_owl(g, ctx) → (conforms, report)   ·   same contract as the SHACL track   ·   "
        "terminates on conforms ∨ k=3 ∨ plateau",
        ha="center", fontsize=8.6, color=MUTED, style="italic", family="monospace")

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, facecolor="white")
print(f"wrote {OUT}")
