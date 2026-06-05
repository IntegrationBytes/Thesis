"""OWL track process figure — input merge, OWL-RL closure, three axiom-level
consistency checks, and the conformance-or-feedback route.

Output: report/figures/fig_owl_process.png
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path("/Users/vincentviitala/Downloads/Thesis/thesiscode/report/figures/fig_owl_process.png")

C_INPUT  = "#dfe4f5"
C_ONTO   = "#fff2c4"
C_CLOSURE= "#fde6d2"
C_CHECK  = "#eef2fb"
C_VIO    = "#f9d6d5"
C_OK     = "#cfe9cf"
C_HDR    = "#e7ebf6"
EDGE     = "#1f2937"
MUTED    = "#6b7280"
RED      = "#b91c1c"
GREEN    = "#15803d"
BLUE     = "#1f3a8a"


def rbox(x, y, w, h, color, text, fs=8.0, fc_text="#111827", bold=False):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4",
                         facecolor=color, edgecolor=EDGE, lw=0.9)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=fc_text)


def codebox(x, y, w, h, text, fs=7.2, color="#ffffff", title=None, title_color=None):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3",
                         facecolor=color, edgecolor=EDGE, lw=0.7)
    ax.add_patch(box)
    if title:
        ax.text(x + 1.5, y + h - 1.5, title,
                ha="left", va="top", fontsize=fs + 0.4, fontweight="bold",
                color=title_color or "#374151")
        ax.text(x + 1.5, y + h - 5, text,
                ha="left", va="top", fontsize=fs, family="monospace",
                color="#111827", linespacing=1.2)
    else:
        ax.text(x + 1.5, y + h - 2, text, ha="left", va="top",
                fontsize=fs, family="monospace", color="#111827", linespacing=1.2)


def arrow(x1, y1, x2, y2, color=EDGE, lw=1.5, style="-|>", curve=0.0):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle=style, mutation_scale=14, lw=lw,
                        color=color,
                        connectionstyle=f"arc3,rad={curve}")
    ax.add_patch(a)


fig, ax = plt.subplots(figsize=(12, 7.0))
ax.set_xlim(0, 240)
ax.set_ylim(0, 145)
ax.axis("off")
ax.set_facecolor("white")

# Title
ax.text(120, 140, "Ontology-track verifier loop  ·  OWL-RL closure over SULO + CHR TBox + candidate ABox",
        ha="center", fontsize=11.5, fontweight="bold", color=BLUE)

# Phase headers
rbox(5, 113, 70, 6, C_HDR, "1 · INPUTS", fs=9.4, bold=True)
rbox(85, 113, 80, 6, C_HDR, "2 · CLOSURE + AXIOM-LEVEL CHECKS", fs=9.4, bold=True)
rbox(175, 113, 60, 6, C_HDR, "3 · ROUTE", fs=9.4, bold=True)

# ===== INPUTS =====
codebox(5, 89, 70, 21, color=C_INPUT,
        title="LLM candidate ABox (cycle 0)",
        title_color=BLUE,
        text=("ex:careplan_2020  a chr:CarePlan ,\n"
              "                  chr:TreatmentPlan ;   # both -> clash\n"
              "  chr:hasPatient ex:patient_M .\n"
              "ex:patient_M  a chr:Person ;\n"
              "  rdfs:label \"Maria\" ."))

codebox(5, 64, 70, 22, color=C_ONTO,
        title="CHR TBox  (chr_ontology.ttl)",
        title_color="#7c2d12",
        text=("25 classes, 20 properties\n"
              "chr:CarePlan       rdfs:subClassOf sulo:InformationObject .\n"
              "chr:TreatmentPlan  rdfs:subClassOf sulo:Process .\n"
              "(maps CHR vocabulary into SULO categories)"))

codebox(5, 38, 70, 22, color=C_ONTO,
        title="SULO upper ontology  (sulo_fetched.ttl)",
        title_color="#7c2d12",
        text=("w3id.org/sulo/  ·  cached locally\n"
              "sulo:InformationObject rdfs:subClassOf sulo:Object .\n"
              "sulo:Object   owl:disjointWith  sulo:Process .\n"
              "sulo:Feature  owl:disjointWith  sulo:SpatialObject .\n"
              "(SULO 1.0 axioms drive deductive closure)"))

# Merge arrow
arrow(75, 100, 90, 100, lw=1.7)
arrow(75, 75, 90, 98, lw=1.5)
arrow(75, 49, 90, 96, lw=1.5)
ax.text(82, 105, "merge", fontsize=8.6, color=MUTED, style="italic")

# ===== CLOSURE =====
rbox(90, 89, 70, 21, C_CLOSURE,
     "OWL-RL deductive closure\n(owlrl.DeductiveClosure)\nmaterialise inferred triples\n+ apply SULO axioms",
     fs=9.6, bold=True)

# Arrow down to checks
arrow(125, 89, 125, 80, lw=1.5)
ax.text(127, 84, "scan inferred + asserted", fontsize=8.2, color=MUTED, style="italic")

# Three checks (clean horizontal row, no overlapping)
check_y = 50
check_w = 22
check_h = 28
check_gap = 1.5
check_xs = [90, 90 + check_w + check_gap, 90 + 2*(check_w + check_gap)]

for cx, label, body, marker in [
    (check_xs[0], "(i)  Disjointness", "subjects with\nmutually exclusive\nclass assertions", "✗"),
    (check_xs[1], "(ii)  FunctionalProp", "subject has >1\ndistinct value on\na functional property", "✗"),
    (check_xs[2], "(iii)  Range", "object fails required\nrange after\nclosure expansion", "✗"),
]:
    box = FancyBboxPatch((cx, check_y), check_w, check_h, boxstyle="round,pad=0.3",
                          facecolor=C_CHECK, edgecolor=EDGE, lw=0.8)
    ax.add_patch(box)
    ax.text(cx + check_w/2, check_y + check_h - 3, label,
            ha="center", va="top", fontsize=8.2, fontweight="bold", color=BLUE)
    ax.text(cx + check_w/2, check_y + check_h - 10, body,
            ha="center", va="top", fontsize=7.4, color="#111827",
            linespacing=1.25)
    ax.text(cx + check_w/2, check_y + 4, marker,
            ha="center", va="center", fontsize=18, color=RED,
            fontweight="bold", alpha=0.5)

# Container hint
ax.plot([89, 161, 161, 89, 89],
        [49, 49, 80, 80, 49],
        ls=":", color=MUTED, lw=0.8)

# Arrow from checks to route
arrow(161, 65, 175, 65, lw=1.6)

# ===== ROUTE =====
# Top: Inconsistent
rbox(175, 89, 60, 12, C_VIO,
     "✗  Inconsistent  (any check fires)",
     fs=10, fc_text=RED, bold=True)
codebox(175, 67, 60, 19, color="#ffffff",
        title="Format inconsistency report",
        title_color=RED,
        text=("• ex:careplan_2020 entailed as both\n"
              "  sulo:Object (via chr:CarePlan) and\n"
              "  sulo:Process (via chr:TreatmentPlan)\n"
              "  -> Object/Process disjointness clash\n"
              "→ append to prompt, retry (k ≤ 3)"))

# Bottom: Consistent
rbox(175, 47, 60, 12, C_OK,
     "✓  Consistent  (no check fires)",
     fs=10, fc_text=GREEN, bold=True)
codebox(175, 25, 60, 19, color="#ffffff",
        title="Emit RDF, record k*",
        title_color=GREEN,
        text=("• cycle k* accepted (first pass)\n"
              "• trace → summary.json\n"
              "• graph → outputs/.../b/v_NNN.ttl\n"
              "• OWL pass-rate ↑ in Fig. 1(b)"))

# Feedback loop: explicit arrow from inconsistent → back to LLM (top of inputs)
# Routed above the phase headers so it doesn't collide with them
arrow(205, 101, 40, 110, lw=1.8, color=RED, curve=-0.35)
ax.text(120, 132, "feedback: violation report → LLM prompt (t+1)",
        ha="center", fontsize=9.5, color=RED, fontweight="bold", style="italic")

# Footnote
ax.text(120, 15,
        "fixed contract :  validate_with_owl(g, ctx) → (conforms: bool, report_text: str)   ·   identical signature to schema track (pyshacl)",
        ha="center", fontsize=8.6, color=MUTED, style="italic", family="monospace")
ax.text(120, 8,
        "termination conditions :  conforms = True   ∨   k = 3   ∨   two-cycle plateau (identical error report twice in a row)",
        ha="center", fontsize=8.6, color=MUTED, style="italic", family="monospace")

fig.tight_layout(pad=0.8)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}")
