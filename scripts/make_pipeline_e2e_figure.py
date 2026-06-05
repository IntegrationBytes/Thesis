"""End-to-end pipeline figure for vignette_001.

Renders Synthea -> FHIR Bundle -> text/gold fork -> LLM extraction ->
SHACL feedback loop -> emit RDF -> score-vs-gold, with the concrete
payloads observed for vignette_001 (Angelic427 Schaden604).
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "report/figures/fig_pipeline_e2e_vignette_001.png"

C_SEED   = "#dfe4f5"
C_BUNDLE = "#dfe4f5"
C_SCRIPT = "#fff2c4"
C_PROSE  = "#fde6d2"
C_GOLD   = "#dceedc"
C_LLM    = "#dfe4f5"
C_TTL    = "#ffffff"
C_VIO    = "#f9d6d5"
C_OK     = "#cfe9cf"
C_HDR    = "#e7ebf6"
C_SCORE  = "#eef2fb"
EDGE     = "#1f2937"
MUTED    = "#6b7280"
RED      = "#b91c1c"
GREEN    = "#15803d"

# Y range 66-134 (68 units) matches content. H is tuned to control padding
# inside boxes: bigger H → each text line occupies fewer data units → more
# breathing room inside fixed-size codeboxes. H≈6.8 hugs text; H≈8.5 gives
# generous padding without touching any box positions.
W, H = 22, 9.2
fig, ax = plt.subplots(figsize=(W, H), dpi=180)
ax.set_xlim(0, 220)
ax.set_ylim(66, 134)
ax.axis("off")


def rbox(x, y, w, h, color, label, fs=10, weight="normal", border=EDGE, lw=1.0, va="center"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.7",
                       fc=color, ec=border, lw=lw)
    ax.add_patch(p)
    if label:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va=va,
                fontsize=fs, family="sans-serif", fontweight=weight)


def codebox(x, y, w, h, text, fs=7.6, color=C_TTL, title=None, title_color=None, border=EDGE):
    """Box hugs content tightly with a small breathing gap between title and content."""
    rbox(x, y, w, h, color, "", border=border, lw=0.9)
    pad_top = 0.45
    if title:
        ax.text(x + 1.0, y + h - pad_top, title, ha="left", va="top",
                fontsize=fs + 0.7, family="monospace", fontweight="bold",
                color=title_color or "#374151")
        pad_top = 1.8
    ax.text(x + 1.0, y + h - pad_top, text, ha="left", va="top",
            fontsize=fs, family="monospace", color="#111827",
            linespacing=0.97)


def arrow(x1, y1, x2, y2, color=EDGE, lw=1.5, style="-|>", curve=0.0):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                        mutation_scale=14, lw=lw, color=color,
                        connectionstyle=f"arc3,rad={curve}")
    ax.add_patch(a)


ax.text(110, 132,
        "One-vignette walkthrough  ·  vignette_001  (Angelic427 Schaden604  ·  body-temp 37.816 Celsius  ·  acute viral pharyngitis)",
        ha="center", fontsize=15, fontweight="bold")

ax.text(110, 128,
        "Every number on this figure is for THIS single vignette only.  "
        "Same Synthea Bundle drives both the LLM prompt and the gold ABox.",
        ha="center", fontsize=10, color=MUTED, style="italic")

COL_HEADERS = [(35, "1 · PREPARE"),
               (110, "2 · EXTRACT + SHACL FEEDBACK LOOP"),
               (185, "3 · SCORE")]
for cx, label in COL_HEADERS:
    rbox(cx - 30, 121, 60, 4.5, C_HDR, label, fs=11.5, weight="bold")

for x in (70, 150):
    ax.plot([x, x], [70, 119.5], "--", color="#cfd4dc", lw=0.8)

# ===== COLUMN 1 =====
rbox(8, 113, 18, 6, C_SEED, "Synthea seed", fs=9.5)
arrow(26, 116, 36, 116)
rbox(36, 111, 24, 9, C_BUNDLE, "FHIR Bundle\n(JSON)", fs=10)

arrow(48, 111, 48, 109.5)

codebox(4, 97, 62, 12,
        '// Patient (id 5e117ba0…)\n'
        '{"name":[{"family":"Schaden604",\n'
        '          "given":["Angelic427","Callie226"]}],\n'
        ' "gender":"female"}\n'
        '// Observation (vital signs)\n'
        '{"code":"Body temperature",\n'
        ' "valueQuantity":{"value":37.816,"unit":"Cel"},\n'
        ' "effectiveDateTime":"2016-08-07T06:24:27+02:00"}\n'
        '// CarePlan\n'
        '{"resourceType":"CarePlan","status":"completed"}',
        title="FHIR Bundle (excerpt)", title_color="#1f2a55")

arrow(35, 97, 18, 95.5, curve=0.0)
arrow(35, 97, 52, 95.5, curve=0.0)

rbox(6, 90, 24, 5.5, C_SCRIPT, "fhir_to_text.py", fs=9)
rbox(40, 90, 24, 5.5, C_SCRIPT, "fhir_to_chr.py", fs=9)

arrow(18, 90, 18, 88.5)
arrow(52, 90, 52, 88.5)

codebox(4, 75, 30, 13,
        'CLINICAL VISIT NOTE\n'
        '— BOSTON MEDICAL CENTER\n'
        'Patient: Angelic427\n'
        ' Callie226 Schaden604\n'
        'Date: 2016-08-07 06:24\n'
        'Body temperature\n'
        ' 37.816 Cel · completed\n'
        'Dr. Willian804 evaluated\n'
        ' Acute viral pharyngitis\n'
        ' (disorder). Severity: Moderate.\n'
        'Care plan established',
        color=C_PROSE,
        title="vignette.txt", title_color="#7a3a14")

codebox(38, 72, 30, 16,
        'ex:measProc_… a chr:\n'
        '   MeasurementProcess ;\n'
        '  chr:hasPatient … ;\n'
        '  chr:hasResult ex:meas_… .\n'
        'ex:meas_… a chr:Measurement ;\n'
        '  chr:hasQuantityValue\n'
        '   "37.816"^^xsd:float ;\n'
        '  chr:hasUnit sulo:Cel .\n'
        'ex:cond_… a chr:\n'
        '   ClinicalCondition ;\n'
        '  rdfs:label "Acute viral\n'
        '   pharyngitis" .\n'
        '(… 22 gold triples total)',
        color=C_GOLD,
        title="gold.ttl", title_color="#1b5a25")

# Prompt arrow — vignette mid → LLM call 1
arrow(34, 81, 80, 117, curve=0.42, lw=1.8)
ax.text(60, 102, "prompt", fontsize=10, color=EDGE, ha="center", style="italic",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.95))

# ===== COLUMN 2 =====
rbox(80, 114, 40, 6, C_LLM, "LLM   ·   call 1", fs=10.5, weight="bold")
arrow(100, 114, 100, 112.5)

codebox(76, 100, 68, 12,
        'ex:visit_2016_08_07 a chr:ClinicalVisit ;\n'
        '  chr:hasPatient        ex:patient_… ;\n'
        '  chr:hasCareProvider   ex:doctor_… .\n'
        'ex:temp_meas_proc_… a chr:MeasurementProcess ;\n'
        '  chr:hasPatient ex:patient_… ;\n'
        '  chr:hasResult  ex:temp_meas_… .\n'
        'ex:eval_proc_… a chr:EvaluationProcess ;\n'
        '  chr:hasObservation ex:diag_… .\n'
        'ex:care_plan_follow_up a chr:CarePlan .   ← no children',
        title="cycle_00.ttl     (k = 0, 18 triples)",
        title_color="#7a3a14")

arrow(100, 100, 100, 98.5)

codebox(76, 91, 68, 7.5,
        'Conforms: False  ·  Constraint Violation (MinCount)\n'
        '  Source Shape: chr:CarePlanCardinalityShape\n'
        '  Focus Node:   ex:care_plan_follow_up\n'
        '  Result Path:  chr:hasMedicalProcedure\n'
        '  Message:  CarePlan should refer to ≥1 MedicalProcedure.',
        color=C_VIO,
        title="SHACL validator     ✗", title_color="#991b1b", border=RED)

arrow(100, 91, 100, 88, color=RED, lw=1.7)
ax.text(101.5, 89.5, "violation report  →  re-prompt  (retry k ≤ 3)",
        fontsize=9, color=RED, ha="left", va="center", style="italic", fontweight="bold")

rbox(80, 82, 40, 6, C_LLM, "LLM   ·   call 2  (retry, k = 1)", fs=10, weight="bold", border=RED)
arrow(100, 82, 100, 80.5)

codebox(76, 72, 68, 8,
        'ex:care_plan_follow_up a chr:CarePlan ;\n'
        '  chr:hasMedicalProcedure\n'
        '      ex:medical_procedure_follow_up .  ← added\n'
        'ex:medical_procedure_follow_up\n'
        '  a chr:MedicalProcedure .              ← added\n'
        '… all triples from cycle_00 retained',
        title="cycle_01.ttl     (k = 1, 20 triples)",
        title_color="#1b5a25", border=GREEN)

# cycle_01 → SHACL ✓
arrow(144, 76, 156, 76, color=GREEN, lw=1.7)

# ===== COLUMN 3 =====
# SHACL ✓ pill — aligned with cycle_01 right side
rbox(156, 72, 58, 8, C_OK, "SHACL  ✓  Conforms   →   emit RDF",
     fs=11, weight="bold", border=GREEN)
arrow(185, 80, 185, 82, color=GREEN, lw=1.6)

# Score panel — tightly sized to its contents
rbox(154, 83, 62, 35, C_SCORE, "", lw=0.8)
ax.text(185, 115, "Score against gold.ttl",
        ha="center", fontsize=11.5, fontweight="bold")
ax.text(185, 112, "(22 gold triples from stage 1)",
        ha="center", fontsize=8.8, color=MUTED, style="italic")

col_x = {"sys": 158, "f1": 180, "shacl": 194, "pc": 208}
ax.text(col_x["sys"], 107.5, "System", fontsize=9.5, fontweight="bold")
ax.text(col_x["f1"], 107.5, "F1", fontsize=9.5, fontweight="bold", ha="center")
ax.text(col_x["shacl"], 107.5, "SHACL", fontsize=9.5, fontweight="bold", ha="center")
ax.text(col_x["pc"], 107.5, "Pat.Cov.", fontsize=9, fontweight="bold", ha="center")
ax.plot([156, 214], [105.5, 105.5], color="#cbd5e1", lw=0.9)

ax.text(col_x["sys"], 102.5, "A  zero-shot", fontsize=9.4)
ax.text(col_x["f1"], 102.5, "0.62", fontsize=9.4, ha="center")
ax.text(col_x["shacl"], 102.5, "✗", fontsize=13, ha="center", color=RED, fontweight="bold")
ax.text(col_x["pc"], 102.5, "0.75", fontsize=9.4, ha="center")

ax.text(col_x["sys"], 98.5, "B  SHACL loop", fontsize=9.4, fontweight="bold")
ax.text(col_x["f1"], 98.5, "0.60", fontsize=9.4, ha="center", fontweight="bold")
ax.text(col_x["shacl"], 98.5, "✓", fontsize=13, ha="center", color=GREEN, fontweight="bold")
ax.text(col_x["pc"], 98.5, "1.00", fontsize=9.4, ha="center", fontweight="bold")

ax.plot([156, 214], [95.5, 95.5], color="#cbd5e1", lw=0.6)
ax.text(158, 93.5, "Detail · vignette_001 only", fontsize=9.3, fontweight="bold",
        color="#1f2a55", family="monospace")

narrative = (
    'STRUCTURE  SHACL ✗→✓  ·  PC 0.75→1.00\n'
    'CONTENT     F1 0.62→0.60 (case-specific; macro avg in §V-B)\n'
    '· CarePlan needed ≥1 MedicalProcedure (cycle 0 had\n'
    '  none); B added one, SHACL passes on cycle 1.\n'
    '· LLM recovered none of the missing source-text facts,\n'
    '  so F1 barely moves — illustrates the decoupling.'
)
ax.text(158, 91, narrative, ha="left", va="top",
        fontsize=7.8, family="monospace", color="#111827", linespacing=1.05)

ax.text(110, 68,
        "llm_calls = 1   ·   trace: evaluation/outputs/chr/schema/full/b/cycles/vignette_001/summary.json",
        ha="center", fontsize=9, color=MUTED, style="italic")

# NOTE: do NOT call ax.set_ylim() again here — that rescales the data range
# and shrinks text in data units, which leaves big empty padding inside every
# codebox. bbox_inches="tight" below crops the saved PNG to actual content.

OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=180, bbox_inches="tight", pad_inches=0.1, facecolor="white")
print(f"wrote {OUT}")
