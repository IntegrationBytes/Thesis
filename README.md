# Thesis Codebase

**SHACL-Guided Self-Correcting Pipeline for LLM-Based RDF Knowledge Graph Extraction**

This codebase tests whether embedding a deterministic structural verifier
(SHACL on the flat schema track, OWL reasoner on the SULO ontology track)
inside the LLM extraction loop — feeding violation reports back as prompt
context — improves the quality of LLM-extracted RDF graphs over zero-shot
LLM output. Evaluated on n=200 Synthea-derived clinical vignettes with two
LLMs (Gemini 2.0 Flash, gpt-oss-120b) and four quality axes (triple-level
F1, SHACL conformance, OWL conformance, LLM-as-judge).

## Research questions

| RQ | Question |
|----|----------|
| RQ1 | Does the verifier-in-loop pipeline improve extraction quality over zero-shot LLM output, and at what additional inference-time compute cost? |
| RQ2 | Does the same effect hold when the validator is changed from custom SHACL shapes (schema track) to an OWL reasoner over the SULO upper ontology (ontology track)? |
| RQ3 | Do structural conformance and content overlap with the gold move together, or are they independent quality axes? |
| RQ4 | Does the choice of validator (custom SHACL vs OWL reasoner) affect retry-loop convergence direction? |

## Layout

```
pipeline/                       Core code
  extract.py                    System A / B runner with track-aware validator dispatch
  evaluate.py                   F1, conformance, hallucination, PC, OC metrics
  prompts.py                    Prompt builders + SchemaContext loader
  iri_normalizer.py             FHIR-grounded IRI matching for triple-level F1
  owl_validator.py              OWL reasoner (primary validator on ontology track)
  llm_judge.py                  Cross-family LLM-as-judge (Llama 3.3 70B by default)
  stats.py / stats_core.py /    Bootstrap CIs, paired sign tests, statistical reports
    stats_report.py
  plots.py                      Per-cycle / conformance / violation-resolution plots

evaluation/
  corpus/
    vignettes/                  200 Synthea-derived clinical-note vignettes
                                  (001-100 general; 101-200 multi-morbidity complex)
    abox_gold/                  200 schema-track gold ABoxes (FHIR-derived)
    fhir_bundles/               source Synthea FHIR Bundles (full reproducibility)
    shapes/                     SHACL constraint files
    tbox/                       Schema TBox + SULO-aligned ontology + cached SULO
  outputs/chr/schema/full/      Gemini 2.0 Flash schema-track results (A, B, D)
  outputs/chr/ontology/         Gemini 2.0 Flash ontology-track results
                                  (A, B with OWL retry; b_shacl/ archived for ablation)
  outputs_freemodel/chr/        gpt-oss-120b results (cross-model robustness)
  outputs/judge_scores.json     LLM-as-judge results across all System B variants
  outputs/connectivity_buckets.json
                                Connectivity-stratified bucket assignments + metrics
  outputs/per_shape_violations.json
                                Per-SHACL-shape and per-OWL-violation breakdown
  outputs/f1_normalizer_ablation.json
                                F1 with vs without IRI normaliser (raw vs normalised)
  outputs/ontology_coverage.json
                                CHR TBox class/property usage stats

scaling_n100/
  fhir_to_chr.py                Deterministic FHIR -> CHR Turtle gold converter
  fhir_to_text.py               Deterministic FHIR -> clinical-note text renderer
  extract_parallel.py           Parallel extraction wrapper (System A + B)

scripts/
  run_system_d.py               System C (LLM looped 4x without verifier feedback;
                                filename retains historical "d" suffix)
  evaluate_system_d.py          A vs B vs C comparison on SHACL + F1
  owl_conformance_survey.py     Parallel OWL conformance across systems and tracks
  connectivity_analysis.py      Connectivity-tertile stratification of all metrics
  per_shape_violation_analysis.py
                                Which SHACL shapes get fixed by B / which resist
  f1_normalizer_ablation.py     F1 with vs without IRI normaliser
  ontology_coverage.py          CHR TBox usage stats
  inspect_vignette.py           Per-vignette diagnostic tool (use during defence Q&A)
  make_n200_stratified_plot.py  Headline A->B conformance plot (4 system variants)
  make_n200_percycle_v2.py      Per-cycle convergence, both metrics on [0,1] scale
  make_n200_decoupling_plot.py  ΔSHACL vs ΔF1 dual-axis chart
  make_connectivity_plot.py     Connectivity-stratified lift figure
  regenerate_all_figures.py     One-shot regen of every thesis figure

report/                         Local thesis writing artefacts (gitignored)
                                  THESIS_WRITING_PACK.md — master copy-paste reference
                                  n200_results_summary.md — detailed tables
                                  cost_and_reproducibility.md — API spend + Synthea seeds
                                  worked_example_vignette_030.md — pipeline trace
                                  per_shape_findings.md — per-shape breakdown
                                  judge_x_connectivity.md — cross-dim finding
                                  f1_normalizer_audit.md — normaliser defence
                                  code_review_notes.md — module-by-module audit
                                  FULL_CODE_RECHECK.md — final audit summary
                                  figures/                — thesis PNGs

tests/                          pytest suite (57 tests, all pass)
```

## How to run

Set the OpenRouter API key in `.env`, then:

```bash
# Schema-track A/B extraction (Gemini 2.0 Flash)
python scaling_n100/extract_parallel.py \
  --schema chr --track schema --prompt full --systems a,b --workers 5

# Ontology-track extraction (uses OWL reasoner — replaces SHACL on this track)
python scaling_n100/extract_parallel.py \
  --schema chr --track ontology --prompt ontology --systems a,b --workers 5

# Run with the alternate model (override defaults, paid endpoint for speed):
OPENROUTER_MODEL_OVERRIDE="openai/gpt-oss-120b" \
OPENROUTER_PROVIDER_SORT="throughput" \
OUTPUTS_ROOT_OVERRIDE="$PWD/evaluation/outputs_freemodel" \
python scaling_n100/extract_parallel.py --schema chr --track schema --prompt full --systems a,b

# System C — compute-fair baseline (4 LLM calls without verifier feedback).
# Scripts retain the historical "system_d" filename; the paper uses "System C".
python scripts/run_system_d.py --schema chr --track schema --workers 5

# F1 + SHACL evaluation on the schema track
python pipeline/evaluate.py --schema chr --track schema --prompt full

# LLM-as-judge (Llama 3.3 70B — cross-family from Gemini and gpt-oss)
JUDGE_MODEL="meta-llama/llama-3.3-70b-instruct" \
python pipeline/llm_judge.py --schema chr --track schema --prompt full --system b

# Statistical CIs and paired sign tests
python pipeline/stats.py

# All supplementary analyses
python scripts/connectivity_analysis.py
python scripts/per_shape_violation_analysis.py
python scripts/f1_normalizer_ablation.py
python scripts/ontology_coverage.py
python scripts/evaluate_system_d.py

# Regenerate every thesis figure from existing outputs
python scripts/regenerate_all_figures.py
```

`pyproject.toml` declares package metadata. Tests run with `python -m pytest tests/`.

## Corpus

n=200 vignettes, all Synthea-derived. Same Synthea FHIR Bundle produces both
the text (via `fhir_to_text.py`) and the gold ABox (via `fhir_to_chr.py`), so
text-and-gold come from a single deterministic source. Stratified into:

- **General (001-100):** standard Synthea modules, mixed encounter types.
- **Complex (101-200):** multi-morbidity, age 60+, disease-specific modules
  (metabolic_syndrome, lung_cancer, breast_cancer_survivor, opioid_addiction,
  sepsis). Mean ~108 entities per gold KG vs ~36 in the general stratum.

Gold uses standardised clinical terminologies (SNOMED CT, LOINC, RxNorm)
inherited from Synthea.

The source FHIR Bundles are committed at `evaluation/corpus/fhir_bundles/`,
so the entire FHIR -> text and FHIR -> CHR projection can be re-derived
byte-identically from the same input.

## IRI normaliser

Triple-level F1 is computed in a canonical-IRI space established by
`pipeline/iri_normalizer.py`. The Synthea gold uses UUID-derived IRIs; the
LLM mints text-span IRIs. The normaliser builds a bijection using
FHIR-grounded matching keys (rdfs:label, literal values, IRI token overlap,
structural anchoring via already-matched neighbours). **Ablation result
(n=200):** without the normaliser, F1 against the Synthea gold is 0.000-0.004
across all 8 model × stratum × system combinations. With it, F1 is 0.40-0.50.
The normaliser is the only path to a measurable triple-level F1 signal.

Full audit in `report/f1_normalizer_audit.md`.

## Validator architecture

`pipeline/extract.py` uses a track-aware `validate_for_track` dispatch:

- **Schema track** -> `validate_with_shacl` (custom `chr_shacl_schema.ttl`)
- **Ontology track** -> `validate_with_owl` (OWL-RL closure + SULO disjointness
  + functional-property + range checks)

This replaces the previous hand-written `chr_shacl_ontology.ttl` shapes as
the primary validator on the ontology track (supervisor's request: validation
should be grounded in SULO axioms, not in our custom shape interpretation).
The archived SHACL-driven System B outputs remain at `outputs*/chr/ontology/ontology/b_shacl/`
for the validator-choice ablation study (RQ4).

## Reproducibility

**LLMs**
- Primary model: `google/gemini-2.0-flash-001` via OpenRouter
- Cross-model replication: `openai/gpt-oss-120b` via OpenRouter (paid endpoint
  with throughput sort for fast extraction; free endpoint hits rate limits)
- Judge: `meta-llama/llama-3.3-70b-instruct` via OpenRouter (cross-family)
- Temperature: 0.1
- Retry budget for System B: k=3
- Termination: conforms / k reached / two-cycle plateau / parse-error recovery

**Corpus generation (Synthea -> FHIR -> text + gold)**
- Synthea version: commit `aa0772fb5e92e48a776c51508c00eddc0d9d27ff` (4.0.1-SNAPSHOT, 2026-03-05)
- Seed: `1777393568786` (both `seed` and `clinicianSeed`)
- State: Massachusetts, end-time `20260428`, modules `*` (general) or
  disease-specific (complex), patient count 6 per run
- 28 generated FHIR Bundles archived at `evaluation/corpus/fhir_bundles/`
- `fhir_to_chr.py` and `fhir_to_text.py` are pure deterministic projections
- All 200 gold ABoxes pass SHACL against their schema-track shapes

Three levels of reproducibility:
1. **Derived artefacts** (vignettes + gold) — committed under `evaluation/corpus/`
2. **Source FHIR Bundles** — committed under `evaluation/corpus/fhir_bundles/`
3. **Upstream Synthea regeneration** — pin Synthea to the SHA above, run with the seed

Full recipe and cost breakdown in `report/cost_and_reproducibility.md`.

## Headline results (n=200)

### Schema track (F1 + SHACL conformance)

| | Gemini A | Gemini B | gpt-oss A | gpt-oss B |
|---|---|---|---|---|
| SHACL conformance | 4% / 3% | **96% / 100%** | 19% / 42% | **87% / 97%** |
| F1 (IRI-normalised) | 0.34 / 0.29 | 0.33 / 0.26 | 0.32 / 0.26 | 0.32 / 0.25 |

(general / complex stratum)

### Ontology track (OWL conformance, primary validator)

| | Gemini A | Gemini B (OWL retry) | gpt-oss A | gpt-oss B (OWL retry) |
|---|---|---|---|---|
| OWL conformance | 65% / 19% | **71% / 59%** | 83% / 94% | **96% / 99%** |

### Decoupling

ΔSHACL spans +55pp to +97pp on schema track. |ΔF1| ≤ 0.024 across all variants.
Decoupling ratio (ΔSHACL / |ΔF1|): 4500× to 11000×. Structural conformance and
content overlap with ground truth are independent quality axes.

### Compute-fairness baseline (System C, formerly System D in code)

System C = LLM called 4 times sequentially with no feedback (same token
budget as B's 1 + 3 retries). On Gemini complex schema: A=3% SHACL, C=0%,
B=100%. The verifier's feedback signal is responsible for nearly all of B's
improvement, not raw inference-time scaling. Scripts under `scripts/` still
use the `system_d` filename for historical reasons.

## Scope of each metric

- **Schema track** has full evaluation: SHACL conformance, triple-level F1
  (IRI-normalised), Property Completeness, hallucination — all in
  `evaluation/outputs/chr/schema/full/eval.json`.
- **Ontology track** reports **OWL conformance** (primary, n=200 stratified)
  AND **SHACL conformance** (for comparability with earlier studies; B/ vs
  b_shacl/ ablation). No gold ABoxes exist on the ontology track for the
  Synthea corpus (`fhir_to_chr.py` emits schema-track gold only), so
  triple-level F1 is not computable on this track.
- **LLM-as-judge** is reported across all 4 System B variants, n=200, with
  three Likert axes (faithfulness, completeness, hallucination). See
  `evaluation/outputs/judge_scores.json`.

## Thesis writing reference

`report/THESIS_WRITING_PACK.md` is the single source of truth for every
number in the thesis manuscript, with section-aligned tables, ready-to-paste
bullet points for the Discussion section, and pre-written defence Q&A
answers for the 7 most likely supervisor questions.
