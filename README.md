# Thesis Codebase

**SHACL-Guided Self-Correcting Pipeline for LLM-Based RDF Knowledge Graph Extraction**

This codebase tests whether embedding a deterministic SHACL validator inside the
extraction loop — feeding violation reports back as prompt context — improves the
structural quality of LLM-extracted RDF graphs.

## Research questions

| RQ | Question |
|----|----------|
| RQ1 | Does SHACL-based feedback bring LLM extraction closer to ground truth? |
| RQ2 | Does the underlying ontology (flat schema vs SULO-aligned) affect extraction quality? |
| RQ3 | Do structural conformance and content overlap with gold improve together, or are they decoupled? |

## Layout

```
pipeline/                       Core code
  extract.py                    A / B / C system runner (with SHACL retry loop)
  evaluate.py                   F1, conformance, hallucination, PC, OC metrics
  prompts.py                    Prompt builders + SchemaContext loader
  iri_normalizer.py             FHIR-grounded IRI matching for triple-level F1
  stats.py / stats_core.py /    Bootstrap CI, paired Wilcoxon, statistical reports
    stats_report.py
  plots.py                      Plot generation

evaluation/
  corpus/
    vignettes/                  100 Synthea-derived clinical-note vignettes
    abox_gold/                  100 schema-track gold ABoxes (FHIR-derived)
    fhir_bundles/               28 source Synthea FHIR Bundles (full reproducibility)
    shapes/                     SHACL constraint files
    tbox/                       Schema + SULO-aligned ontology
  outputs/chr/schema/full/      Gemini 2.0 Flash schema-track results
  outputs/chr/ontology/         Gemini 2.0 Flash ontology-track results
  outputs_freemodel/chr/        gpt-oss-120b:free results (multi-model robustness)
  diagrams/                     Figures embedded in the thesis

scaling_n100/
  fhir_to_chr.py                Deterministic FHIR → CHR Turtle gold converter
  fhir_to_text.py               Deterministic FHIR → clinical-note text renderer
  extract_parallel.py           Parallel extraction wrapper

tests/                          pytest test suite
```

## How to run

Set the OpenRouter API key in `.env`, then:

```bash
# Schema-track A/B/C extraction (Gemini 2.0 Flash)
python pipeline/extract.py --schema chr --track schema --prompt full --systems a,b,c

# Ontology-track extraction (uses chr_shacl_ontology.ttl)
python pipeline/extract.py --schema chr --track ontology --prompt ontology --systems a,b

# Run with a different model (override defaults):
OPENROUTER_MODEL_OVERRIDE="openai/gpt-oss-120b:free" \
OUTPUTS_ROOT_OVERRIDE="$PWD/evaluation/outputs_freemodel" \
python pipeline/extract.py --schema chr --track schema --prompt full --systems a,b

# Evaluate (IRI-normalised F1, conformance, hallucination):
python pipeline/evaluate.py --schema chr --track schema --prompt full
```

`Makefile` wraps the common cases. `make evaluate` re-scores every output cell.

## Corpus

100 vignettes, all Synthea-derived. Same Synthea FHIR Bundle produces both the
text (via `fhir_to_text.py`) and the gold ABox (via `fhir_to_chr.py`), so
text-and-gold come from a single deterministic source. Gold uses standardised
clinical terminologies (SNOMED CT, LOINC, RxNorm) inherited from Synthea.

The 28 source FHIR Bundles are committed at `evaluation/corpus/fhir_bundles/`,
so the entire FHIR → text and FHIR → CHR projection can be re-derived
byte-identically from the same input.

## IRI normaliser

Triple-level F1 is computed in a canonical-IRI space established by
`pipeline/iri_normalizer.py`. The Synthea gold uses UUID-derived IRIs; the
LLM mints text-span IRIs. The normaliser builds a bijection using
FHIR-grounded matching keys (rdfs:label, literal values, IRI token overlap,
structural anchoring via already-matched neighbours). Without it, F1 against
the Synthea gold is artificially zero.

## Reproducibility

**LLM**
- Primary model: `google/gemini-2.0-flash-001` via OpenRouter
- Replication model: `openai/gpt-oss-120b:free` via OpenRouter
- Temperature: 0.1
- Retry budget: k=3
- Termination: conforms / k reached / two-cycle plateau

**Corpus generation (Synthea → FHIR → text + gold)**
- Synthea version: commit `aa0772fb5e92e48a776c51508c00eddc0d9d27ff` (4.0.1-SNAPSHOT, 2026-03-05)
- Seed: `1777393568786` (both `seed` and `clinicianSeed`)
- State: Massachusetts, end-time `20260428`, modules `*`, patient count 6 per run
- 28 generated FHIR Bundles archived at `evaluation/corpus/fhir_bundles/`
- `fhir_to_chr.py` and `fhir_to_text.py` are pure deterministic projections
- All 100 gold ABoxes pass SHACL against their schema-track shapes (100/100)

Three levels of reproducibility:
1. **Derived artefacts** (vignettes + gold) — committed under `evaluation/corpus/`
2. **Source FHIR Bundles** — committed under `evaluation/corpus/fhir_bundles/`
3. **Upstream Synthea regeneration** — pin Synthea to the SHA above, run with the seed

## Headline results

| | Gemini A | Gemini B | gpt-oss A | gpt-oss B |
|---|---|---|---|---|
| SHACL conformance (schema)   | 4%  | **96%** | 20% | **87%** |
| SHACL conformance (ontology) | 79% | **94%** | 29% | **97%** |
| F1 (IRI-normalised, schema)  | 0.336 | 0.330 | 0.317 | 0.319 |

SHACL feedback raises conformance by 15–92 percentage points; F1 movement is
< 1 percentage point in either direction. The two quality axes are decoupled
in magnitude — SHACL is a *structural safety net*, not a content booster.

### Scope of each metric

- **Schema track** has full evaluation: SHACL conformance, triple-level F1
  (IRI-normalised), Property Completeness, hallucination — all in
  `evaluation/outputs/chr/schema/full/eval.json`.
- **Ontology track** reports **SHACL conformance only**. No gold ABoxes
  exist on the ontology track for the Synthea corpus (`fhir_to_chr.py`
  emits schema-track gold only), so triple-level F1 is not computable.
  Conformance numbers are derived by running pyshacl against the System A
  and System B output `.ttl` files in
  `evaluation/outputs/chr/ontology/ontology/{a,b}/` with the
  `chr_shacl_ontology.ttl` shapes; System B's per-cycle conformance is
  recorded in `b/cycles/vignette_NNN/summary.json`.
