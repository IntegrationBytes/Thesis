# Handoff — Thesis Codebase Session (2026-06-02)

## Repository

BSc thesis by Vincent Viitala (supervisors: Remzi Çelebi, Michel Dumontier) —
LLM-based clinical knowledge graph extraction with SHACL/OWL validator feedback.
Pipeline extracts RDF ABoxes from synthetic clinical notes (Synthea FHIR) and
evaluates with SHACL conformance, triple-level F1, property completeness, and
hallucination metrics across three systems (A: baseline, B: validator feedback loop,
C: compute-fair control).

---

## Changes Made This Session

### Round 1 — Schema gaps from vignette_001 analysis

| # | Issue | Fix |
|---|---|---|
| 1 | `hasSeverity` declared in TBox/SHACL but never populated in gold ABoxes | `fhir_to_chr.py`: emit `chr:hasSeverity` → `chr:Severity` from `condition.severity`; no triple when absent |
| 2 | `ClinicalVisit` had no date property | Added `chr:hasDate` (domain: `ClinicalVisit`, range: `xsd:dateTime`) to TBox, SHACL, `prompts.py`, `fhir_to_chr.py` |
| 3 | `ClinicalVisit` not linked to its procedures | Added `chr:hasProcedure` (domain: `ClinicalVisit`, range: `MedicalProcedure`) to TBox, SHACL, `prompts.py`, `fhir_to_chr.py` — emitted for every `MeasurementProcess`, `EvaluationProcess`, `MedicationAdministration` |
| 4 | `fhir_to_text.py` hardcoded severity "Moderate" for every condition | Fixed to read `condition.severity`; sentence omitted entirely when FHIR field is absent |

### Round 2 — Schema gaps from vignette_002 / vignette_143 / vignette_030 analysis

| # | Issue | Fix |
|---|---|---|
| 5 | `chr:Measurement` carried no label — all measurements anonymous in the KG | `fhir_to_chr.py`: emit `rdfs:label` on each `chr:Measurement` from FHIR observation code text |
| 6 | `ClinicalCondition` nodes were disconnected — no traversal path from visit | Added `chr:hasCondition` (domain: `EvaluationProcess`, range: `ClinicalCondition`) to TBox, SHACL, `prompts.py`, `fhir_to_chr.py`; full path now `visit → hasProcedure → evalProc → hasCondition → cond` |
| 7 | Unit IRI bug: `_slug()` dropped `%`, `{…}`, `/` → broken or ambiguous IRIs | Replaced with `_ucum_uri(unit)`: primary lookup in 874-entry SPHN UCUM ontology dict (`scaling_n100/ucum_units.py`); fallback to BioMedIT character-map encoding. Unit namespace now `https://biomedit.ch/rdf/sphn-resource/ucum/`. FHIR non-standard codes normalised: `[iU]/L` → `[IU]/L`, `K/uL` → `10*3/uL`. |
| 8 | No formal terminology codes on clinical entities | Added `chr:hasCode` property to TBox, SHACL, `prompts.py`, `fhir_to_chr.py`. Emits SNOMED IRI (`http://snomed.info/id/{code}`) on `ClinicalCondition` + `DiagnosticStatement`, LOINC IRI (`https://loinc.org/{code}`) on `Measurement`, and RxNorm when present on `PharmaceuticalProduct`. Source codings from `Condition.code.coding` and `Observation.code.coding`. |

### Round 3 — Evaluation pipeline improvements (vignette_001 single-example analysis)

| # | Issue | Fix |
|---|---|---|
| 9 | Vignette text truncated timestamps to `HH:MM` → LLM couldn't emit correct dateTime | `fhir_to_text.py`: `_format_dt()` now returns full ISO-8601 string preserving seconds and timezone |
| 10 | Vignette text had no SNOMED/LOINC codes → LLM hallucinated wrong codes | `fhir_to_text.py`: inline codes now appended after condition/observation names: `"Acute viral pharyngitis (disorder) (SNOMED: 195662009)"`, `"Body temperature (LOINC: 8310-5)"` |
| 11 | LLM not emitting `rdfs:label` on all individuals | Added hard rule 7 to `prompts.py`: every named individual must carry `rdfs:label` |
| 12 | LLM using wrong IRI patterns for SNOMED/LOINC codes | Added hard rule 8 to `prompts.py` with exact patterns: `<http://snomed.info/id/{code}>` and `<https://loinc.org/{code}>` |
| 13 | Datetime literal mismatch — LLM `T06:24:00` vs gold `T06:24:27+02:00` | `evaluate.py`: `_normalize_datetime()` truncates both sides to `YYYY-MM-DDTHH:MM` before triple comparison |
| 14 | Unit IRI mismatch — LLM `ex:unit_Cel` vs gold `ucum:Cel` | `evaluate.py`: `_normalize_unit_iri()` maps all unit IRI variants to `__unit__{localname}` key before comparison |
| 15 | Terminology IRI variant mismatch — `http://loinc.org/id/X` vs `https://loinc.org/X` | `evaluate.py`: `_normalize_terminology_iri()` normalises LOINC and SNOMED IRI variants to canonical forms |
| 16 | IRI normalizer: no code-based entity matching | `iri_normalizer.py`: `chr:hasCode` IRI added to `_entity_keys`; matching gives +2.0 score for shared code (beats all other signals). Code variants normalised via `_norm_code()` |
| 17 | IRI normalizer: patient label mismatch (`"Ms. Angelic427"` vs `"Angelic427"`) | `iri_normalizer.py`: added label-token Jaccard to `_score` — matches on shared name tokens regardless of title prefixes |
| 18 | IRI normalizer: unit not used to disambiguate measurements | `iri_normalizer.py`: added `"hasUnit"` to `_VALUE_KEYS[Measurement]` — unit IRI now a matching signal alongside value and date |

### F1 progression on vignette_001 (Gemini 3.5 Flash, schema track)

| Stage | F1 | TP | FP | FN | Gold | Notes |
|---|---|---|---|---|---|---|
| Original committed outputs | 0.410 | 8 | 4 | 19 | 27 | SHACL FAIL |
| + Schema properties in prompt | 0.553 | 13 | 7 | 14 | 27 | SHACL OK |
| + rdfs:label rule | 0.600 | 18 | 15 | 9 | 27 | |
| + DT / unit normalisation in evaluator | 0.800 | 24 | 9 | 3 | 27 | |
| + hasCode in gold (gold grew 27→32) | 0.676 | 24 | 17 | 6 | 32 | gold expanded |
| + Codes in vignette text + prompt rules | 0.738 | 24 | 11 | 6 | 32 | |
| + chr:Unit class + UCUM code on units | **0.829** | **29** | **9** | **3** | **32** | best |

### Round 4 — IRI normalizer improvements

| # | Change | Effect |
|---|---|---|
| 19 | Label extraction suppressed for all clinical types (`Measurement`, `EvaluationProcess`, `MeasurementProcess`, `ClinicalVisit`, etc.) — only `Person`, `CareUnit`, `ProcessStatus`, `Severity`, `Device`, `AnatomicalStructure`, `PharmaceuticalProduct` use labels | Prevents label cross-matches on clinical entities |
| 20 | `chr:Unit` excluded from label matching — identity via `chr:hasCode` (UCUM) only | |
| 21 | `Measurement` matching uses `hasMeasuredDate` + `hasCode` (LOINC +2.0); removed `hasQuantityValue` and `hasUnit` from value keys | Cleaner type-based identity |
| 22 | `ClinicalCondition` and `DiagnosticStatement` code-only matching removed from `_score` (now handled at extraction via `_NAMED_ENTITY_TYPES`) | Consolidates logic |
| 23 | URL-decoded unit IRIs now normalised correctly: `%25`→`percent`, `%2F`→`/`→stripped, `%7B`/`%7D`→stripped. `_unit_bare_key` URL-decodes before stripping, handles `%` → `percent` specially | Fixes LLM percent-encoding of UCUM special chars |

### Round 5 — `chr:Unit` class added to schema

| # | Change | Files |
|---|---|---|
| 24 | Added `chr:Unit` class; `chr:hasUnit` range changed from `sulo:Unit` to `chr:Unit` | TBox, SHACL |
| 25 | `UnitCodeShape`: every `chr:Unit` must have `chr:hasCode` with UCUM IRI (`sh:pattern`) | SHACL |
| 26 | `chr:Unit` added to class list; `hasUnit` range updated; hard rule 9 added | `prompts.py` |
| 27 | Unit instances now `ex:unit_{slug} a chr:Unit ; rdfs:label "..." ; chr:hasCode <ucum:...>` | `fhir_to_chr.py` |

### Files changed (cumulative)

| File | Changes |
|---|---|
| `evaluation/corpus/tbox/chr_schema.ttl` | Added `chr:hasDate`, `chr:hasProcedure`, `chr:hasCondition`, `chr:hasCode`, `chr:Unit`; `hasUnit` range → `chr:Unit` |
| `evaluation/corpus/shapes/chr_shacl_schema.ttl` | Shapes for `hasDate`, `hasProcedure`, `hasCondition`, `hasCode`, `UnitCodeShape` |
| `pipeline/prompts.py` | `hasDate`, `hasProcedure`, `hasCondition`, `hasCode` in properties; `chr:Unit` in class list; hard rules 7–9 |
| `pipeline/evaluate.py` | `_normalize_datetime`, `_normalize_unit_iri` (URL-decode + bare-key), `_normalize_terminology_iri` |
| `pipeline/iri_normalizer.py` | Code matching (+2.0), `_NAMED_ENTITY_TYPES` for label suppression, `hasUnit` bare-key with URL-decode, `Measurement` value keys = `hasMeasuredDate` only |
| `scaling_n100/fhir_to_chr.py` | Severity, `hasDate`, `hasProcedure`, `hasCondition`, measurement labels, UCUM units, `chr:hasCode` on conditions/measurements/drugs, `chr:Unit` instances |
| `scaling_n100/fhir_to_text.py` | Full ISO-8601 timestamps; inline SNOMED/LOINC codes; severity FHIR-driven |
| `scaling_n100/ucum_units.py` | New — 874 UCUM label→IRI entries from `sphn_ucum_2023-1.ttl` (SPHN/SIB) |

### Corpus regenerated (8×)

All 200 gold ABoxes and 200 vignette texts regenerated from source FHIR bundles.
Mapping reconstructed from gold ABox IRIs: patient UUID → bundle filename;
encounter UUID → FHIR encounter id.

---

## Remaining Known Issues

1. No link between `DiagnosticStatement` and `ClinicalCondition` (parallel nodes off `EvaluationProcess`)
2. `rdfs:label` missing on `MeasurementProcess` (present on `Measurement` only)
3. `PharmaceuticalProduct` dose not structured — free-text label only; `MedicationAdministration.dosage` ignored
4. No schema property for causal measurement→condition links (e.g. DAST-10 → "Misuses drugs")
5. `Encounter.type` not representable — no `chr:` property
6. `{nominal}` and `pH` units not in SPHN UCUM ontology — fall back to `sulo:Unit/` namespace
7. LLM consistently misses `MeasurementProcess → chr:hasPerformer` — inferred from encounter participant, not in `Observation.performer`; may need explicit prompt example
8. LLM emits percent-encoded UCUM IRIs (e.g. `ucum:kg%2Fm2`) — normalizer handles this at evaluation time, but prompt rule 8 could add encoded examples to guide the LLM toward the SPHN form
9. `ClinicalVisit → hasProcedure` links (35 in vignette_030) consistently missed — LLM links only hallucinated encounter procedure instead of real measProc/evalProc/medAdmin nodes

---

## How to Re-Regenerate the Corpus

```python
import json, re, sys
from pathlib import Path
from rdflib import Graph
sys.path.insert(0, '.')
from scaling_n100.fhir_to_chr import convert_encounter, _build_resource_index

GOLD_DIR   = Path('evaluation/corpus/abox_gold')
BUNDLE_DIR = Path('evaluation/corpus/fhir_bundles')
UUID_RE = re.compile(r'[0-9a-f]{8}(?:[_-][0-9a-f]{4}){3}[_-][0-9a-f]{8,}', re.IGNORECASE)

def iri_to_uuid(local):
    m = UUID_RE.search(local); return m.group(0).replace('_', '-') if m else ''

bundle_by_patient = {
    UUID_RE.search(p.stem).group(0).lower(): p
    for p in BUNDLE_DIR.glob('*.json') if UUID_RE.search(p.stem)
}
for gold_path in sorted(GOLD_DIR.glob('vignette_*_gold_schema.ttl')):
    vid = gold_path.name.replace('_gold_schema.ttl', '')
    g = Graph(); g.parse(gold_path.as_posix(), format='turtle')
    pat_uuid = visit_uuid = ''
    for s in g.subjects():
        local = str(s).split('/')[-1]
        if local.startswith('patient_'): pat_uuid = iri_to_uuid(local)
        elif local.startswith('visit_'): visit_uuid = iri_to_uuid(local)
    bundle = json.loads(bundle_by_patient[pat_uuid.lower()].read_text())
    res_idx = _build_resource_index(bundle)
    enc = next(e['resource'] for e in bundle['entry']
               if e['resource'].get('resourceType') == 'Encounter'
               and e['resource'].get('id', '').lower() == visit_uuid.lower())
    convert_encounter(bundle, enc, res_idx, vid).serialize(gold_path.as_posix(), format='turtle')
```

Same pattern for vignette texts — replace `convert_encounter` with `render` from
`scaling_n100/fhir_to_text.py` and write to `evaluation/corpus/vignettes/{vid}.txt`.

## How to Re-Run Single-Vignette Extraction

```bash
OPENROUTER_MODEL_OVERRIDE="google/gemini-3.5-flash" \
python3 pipeline/extract.py --schema chr --track schema --systems a,b --only vignette_001
```

Requires `OPENROUTER_API_KEY` in `.env`. Default model `google/gemini-2.0-flash-001`
is no longer available on OpenRouter as of 2026-06-02.
