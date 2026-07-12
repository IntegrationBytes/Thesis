# Thesis Codebase — Reproduction Guide

**SHACL and OWL Validator Feedback for LLM-Based Knowledge Graph Extraction**
Vincent Viitala — BSc Data Science and Artificial Intelligence, Maastricht University, 2026
Supervisors: Remzi Çelebi, Michel Dumontier

This README is the recipe to reproduce every result this codebase
produces: evaluation metrics, statistics, and figures.

---

## 1. Prerequisites

- macOS or Linux with Python **3.13**
- An [OpenRouter](https://openrouter.ai/) API key. The pipeline calls two
  models: the extractor **gpt-oss-120b** (open-weight, Apache 2.0) and the
  cross-family LLM-as-judge **Llama 3.3 70B**.
- ~10 USD of OpenRouter credit for a full end-to-end run (see budget table
  below). The repo already contains all model outputs, so reproduction
  without re-running the LLM is free.

> **Model note.** The thesis uses a single extractor, **gpt-oss-120b**, whose
> outputs are committed under `evaluation/outputs/`. The codebase still
> contains historical scaffolding from an earlier dual-model exploration
> (a `evaluation/outputs_freemodel/` directory and a Gemini default model
> env var); **these are not used by the final single-model pipeline** and can
> be ignored. Every result is computed from `evaluation/outputs/`.

## 2. Install

```bash
git clone <repo-url> thesiscode
cd thesiscode
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[analysis,dev]"     # core + matplotlib + pytest
cp .env.example .env                  # then edit .env to add OPENROUTER_API_KEY
```

Verify the install:

```bash
python -m pytest tests/
```

## 3. Reproduce results without re-running the LLM (fast path)

The repo ships with every gpt-oss-120b output committed under
`evaluation/outputs/`. To regenerate the evaluation tables and figures
from those outputs only:

```bash
make evaluate                                       # F1, SHACL, OWL, PC -> eval.json
make stats                                          # paired sign tests, bootstrap CIs
python scripts/make_percycle_v3.py                  # per-cycle dynamics
python scripts/make_n200_decoupling_plot.py         # structure-vs-content decoupling
python scripts/make_pipeline_e2e_figure.py          # single-case walkthrough
python scripts/make_violation_resolution_figure.py  # per-violation resolution, both tracks
python scripts/make_owl_process_figure.py           # ontology-track verifier loop
```

PNGs land in `report/figures/`.

## 4. Reproduce results from scratch (full re-run)

If you want to re-extract every graph with the LLM, run the steps below in
order. Total wall-clock is ~45 minutes on the paid gpt-oss-120b endpoint with
a 20-worker pool.

### 4.1 Re-derive the corpus (optional)

The 200 vignettes and gold ABoxes are committed under
`evaluation/corpus/`. To regenerate them from the source Synthea FHIR
bundles (also committed, at `evaluation/corpus/fhir_bundles/`):

```bash
python scaling_n100/fhir_to_text.py   # FHIR -> clinical-note text
python scaling_n100/fhir_to_chr.py    # FHIR -> schema-track gold ABox
```

To regenerate the FHIR bundles themselves you need Synthea pinned to
commit `aa0772fb5e92e48a776c51508c00eddc0d9d27ff`, seed `1777393568786`
(used for both `seed` and `clinicianSeed`), state Massachusetts, end-time
`20260428`, patient count 6 per run. General-stratum runs use module
glob `*`; complex-stratum runs use the disease modules
`metabolic_syndrome`, `lung_cancer`, `breast_cancer_survivor`,
`opioid_addiction`, `sepsis`.

### 4.2 Run extraction — gpt-oss-120b (System A + System B)

The codebase's default model env var is a now-deprecated Gemini id for
historical reasons, so the extractor must be passed explicitly. Outputs
go to the default root `evaluation/outputs/` (do not override it; that is
where the committed results come from).

```bash
OPENROUTER_MODEL_OVERRIDE="openai/gpt-oss-120b" \
OPENROUTER_PROVIDER_SORT="throughput" \
python scaling_n100/extract_parallel.py \
    --schema chr --track schema --prompt full --systems a,b --workers 20

OPENROUTER_MODEL_OVERRIDE="openai/gpt-oss-120b" \
OPENROUTER_PROVIDER_SORT="throughput" \
python scaling_n100/extract_parallel.py \
    --schema chr --track ontology --prompt ontology --systems a,b --workers 20
```

### 4.3 Run compute-fair control (System C)

```bash
OPENROUTER_MODEL_OVERRIDE="openai/gpt-oss-120b" \
OPENROUTER_PROVIDER_SORT="throughput" \
python scripts/run_system_d.py --schema chr --track schema --workers 20
# The script file is named system_d for historical reasons; this is
# the compute-fair control (System C).
```

### 4.4 Sanitise outputs (Turtle syntax normalisation)

gpt-oss-120b emits Turtle-invalid IRI local names (a '+' or extra ':' from
ISO-8601 timezones) on ~14% of cases. Run the deterministic sanitiser pass
before scoring. It is idempotent on already-clean files.

```bash
python scripts/sanitize_outputs.py
```

### 4.5 Score everything

```bash
make evaluate                         # F1, SHACL conformance, OWL, PC
make stats                            # paired sign tests, bootstrap CIs
```

### 4.6 Run the LLM-as-judge

```bash
JUDGE_MODEL="meta-llama/llama-3.3-70b-instruct" \
python pipeline/llm_judge.py --schema chr --track schema --prompt full --system a
JUDGE_MODEL="meta-llama/llama-3.3-70b-instruct" \
python pipeline/llm_judge.py --schema chr --track schema --prompt full --system b
```

### 4.7 Supplementary analyses (per-shape, normaliser ablation, OWL conformance)

```bash
python scripts/per_shape_violation_analysis.py
python scripts/owl_conformance_survey.py
python scripts/f1_normalizer_ablation.py
```

Then re-run the figure scripts from section 3 to refresh the PNGs.

### 4.8 Supervisor-revision analyses (fast path, no LLM calls)

Back appendices A, B, C. Outputs land in `evaluation/outputs/`.

```bash
python scripts/er_ablation.py            # entity-resolution signal ablation
python scripts/owl_on_schema_abox.py     # OWL-RL on the schema-track ABox
python scripts/owl_fault_injection.py    # OWL fault-injection (detection / false-positive)
python scripts/shacl_structural_prf1.py  # graded entity-level SHACL P/R/F1
```

## 5. Configuration knobs

All set in `.env` or as environment variables:

| Variable | Default | Notes |
|---|---|---|
| `OPENROUTER_API_KEY` | _(required)_ | OpenRouter credential |
| `OPENROUTER_MODEL_OVERRIDE` | `google/gemini-2.0-flash-001` | Deprecated historical default; **set to `openai/gpt-oss-120b`** to reproduce the committed results |
| `OPENROUTER_PROVIDER_SORT` | _(unset)_ | Set to `throughput` for gpt-oss-120b on the paid endpoint |
| `OUTPUTS_ROOT_OVERRIDE` | `evaluation/outputs` | The gpt-oss-120b outputs live here. **Leave at default**; the legacy `evaluation/outputs_freemodel` directory is unused historical scaffolding |
| `JUDGE_MODEL` | `meta-llama/llama-3.3-70b-instruct` | Cross-family judge |

Hard-coded in `pipeline/extract.py`: temperature `0.1`, retry budget `k=3`,
plateau heuristic (two identical violation reports terminates early).

## 6. Smoke tests

```bash
python -m pytest tests/                       # unit tests
python scripts/verify_experiments_e2e.py      # end-to-end claim verification
```

`verify_experiments_e2e.py` reads `evaluation/outputs/*.json` and asserts
every quantitative result still holds against the committed data. If you
re-run extraction and any number drifts, this is the first thing to run.

## 7. Cost budget (if running from scratch)

| Step | Calls | Model | ~USD |
|---|---|---|---|
| 4.2 gpt-oss schema + ontology, A+B | ~1,200 | gpt-oss-120b (paid) | 5 |
| 4.3 System C (control) | ~800 | gpt-oss-120b (paid) | 3 |
| 4.6 Judge (4 cells × 200 cases) | ~800 | llama-3.3-70b | 2 |
| **Total** | ~2,800 | | **~10** |

---

Questions about the code or repository: viitala.vincent@gmail.com.
