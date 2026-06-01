# Thesis Codebase — Reproduction Guide

**SHACL and OWL Validator Feedback for LLM-Based Knowledge Graph Extraction**
Vincent Viitala — BSc Data Science and Artificial Intelligence, Maastricht University, 2026
Supervisors: Remzi Çelebi, Michel Dumontier

This README contains the recipe to reproduce every number, table, and figure
in the report.

---

## 1. Prerequisites

- macOS or Linux with Python **3.13**
- An [OpenRouter](https://openrouter.ai/) API key (the pipeline calls three
  models through OpenRouter: Gemini 2.0 Flash, gpt-oss-120b, Llama 3.3 70B)
- ~20 USD of OpenRouter credit for a full end-to-end run (see budget table
  below). The repo already contains all model outputs, so reproduction
  without re-running the LLM is free.

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
python -m pytest tests/               # 57 tests, all pass
```

## 3. Reproduce results without re-running the LLM (fast path)

The repo ships with every model output committed under `evaluation/outputs/`
and `evaluation/outputs_freemodel/`. To regenerate the paper's tables and
figures from those outputs only:

```bash
make all                              # evaluate → stats → plots (~3 min)
python scripts/make_n200_decoupling_plot.py
python scripts/make_n200_percycle_v2.py
python scripts/make_pipeline_e2e_figure.py
python scripts/make_violation_resolution_figure.py
```

The four figure scripts above produce the four PNGs the paper actually
imports (see §5 for the mapping).

## 4. Reproduce results from scratch (full re-run)

If you want to re-extract every graph with the LLM, run the steps below in
order. Total wall-clock is ~6 hours on a 5-worker pool against OpenRouter
free tiers; ~45 minutes on the paid gpt-oss endpoint.

### 4.1 Re-derive the corpus (optional)

The 200 vignettes and gold ABoxes are committed under
`evaluation/corpus/`. To regenerate them from the source Synthea FHIR
bundles (also committed, at `evaluation/corpus/fhir_bundles/`):

```bash
python scaling_n100/fhir_to_text.py   # FHIR → clinical-note text
python scaling_n100/fhir_to_chr.py    # FHIR → schema-track gold ABox
```

To regenerate the FHIR bundles themselves you need Synthea pinned to
commit `aa0772fb5e92e48a776c51508c00eddc0d9d27ff`, seed `1777393568786`
(used for both `seed` and `clinicianSeed`), state Massachusetts, end-time
`20260428`, patient count 6 per run. General-stratum runs use module
glob `*`; complex-stratum runs use the disease modules
`metabolic_syndrome`, `lung_cancer`, `breast_cancer_survivor`,
`opioid_addiction`, `sepsis`.

### 4.2 Run extraction — gpt-oss-120b (primary, open-weight)

The codebase's default model env var is set to Gemini for historical
reasons, so the primary model has to be passed explicitly. The
`outputs_freemodel/` directory name is also historical — these are the
primary-model outputs the paper leads with.

```bash
OPENROUTER_MODEL_OVERRIDE="openai/gpt-oss-120b" \
OPENROUTER_PROVIDER_SORT="throughput" \
OUTPUTS_ROOT_OVERRIDE="$PWD/evaluation/outputs_freemodel" \
python scaling_n100/extract_parallel.py \
    --schema chr --track schema --prompt full --systems a,b --workers 5

OPENROUTER_MODEL_OVERRIDE="openai/gpt-oss-120b" \
OPENROUTER_PROVIDER_SORT="throughput" \
OUTPUTS_ROOT_OVERRIDE="$PWD/evaluation/outputs_freemodel" \
python scaling_n100/extract_parallel.py \
    --schema chr --track ontology --prompt ontology --systems a,b --workers 5
```

### 4.3 Run extraction — Gemini 2.0 Flash (cross-model replication)

```bash
# Schema track (SHACL validator)
python scaling_n100/extract_parallel.py \
    --schema chr --track schema --prompt full --systems a,b --workers 5

# Ontology track (OWL-RL reasoner)
python scaling_n100/extract_parallel.py \
    --schema chr --track ontology --prompt ontology --systems a,b --workers 5
```

### 4.4 Run compute-fair control (System C)

```bash
python scripts/run_system_d.py --schema chr --track schema --workers 5
# The script file is named system_d for historical reasons; the paper
# calls this System C.
```

### 4.5 Score everything

```bash
make evaluate                         # F1, SHACL conformance, PC, hallucination
make stats                            # paired sign tests, bootstrap CIs
```

### 4.6 Run the LLM-as-judge

```bash
JUDGE_MODEL="meta-llama/llama-3.3-70b-instruct" \
python pipeline/llm_judge.py --schema chr --track schema --prompt full --system a
JUDGE_MODEL="meta-llama/llama-3.3-70b-instruct" \
python pipeline/llm_judge.py --schema chr --track schema --prompt full --system b
# Repeat with OUTPUTS_ROOT_OVERRIDE for the gpt-oss outputs.
```

### 4.7 Supplementary analyses (per-shape, normaliser ablation, OWL conformance)

```bash
python scripts/per_shape_violation_analysis.py
python scripts/owl_conformance_survey.py
python scripts/f1_normalizer_ablation.py
```

### 4.8 Regenerate the four paper figures

```bash
python scripts/make_n200_decoupling_plot.py        # Figure 2
python scripts/make_n200_percycle_v2.py            # Figure 1
python scripts/make_pipeline_e2e_figure.py         # Figure 4
python scripts/make_violation_resolution_figure.py # Figure 3
```

## 5. Where each paper element lives

| Paper element | Source |
|---|---|
| Figure 1 (per-cycle dynamics) | `scripts/make_n200_percycle_v2.py` → `report/figures/fig_percycle_v2.png` |
| Figure 2 (decoupling) | `scripts/make_n200_decoupling_plot.py` → `report/figures/fig_decoupling_n200.png` |
| Figure 3 (violation resolution) | `scripts/make_violation_resolution_figure.py` → `report/figures/fig_violation_resolution.png` |
| Figure 4 (end-to-end worked example) | `scripts/make_pipeline_e2e_figure.py` → `report/figures/fig_pipeline_e2e_vignette_001.png` |
| Table I (main results) | `make evaluate` → `evaluation/outputs/**/eval.json` and `evaluation/outputs_freemodel/**/eval.json` |
| Table II (LLM-as-judge) | `pipeline/llm_judge.py` → `evaluation/outputs/judge_scores.json` |
| Statistical claims (p-values, CIs) | `make stats` → `evaluation/outputs/stats.json` |
| Per-shape breakdown (§V-K) | `scripts/per_shape_violation_analysis.py` → `evaluation/outputs/per_shape_violations.json` |
| OWL conformance survey | `scripts/owl_conformance_survey.py` → printed to stdout (A vs B per stratum, both models) |
| IRI normaliser ablation (§IV-D) | `scripts/f1_normalizer_ablation.py` → `evaluation/outputs/f1_normalizer_ablation.json` |

## 6. Configuration knobs

All set in `.env` or as environment variables:

| Variable | Default | Notes |
|---|---|---|
| `OPENROUTER_API_KEY` | _(required)_ | OpenRouter credential |
| `OPENROUTER_MODEL_OVERRIDE` | `google/gemini-2.0-flash-001` | Codebase default; set to `openai/gpt-oss-120b` for the primary-model runs |
| `OPENROUTER_PROVIDER_SORT` | _(unset)_ | Set to `throughput` when running gpt-oss-120b (primary) on the paid endpoint |
| `OUTPUTS_ROOT_OVERRIDE` | `evaluation/outputs` | Holds Gemini (replication) outputs. Set to `evaluation/outputs_freemodel` for gpt-oss-120b (primary) outputs |
| `JUDGE_MODEL` | `meta-llama/llama-3.3-70b-instruct` | Cross-family judge |

Hard-coded in `pipeline/extract.py`: temperature `0.1`, retry budget `k=3`,
plateau heuristic (two identical violation reports terminates early).

## 7. Smoke tests

```bash
python -m pytest tests/                       # unit tests
python scripts/verify_experiments_e2e.py      # end-to-end claim verification
```

`verify_experiments_e2e.py` reads `evaluation/outputs/*.json` and asserts
every quantitative claim in the paper still holds against the committed
data. If you re-run extraction and any number drifts, this is the first
thing to run.

## 8. Cost budget (if running from scratch)

| Step | Calls | Model | ~USD |
|---|---|---|---|
| 4.2 gpt-oss schema + ontology, A+B (primary) | ~1,200 | gpt-oss-120b (paid) | 8 |
| 4.3 Gemini schema + ontology, A+B (replication) | ~1,200 | gemini-2.0-flash | 2 |
| 4.4 System C, both models | ~1,600 | both | 6 |
| 4.6 Judge (4 cells × 200 cases × 2 systems) | ~1,600 | llama-3.3-70b | 3 |
| **Total** | ~5,600 | | **~19** |

---

Questions about the code or repository: viitala.vincent@gmail.com.
