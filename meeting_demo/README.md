# Clinical Concept Extraction & Validation Demo

This demo showcases two key components of the Knowledge Graph pipeline:
1.  **Extraction**: Using LLMs to extract structured clinical concepts from unstructured text.
2.  **Validation**: Using SHACL to enforce data quality rules on the extracted graph.

## Setup

Ensure you have the virtual environment activated and dependencies installed:

```bash
# From the project root
source ../venv/bin/activate
pip install langchain-openai langchain-core python-dotenv rdflib pyshacl openai PyPDF2
```

## 1. Concept Extraction (`extract.py`)

This script uses an LLM (Gemini via OpenRouter) to extract 20 distinct clinical concepts from Maria's case history.

**Run:**
```bash
python3 extract.py
```

## 2. SULO KG Extraction (`sulo_extract.py`)

Reads the clinical narrative from `Clinical text.pdf`, sends it to the LLM with SULO ontological instructions, and writes RDF Turtle to `maria_clinical_evolution.ttl`. Requires `OPENROUTER_API_KEY` in `.env`.

**Run:**
```bash
python3 sulo_extract.py
```

For a step-by-step explanation of how the pipeline works and how it mirrors the SULO paper, see **HOW_IT_WORKS.md**.

## 3. SHACL Validation (`validate_shacl.py`)

This script demonstrates how we enforce data integrity rules on the Knowledge Graph.

**Run:**
```bash
python3 validate_shacl.py
```

**What to look for:**
- **Pass 1**: The graph is valid because all diagnoses have evidence and timestamps.
- **Pass 2**: The script intentionally removes a link to prove the validator catches the error ("Every diagnosis must be traceable...").

## Reference

- **SULO**: Dumontier, M., Çelebi, R., Gilani, K., de Zegher, I., Serafimova, K., Martínez Costa, C., & Schulz, S. (2025). *SULO – a simplified upper-level ontology*. Proceedings of the Joint Ontology Workshops (JOWO), co-located with FOIS 2025, Catania. SULO namespace: `https://w3id.org/sulo/`.
