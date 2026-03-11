import os
from openai import OpenAI
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from rdflib import Graph, RDF
from sulo_spec import (
    SULO_NAMESPACE,
    get_paper_overview_for_prompt,
    get_predicate_rules_for_prompt,
)
load_dotenv()
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "google/gemini-2.0-flash-001"

def read_clinical_narrative_from_pdf(pdf_path: str) -> str:
    """Extract text from all pages of a PDF. Fails loudly if the file cannot be read."""
    try:
        reader = PdfReader(pdf_path)
        pages_text = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            pages_text.append(page_text.strip())
        return "\n\n".join(t for t in pages_text if t)
    except Exception as e:
        raise RuntimeError(f"Failed to read clinical narrative from PDF: {e}") from e


def build_sulo_prompt(clinical_text: str) -> str:
    """Assemble the full LLM prompt from the ontology spec and the narrative."""
    predicate_rules = get_predicate_rules_for_prompt()
    paper_overview = get_paper_overview_for_prompt()
    return f"""You are a Clinical Knowledge Engineer. Build an RDF Turtle Knowledge Graph that strictly follows the SULO ontology (Dumontier et al., FOIS/JOWO 2025).

PAPER OVERVIEW:
{paper_overview}

NAMESPACES (use exactly these):
@prefix : <http://example.org/maria#> .
@prefix sulo: <{SULO_NAMESPACE}> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

TASK:
1. From the narrative, identify EXACTLY 20 salient domain concepts to represent as SULO individuals.
   - For each, invent a concise local name after the ":" prefix (e.g. :patient1, :primaryFinding, :initialBiopsy, :time_Diagnosis).
   - Assign each a single most-appropriate SULO class (Process, SpatialObject, InformationObject, Role, Quality, Capability, TimeInstant, TimeInterval, Duration, Unit, Collection, Quantity, etc.).
   - Reuse the same 20 individuals consistently in all triples you generate. Do not create additional individuals.

PREDICATE RULES (from paper relations and patterns):
{predicate_rules}

OUTPUT:
• Return ONLY valid RDF Turtle. No markdown, no fences, no prose.
• Use EXACTLY 20 individuals in the http://example.org/maria# namespace (no more, no fewer).
• For unknown dates you may use "9999-01-01"^^xsd:date with hasValue (SOLID) or another placeholder literal, but still keep the individual count at 20.

CLINICAL NARRATIVE TO PROCESS:
{clinical_text}
"""

def call_llm_for_turtle(prompt: str) -> str:
    """Send the SULO prompt to the LLM and return the raw response content."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set. Add it to .env or your environment.")

    client = OpenAI(base_url=OPENROUTER_BASE, api_key=api_key)
    response = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return response.choices[0].message.content or ""


def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences and optional language tag from LLM output."""
    if "```" not in text:
        return text.strip()
    parts = text.split("```")
    # First fenced block (index 1); drop first line if it looks like "turtle" or "ttl"
    block = parts[1].strip()
    lines = block.splitlines()
    if lines and lines[0].strip().lower() in ("turtle", "ttl", "rdf"):
        block = "\n".join(lines[1:])
    return block.strip()

def generate_sulo_kg() -> None:
    """Run the full pipeline: PDF → spec-based prompt → LLM → Turtle file."""
    # Step 1: Load narrative from PDF
    print("Step 1: Reading clinical narrative from Clinical text.pdf...")
    pdf_path = os.path.join(os.path.dirname(__file__), "Clinical text.pdf")
    clinical_text = read_clinical_narrative_from_pdf(pdf_path)
    print(f"       Extracted {len(clinical_text)} characters.")
    print("Step 2: Building SULO prompt from ontology spec...")
    prompt = build_sulo_prompt(clinical_text)
    print("Step 3: Calling LLM (OpenRouter) for RDF Turtle generation...")
    try:
        raw_content = call_llm_for_turtle(prompt)
    except Exception as e:
        print(f"Error during LLM call: {e}")
        return
    print("Step 4: Normalising output and writing Turtle file...")
    ttl_content = strip_markdown_fences(raw_content)
    filename = "maria_clinical_evolution.ttl"
    out_path = os.path.join(os.path.dirname(__file__), filename)
    with open(out_path, "w") as f:
        f.write(ttl_content)

    print(f"Done. SULO Knowledge Graph saved to {filename}")

    # Step 5: Quick summary of what was extracted (concepts and predicates)
    try:
        g = Graph()
        g.parse(out_path, format="turtle")

        # Collect individuals and their SULO types
        type_triples = [
            t
            for t in g.triples((None, RDF.type, None))
            if str(t[0]).startswith("http://example.org/maria#")
        ]
        type_map = {}
        for subj, _, obj in type_triples:
            local_name = str(subj).split("#")[-1]
            sulo_type = str(obj).split("/")[-1]
            type_map.setdefault(local_name, set()).add(sulo_type)

        individuals = sorted(type_map.keys())
        print(f"\nSummary of extracted SULO individuals ({len(individuals)}):")
        for name in individuals:
            classes = ", ".join(sorted(type_map[name]))
            print(f"  - {name}  :  sulo:{classes}")

        preds = sorted(
            set(
                str(p).split("/")[-1]
                for p in g.predicates()
                if str(p).startswith("https://w3id.org/sulo/")
            )
        )
        print("\nSULO predicates used in the graph:")
        print("  " + ", ".join(preds) if preds else "  (none)")
    except Exception as e:
        print(f"\n[WARN] Could not summarise extracted concepts: {e}")

    print("\nYou can now run your SHACL validator on this file.")


if __name__ == "__main__":
    generate_sulo_kg()
