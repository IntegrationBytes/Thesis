#!/usr/bin/env python3
"""
Demo sanity checks: spec, TTL parse, and optional env/PDF checks.
Run from meeting_demo with venv activated: python test_demo_checks.py
"""
import os
import sys

def main():
    errors = []
    base = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base)

    # 1. Spec: 20 individuals, classification flow
    try:
        from sulo_spec import CLINICAL_INDIVIDUALS, SULO_CLASSIFICATION_FLOW
        if len(CLINICAL_INDIVIDUALS) != 20:
            errors.append(f"CLINICAL_INDIVIDUALS: expected 20, got {len(CLINICAL_INDIVIDUALS)}")
        if len(SULO_CLASSIFICATION_FLOW) != 6:
            errors.append(f"SULO_CLASSIFICATION_FLOW: expected 6 steps, got {len(SULO_CLASSIFICATION_FLOW)}")
        print("[OK] Spec: 20 individuals, 6-step classification flow")
    except Exception as e:
        errors.append(f"Spec import: {e}")
        print("[FAIL] Spec import:", e)

    # 2. TTL exists and parses; has 20 individuals and expected predicates
    ttl_path = os.path.join(base, "maria_clinical_evolution.ttl")
    if not os.path.isfile(ttl_path):
        errors.append("maria_clinical_evolution.ttl not found")
        print("[SKIP] No TTL file — run sulo_extract.py first")
    else:
        try:
            from rdflib import Graph, RDF
            g = Graph()
            g.parse(ttl_path, format="turtle")
            type_triples = [
                t for t in g.triples((None, RDF.type, None))
                if str(t[0]).startswith("http://example.org/maria#")
            ]
            found = set(t[0].split("#")[-1] for t in type_triples)
            if len(found) != 20:
                errors.append(f"TTL should define exactly 20 individuals in the : namespace, found {len(found)}")
            preds = set(str(p).split("/")[-1] for p in g.predicates())
            allowed = {"type", "atTime", "hasPart", "hasParticipant", "hasValue", "isIn", "refersTo"}
            bad_preds = preds - allowed - {"22-rdf-syntax-ns#type"}
            if bad_preds:
                errors.append(f"TTL uses unexpected predicates: {bad_preds}")
            if not errors:
                print("[OK] TTL parses; 20 individuals; predicates in expected set")
            else:
                print("[FAIL] TTL checks")
        except ImportError as e:
            errors.append(f"rdflib not available: {e}")
            print("[SKIP] rdflib not installed — activate venv and pip install rdflib")
        except Exception as e:
            errors.append(f"TTL parse/check: {e}")
            print("[FAIL] TTL:", e)

    # 3. Optional: PDF and .env
    pdf_path = os.path.join(base, "Clinical text.pdf")
    if not os.path.isfile(pdf_path):
        print("[WARN] Clinical text.pdf missing — extraction will fail")
    else:
        print("[OK] Clinical text.pdf present")
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(base, ".env"))
        if not os.getenv("OPENROUTER_API_KEY"):
            print("[WARN] OPENROUTER_API_KEY not set — extraction will fail")
        else:
            print("[OK] OPENROUTER_API_KEY set")
    except ImportError:
        print("[SKIP] python-dotenv not installed")

    if errors:
        print("\nErrors:", errors)
        sys.exit(1)
    print("\nAll checks passed.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
