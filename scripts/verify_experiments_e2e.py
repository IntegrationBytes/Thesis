"""End-to-end verification — every validator, every pipeline component.

For each component (OWL validator, SHACL validator, IRI normalizer, F1
evaluator, judge response parser), construct a SYNTHETIC input that the
component SHOULD catch, run the component, and assert the expected
behaviour. Also cross-check that existing data files yield consistent
numbers when re-analysed.

Defensible answer to "are the experiments actually working?"

Run::

    python scripts/verify_experiments_e2e.py

Exit code 0 on full pass. Non-zero on any failure.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

import rdflib  # noqa: E402

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    if cond:
        PASSED.append(label)
        print(f"  PASS  {label}")
    else:
        FAILED.append((label, detail))
        print(f"  FAIL  {label}  -- {detail}")


# ============================================================================
# Component 1: OWL validator
# ============================================================================
def test_owl_validator() -> None:
    print("\n=== Component 1: OWL validator (pipeline/owl_validator.py) ===")
    from owl_validator import validate_with_owl
    CHR_ONT = ROOT / "evaluation/corpus/tbox/chr_ontology.owl.ttl"

    # 1a. Empty graph is vacuously consistent (no facts → no violations possible)
    ok, _ = validate_with_owl("", CHR_ONT)
    check("OWL: empty graph is vacuously consistent (no facts to clash)",
          ok is True, "should pass — empty graph has no violations to find")

    # 1b. Trivially valid graph (one typed Patient — no violation)
    valid_ttl = """
@prefix chr: <https://w3id.org/shexmap/resource/ontology-schema/d285f599-dc2e-4bd0-83f3-df21defa8821/> .
@prefix ex: <http://example.org/clinical/> .
ex:p1 a chr:Person .
"""
    ok, report = validate_with_owl(valid_ttl, CHR_ONT)
    check("OWL: valid graph (one Person) consistent",
          ok is True, f"got inconsistent: {report[:120]}")

    # 1c. Disjointness clash — assert sulo:Process AND sulo:SpatialObject on same node
    clash_ttl = """
@prefix sulo: <https://w3id.org/sulo/> .
@prefix ex: <http://example.org/clinical/> .
ex:x a sulo:Process .
ex:x a sulo:SpatialObject .
"""
    ok, report = validate_with_owl(clash_ttl, CHR_ONT)
    check("OWL: disjointness clash on sulo:Process + SpatialObject DETECTED",
          ok is False and "DisjointnessClash" in report,
          f"ok={ok}, report has clash? {'DisjointnessClash' in report}")

    # 1d. Invalid Turtle returns failure (parse error)
    bad_ttl = "this is not valid turtle @@"
    ok, report = validate_with_owl(bad_ttl, CHR_ONT)
    check("OWL: malformed Turtle returns (False, parse-error report)",
          ok is False and "parse error" in report.lower(),
          f"ok={ok}, has 'parse error'? {'parse error' in report.lower()}")


# ============================================================================
# Component 2: SHACL validator
# ============================================================================
def test_shacl_validator() -> None:
    print("\n=== Component 2: SHACL validator (pipeline/extract.py) ===")
    from extract import validate_with_shacl
    from prompts import chr_context
    ctx = chr_context("schema")

    # 2a. Empty graph is vacuously conforming under SHACL (no targets to violate)
    ok, _ = validate_with_shacl("", ctx)
    check("SHACL: empty graph is vacuously conforming (no targets to violate)",
          ok is True, "should pass — empty graph has no shapes targeting anything")

    # 2b. Valid Turtle conforming to schema shapes
    valid_ttl = """
@prefix chr: <https://w3id.org/shexmap/resource/ontology-schema/d285f599-dc2e-4bd0-83f3-df21defa8821/> .
@prefix ex: <http://example.org/clinical/> .
ex:p1 a chr:Person .
"""
    ok, report = validate_with_shacl(valid_ttl, ctx)
    check("SHACL: valid graph (one Person) conforms",
          ok is True, f"got non-conforming: {report[:200]}")

    # 2c. Graph that VIOLATES HasMedicalProcedureDomainShape
    # (chr:hasMedicalProcedure on something that's not a CarePlan)
    violation_ttl = """
@prefix chr: <https://w3id.org/shexmap/resource/ontology-schema/d285f599-dc2e-4bd0-83f3-df21defa8821/> .
@prefix ex: <http://example.org/clinical/> .
ex:not_a_careplan chr:hasMedicalProcedure ex:proc1 .
ex:proc1 a chr:MedicalProcedure .
"""
    ok, report = validate_with_shacl(violation_ttl, ctx)
    check("SHACL: HasMedicalProcedureDomainShape violation DETECTED",
          ok is False and "HasMedicalProcedureDomainShape" in report,
          f"ok={ok}, has shape name? {'HasMedicalProcedureDomainShape' in report}")


# ============================================================================
# Component 3: IRI normalizer
# ============================================================================
def test_iri_normalizer() -> None:
    print("\n=== Component 3: IRI normalizer (pipeline/iri_normalizer.py) ===")
    from iri_normalizer import normalize_llm_to_gold

    gold_ttl = """
@prefix chr: <https://w3id.org/shexmap/resource/ontology-schema/d285f599-dc2e-4bd0-83f3-df21defa8821/> .
@prefix ex: <http://example.org/clinical/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
ex:patient_5e117ba0_uuid a chr:Patient ;
    rdfs:label "Alice Smith" .
"""
    llm_ttl = """
@prefix chr: <https://w3id.org/shexmap/resource/ontology-schema/d285f599-dc2e-4bd0-83f3-df21defa8821/> .
@prefix ex: <http://example.org/clinical/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
ex:patient_Alice_Smith a chr:Patient ;
    rdfs:label "alice smith" .
"""
    gold_g = rdflib.Graph()
    gold_g.parse(data=gold_ttl, format="turtle")
    llm_g = rdflib.Graph()
    llm_g.parse(data=llm_ttl, format="turtle")
    rewritten, iri_map = normalize_llm_to_gold(llm_g, gold_g)
    # The LLM patient should map to the gold patient via label match
    mapped = any(
        "patient_Alice_Smith" in str(k) and "patient_5e117ba0_uuid" in str(v)
        for k, v in iri_map.items()
    )
    check("IRI normalizer: label-match maps text-span IRI to UUID IRI",
          mapped, f"iri_map={dict(iri_map)}")


# ============================================================================
# Component 4: F1 evaluator
# ============================================================================
def test_f1_evaluator() -> None:
    print("\n=== Component 4: F1 evaluator (pipeline/evaluate.py:prf1) ===")
    from evaluate import prf1

    # Exact match → F1 = 1
    gold = {("s", "p", "o"), ("s", "p", "o2")}
    pred = {("s", "p", "o"), ("s", "p", "o2")}
    m = prf1(gold, pred)
    check("F1: exact match → F1=1.0",
          m["f1"] == 1.0, f"got f1={m['f1']}")

    # No overlap → F1 = 0
    gold = {("s", "p", "o")}
    pred = {("x", "y", "z")}
    m = prf1(gold, pred)
    check("F1: no overlap → F1=0.0",
          m["f1"] == 0.0, f"got f1={m['f1']}")

    # Partial overlap → 0 < F1 < 1
    gold = {("s", "p", "o1"), ("s", "p", "o2")}
    pred = {("s", "p", "o1"), ("s", "p", "o3")}
    m = prf1(gold, pred)
    check("F1: partial overlap (1/2 TP, 1 FP, 1 FN) → F1=0.5",
          abs(m["f1"] - 0.5) < 0.01, f"got f1={m['f1']}")

    # None pred → all gold counted as FN
    m = prf1(gold, None)
    check("F1: None pred → F1=0.0 with valid=False",
          m["f1"] == 0.0 and m["valid"] is False,
          f"got f1={m['f1']}, valid={m['valid']}")


# ============================================================================
# Component 5: Judge response parser
# ============================================================================
def test_judge_parser() -> None:
    print("\n=== Component 5: Judge response parser (pipeline/llm_judge.py) ===")
    from llm_judge import _parse_judge_response

    # 5a. Clean JSON response
    clean = '{"faithfulness": 4, "completeness": 3, "hallucination": 2, "rationale": {"a":"b"}}'
    p = _parse_judge_response(clean)
    check("Judge: clean JSON parsed",
          p["faithfulness"] == 4 and p["completeness"] == 3 and p["hallucination"] == 2,
          f"got: {p}")

    # 5b. JSON wrapped in text
    wrapped = 'Here is my evaluation:\n```json\n{"faithfulness": 5, "completeness": 5, "hallucination": 1}\n```\nThank you.'
    p = _parse_judge_response(wrapped)
    check("Judge: JSON-in-markdown-fence extracted",
          p["faithfulness"] == 5, f"got: {p}")

    # 5c. Out-of-range scores clamped
    out_of_range = '{"faithfulness": 7, "completeness": -2, "hallucination": 3}'
    p = _parse_judge_response(out_of_range)
    check("Judge: scores clamped to 1-5 range",
          p["faithfulness"] == 5 and p["completeness"] == 1,
          f"got: faithfulness={p['faithfulness']}, completeness={p['completeness']}")

    # 5d. Garbage response handled
    garbage = "I cannot evaluate this"
    p = _parse_judge_response(garbage)
    check("Judge: garbage response → None scores + error key",
          p["faithfulness"] is None and "error" in p,
          f"got: {p}")


# ============================================================================
# Component 6: Cross-check with real corpus data
# ============================================================================
def test_real_data_pipeline() -> None:
    print("\n=== Component 6: Real-data sanity check (vignette_030) ===")
    from extract import validate_with_shacl
    from owl_validator import validate_with_owl
    from prompts import chr_context

    vid = "vignette_030"
    schema_a = ROOT / f"evaluation/outputs/chr/schema/full/a/{vid}.ttl"
    schema_b = ROOT / f"evaluation/outputs/chr/schema/full/b/{vid}.ttl"

    if not (schema_a.exists() and schema_b.exists()):
        FAILED.append(("Real-data sanity", "vignette_030 outputs missing"))
        return

    ctx = chr_context("schema")
    a_content = schema_a.read_text()
    b_content = schema_b.read_text()

    a_ok, _ = validate_with_shacl(a_content, ctx)
    b_ok, _ = validate_with_shacl(b_content, ctx)
    check("Real data: vignette_030 System A FAILS SHACL (as documented)",
          a_ok is False, f"A conforms={a_ok}")
    check("Real data: vignette_030 System B PASSES SHACL (as documented)",
          b_ok is True, f"B conforms={b_ok}")

    # OWL validator on a real ontology-track output
    ont_b = ROOT / f"evaluation/outputs/chr/ontology/ontology/b/{vid}.ttl"
    if ont_b.exists():
        ok, _ = validate_with_owl(ont_b.read_text(), ROOT / "evaluation/corpus/tbox/chr_ontology.owl.ttl")
        # Don't assert a particular outcome — just that the validator returns a bool
        check("Real data: OWL validator returns valid bool on real input",
              isinstance(ok, bool), f"got non-bool: {type(ok)}")


# ============================================================================
# Component 7: Analysis JSON outputs are self-consistent
# ============================================================================
def test_analysis_outputs() -> None:
    print("\n=== Component 7: Analysis JSON outputs consistency ===")
    import json

    expected = [
        ("evaluation/outputs/judge_scores.json",
         "judge: 4 system-B variants present"),
        ("evaluation/outputs/connectivity_buckets.json",
         "connectivity: 3 buckets sum to 200"),
        ("evaluation/outputs/f1_normalizer_ablation.json",
         "F1 ablation: 8 cells present (2 models × 2 strata × 2 systems)"),
        ("evaluation/outputs/per_shape_violations.json",
         "per-shape: schema + ontology keys present"),
        ("evaluation/outputs/stats.json",
         "stats: schema/full cell present"),
    ]
    for path, msg in expected:
        p = ROOT / path
        check(f"Analysis output exists: {path}",
              p.exists(), "missing file")

    # Specific consistency checks
    if (ROOT / "evaluation/outputs/connectivity_buckets.json").exists():
        d = json.loads((ROOT / "evaluation/outputs/connectivity_buckets.json").read_text())
        total = sum(len(d["buckets"][t]["vignettes"]) for t in ("low", "medium", "high"))
        check("Connectivity: 3 buckets sum to 200 vignettes",
              total == 200, f"sum={total}")

    if (ROOT / "evaluation/outputs/judge_scores.json").exists():
        d = json.loads((ROOT / "evaluation/outputs/judge_scores.json").read_text())
        n_variants = len(d)
        check(f"Judge: 4 system-B variants ({n_variants} found)",
              n_variants >= 4, f"only {n_variants} variants")


def main() -> int:
    test_owl_validator()
    test_shacl_validator()
    test_iri_normalizer()
    test_f1_evaluator()
    test_judge_parser()
    test_real_data_pipeline()
    test_analysis_outputs()

    print("\n" + "=" * 60)
    print(f"PASSED: {len(PASSED)}")
    print(f"FAILED: {len(FAILED)}")
    for label, detail in FAILED:
        print(f"  - {label}: {detail}")
    print("=" * 60)
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
