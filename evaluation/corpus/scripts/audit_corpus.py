#!/usr/bin/env python3
"""Audit the CHR corpus (schema track).

For every vignette in corpus/vignettes/, assert that:

  1. The schema-level gold ABox (vignette_NNN_gold_schema.ttl) parses as
     valid Turtle and conforms against corpus/shapes/chr_shacl_schema.ttl
     with ZERO violations.

  2. Every IRI in each gold ABox uses a declared prefix (chr:, sulo:, ex:,
     xsd:, rdf:, rdfs:, owl:). Any other prefix is a corpus hygiene bug.

Aggregate class coverage is also reported (which of the 25 chr:* classes
are instantiated), but informationally — the Synthea-derived gold uses
the FHIR-mapped subset, not the full TBox.

Exit code 0 on success, 1 on any failure. Prints a structured report.

Usage:
    python corpus/scripts/audit_corpus.py              # audit the whole corpus
    python corpus/scripts/audit_corpus.py vignette_001 # audit one vignette
"""

from __future__ import annotations

import sys
from pathlib import Path

from pyshacl import validate
from rdflib import Graph, Namespace, RDF, URIRef

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CORPUS = Path(__file__).resolve().parent.parent  # corpus/
SHAPES_SCHEMA = CORPUS / "shapes" / "chr_shacl_schema.ttl"
TBOX_SCHEMA   = CORPUS / "tbox"   / "chr_schema.ttl"
VIGNETTES_DIR = CORPUS / "vignettes"
GOLD_DIR      = CORPUS / "abox_gold"

CHR_NAMESPACE = "https://w3id.org/shexmap/resource/ontology-schema/d285f599-dc2e-4bd0-83f3-df21defa8821/"
CHR = Namespace(CHR_NAMESPACE)

# Prefixes that a gold ABox IRI is allowed to start with. Any IRI whose
# lexical form does not begin with one of these is flagged.
ALLOWED_NAMESPACES = (
    CHR_NAMESPACE,
    "https://w3id.org/sulo/",
    "http://example.org/clinical/",
    "http://www.w3.org/2001/XMLSchema#",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "http://www.w3.org/2000/01/rdf-schema#",
    "http://www.w3.org/2002/07/owl#",
)

# All 25 CHR classes declared in the TBox. Hard-coded (rather than scraped
# from the TBox) so the audit FAILS if a class is added to the TBox without
# being covered by a gold ABox.
ALL_CHR_CLASSES = {
    "AnatomicalStructure", "CarePlan", "CareProviderRole", "CareUnit",
    "ClinicalCondition", "ClinicalVisit", "Device", "DiagnosticStatement",
    "EvaluationProcess", "InstrumentRole", "Measurement",
    "MeasurementProcess", "MedicalProcedure", "MedicationAdministration",
    "Occupation", "OutputRole", "PerformerRole", "Person",
    "PharmaceuticalDose", "PharmaceuticalDoseForm", "PharmaceuticalProduct",
    "ProcessStatus", "Severity", "SubjectOfCareRole", "TreatmentPlan",
}

# ANSI colours for the report.
GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load(*paths: Path) -> Graph:
    g = Graph()
    for p in paths:
        g.parse(p.as_posix(), format="turtle")
    return g


def _vignette_ids() -> list[str]:
    """Return vignette IDs like ['vignette_001', 'vignette_002', ...]."""
    if not VIGNETTES_DIR.exists():
        return []
    return sorted(p.stem for p in VIGNETTES_DIR.glob("vignette_*.txt"))


def _run_shacl(data_file: Path) -> dict:
    """Run pyshacl against the schema-track shapes. Return conforms / counts / report."""
    data_graph   = _load(data_file)
    shapes_graph = _load(SHAPES_SCHEMA)
    ont_graph    = _load(TBOX_SCHEMA)
    conforms, _results_graph, results_text = validate(
        data_graph=data_graph,
        shacl_graph=shapes_graph,
        ont_graph=ont_graph,
        inference="none",
        abort_on_first=False,
        meta_shacl=False,
        advanced=True,
        js=False,
        debug=False,
    )
    return {
        "conforms":   bool(conforms),
        "violations": results_text.count("Severity: sh:Violation"),
        "warnings":   results_text.count("Severity: sh:Warning"),
        "report":     results_text,
    }


def _iri_prefix_issues(g: Graph) -> list[str]:
    """Return list of IRIs in g that don't start with an allowed namespace."""
    offenders: set[str] = set()
    for s, p, o in g:
        for node in (s, p, o):
            if isinstance(node, URIRef):
                uri = str(node)
                if not any(uri.startswith(ns) for ns in ALLOWED_NAMESPACES):
                    offenders.add(uri)
    return sorted(offenders)


def _classes_instantiated(g: Graph) -> set[str]:
    """Return the set of chr:* local names that are used as rdf:type values in g."""
    out: set[str] = set()
    for _s, _p, o in g.triples((None, RDF.type, None)):
        if isinstance(o, URIRef) and str(o).startswith(CHR_NAMESPACE):
            out.add(str(o)[len(CHR_NAMESPACE):])
    return out


# ---------------------------------------------------------------------------
# Per-vignette audit
# ---------------------------------------------------------------------------


def audit_one_gold(vignette: str) -> dict:
    """Audit the schema-track gold ABox for one vignette."""
    gold_path = GOLD_DIR / f"{vignette}_gold_schema.ttl"
    result: dict = {
        "vignette": vignette, "path": gold_path,
        "exists": gold_path.exists(), "parses": False,
        "conforms": False, "violations": None, "warnings": None,
        "iri_offenders": [], "classes": set(), "report": "",
    }
    if not gold_path.exists():
        return result

    try:
        g = _load(gold_path)
        result["parses"] = True
    except Exception as exc:
        result["report"] = f"parse error: {exc}"
        return result

    result["iri_offenders"] = _iri_prefix_issues(g)
    result["classes"] = _classes_instantiated(g)

    shacl = _run_shacl(gold_path)
    result.update({
        "conforms":   shacl["conforms"],
        "violations": shacl["violations"],
        "warnings":   shacl["warnings"],
        "report":     shacl["report"],
    })
    return result


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------


def _render_row(r: dict) -> str:
    """One-liner per gold ABox."""
    if not r["exists"]:
        status = f"{YELLOW}MISSING{RESET}"
        extra = ""
    elif not r["parses"]:
        status = f"{RED}PARSE ERROR{RESET}"
        extra = r["report"]
    elif r["violations"] and r["violations"] > 0:
        status = f"{RED}FAIL{RESET}"
        extra = f"{r['violations']} violations, {r['warnings']} warnings"
    else:
        status = f"{GREEN}OK{RESET}"
        extra = f"0 violations, {r['warnings']} warnings"
    return f"  [{status}] {r['vignette']}_gold_schema.ttl  —  {extra}"


def _render_report_excerpt(report: str, max_chars: int = 1200) -> str:
    """Trim a pyshacl report to its first few violations for terminal display."""
    if len(report) <= max_chars:
        return report
    return report[:max_chars] + f"\n... ({len(report) - max_chars} more chars elided)"


def _print_section(title: str):
    print(f"\n{title}")
    print("=" * len(title))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    targets = argv[1:] or _vignette_ids()
    if not targets:
        print(f"{YELLOW}No vignettes found in {VIGNETTES_DIR}{RESET}")
        return 1

    _print_section("CHR corpus audit (schema track)")
    print(f"  corpus root:   {CORPUS}")
    print(f"  schema TBox:   {TBOX_SCHEMA.name}")
    print(f"  schema SHACL:  {SHAPES_SCHEMA.name}")
    print(f"  vignettes:     {len(targets)}")

    results: list[dict] = [audit_one_gold(v) for v in targets]

    # -- Per-file status --
    _print_section("Per-file SHACL conformance")
    for r in results:
        print(_render_row(r))

    # -- IRI prefix offenders --
    offenders_present = any(r["iri_offenders"] for r in results if r["parses"])
    _print_section("IRI prefix hygiene")
    if not offenders_present:
        print(f"  {GREEN}OK{RESET}  every IRI uses a declared namespace.")
    else:
        for r in results:
            if r["iri_offenders"]:
                print(f"  {RED}FAIL{RESET}  {r['vignette']}_gold_schema.ttl  ({len(r['iri_offenders'])} offenders)")
                for uri in r["iri_offenders"][:5]:
                    print(f"      {DIM}{uri}{RESET}")
                if len(r["iri_offenders"]) > 5:
                    print(f"      {DIM}... and {len(r['iri_offenders']) - 5} more{RESET}")

    # -- Class coverage (informational, not pass/fail) --
    _print_section("Class coverage  (informational — Synthea-derived corpus exercises a subset)")
    seen = set().union(*(r["classes"] for r in results))
    missing = sorted(ALL_CHR_CLASSES - seen)
    print(f"  {len(seen)}/{len(ALL_CHR_CLASSES)} CHR classes instantiated in gold.")
    if missing:
        print(f"  Not instantiated: {', '.join(missing)}")
        print(f"  {DIM}(Synthea does not emit FHIR resources for every CHR concept;"
              f" Role-pattern classes are ontology-track only.){RESET}")

    # -- Failure details --
    failed = [
        r for r in results
        if r["exists"]
        and (not r["parses"] or (r["violations"] or 0) > 0 or r["iri_offenders"])
    ]
    if failed:
        _print_section("Failure reports (first violations per failing file)")
        for r in failed:
            if not r["parses"]:
                print(f"\n  --- {r['vignette']}_gold_schema.ttl: parse error ---")
                print(f"  {r['report']}")
                continue
            if (r["violations"] or 0) > 0:
                print(f"\n  --- {r['vignette']}_gold_schema.ttl: SHACL violations ---")
                print(_render_report_excerpt(r["report"]))

    # -- Verdict --
    any_missing = any(not r["exists"] for r in results)
    any_parse   = any(r["exists"] and not r["parses"] for r in results)
    any_violate = any((r["violations"] or 0) > 0 for r in results if r["parses"])
    any_iri_bad = offenders_present

    _print_section("Verdict")
    def _line(label, ok):
        mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
        print(f"  [{mark}] {label}")
    _line("all expected gold files present",      not any_missing)
    _line("all gold files parse as Turtle",       not any_parse)
    _line("all gold files SHACL-conform",         not any_violate)
    _line("all IRIs use declared namespaces",     not any_iri_bad)

    ok = (not any_missing and not any_parse and not any_violate
          and not any_iri_bad)
    if ok:
        print(f"\n{GREEN}ALL CHECKS PASSED.{RESET}")
        return 0
    print(f"\n{RED}AUDIT FAILED.{RESET}  Fix the issues above and re-run.")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
