"""Fault-injection (mutation) validation of the OWL reasoner.

Natural logical errors are rare on the flat clinical ABox (2/200), so we
validate the reasoner's detection capability directly: inject KNOWN violations
into clean, OWL-consistent System-B graphs and measure detection vs false
positives.

Conditions (per graph, mutating a chr:Person node):
  orig       : unchanged                          -> expect CONSISTENT (no false positive)
  disjoint   : + a chr:MedicalProcedure type      -> Person(SpatialObject<=Object) + Process,
                                                      Object owl:disjointWith Process -> DisjointnessClash
  control    : + a chr:Device type (SpatialObject) -> same SULO parent -> expect CONSISTENT (specificity)
  functional : + two sulo:hasValue literals        -> sulo:hasValue is owl:FunctionalProperty
                                                      -> FunctionalPropertyConflict

Output: evaluation/outputs/owl_fault_injection.json
"""
import sys, json, collections
sys.path.insert(0, "pipeline")
from pathlib import Path
from rdflib import Graph, RDF, URIRef, Literal, XSD
from owl_validator import validate_with_owl

CHR = "https://w3id.org/shexmap/resource/ontology-schema/d285f599-dc2e-4bd0-83f3-df21defa8821/"
SULO = "https://w3id.org/sulo/"
Person = URIRef(CHR + "Person")
MedProc = URIRef(CHR + "MedicalProcedure")
Device = URIRef(CHR + "Device")
hasValue = URIRef(SULO + "hasValue")
CHR_ONT = Path("evaluation/corpus/tbox/chr_ontology.owl.ttl")

N = 50
files = sorted(Path("evaluation/outputs/chr/schema/full/b").glob("vignette_*.ttl"))[:N]


def mutate(ttl, kind):
    g = Graph()
    try:
        g.parse(data=ttl, format="turtle")
    except Exception:
        return None
    persons = list(g.subjects(RDF.type, Person))
    if not persons:
        return None
    n = persons[0]
    if kind == "disjoint":
        g.add((n, RDF.type, MedProc))
    elif kind == "control":
        g.add((n, RDF.type, Device))
    elif kind == "functional":
        g.add((n, hasValue, Literal("1.0", datatype=XSD.float)))
        g.add((n, hasValue, Literal("2.0", datatype=XSD.float)))
    return g.serialize(format="turtle")


res = {k: {"n": 0, "flagged": 0, "by_type": collections.Counter()}
       for k in ("orig", "disjoint", "control", "functional")}

for f in files:
    ttl = f.read_text()
    for kind in res:
        mut = ttl if kind == "orig" else mutate(ttl, kind)
        if mut is None:
            continue
        ok, rep = validate_with_owl(mut, CHR_ONT)
        if "parse error" in rep.lower() or "closure error" in rep.lower():
            continue
        res[kind]["n"] += 1
        if not ok:
            res[kind]["flagged"] += 1
            for et in ("DisjointnessClash", "FunctionalPropertyConflict", "RangeViolation"):
                if et in rep:
                    res[kind]["by_type"][et] += 1

print("=== FAULT-INJECTION: OWL reasoner detection on the (flat) schema ABox ===\n")
out = {}
for kind in ("orig", "disjoint", "control", "functional"):
    r = res[kind]
    if r["n"] == 0:
        continue
    rate = 100 * r["flagged"] / r["n"]
    out[kind] = {"n": r["n"], "flagged": r["flagged"], "rate_pct": round(rate, 1),
                 "by_type": dict(r["by_type"])}
    print(f"  {kind:11}: flagged {r['flagged']:>3}/{r['n']:<3} ({rate:5.1f}%)   types={dict(r['by_type'])}")

print("\nInterpretation:")
print("  orig       low  = few false positives on clean data")
print("  disjoint   ~100 = reliably detects injected Object/Process contradictions")
print("  control    ~0   = specificity (same-SULO-parent double typing is NOT flagged)")
print("  functional ~100 = reliably detects injected functional-property conflicts")

Path("evaluation/outputs/owl_fault_injection.json").write_text(json.dumps(out, indent=2))
print("\n[OK] wrote evaluation/outputs/owl_fault_injection.json")
