"""Finer-grained, SHACL-derived precision / recall / F1 (not a binary pass-rate).

The thesis reports SHACL conformance as a binary graph-level pass-rate (does the
graph satisfy ALL hard shapes? yes/no). Per reviewer guidance, we add a graded,
entity-level structural metric derived from SHACL validation:

  Structural PRECISION = fraction of generated entities that are well-formed
                         (zero hard-shape violations).
  Structural RECALL    = fraction of gold (required) entities whose normalised
                         counterpart is present and well-formed.
  Structural F1        = harmonic mean.

Run per system (A/B) and stratum (general/complex), with the binary pass-rate
alongside for contrast.

Output: evaluation/outputs/shacl_structural_prf1.json
"""
import sys, json
sys.path.insert(0, "pipeline")
from pathlib import Path
from rdflib import Graph, RDF, URIRef, Namespace
import pyshacl
from iri_normalizer import normalize_llm_to_gold, EX_PREFIX

SH = Namespace("http://www.w3.org/ns/shacl#")
SHAPES = Graph().parse("evaluation/corpus/shapes/chr_shacl_schema.ttl", format="turtle")
ONT = Graph().parse("evaluation/corpus/tbox/chr_ontology.owl.ttl", format="turtle")
GOLD = Path("evaluation/corpus/abox_gold")


def ex_entities(g):
    return {s for s in set(g.subjects())
            if isinstance(s, URIRef) and str(s).startswith(EX_PREFIX)}


def validate(g):
    conforms, rg, _ = pyshacl.validate(
        g, shacl_graph=SHAPES, ont_graph=ONT, inference="none",
        abort_on_first=False, advanced=True, js=False, meta_shacl=False)
    viol = set()
    for res in rg.subjects(RDF.type, SH.ValidationResult):
        sev = next(rg.objects(res, SH.resultSeverity), None)
        if sev is None or str(sev).endswith("Violation"):   # hard violations only
            fn = next(rg.objects(res, SH.focusNode), None)
            if isinstance(fn, URIRef):
                viol.add(fn)
    return conforms, viol


def stratum(vid):
    return "general" if int(vid.split("_")[1]) <= 100 else "complex"


result = {}
for sysn in ("a", "b"):
    bdir = Path(f"evaluation/outputs/chr/schema/full/{sysn}")
    agg = {st: {"conf_gen": 0, "all_gen": 0, "conf_matched": 0, "all_gold": 0,
                "graphs": 0, "graphs_pass": 0} for st in ("general", "complex")}
    for gf in sorted(GOLD.glob("*_gold_schema.ttl")):
        vid = gf.name.replace("_gold_schema.ttl", "")
        lf = bdir / f"{vid}.ttl"
        if not lf.exists():
            continue
        try:
            llm = Graph().parse(lf.as_posix(), format="turtle")
            gold = Graph().parse(gf.as_posix(), format="turtle")
        except Exception:
            continue
        rew, _ = normalize_llm_to_gold(llm, gold)
        gen_ents, gold_ents = ex_entities(rew), ex_entities(gold)
        conforms, viol = validate(rew)
        conforming = gen_ents - viol
        st = stratum(vid); a = agg[st]
        a["conf_gen"] += len(conforming); a["all_gen"] += len(gen_ents)
        a["conf_matched"] += len(conforming & gold_ents); a["all_gold"] += len(gold_ents)
        a["graphs"] += 1; a["graphs_pass"] += int(bool(conforms))
    result[sysn] = {}
    for st in ("general", "complex"):
        a = agg[st]
        P = a["conf_gen"] / a["all_gen"] if a["all_gen"] else 0.0
        R = a["conf_matched"] / a["all_gold"] if a["all_gold"] else 0.0
        F = 2 * P * R / (P + R) if P + R else 0.0
        binrate = a["graphs_pass"] / a["graphs"] if a["graphs"] else 0.0
        result[sysn][st] = {"struct_P": round(P, 3), "struct_R": round(R, 3),
                            "struct_F1": round(F, 3), "binary_passrate": round(binrate, 3)}
        print(f"System {sysn.upper()} {st:8}: structural P={P:.3f} R={R:.3f} F1={F:.3f}   "
              f"(binary pass-rate {binrate:.0%})")

Path("evaluation/outputs/shacl_structural_prf1.json").write_text(json.dumps(result, indent=2))
print("\n[OK] wrote evaluation/outputs/shacl_structural_prf1.json")
print("Note: graded structural F1 shows the loop's effect on a continuous scale,")
print("      far richer than the all-or-nothing binary pass-rate.")
