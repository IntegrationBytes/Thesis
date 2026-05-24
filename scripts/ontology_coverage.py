"""CHR ontology coverage stats — what fraction of the ontology is actually used?

Counts instances of each CHR class and uses of each CHR property in the
n=200 gold corpus. Helps frame the evaluation scope: "we cover X of Y
ontology classes" is a defensible scope statement.

Output: stdout table + JSON dump at
``evaluation/outputs/ontology_coverage.json``.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import rdflib

ROOT = Path(__file__).resolve().parent.parent
GOLD_DIR = ROOT / "evaluation/corpus/abox_gold"
SCHEMA_TBOX = ROOT / "evaluation/corpus/tbox/chr_schema.ttl"
ONT_TBOX = ROOT / "evaluation/corpus/tbox/chr_ontology.owl.ttl"
SHAPES_SCHEMA = ROOT / "evaluation/corpus/shapes/chr_shacl_schema.ttl"


def _load_tbox_classes_properties(tbox_path: Path) -> tuple[set, set]:
    """Return (classes, properties) declared in the TBox."""
    g = rdflib.Graph()
    g.parse(tbox_path.as_posix(), format="turtle")
    classes = set()
    properties = set()
    for c in g.subjects(rdflib.RDF.type, rdflib.OWL.Class):
        if isinstance(c, rdflib.URIRef):
            classes.add(str(c))
    for c in g.subjects(rdflib.RDF.type, rdflib.RDFS.Class):
        if isinstance(c, rdflib.URIRef):
            classes.add(str(c))
    for p in g.subjects(rdflib.RDF.type, rdflib.OWL.ObjectProperty):
        if isinstance(p, rdflib.URIRef):
            properties.add(str(p))
    for p in g.subjects(rdflib.RDF.type, rdflib.OWL.DatatypeProperty):
        if isinstance(p, rdflib.URIRef):
            properties.add(str(p))
    return classes, properties


def main() -> None:
    print("=== TBox declarations ===")
    schema_classes, schema_props = _load_tbox_classes_properties(SCHEMA_TBOX)
    ont_classes, ont_props = _load_tbox_classes_properties(ONT_TBOX)
    print(f"  Schema TBox: {len(schema_classes)} classes, {len(schema_props)} properties")
    print(f"  Ontology TBox: {len(ont_classes)} classes, {len(ont_props)} properties")

    # Count usage in gold
    print("\n=== Counting usage in n=200 gold corpus...")
    class_counter: Counter = Counter()
    property_counter: Counter = Counter()
    for gold in sorted(GOLD_DIR.glob("vignette_*_gold_schema.ttl")):
        g = rdflib.Graph()
        try:
            g.parse(gold.as_posix(), format="turtle")
        except Exception:
            continue
        for s, p, o in g:
            if p == rdflib.RDF.type and isinstance(o, rdflib.URIRef):
                class_counter[str(o)] += 1
            if isinstance(p, rdflib.URIRef):
                property_counter[str(p)] += 1
    print(f"  Done")

    # Schema-track coverage
    print("\n=== Schema-track class coverage (used / declared) ===")
    used_schema_classes = schema_classes & set(class_counter)
    print(f"  Used: {len(used_schema_classes)} / declared: {len(schema_classes)}")
    rows = [(c, class_counter[c]) for c in used_schema_classes]
    rows.sort(key=lambda x: -x[1])
    for cls, n in rows[:15]:
        short = cls.rsplit("/", 1)[-1]
        print(f"    {short:<35} {n:>6} instances")
    unused = schema_classes - set(class_counter)
    if unused:
        print(f"\n  Declared but UNUSED in gold ({len(unused)}):")
        for c in sorted(unused):
            short = c.rsplit("/", 1)[-1]
            print(f"    {short}")

    # Schema-track property coverage
    print("\n=== Schema-track property coverage (used / declared) ===")
    used_schema_props = schema_props & set(property_counter)
    print(f"  Used: {len(used_schema_props)} / declared: {len(schema_props)}")
    rows = [(p, property_counter[p]) for p in used_schema_props]
    rows.sort(key=lambda x: -x[1])
    for p, n in rows[:15]:
        short = p.rsplit("/", 1)[-1]
        print(f"    {short:<35} {n:>6} uses")
    unused = schema_props - set(property_counter)
    if unused:
        print(f"\n  Declared but UNUSED in gold ({len(unused)}):")
        for p in sorted(unused):
            short = p.rsplit("/", 1)[-1]
            print(f"    {short}")

    # Dump JSON
    out = {
        "schema_tbox": {
            "n_classes_declared": len(schema_classes),
            "n_classes_used": len(used_schema_classes),
            "n_properties_declared": len(schema_props),
            "n_properties_used": len(used_schema_props),
            "class_instance_counts": {k: class_counter[k] for k in used_schema_classes},
            "property_use_counts": {k: property_counter[k] for k in used_schema_props},
            "unused_classes": sorted(schema_classes - set(class_counter)),
            "unused_properties": sorted(schema_props - set(property_counter)),
        },
        "ontology_tbox": {
            "n_classes_declared": len(ont_classes),
            "n_properties_declared": len(ont_props),
        },
    }
    out_path = ROOT / "evaluation/outputs/ontology_coverage.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[OK] dumped {out_path}")


if __name__ == "__main__":
    main()
