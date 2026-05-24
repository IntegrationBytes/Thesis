"""OWL-reasoner-based validation for the SULO ontology track.

This module replaces the hand-written SHACL shapes
(``chr_shacl_ontology.ttl``) as the primary validator on the ontology
track. The supervisor's argument was that the SHACL was too narrow:
it captured only what we explicitly wrote, while the SULO ontology
itself carries richer constraints (class disjointness, functional
properties, transitive closures) that an OWL reasoner can enforce
directly.

Approach
--------
1. Merge (data graph, full SULO ontology, CHR ontology import).
2. Apply OWL-RL deductive closure (materialises subClassOf, inverseOf,
   transitive, equivalent-class entailments).
3. Run three explicit consistency checks against the closed graph:
     a. **Disjointness clashes** — any subject typed as both members of
        an ``owl:disjointWith`` pair.
     b. **Functional-property conflicts** — any subject with two
        distinct values for a property declared
        ``owl:FunctionalProperty``.
     c. **Range violations** — any (s p o) where ``rdfs:range`` declares
        a type for ``o`` and the closure does not contain that type
        assertion. (Useful as a "soft" constraint; the closure expands
        the range automatically when the LLM provides enough context.)

Violations are returned as a human-readable text block in the same
shape that ``validate_with_shacl`` returns, so callers (the System B
retry loop, evaluate.py) can swap one for the other without code
changes elsewhere.

The full SULO ontology is fetched once on import and cached on disk at
``evaluation/corpus/tbox/sulo_fetched.ttl`` so re-runs are reproducible.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

from rdflib import Graph, OWL, RDF, RDFS, URIRef

try:
    import owlrl
    _HAS_OWLRL = True
except ImportError:
    _HAS_OWLRL = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SULO_URL = "https://w3id.org/sulo/"
SULO_CACHE = PROJECT_ROOT / "evaluation" / "corpus" / "tbox" / "sulo_fetched.ttl"


# ---------------------------------------------------------------------------
# SULO bootstrap.
# ---------------------------------------------------------------------------
def _ensure_sulo() -> Graph:
    """Load SULO from disk cache, fetching from the web if missing."""
    if not SULO_CACHE.exists():
        print(f"[owl_validator] fetching SULO from {SULO_URL} ...")
        req = urllib.request.Request(
            SULO_URL, headers={"Accept": "text/turtle"}
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            data = r.read().decode()
        SULO_CACHE.parent.mkdir(parents=True, exist_ok=True)
        SULO_CACHE.write_text(data)
        print(f"[owl_validator] cached {len(data)} bytes -> {SULO_CACHE}")
    g = Graph()
    g.parse(SULO_CACHE.as_posix(), format="turtle")
    return g


_SULO_GRAPH: Graph | None = None


def _get_sulo() -> Graph:
    """Cached SULO graph singleton."""
    global _SULO_GRAPH
    if _SULO_GRAPH is None:
        _SULO_GRAPH = _ensure_sulo()
    return _SULO_GRAPH


# ---------------------------------------------------------------------------
# Closure + checks.
# ---------------------------------------------------------------------------
def _materialise(data: Graph, ontology: Graph) -> Graph:
    """Apply OWL-RL closure to ``data`` after merging ``ontology`` axioms.

    Returns the same ``data`` object (mutated in place) for convenience.
    """
    if not _HAS_OWLRL:
        return data
    sulo = _get_sulo()
    for triple in ontology:
        data.add(triple)
    for triple in sulo:
        data.add(triple)
    owlrl.DeductiveClosure(
        owlrl.OWLRL_Semantics,
        rdfs_closure=True,
        axiomatic_triples=False,
        datatype_axioms=False,
    ).expand(data)
    return data


def _find_disjoint_clashes(g: Graph) -> list[str]:
    """Return human-readable messages for every owl:disjointWith clash."""
    msgs: list[str] = []
    # Collect all disjoint pairs (symmetric)
    disjoint_pairs: set[tuple[URIRef, URIRef]] = set()
    for c1, _, c2 in g.triples((None, OWL.disjointWith, None)):
        if isinstance(c1, URIRef) and isinstance(c2, URIRef):
            disjoint_pairs.add((c1, c2))
            disjoint_pairs.add((c2, c1))

    # For each subject, get its named types, then check pairwise disjointness
    subject_types: dict[URIRef, set[URIRef]] = {}
    for s, _, o in g.triples((None, RDF.type, None)):
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            subject_types.setdefault(s, set()).add(o)

    for s, types in subject_types.items():
        for t1 in types:
            for t2 in types:
                if t1 == t2:
                    continue
                if (t1, t2) in disjoint_pairs:
                    msg = (
                        f"DisjointnessClash on {s}: asserted as both "
                        f"<{t1}> and <{t2}>, which are owl:disjointWith"
                    )
                    if msg not in msgs:
                        msgs.append(msg)
    return msgs


def _find_functional_property_conflicts(g: Graph) -> list[str]:
    """Subjects with >1 distinct value for an owl:FunctionalProperty."""
    msgs: list[str] = []
    functional_props = {
        p for p in g.subjects(RDF.type, OWL.FunctionalProperty)
        if isinstance(p, URIRef)
    }
    for p in functional_props:
        # Group by subject
        by_subject: dict[URIRef, set] = {}
        for s, _, o in g.triples((None, p, None)):
            if isinstance(s, URIRef):
                by_subject.setdefault(s, set()).add(o)
        for s, values in by_subject.items():
            if len(values) > 1:
                vals_str = ", ".join(str(v) for v in list(values)[:3])
                msgs.append(
                    f"FunctionalPropertyConflict on {s} via <{p}>: "
                    f"{len(values)} distinct values ({vals_str})"
                )
    return msgs


def _find_range_violations(g: Graph) -> list[str]:
    """Triples whose object lacks a declared rdfs:range type.

    Returns at most 20 entries to keep the prompt manageable. Only
    fires when the range type is a named class (not a literal datatype
    or a blank-node restriction).
    """
    msgs: list[str] = []
    # Collect property -> set of range types (named only)
    range_of: dict[URIRef, set[URIRef]] = {}
    for p, _, r in g.triples((None, RDFS.range, None)):
        if isinstance(p, URIRef) and isinstance(r, URIRef):
            range_of.setdefault(p, set()).add(r)

    for p, ranges in range_of.items():
        for s, _, o in g.triples((None, p, None)):
            if not isinstance(o, URIRef):
                continue  # literal — handled by datatype check separately
            # If closure already classified o as one of the range types, OK
            actual_types = set(g.objects(o, RDF.type))
            if actual_types & ranges:
                continue
            range_str = ", ".join(str(r).rsplit("/", 1)[-1] for r in ranges)
            msgs.append(
                f"RangeViolation: <{p}> expects ({range_str}), "
                f"but {o} has no such type (used by {s})"
            )
            if len(msgs) >= 20:
                return msgs
    return msgs


# ---------------------------------------------------------------------------
# Public API — same signature as validate_with_shacl.
# ---------------------------------------------------------------------------
def validate_with_owl(
    ttl_content: str,
    chr_ontology_path: Path,
) -> tuple[bool, str]:
    """Run OWL reasoning on ``ttl_content`` and return (consistent, report).

    Parameters
    ----------
    ttl_content
        The LLM-generated Turtle string to validate.
    chr_ontology_path
        Path to the CHR ontology TBox (chr_ontology.owl.ttl). SULO is
        fetched separately via _get_sulo.

    Returns
    -------
    (consistent, report_text)
        - ``consistent`` is True iff zero violations were found.
        - ``report_text`` is a human-readable block listing violations,
          formatted similarly to pyshacl's text report so the System B
          retry-loop prompt-builder can pass it back to the LLM
          unchanged.
    """
    try:
        data = Graph()
        data.parse(data=ttl_content, format="turtle")
    except Exception as parse_err:
        return False, f"Turtle parse error (not valid RDF): {parse_err}"

    try:
        ontology = Graph()
        ontology.parse(chr_ontology_path.as_posix(), format="turtle")
        _materialise(data, ontology)
    except Exception as exc:
        return False, f"OWL closure error: {exc}"

    disjoint = _find_disjoint_clashes(data)
    functional = _find_functional_property_conflicts(data)
    ranges = _find_range_violations(data)
    all_violations = disjoint + functional + ranges

    if not all_violations:
        return True, "Validation Report\nConforms: True\nResults (0):\n"

    lines = [
        "Validation Report",
        "Conforms: False",
        f"Results ({len(all_violations)}):",
    ]
    for i, msg in enumerate(all_violations, 1):
        lines.append(f"Constraint Violation in OWLReasoning ({i}):")
        lines.append(f"\tSeverity: sh:Violation")
        lines.append(f"\tMessage: {msg}")
        lines.append("")
    return False, "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI smoke test.
# ---------------------------------------------------------------------------
def _smoke() -> None:
    """Run validator on one ontology output to verify wiring."""
    sample = (
        PROJECT_ROOT
        / "evaluation/outputs/chr/ontology/ontology/b/vignette_001.ttl"
    )
    chr_ont = (
        PROJECT_ROOT / "evaluation/corpus/tbox/chr_ontology.owl.ttl"
    )
    if not sample.exists():
        print(f"sample not found: {sample}")
        return
    ttl = sample.read_text()
    ok, report = validate_with_owl(ttl, chr_ont)
    print(f"\nSample: {sample.name}")
    print(f"Consistent: {ok}")
    print(report[:1200])


if __name__ == "__main__":
    _smoke()
