"""FHIR-grounded IRI normaliser for triple-level F1 across the full corpus.

Problem
-------
The FHIR-to-CHR converter mints UUID-derived IRIs:
    ex:patient_5e117ba0_820d_777a_7d5f_eb972d065bfa

The LLM, extracting from the rendered text, mints text-span IRIs:
    ex:patient_Angelic_Callie

Both denote the same Person in the same FHIR Bundle. Exact-triple F1 fails
because the IRIs differ even though the semantic content is identical.

Approach
--------
Build a bijection ``LLM_iri -> gold_iri`` per vignette, grounded in the
semantic content FHIR already carries:

1. ``rdfs:label`` exact match (case-insensitive). Gold entities derived from
   FHIR resources carry the resource's name/display as ``rdfs:label`` — the
   strongest semantic key.

2. Literal-value match. For Measurements, match by ``chr:hasQuantityValue``
   (e.g., ``37.816``). For dated processes, match by ``chr:hasMeasuredDate /
   hasRecordDate / hasPerformedDate`` (date prefix).

3. IRI local-name exact match. Shared conventions like ``ex:status_completed``
   are emitted identically by the converter and the LLM.

4. Token overlap in IRI local names. For entities without explicit labels,
   tokenise the LLM's IRI fragment and the gold's label and match by token
   overlap (Jaccard) above a threshold.

Then rewrite the LLM graph with gold IRIs where a match exists. Unmatched
LLM entities keep their original IRIs and contribute to false positives in F1
(which is the right behaviour — they're entities the LLM emitted that the
gold doesn't have).

Defending the approach
----------------------
Every matching key is grounded in semantic content the FHIR source emits:
labels come from FHIR resource displays; values come from FHIR Observation
values; status local-names come from FHIR Resource.status. The normaliser
recognises equivalence that already exists in the FHIR semantics — it does
not invent it.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

from rdflib import Graph, Namespace, RDF, RDFS, URIRef, Literal, BNode

CHR = Namespace("https://w3id.org/shexmap/resource/ontology-schema/d285f599-dc2e-4bd0-83f3-df21defa8821/")
EX_PREFIX = "http://example.org/clinical/"

# Per-type value-bearing properties used as additional matching keys.
# All keys are FHIR-derived (the converter only emits them from a FHIR source).
_VALUE_KEYS: dict[str, tuple[str, ...]] = {
    str(CHR.Measurement):              ("hasQuantityValue", "hasMeasuredDate"),
    str(CHR.MeasurementProcess):       ("hasPerformedDate",),
    str(CHR.EvaluationProcess):        ("hasPerformedDate",),
    str(CHR.MedicalProcedure):         ("hasPerformedDate",),
    str(CHR.MedicationAdministration): ("hasPerformedDate",),
    str(CHR.ClinicalCondition):        ("hasRecordDate", "hasConditionEndDate"),
    str(CHR.ClinicalVisit):            (),  # match purely structurally / via patient label
}

# A few "stop words" to drop from IRI fragments before token matching.
_STOP = {"patient", "person", "visit", "physician", "doctor", "nurse",
         "provider", "device", "unit", "careunit", "measurement", "meas",
         "measurementprocess", "measproc", "evaluation", "evalproc", "diag",
         "condition", "cond", "treatment", "treatmentplan", "trt",
         "pharmaceutical", "drug", "medication", "med", "status", "ex"}

_WORD_RE = re.compile(r"[A-Za-z]+|[0-9]+")  # alpha and numeric tokens separately


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------
def _iri_local(uri: str) -> str:
    """Extract the local name after the EX_PREFIX, or after the last / or #."""
    if uri.startswith(EX_PREFIX):
        return uri[len(EX_PREFIX):]
    for sep in ("#", "/"):
        idx = uri.rfind(sep)
        if idx >= 0:
            return uri[idx + 1:]
    return uri


def _tokens(s: str) -> set[str]:
    """Lowercase alpha+digit tokens, with stop-words removed.

    Splits on underscore/CamelCase/non-alnum so
    ``measurement_Temp_20160807_0624`` -> {temp, 20160807, 0624}.
    """
    # Split CamelCase: "BostonMedicalCenter" -> "Boston Medical Center"
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    toks = {t.lower() for t in _WORD_RE.findall(s)}
    return toks - _STOP


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Entity-key extraction
# ---------------------------------------------------------------------------
def _entities_by_type(g: Graph) -> dict[str, list[URIRef]]:
    """Group ex: subjects by their rdf:type."""
    out: dict[str, list[URIRef]] = defaultdict(list)
    for s in set(g.subjects()):
        if isinstance(s, BNode):
            continue
        if not str(s).startswith(EX_PREFIX):
            continue
        for t in g.objects(s, RDF.type):
            if isinstance(t, URIRef):
                out[str(t)].append(s)
    return out


def _entity_keys(g: Graph, e: URIRef, type_uri: str) -> dict[str, object]:
    """Pull every key that's useful for matching this entity.

    Returns a dict with:
      label: str | None     (lowercased rdfs:label)
      local_name: str       (IRI fragment after EX_PREFIX)
      local_tokens: set[str]
      label_tokens: set[str]
      values: dict[predicate_localname, str]  (literal values, date prefixes)
    """
    label_lit = next(g.objects(e, RDFS.label), None)
    label = str(label_lit).lower().strip() if label_lit else None
    local = _iri_local(str(e))

    label_tokens = _tokens(label) if label else set()
    local_tokens = _tokens(local)

    values: dict[str, str] = {}
    for pred_local in _VALUE_KEYS.get(type_uri, ()):
        pred_uri = URIRef(str(CHR) + pred_local)
        v = next(g.objects(e, pred_uri), None)
        if v is None:
            continue
        v_str = str(v).strip()
        # Normalise dates to the date-only prefix so 09:24:27 vs 09:24:00 doesn't break the match.
        if pred_local.endswith("Date") and "T" in v_str:
            v_str = v_str.split("T")[0]
        values[pred_local] = v_str

    return {
        "label": label,
        "local_name": local,
        "label_tokens": label_tokens,
        "local_tokens": local_tokens,
        "values": values,
    }


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
def _score(llm_keys: dict, gold_keys: dict) -> float:
    """Higher = better match. Returns a score in [0, 1+] (capped).

    Order of evidence:
      +1.0 — local_name exact match (for shared conventions like status_completed)
      +1.0 — any value key matches (e.g., hasQuantityValue)
      +0.9 — label exact match (LLM rarely produces this — bonus when it does)
      +score in [0, 0.8] — Jaccard token overlap (LLM local-name tokens vs. gold label tokens)
    """
    score = 0.0

    # Local-name exact match — shared convention bonus
    if llm_keys["local_name"] == gold_keys["local_name"]:
        score += 1.0

    # Value key match — at least one shared value
    llm_vals = llm_keys["values"]
    gold_vals = gold_keys["values"]
    shared_keys = set(llm_vals) & set(gold_vals)
    for k in shared_keys:
        if llm_vals[k] == gold_vals[k]:
            score += 1.0
            break  # one value match is enough

    # Label vs label exact match
    if llm_keys["label"] and gold_keys["label"]:
        if llm_keys["label"] == gold_keys["label"]:
            score += 0.9

    # Jaccard between LLM IRI tokens and gold label tokens
    j = _jaccard(llm_keys["local_tokens"], gold_keys["label_tokens"])
    if j > 0:
        score += min(j, 0.8)

    return score


def _greedy_bijection(scores: dict[tuple[URIRef, URIRef], float],
                      llm_ents: list[URIRef],
                      gold_ents: list[URIRef],
                      min_score: float = 0.3) -> dict[URIRef, URIRef]:
    """Greedy bijective matching: pick highest-scoring pair, remove both, repeat."""
    mapping: dict[URIRef, URIRef] = {}
    remaining_pairs = [(p, s) for p, s in scores.items() if s >= min_score]
    remaining_pairs.sort(key=lambda kv: kv[1], reverse=True)
    used_llm: set[URIRef] = set()
    used_gold: set[URIRef] = set()
    for (llm_e, gold_e), _ in remaining_pairs:
        if llm_e in used_llm or gold_e in used_gold:
            continue
        mapping[llm_e] = gold_e
        used_llm.add(llm_e)
        used_gold.add(gold_e)
    return mapping


def _structural_score(llm_graph: Graph, gold_graph: Graph,
                       le: URIRef, ge: URIRef,
                       current_map: dict[URIRef, URIRef]) -> float:
    """Score how many already-matched neighbours these two entities share.

    For each (predicate, object) where the object is also a URI that's already
    been mapped via current_map, check if the gold has the same (predicate,
    mapped_object). One point per match. Symmetric for inverse (when le is
    object).
    """
    score = 0.0
    # Outgoing edges
    for p, o in llm_graph.predicate_objects(le):
        if isinstance(o, URIRef) and o in current_map:
            if (ge, p, current_map[o]) in gold_graph:
                score += 1.0
    # Incoming edges
    for s, p in llm_graph.subject_predicates(le):
        if isinstance(s, URIRef) and s in current_map:
            if (current_map[s], p, ge) in gold_graph:
                score += 1.0
    return score


def build_iri_map(llm_graph: Graph, gold_graph: Graph) -> dict[URIRef, URIRef]:
    """Build LLM_iri -> gold_iri mapping using per-type matching.

    Two passes:
      Pass 1 — direct semantic-key matching (labels, literal values, IRI tokens).
      Pass 2 — structural matching using already-matched neighbours (anchored
               by Pass 1) for entities like Visit / Process that lack labels
               or distinguishing literal values on their own.
    """
    gold_by_type = _entities_by_type(gold_graph)
    llm_by_type = _entities_by_type(llm_graph)

    full_map: dict[URIRef, URIRef] = {}

    # Pass 1: direct semantic-key matching
    for type_uri, llm_ents in llm_by_type.items():
        gold_ents = gold_by_type.get(type_uri, [])
        if not gold_ents:
            continue

        scores: dict[tuple[URIRef, URIRef], float] = {}
        for le in llm_ents:
            lkeys = _entity_keys(llm_graph, le, type_uri)
            for ge in gold_ents:
                gkeys = _entity_keys(gold_graph, ge, type_uri)
                scores[(le, ge)] = _score(lkeys, gkeys)

        per_type_map = _greedy_bijection(scores, llm_ents, gold_ents)
        full_map.update(per_type_map)

    # Pass 2: structural matching for entities not yet mapped, leveraging Pass 1's anchors
    # Iterate to a fixed point (matches in pass 2 can enable further matches)
    for _iteration in range(3):  # usually converges in 1-2 iterations
        added_this_round = 0
        for type_uri, llm_ents in llm_by_type.items():
            gold_ents = gold_by_type.get(type_uri, [])
            unmatched_llm = [e for e in llm_ents if e not in full_map]
            mapped_gold_vals = set(full_map.values())
            unmatched_gold = [e for e in gold_ents if e not in mapped_gold_vals]
            if not unmatched_llm or not unmatched_gold:
                continue

            # Compute structural scores against unmatched candidates
            scores: dict[tuple[URIRef, URIRef], float] = {}
            for le in unmatched_llm:
                lkeys = _entity_keys(llm_graph, le, type_uri)
                for ge in unmatched_gold:
                    gkeys = _entity_keys(gold_graph, ge, type_uri)
                    base = _score(lkeys, gkeys)
                    struct = _structural_score(llm_graph, gold_graph, le, ge, full_map)
                    scores[(le, ge)] = base + struct

            per_type_map = _greedy_bijection(scores, unmatched_llm,
                                              unmatched_gold, min_score=1.0)
            full_map.update(per_type_map)
            added_this_round += len(per_type_map)

        if added_this_round == 0:
            break

    return full_map


# ---------------------------------------------------------------------------
# Graph rewriting
# ---------------------------------------------------------------------------
def rewrite_iris(llm_graph: Graph, iri_map: dict[URIRef, URIRef]) -> Graph:
    """Return a new Graph where every matched LLM IRI is replaced with its
    gold counterpart. Unmatched IRIs are preserved as-is.
    """
    if not iri_map:
        return llm_graph  # nothing to do

    out = Graph()
    for prefix, namespace in llm_graph.namespaces():
        out.bind(prefix, namespace)

    for s, p, o in llm_graph:
        new_s = iri_map.get(s, s) if isinstance(s, URIRef) else s
        new_o = iri_map.get(o, o) if isinstance(o, URIRef) else o
        out.add((new_s, p, new_o))

    return out


def normalize_llm_to_gold(llm_graph: Graph, gold_graph: Graph) -> tuple[Graph, dict[URIRef, URIRef]]:
    """One-shot helper: build map, rewrite graph, return both."""
    iri_map = build_iri_map(llm_graph, gold_graph)
    rewritten = rewrite_iris(llm_graph, iri_map)
    return rewritten, iri_map


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import sys
    if len(sys.argv) != 3:
        print("Usage: python iri_normalizer.py <llm.ttl> <gold.ttl>")
        sys.exit(1)
    llm_g = Graph().parse(sys.argv[1], format="turtle")
    gold_g = Graph().parse(sys.argv[2], format="turtle")
    rewritten, m = normalize_llm_to_gold(llm_g, gold_g)
    print(f"Mapped {len(m)} LLM entities to gold counterparts:")
    for llm_iri, gold_iri in m.items():
        print(f"  {_iri_local(str(llm_iri))} -> {_iri_local(str(gold_iri))}")
    # Compute F1 with and without normalisation
    def trips(g):
        return {(str(s), str(p), str(o)) for s, p, o in g if str(p) != str(RDF.type)}
    raw = trips(llm_g)
    norm = trips(rewritten)
    gold = trips(gold_g)
    def f1(pred, gold):
        if not pred:
            return 0.0
        tp = len(pred & gold)
        p = tp / len(pred) if pred else 0.0
        r = tp / len(gold) if gold else 0.0
        return 2 * p * r / (p + r) if (p + r) else 0.0
    print(f"\nF1 raw:        {f1(raw, gold):.3f}")
    print(f"F1 normalised: {f1(norm, gold):.3f}")
