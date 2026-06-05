"""Schema-agnostic evaluator for the A/B/C ablation.

Reads extraction outputs from
``evaluation/outputs/<schema>/<track>/<prompt>/<system>/<vignette>.ttl``,
compares them to corpus gold at
``evaluation/corpus/abox_gold/<vignette>_gold_<track>.ttl``, and reports:

    - Triple-level Precision / Recall / F1 (exact (s,p,o) match,
      excluding rdf:type so type assertions don't dominate the score).
    - Ontology Conformance (OC): fraction of predicates that are declared
      in the schema's TBox or referenced by the SHACL shapes. The "valid
      predicate" set is derived from the context — no hard-coded list.
    - Relation Hallucination (RH = 1 - OC).
    - SHACL conformance (did the graph pass pyshacl? separate from F1).

Outputs a terminal summary + JSON at::

    evaluation/outputs/<schema>/<track>/<prompt>/eval.json

Usage::

    python pipeline/evaluate.py --schema chr --track schema --prompt full
    python pipeline/evaluate.py --schema chr --track ontology --prompt full

"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from rdflib import Graph, Namespace, RDF, OWL, RDFS, URIRef

_PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_PIPELINE_DIR))
from prompts import SchemaContext, chr_context  # noqa: E402
from iri_normalizer import normalize_llm_to_gold  # noqa: E402

PROJECT_ROOT = _PIPELINE_DIR.parent
CORPUS_GOLD_DIR = PROJECT_ROOT / "evaluation" / "corpus" / "abox_gold"
CORPUS_VIGNETTES_DIR = PROJECT_ROOT / "evaluation" / "corpus" / "vignettes"
# Outputs root redirectable via env var so an alternate rerun's outputs
# can be evaluated without overwriting the main eval.json.
OUTPUTS_ROOT = Path(os.getenv("OUTPUTS_ROOT_OVERRIDE",
                              str(PROJECT_ROOT / "evaluation" / "outputs")))

SH = Namespace("http://www.w3.org/ns/shacl#")

# Namespace for instance IRIs. All canonicalization below applies only to
# URIs under this prefix — schema vocabulary (chr:, sulo:, rdf:, xsd:) is
# left exactly as declared so predicate URIs never drift.
EX_PREFIX = "http://example.org/clinical/"


def _canonicalize_iri(uri: str) -> str:
    """Normalize the local name of any ``ex:`` instance IRI.

    The gold annotator and the LLM emit the same entities with different
    casing (``ex:cond_Hypertension`` vs ``ex:cond_hypertension``), which
    makes strict triple-overlap F1 underreport semantic overlap by roughly
    3-5x on this corpus. Lowercasing the local name of every instance
    IRI is the minimum pre-pass that rescues those specific pairs without
    pretending to solve the wider "provider vs doctor" naming-convention
    mismatch (which is documented as a separate open item).

    Schema-vocabulary URIs (chr:, sulo:, rdfs:, rdf:, xsd:, owl:) pass
    through unchanged so predicate identity is never altered.
    """
    if uri.startswith(EX_PREFIX):
        local = uri[len(EX_PREFIX):]
        return EX_PREFIX + local.lower()
    return uri


import re as _re

_DATETIME_PREFIX_RE = _re.compile(
    r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2})"  # capture YYYY-MM-DDTHH:MM
    r"(?::\d{2}(?:\.\d+)?)?(?:[+-]\d{2}:\d{2}|Z)?"  # drop seconds + timezone
)

_UCUM_BASE = "https://biomedit.ch/rdf/sphn-resource/ucum/"
_SULO_UNIT_BASE = "https://w3id.org/sulo/Unit/"

_HAS_UNIT_PRED = "https://w3id.org/shexmap/resource/ontology-schema/d285f599-dc2e-4bd0-83f3-df21defa8821/hasUnit"
_HAS_CODE_PRED = "https://w3id.org/shexmap/resource/ontology-schema/d285f599-dc2e-4bd0-83f3-df21defa8821/hasCode"

# LOINC IRI variants the LLM sometimes emits → canonical https://loinc.org/{code}
_LOINC_VARIANT_RES = [
    _re.compile(r"^https?://loinc\.org/(?:id/)?(.+)$"),
]
# SNOMED IRI variants → canonical http://snomed.info/id/{code}
_SNOMED_VARIANT_RES = [
    _re.compile(r"^https?://snomed\.info/(?:id/|sct/)?(.+)$"),
]


def _normalize_terminology_iri(uri: str) -> str:
    """Normalise LOINC and SNOMED IRIs to their canonical forms.

    LOINC canonical:  https://loinc.org/{code}
    SNOMED canonical: http://snomed.info/id/{code}
    """
    for pattern in _LOINC_VARIANT_RES:
        m = pattern.match(uri)
        if m:
            return f"https://loinc.org/{m.group(1)}"
    for pattern in _SNOMED_VARIANT_RES:
        m = pattern.match(uri)
        if m:
            return f"http://snomed.info/id/{m.group(1)}"
    return uri


def _normalize_datetime(val: str) -> str:
    """Truncate an xsd:dateTime literal to minute granularity.

    ``"2016-08-07T06:24:27+02:00"`` and ``"2016-08-07T06:24:00"`` both
    become ``"2016-08-07T06:24"``, making LLM-truncated timestamps match
    gold FHIR timestamps that include seconds and timezone offset.
    """
    m = _DATETIME_PREFIX_RE.match(val)
    return m.group(1) if m else val


# UCUM character-map tokens to strip when normalising unit local names.
# Removing these (plus underscores) reduces both SPHN-encoded IRIs and
# LLM-slugged IRIs to the same bare alphanumeric key:
#   ucum:mmolperL  -> strip "per" -> "mmolL" -> lower -> "mmoll"
#   ex:unit_mmol_l -> strip "_"   -> "mmoll" -> lower -> "mmoll"  ✓
_UCUM_STRIP_TOKENS = sorted(
    ["per", "dot", "exp", "cbl", "cbr", "sbl", "sbr", "nb", "rbl", "rbr", "apo"],
    key=len, reverse=True,  # longest first to avoid partial replacements
)


_UCUM_RAW_SPECIAL = list("./%{}[]#'()*")  # raw UCUM chars stripped after URL-decoding


def _unit_bare_key(local: str) -> str:
    """Reduce a unit IRI local name to a bare alphanumeric key.

    Handles three encoding styles:
    - SPHN-encoded  (``mmolperL``)  — strip encoded tokens (``per``, ``dot``, …)
    - URL-encoded   (``kg%2Fm2``)   — URL-decode first, then strip raw chars
    - LLM-slugged   (``mmol_l``)    — strip underscores
    """
    from urllib.parse import unquote as _unquote
    s = _unquote(local).lower()  # URL-decode first
    if s.startswith("unit_"):
        s = s[len("unit_"):]
    s = s.replace("%", "percent")             # % is a unit symbol; map to word before stripping
    for token in _UCUM_STRIP_TOKENS:           # strip encoded tokens (per, dot, …)
        s = s.replace(token, "")
    for ch in _UCUM_RAW_SPECIAL:               # strip remaining raw UCUM special characters
        s = s.replace(ch, "")
    s = s.replace("_", "").replace("-", "")
    return "__unit__" + s


def _normalize_unit_iri(uri: str) -> str:
    """Normalise unit IRIs to a bare-alphanumeric key for comparison.

    Handles SPHN UCUM IRIs (``ucum:mmolperL``), sulo IRIs (``sulo:Unit/Cel``),
    and LLM-minted local IRIs (``ex:unit_mmol_l``) — all reduce to the same
    key (e.g. ``__unit__mmoll``) by stripping UCUM encoding tokens and
    underscores before lowercasing.
    """
    for base in (EX_PREFIX, _UCUM_BASE, _SULO_UNIT_BASE):
        if uri.startswith(base):
            return _unit_bare_key(uri[len(base):])
    return uri


def load_schema(name: str, track: str) -> SchemaContext:
    if name == "chr":
        return chr_context(track)
    raise ValueError(f"unknown schema {name!r}")


# ---------------------------------------------------------------------------
# Triple loading.
# ---------------------------------------------------------------------------
# We compare triples as (s, p, o) string tuples — not rdflib nodes —
# because the LLM may emit syntactically equivalent but textually distinct
# literals (e.g. "152.0"^^xsd:float vs "152"^^xsd:decimal). String-exact
# match is the strictest reasonable baseline.

def _triples(
    path: Path,
    exclude_type: bool = True,
    canonicalize: bool = True,
) -> set[tuple[str, str, str]] | None:
    """Parse a Turtle file -> set of (s, p, o) string tuples.

    Returns ``None`` if the file is missing or unparseable (the caller uses
    that to score the graph as zero-TP / zero-FP / full-FN). If
    ``exclude_type`` is True (default), rdf:type triples are dropped so the
    F1 score measures domain content, not taxonomy repetition.

    If ``canonicalize`` is True (default), instance-IRI local names are
    lowercased via :func:`_canonicalize_iri` before set construction. This
    fixes ``ex:cond_Hypertension`` vs ``ex:cond_hypertension`` drift in
    LLM output. Set ``canonicalize=False`` to recover the strict-casing
    behaviour (sanity check / regression).
    """
    if not path.exists():
        return None
    try:
        g = Graph()
        g.parse(path.as_posix(), format="turtle")
    except Exception:
        return None
    out: set[tuple[str, str, str]] = set()
    for s, p, o in g:
        if exclude_type and p == RDF.type:
            continue
        s_str, p_str, o_str = str(s), str(p), str(o)
        if canonicalize:
            s_str = _canonicalize_iri(s_str)
            p_str = _canonicalize_iri(p_str)
            o_str = _canonicalize_iri(o_str)
        o_str = _normalize_datetime(o_str)
        if p_str == _HAS_UNIT_PRED:
            o_str = _normalize_unit_iri(o_str)
        if p_str == _HAS_CODE_PRED:
            o_str = _normalize_terminology_iri(o_str)
        out.add((s_str, p_str, o_str))
    return out


def _all_triples(path: Path, canonicalize: bool = True) -> set[tuple[str, str, str]] | None:
    """Every triple, including rdf:type — used for OC/RH."""
    return _triples(path, exclude_type=False, canonicalize=canonicalize)


def _triples_normalized(
    gen_path: Path,
    gold_path: Path,
    exclude_type: bool = True,
    canonicalize: bool = True,
) -> set[tuple[str, str, str]] | None:
    """Parse LLM Turtle, normalize its IRIs against gold (FHIR-grounded), return triples.

    Uses the FHIR-grounded IRI normaliser from ``iri_normalizer.py`` to map
    LLM-minted text-span IRIs (e.g. ``ex:patient_JohnAndersson``) to gold
    UUID-derived IRIs (e.g. ``ex:patient_5e117ba0_...``) by matching on:
        - rdfs:label exact match
        - literal value equality (e.g. hasQuantityValue, hasMeasuredDate)
        - IRI token overlap (Jaccard)
        - structural anchoring via already-matched neighbours

    This recognises the semantic equivalence FHIR already encodes, enabling
    triple-level F1 across the full corpus.
    """
    if not gen_path.exists() or not gold_path.exists():
        return None
    try:
        llm_g = Graph()
        llm_g.parse(gen_path.as_posix(), format="turtle")
        gold_g = Graph()
        gold_g.parse(gold_path.as_posix(), format="turtle")
    except Exception:
        return None

    rewritten, _ = normalize_llm_to_gold(llm_g, gold_g)

    out: set[tuple[str, str, str]] = set()
    for s, p, o in rewritten:
        if exclude_type and p == RDF.type:
            continue
        s_str, p_str, o_str = str(s), str(p), str(o)
        if canonicalize:
            s_str = _canonicalize_iri(s_str)
            p_str = _canonicalize_iri(p_str)
            o_str = _canonicalize_iri(o_str)
        o_str = _normalize_datetime(o_str)
        if p_str == _HAS_UNIT_PRED:
            o_str = _normalize_unit_iri(o_str)
        out.add((s_str, p_str, o_str))
    return out


_UUIDISH_LOCAL_RE = re.compile(
    r"[0-9a-f]{8}[_-][0-9a-f]{4}[_-][0-9a-f]{4}",
    flags=re.IGNORECASE,
)


def _gold_f1_comparable(gold_all: set[tuple[str, str, str]] | None) -> bool:
    """Return whether exact-match F1 is meaningful for this gold graph.

    The manually-authored vignettes use human-readable ex: IRIs similar to
    the LLM output, so exact triple overlap is a fair signal. The larger
    Synthea-derived gold set uses UUID fragments from FHIR source IDs
    (e.g. ``ex:visit_5e117ba0_820d_777a_...``). The LLM sees the clinical
    note, not those UUIDs, so subject/object exact-match F1 is zero by
    construction even when the graph is semantically plausible. SHACL and PC
    remain valid for those vignettes because they do not depend on IRI overlap.
    """
    if gold_all is None:
        return False
    for s, _, o in gold_all:
        for node in (s, o):
            if node.startswith(EX_PREFIX):
                local = node[len(EX_PREFIX):]
                if _UUIDISH_LOCAL_RE.search(local):
                    return False
    return True


# ---------------------------------------------------------------------------
# Valid predicates — derived from TBox + SHACL.
# ---------------------------------------------------------------------------
def valid_predicates(ctx: SchemaContext) -> set[str]:
    """Union of all property URIs declared in TBox or referenced by SHACL.

    The TBox sweep picks up any ``rdf:Property``, ``owl:ObjectProperty``,
    or ``owl:DatatypeProperty`` subject. The SHACL sweep picks up any
    ``sh:path`` object that isn't a blank node (sequence paths are
    lists-of-URIs — we flatten those too).

    We add ``rdf:type``, ``rdfs:subClassOf``, ``rdfs:label``, and
    ``rdfs:comment`` by convention so taxonomy / annotation triples don't
    count as hallucinations.
    """
    preds: set[str] = {
        str(RDF.type),
        str(RDFS.subClassOf),
        str(RDFS.label),
        str(RDFS.comment),
    }

    tbox = Graph().parse(ctx.tbox_path.as_posix(), format="turtle")
    # Schema track's TBox uses rdfs:Property (non-standard but valid — the
    # 22 April supervisor file emits `a rdfs:Property`); ontology track
    # uses owl:ObjectProperty / owl:DatatypeProperty. Probe all four so
    # one evaluator handles both. RDFS.Property isn't exposed as an
    # attribute by rdflib's DefinedNamespace (standard is rdf:Property),
    # so construct its URI directly.
    property_types = (
        RDF.Property,
        URIRef("http://www.w3.org/2000/01/rdf-schema#Property"),
        OWL.ObjectProperty,
        OWL.DatatypeProperty,
    )
    for prop_type in property_types:
        for p in tbox.subjects(RDF.type, prop_type):
            preds.add(str(p))

    shapes = Graph().parse(ctx.shapes_path.as_posix(), format="turtle")
    for path in shapes.objects(None, SH.path):
        if isinstance(path, URIRef):
            preds.add(str(path))
        else:
            # Sequence path (RDF list). Walk it and pick up each URI step.
            preds.update(_flatten_rdf_list(shapes, path))

    return preds


def _flatten_rdf_list(g: Graph, head) -> set[str]:
    """Walk an RDF list and return URIRef members as strings (order not preserved)."""
    out: set[str] = set()
    node = head
    while node is not None and node != RDF.nil:
        first = next(g.objects(node, RDF.first), None)
        if isinstance(first, URIRef):
            out.add(str(first))
        node = next(g.objects(node, RDF.rest), None)
    return out


def _flatten_rdf_list_ordered(g: Graph, head) -> list[str]:
    """Walk an RDF list and return URIRef members in order (for sequence paths)."""
    out: list[str] = []
    node = head
    while node is not None and node != RDF.nil:
        first = next(g.objects(node, RDF.first), None)
        if isinstance(first, URIRef):
            out.append(str(first))
        node = next(g.objects(node, RDF.rest), None)
    return out


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------
def prf1(gold: set[tuple], pred: set[tuple] | None) -> dict:
    if pred is None:
        return {
            "tp": 0, "fp": 0, "fn": len(gold),
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "valid": False,
        }
    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": p, "recall": r, "f1": f,
        "valid": True,
    }


def oc(all_trips: set[tuple] | None, valid: set[str]) -> dict:
    """Ontology Conformance: fraction of triples whose predicate is declared."""
    if not all_trips:
        return {"oc": 1.0, "rh": 0.0, "conforming": 0, "total": 0}
    total = len(all_trips)
    conforming = sum(1 for _, p, _ in all_trips if p in valid)
    return {
        "oc": conforming / total,
        "rh": 1.0 - conforming / total,
        "conforming": conforming,
        "total": total,
    }


# ---------------------------------------------------------------------------
# Property Completeness (PC).
# ---------------------------------------------------------------------------
# Complements OC/SHACL conformance: a graph with zero triples has no
# violations yet contributes nothing. PC asks the opposite question —
# for each typed instance that *does* exist, what fraction of its
# SHACL-declared property slots are populated?
#
# A slot is a sh:property path declared on the class's NodeShape. Sequence
# paths (e.g. sulo:atTime → sulo:hasValue) are traversed hop-by-hop; a slot
# counts as covered if *any* value is reachable at the end of the path.
#
# The score is the mean slot-coverage rate over all typed instances in the
# generated graph that have at least one declared slot. Instances of classes
# with no declared sh:property (leaf classes like Person, Device) are skipped
# — they cannot be judged incomplete by this metric.
# ---------------------------------------------------------------------------

def _path_populated(g: Graph, start: URIRef, path: list[str]) -> bool:
    """Return True if any value is reachable from *start* by following *path*."""
    current: set = {start}
    for pred_uri in path:
        p = URIRef(pred_uri)
        nxt: set = set()
        for node in current:
            nxt.update(g.objects(node, p))
        if not nxt:
            return False
        current = nxt
    return True


def _shapes_to_slot_map(ctx: SchemaContext) -> dict[str, list[list[str]]]:
    """Parse SHACL shapes → {targetClass_URI: [[pred1, pred2, ...], ...]}

    Each inner list is one sh:property path (single-element for simple paths,
    multi-element for sequence paths). Classes with no declared sh:property
    are omitted entirely.
    """
    shapes_g = Graph().parse(ctx.shapes_path.as_posix(), format="turtle")
    result: dict[str, list[list[str]]] = {}
    for shape in shapes_g.subjects(RDF.type, SH.NodeShape):
        cls = next(shapes_g.objects(shape, SH.targetClass), None)
        if cls is None:
            continue
        paths: list[list[str]] = []
        for prop_bn in shapes_g.objects(shape, SH.property):
            sh_path = next(shapes_g.objects(prop_bn, SH.path), None)
            if sh_path is None:
                continue
            if isinstance(sh_path, URIRef):
                paths.append([str(sh_path)])
            else:
                seq = _flatten_rdf_list_ordered(shapes_g, sh_path)
                if seq:
                    paths.append(seq)
        if paths:
            result[str(cls)] = paths
    return result


def property_completeness(gen_path: Path, ctx: SchemaContext) -> dict:
    """Property Completeness: for typed instances, fraction of expected slots filled.

    Returns:
        pc            — mean slot-fill rate across all judged instances (None if no
                        instances of slot-bearing classes were found).
        total_slots   — sum of (expected paths × instances) across all classes.
        covered_slots — how many of those slots had at least one value.
        per_class     — {class_local_name: {"instances": n, "slots": t, "covered": c}}
    """
    empty = {
        "pc": None,
        "total_slots": 0,
        "covered_slots": 0,
        "per_class": {},
    }
    if not gen_path.exists():
        return empty
    try:
        gen_g = Graph()
        gen_g.parse(gen_path.as_posix(), format="turtle")
    except Exception:
        return empty

    slot_map = _shapes_to_slot_map(ctx)
    if not slot_map:
        return empty

    total_slots = 0
    covered_slots = 0
    per_class: dict[str, dict] = {}

    for instance, _, cls_node in gen_g.triples((None, RDF.type, None)):
        cls_uri = str(cls_node)
        if cls_uri not in slot_map:
            continue
        if not isinstance(instance, URIRef):
            continue
        paths = slot_map[cls_uri]
        local = cls_uri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
        entry = per_class.setdefault(local, {"instances": 0, "slots": 0, "covered": 0})
        entry["instances"] += 1
        for path in paths:
            total_slots += 1
            entry["slots"] += 1
            if _path_populated(gen_g, instance, path):
                covered_slots += 1
                entry["covered"] += 1

    if total_slots == 0:
        return empty
    return {
        "pc": covered_slots / total_slots,
        "total_slots": total_slots,
        "covered_slots": covered_slots,
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# Text2KGBench-aligned hallucination metrics (SH / OH).
# ---------------------------------------------------------------------------
# Definition (Mihindukulasooriya et al., ISWC 2023):
#   SH/OH check if the subject/object of a generated triple is present in
#   either the source sentence OR the ontology concepts. Stemming is used
#   to account for inflected forms.
#
# We follow the same intuition with a slightly more permissive matcher
# (lowercase + underscore-tokenize + suffix-strip) instead of full Porter
# stemming, because the LLM emits CamelCase / underscore_separated local
# names that need tokenization before any stemming would even apply. A
# hit on ANY token of the entity counts as "present" — this matches the
# semantic-overlap spirit of the original definition while staying robust
# to URI-naming drift.

_SUFFIXES = ("ies", "es", "ed", "ing", "ly", "s")


def _normalize_token(t: str) -> str:
    t = t.lower()
    for suf in _SUFFIXES:
        if len(t) > len(suf) + 2 and t.endswith(suf):
            return t[: -len(suf)]
    return t


def _tokenize(text: str) -> set[str]:
    """Lowercase + split on non-alphanumerics + light suffix-strip."""
    raw = re.split(r"[^a-zA-Z0-9]+", text)
    return {_normalize_token(t) for t in raw if len(t) >= 2}


def _entity_tokens(uri: str) -> set[str]:
    """Tokenize the local name of an instance URI. Schema URIs return empty
    so they don't dominate the match (they'd always match the ontology)."""
    if not uri.startswith(EX_PREFIX):
        return set()
    local = uri[len(EX_PREFIX):]
    return _tokenize(local)


def _vignette_text(vid: str) -> str:
    """Load the vignette text for ``vid`` (e.g. ``vignette_001``)."""
    path = CORPUS_VIGNETTES_DIR / f"{vid}.txt"
    return path.read_text() if path.exists() else ""


def _ontology_token_set(ctx: SchemaContext) -> set[str]:
    """Tokens harvested from every class and property local-name in the
    TBox plus every shape in SHACL. Cached per-context."""
    g = Graph().parse(ctx.tbox_path.as_posix(), format="turtle")
    g += Graph().parse(ctx.shapes_path.as_posix(), format="turtle")
    out: set[str] = set()
    for s in set(g.subjects()) | set(g.predicates()) | set(g.objects()):
        if isinstance(s, URIRef):
            uri = str(s)
            # Take the part after #, /, or :
            local = uri.rsplit("#", 1)[-1].rsplit("/", 1)[-1]
            out.update(_tokenize(local))
    return out


_ONT_CACHE: dict[str, set[str]] = {}


def hallucination(
    all_trips: set[tuple] | None,
    vignette_id: str,
    ctx: SchemaContext,
) -> dict:
    """Text2KGBench Subject/Object Hallucination metrics.

    SH/OH = (# triples whose subject/object tokens are NOT found in the
    source sentence OR ontology concept set) / (total triples).

    Lower is better. SH = OH = 0.0 means every entity in the graph traces
    back to either the input text or the schema vocabulary.
    """
    if not all_trips:
        return {"sh": 0.0, "oh": 0.0, "subj_halluc": 0, "obj_halluc": 0, "total": 0}

    text_tokens = _tokenize(_vignette_text(vignette_id))
    cache_key = str(ctx.tbox_path) + str(ctx.shapes_path)
    if cache_key not in _ONT_CACHE:
        _ONT_CACHE[cache_key] = _ontology_token_set(ctx)
    onto_tokens = _ONT_CACHE[cache_key]
    grounded = text_tokens | onto_tokens

    subj_h = 0
    obj_h = 0
    for s, _, o in all_trips:
        s_toks = _entity_tokens(s)
        if s_toks and not (s_toks & grounded):
            subj_h += 1
        # Object may be a literal or a schema URI (rdf:type's o is a class
        # URI which we don't ex:-tokenize). Treat literals as non-instance
        # entities — they're checked against the source text directly.
        if o.startswith(EX_PREFIX):
            o_toks = _entity_tokens(o)
            if o_toks and not (o_toks & grounded):
                obj_h += 1
        elif o.startswith(("http://", "https://")):
            # schema-vocabulary URI; never count as hallucination
            pass
        else:
            # literal — match its tokens against grounded set
            o_toks = _tokenize(o)
            if o_toks and not (o_toks & grounded):
                obj_h += 1

    total = len(all_trips)
    return {
        "sh": subj_h / total,
        "oh": obj_h / total,
        "subj_halluc": subj_h,
        "obj_halluc": obj_h,
        "total": total,
    }


# ---------------------------------------------------------------------------
# SHACL conformance — lightweight wrapper so we can report pass-rate per
# system without re-running extract.py.
# ---------------------------------------------------------------------------
def shacl_conforms(ttl_path: Path, ctx: SchemaContext) -> bool | None:
    """Return True/False, or None if the file is missing / unparseable.

    On the ontology track, applies OWL-RL closure to the data graph before
    SHACL validation so subclass / inverseOf entailments are visible to
    the shapes. The schema track does NOT use closure (cf. extract.py for
    rationale: chr:hasPatient's multi-domain declaration produces a flood
    of spurious cross-class violations under rdfs-inference).
    """
    if not ttl_path.exists():
        return None
    try:
        import pyshacl  # local import keeps import-time cost off eval startup
        data = Graph().parse(ttl_path.as_posix(), format="turtle")
        shapes = Graph().parse(ctx.shapes_path.as_posix(), format="turtle")
        ont = Graph().parse(ctx.tbox_path.as_posix(), format="turtle")
        if ctx.track == "ontology":
            try:
                import owlrl
                for s, p, o in ont:
                    data.add((s, p, o))
                owlrl.DeductiveClosure(
                    owlrl.OWLRL_Semantics,
                    rdfs_closure=True,
                    axiomatic_triples=False,
                    datatype_axioms=False,
                ).expand(data)
            except ImportError:
                pass  # owlrl missing — fall back to no-closure SHACL
        conforms, _, _ = pyshacl.validate(
            data, shacl_graph=shapes, ont_graph=ont,
            inference="none", abort_on_first=True,
            meta_shacl=False, advanced=True, js=False,
        )
        return bool(conforms)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Runner.
# ---------------------------------------------------------------------------
def _vignettes_with_gold(track: str) -> list[str]:
    """Return vignette IDs for which a gold-on-this-track exists."""
    pattern = f"*_gold_{track}.ttl"
    return sorted(
        p.name.replace(f"_gold_{track}.ttl", "")
        for p in CORPUS_GOLD_DIR.glob(pattern)
    )


def _system_dirs(schema: str, track: str, prompt: str) -> dict[str, Path]:
    base = OUTPUTS_ROOT / schema / track / prompt
    return {s: base / s for s in ("a", "b", "c")}


def evaluate(schema: str, track: str, prompt: str, canonicalize: bool = True) -> dict:
    ctx = load_schema(schema, track)
    preds = valid_predicates(ctx)
    print(f"\nEvaluate {schema}/{track}/{prompt}")
    print(f"  shapes:      {ctx.shapes_path.relative_to(PROJECT_ROOT)}")
    print(f"  tbox:        {ctx.tbox_path.relative_to(PROJECT_ROOT)}")
    print(f"  valid preds: {len(preds)} (TBox + SHACL union)")
    print(f"  IRI canon:   {'ON (ex:-local lowercased)' if canonicalize else 'OFF (strict casing)'}")

    vignette_ids = _vignettes_with_gold(track)
    if not vignette_ids:
        raise FileNotFoundError(
            f"no gold files matching *_gold_{track}.ttl in {CORPUS_GOLD_DIR}"
        )
    sys_dirs = _system_dirs(schema, track, prompt)
    active_systems = [s for s, d in sys_dirs.items() if d.exists() and any(d.glob("*.ttl"))]
    if not active_systems:
        raise FileNotFoundError(
            f"no system outputs at {OUTPUTS_ROOT / schema / track / prompt}"
        )
    print(f"  vignettes:   {len(vignette_ids)} ({', '.join(vignette_ids)})")
    print(f"  systems:     {','.join(active_systems)}")

    per_vig: list[dict] = []
    for vid in vignette_ids:
        gold_path = CORPUS_GOLD_DIR / f"{vid}_gold_{track}.ttl"
        gold_trips = _triples(gold_path, canonicalize=canonicalize)
        gold_all = _all_triples(gold_path, canonicalize=canonicalize)
        if gold_trips is None:
            print(f"  [WARN] could not parse gold {gold_path.name}, skipping")
            continue
        # IRI normalisation now handles UUID-bearing gold (Synthea-derived),
        # so every gold ABox is F1-comparable. The legacy _gold_f1_comparable
        # check is kept for reference but is overridden to True.
        row: dict = {
            "vignette": vid,
            "gold_triples": len(gold_trips),
            "f1_comparable": True,
            "f1_comparable_legacy": _gold_f1_comparable(gold_all),
        }
        for s in active_systems:
            gen_path = sys_dirs[s] / f"{vid}.ttl"
            present = gen_path.exists()
            # F1 triples: normalize LLM IRIs against gold via FHIR-grounded
            # matching (rdfs:label, literal values, IRI token overlap, structural
            # anchoring). Non-F1 metrics (OC/RH/SH/OH) keep the raw triples.
            gen_trips = _triples_normalized(gen_path, gold_path, canonicalize=canonicalize)
            gen_all = _all_triples(gen_path, canonicalize=canonicalize)
            m = prf1(gold_trips, gen_trips)
            o = oc(gen_all, preds)
            h = hallucination(gen_all, vid, ctx)
            conforms = shacl_conforms(gen_path, ctx)
            pc_result = property_completeness(gen_path, ctx)
            row[s] = {
                # `output_present` distinguishes "system did not run" from
                # "system ran and produced an empty graph". Stats and plots
                # filter on this so missing runs do not silently dilute
                # PC/SHACL means with zero values.
                "output_present": present,
                **m,
                "oc": o["oc"],
                "rh": o["rh"],
                "sh": h["sh"],
                "oh": h["oh"],
                "subj_halluc": h["subj_halluc"],
                "obj_halluc": h["obj_halluc"],
                "total_triples": o["total"],
                "shacl_conforms": conforms,
                "pc": pc_result["pc"],
                "pc_slots": pc_result["total_slots"],
                "pc_covered": pc_result["covered_slots"],
                "pc_per_class": pc_result["per_class"],
            }
        per_vig.append(row)

    # ------------------------------------------------------------------ TABLE
    hdr = (
        f"{'Vignette':<14} {'Sys':>3}  "
        f"{'P':>5} {'R':>5} {'F1':>5}  "
        f"{'OC':>5} {'RH':>5} {'SH':>5} {'OH':>5}  "
        f"{'PC':>5} {'Slots':>5}  "
        f"{'Tn':>3} {'SHACL':>7}"
    )
    sep = "-" * len(hdr)
    print(f"\n{sep}\n{hdr}\n{sep}")
    for row in per_vig:
        for s in active_systems:
            m = row[s]
            conforms_mark = (
                " -" if m["shacl_conforms"] is None
                else "  OK" if m["shacl_conforms"]
                else "FAIL"
            )
            pc_val = m.get("pc")
            pc_str = f"{pc_val:.2f}" if pc_val is not None else "  -  "
            print(
                f"{row['vignette']:<14} {s:>3}  "
                f"{m['precision']:.2f}  {m['recall']:.2f}  {m['f1']:.2f}  "
                f"{m['oc']:.2f}  {m['rh']:.2f}  {m['sh']:.2f}  {m['oh']:.2f}  "
                f"{pc_str:>5} {m.get('pc_slots', 0):>5}  "
                f"{m['total_triples']:>3} {conforms_mark:>7}"
            )
    print(sep)

    # -------------------------------------------------------------- AGGREGATE
    # F1 is averaged only over vignettes with f1_comparable=true (the manually
    # annotated subset whose gold IRIs match the LLM's naming convention).
    # OC / SH / OH / PC / SHACL are averaged only over vignettes where the
    # system actually produced output (output_present=true) so a system that
    # was never run does not dilute the means with zeros.
    agg: dict[str, dict] = {}
    for s in active_systems:
        present_rows = [row for row in per_vig if row[s].get("output_present")]
        f1_rows = [row for row in per_vig if row.get("f1_comparable", True)]
        n_present = len(present_rows)
        n_f1 = len(f1_rows)

        def _mean(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        p = _mean([row[s]["precision"] for row in f1_rows])
        r = _mean([row[s]["recall"] for row in f1_rows])
        f = _mean([row[s]["f1"] for row in f1_rows])
        o_avg = _mean([row[s]["oc"] for row in present_rows])
        rh_avg = _mean([row[s]["rh"] for row in present_rows])
        sh_avg = _mean([row[s]["sh"] for row in present_rows])
        oh_avg = _mean([row[s]["oh"] for row in present_rows])
        conforming_count = sum(
            1 for row in present_rows if row[s]["shacl_conforms"] is True
        )
        pc_vals = [
            row[s]["pc"] for row in present_rows if row[s].get("pc") is not None
        ]
        pc_avg = sum(pc_vals) / len(pc_vals) if pc_vals else None
        total_slots_sum = sum(row[s].get("pc_slots", 0) for row in present_rows)
        covered_slots_sum = sum(row[s].get("pc_covered", 0) for row in present_rows)
        agg[s] = {
            "avg_precision": round(p, 3),
            "avg_recall": round(r, 3),
            "avg_f1": round(f, 3),
            "avg_oc": round(o_avg, 3),
            "avg_rh": round(rh_avg, 3),
            "avg_sh": round(sh_avg, 3),
            "avg_oh": round(oh_avg, 3),
            "avg_pc": round(pc_avg, 3) if pc_avg is not None else None,
            "total_slots": total_slots_sum,
            "covered_slots": covered_slots_sum,
            "shacl_conforms_n": conforming_count,
            "n": len(per_vig),
            "n_present": n_present,
            "n_f1_comparable": n_f1,
        }

    print("\nAggregate (F1 over f1_comparable; OC/SH/OH/PC/SHACL over present output):")
    print(
        f"  {'Sys':>3}  {'P':>5} {'R':>5} {'F1':>5}  "
        f"{'OC':>5} {'RH':>5} {'SH':>5} {'OH':>5}  "
        f"{'PC':>5}  {'SHACL-pass':>12}  {'n_pres':>6} {'n_f1':>5}"
    )
    for s in active_systems:
        a = agg[s]
        pc_str = f"{a['avg_pc']:.3f}" if a["avg_pc"] is not None else "  -  "
        print(
            f"  {s:>3}  "
            f"{a['avg_precision']:.3f}  {a['avg_recall']:.3f}  {a['avg_f1']:.3f}  "
            f"{a['avg_oc']:.3f}  {a['avg_rh']:.3f}  {a['avg_sh']:.3f}  {a['avg_oh']:.3f}  "
            f"{pc_str:>5}  "
            f"{a['shacl_conforms_n']}/{a['n_present']:<3}  "
            f"{a['n_present']:>6} {a['n_f1_comparable']:>5}"
        )

    # Pairwise F1 deltas — the comparison the ablation study hangs on.
    if len(active_systems) > 1:
        print("\nPairwise F1 deltas (b-a, c-a, c-b):")
        for i in range(len(active_systems)):
            for j in range(i + 1, len(active_systems)):
                s1, s2 = active_systems[i], active_systems[j]
                d = agg[s2]["avg_f1"] - agg[s1]["avg_f1"]
                winner = s2.upper() if d > 0 else (s1.upper() if d < 0 else "tied")
                print(f"  {s2} - {s1}: {d:+.3f}  ({winner})")

    # ---------------------------------------------------------------- PERSIST
    out_dir = OUTPUTS_ROOT / schema / track / prompt
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema": schema,
        "track": track,
        "prompt": prompt,
        "canonicalize_iris": canonicalize,
        "n_vignettes": len(per_vig),
        "per_vignette": per_vig,
        "aggregate": agg,
    }
    out = out_dir / "eval.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {out.relative_to(PROJECT_ROOT)}")
    return summary


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schema", default="chr")
    ap.add_argument("--track", choices=("schema", "ontology"), default="schema")
    ap.add_argument("--prompt", default="full",
                    help="Prompt variant (default: full; must match extract.py's --prompt).")
    ap.add_argument(
        "--strict-iris", action="store_true",
        help="Disable the ex:-local-name lowercasing pre-pass "
             "(useful to measure canonicalization's F1 delta).",
    )
    args = ap.parse_args()
    evaluate(args.schema, args.track, args.prompt, canonicalize=not args.strict_iris)


if __name__ == "__main__":
    main()
