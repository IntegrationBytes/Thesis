"""Unit tests for pure helpers in pipeline/evaluate.py.

These are the metric primitives the thesis reports — they need to be
correct at the boundary cases (empty gold, empty pred, no overlap,
perfect overlap) before any aggregated number can be trusted.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from rdflib import Graph, Namespace, RDF, URIRef

from pipeline.evaluate import (
    _canonicalize_iri,
    _gold_f1_comparable,
    _path_populated,
    _shapes_to_slot_map,
    oc,
    prf1,
    property_completeness,
)

EX = Namespace("http://example.org/clinical/")
CHR = Namespace("https://w3id.org/shexmap/resource/ontology-schema/d285f599-dc2e-4bd0-83f3-df21defa8821/")
SH = Namespace("http://www.w3.org/ns/shacl#")


# ---------------------------------------------------------------------------
# Helpers — write a minimal Turtle file to a tmp_path and return the Path.
# ---------------------------------------------------------------------------

def _write_ttl(tmp_path: Path, filename: str, content: str) -> Path:
    p = tmp_path / filename
    p.write_text(textwrap.dedent(content))
    return p


# ---------------------------------------------------------------------------
# _canonicalize_iri — ex: instance URIs are lowercased; schema URIs pass through.
# ---------------------------------------------------------------------------
class TestCanonicalizeIri:
    def test_ex_uri_is_lowercased(self):
        # The motivating case: ex:cond_Hypertension vs ex:cond_hypertension.
        assert (
            _canonicalize_iri("http://example.org/clinical/cond_Hypertension")
            == "http://example.org/clinical/cond_hypertension"
        )

    def test_ex_uri_already_lowercase_is_unchanged(self):
        uri = "http://example.org/clinical/patient_johnandersson"
        assert _canonicalize_iri(uri) == uri

    def test_chr_predicate_passes_through(self):
        # Schema vocabulary must never be rewritten — predicate identity
        # would drift otherwise.
        uri = "https://w3id.org/shexmap/resource/ontology-schema/d285f599-dc2e-4bd0-83f3-df21defa8821/hasPatient"
        assert _canonicalize_iri(uri) == uri

    def test_rdf_type_passes_through(self):
        uri = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
        assert _canonicalize_iri(uri) == uri

    def test_xsd_datatype_passes_through(self):
        uri = "http://www.w3.org/2001/XMLSchema#dateTime"
        assert _canonicalize_iri(uri) == uri


# ---------------------------------------------------------------------------
# prf1 — set-based precision/recall/F1 with pred=None → all-zero-but-valid=False.
# ---------------------------------------------------------------------------
class TestPrf1:
    def test_perfect_match(self):
        trips = {("s", "p", "o"), ("a", "b", "c")}
        result = prf1(trips, trips)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0
        assert result["tp"] == 2
        assert result["fp"] == 0
        assert result["fn"] == 0
        assert result["valid"] is True

    def test_no_overlap(self):
        gold = {("s", "p", "o")}
        pred = {("x", "y", "z")}
        result = prf1(gold, pred)
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0
        assert result["tp"] == 0
        assert result["fp"] == 1
        assert result["fn"] == 1

    def test_half_overlap(self):
        gold = {("a", "b", "c"), ("d", "e", "f")}
        pred = {("a", "b", "c"), ("x", "y", "z")}
        result = prf1(gold, pred)
        assert result["tp"] == 1
        assert result["fp"] == 1
        assert result["fn"] == 1
        assert result["precision"] == pytest.approx(0.5)
        assert result["recall"] == pytest.approx(0.5)
        assert result["f1"] == pytest.approx(0.5)

    def test_pred_none_means_invalid(self):
        """A missing or unparseable output → zero TP, full FN, valid=False."""
        gold = {("a", "b", "c"), ("d", "e", "f")}
        result = prf1(gold, None)
        assert result["valid"] is False
        assert result["tp"] == 0
        assert result["fp"] == 0
        assert result["fn"] == 2
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0

    def test_empty_gold_and_empty_pred(self):
        """Edge case: nothing to compare — metrics stay at 0, not nan."""
        result = prf1(set(), set())
        assert result["tp"] == 0
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0
        assert result["valid"] is True


# ---------------------------------------------------------------------------
# oc — Ontology Conformance = (predicate-in-valid-set) / total triples.
# ---------------------------------------------------------------------------
class TestOc:
    def test_all_conforming(self):
        trips = {("s", "p1", "o"), ("s", "p2", "o")}
        valid = {"p1", "p2"}
        result = oc(trips, valid)
        assert result["oc"] == 1.0
        assert result["rh"] == 0.0
        assert result["conforming"] == 2
        assert result["total"] == 2

    def test_none_conforming(self):
        trips = {("s", "p1", "o")}
        valid = {"p99"}
        result = oc(trips, valid)
        assert result["oc"] == 0.0
        assert result["rh"] == 1.0

    def test_half_conforming(self):
        trips = {("s", "p1", "o"), ("s", "p2", "o")}
        valid = {"p1"}
        result = oc(trips, valid)
        assert result["oc"] == pytest.approx(0.5)
        assert result["rh"] == pytest.approx(0.5)
        assert result["conforming"] == 1
        assert result["total"] == 2

    def test_empty_trips_is_vacuously_conformant(self):
        """No triples → oc=1.0, rh=0.0 (nothing to hallucinate)."""
        result = oc(set(), {"p1"})
        assert result["oc"] == 1.0
        assert result["rh"] == 0.0
        assert result["total"] == 0

    def test_none_trips_is_vacuously_conformant(self):
        result = oc(None, {"p1"})
        assert result["oc"] == 1.0
        assert result["rh"] == 0.0


# ---------------------------------------------------------------------------
# _gold_f1_comparable — exact F1 is invalid for UUID-based Synthea IRIs.
# ---------------------------------------------------------------------------
class TestGoldF1Comparable:
    def test_human_readable_gold_is_comparable(self):
        trips = {
            (
                "http://example.org/clinical/cond_transient_ischaemic_attack",
                str(CHR.hasLocatedIn),
                "http://example.org/clinical/site_left_middle_cerebral_artery",
            )
        }
        assert _gold_f1_comparable(trips) is True

    def test_uuidish_gold_subject_is_not_comparable(self):
        trips = {
            (
                "http://example.org/clinical/visit_5e117ba0_820d_777a_6f36_fd808fb45dbb",
                str(CHR.hasCareUnit),
                "http://example.org/clinical/unit_github_com_syntheti",
            )
        }
        assert _gold_f1_comparable(trips) is False

    def test_uuidish_gold_object_is_not_comparable(self):
        trips = {
            (
                "http://example.org/clinical/meas_proc_temperature",
                str(CHR.hasPatient),
                "http://example.org/clinical/patient_5e117ba0-820d-777a-7d5f-eb972d065bfa",
            )
        }
        assert _gold_f1_comparable(trips) is False

    def test_missing_gold_is_not_comparable(self):
        assert _gold_f1_comparable(None) is False


# ---------------------------------------------------------------------------
# _path_populated — simple and sequence property path traversal.
# ---------------------------------------------------------------------------
class TestPathPopulated:
    def test_simple_path_present(self, tmp_path):
        """A single-hop path is populated when the predicate has any value."""
        g = Graph()
        s = EX.instance1
        p = CHR.hasValue
        g.add((s, p, URIRef("http://example.org/clinical/val")))
        assert _path_populated(g, s, [str(p)]) is True

    def test_simple_path_absent(self, tmp_path):
        g = Graph()
        s = EX.instance1
        assert _path_populated(g, s, [str(CHR.hasValue)]) is False

    def test_sequence_path_both_hops_present(self):
        """A two-hop path (atTime → hasValue) is populated when the full chain exists."""
        g = Graph()
        s = EX.meas1
        intermediate = EX.time1
        g.add((s, CHR.atTime, intermediate))
        g.add((intermediate, CHR.hasValue, URIRef("http://example.org/clinical/v")))
        assert _path_populated(g, s, [str(CHR.atTime), str(CHR.hasValue)]) is True

    def test_sequence_path_first_hop_only(self):
        """A two-hop path where only the first hop exists returns False."""
        g = Graph()
        s = EX.meas1
        intermediate = EX.time1
        g.add((s, CHR.atTime, intermediate))
        # No second hop
        assert _path_populated(g, s, [str(CHR.atTime), str(CHR.hasValue)]) is False

    def test_empty_path_returns_true(self):
        """An empty path trivially succeeds (start node itself is the terminus)."""
        g = Graph()
        assert _path_populated(g, EX.any, []) is True


# ---------------------------------------------------------------------------
# property_completeness — integration test using real SHACL shapes + tmp TTL.
# ---------------------------------------------------------------------------
class TestPropertyCompleteness:
    """Integration tests that write minimal Turtle graphs and check PC."""

    @pytest.fixture()
    def ctx(self):
        from pipeline.prompts import chr_context
        return chr_context("schema")

    def test_missing_file_returns_none_pc(self, tmp_path, ctx):
        result = property_completeness(tmp_path / "nonexistent.ttl", ctx)
        assert result["pc"] is None
        assert result["total_slots"] == 0
        assert result["covered_slots"] == 0

    def test_empty_graph_returns_none_pc(self, tmp_path, ctx):
        """A graph with no typed instances has no slots to judge."""
        ttl = _write_ttl(tmp_path, "empty.ttl", """
            @prefix ex: <http://example.org/clinical/> .
        """)
        result = property_completeness(ttl, ctx)
        assert result["pc"] is None

    def test_instance_with_all_slots_filled_scores_1(self, tmp_path, ctx):
        """A ClinicalCondition with all expected properties present → PC=1.0."""
        # From the schema shapes: ClinicalCondition expects hasRecordDate.
        ttl = _write_ttl(tmp_path, "full.ttl", """
            @prefix ex:  <http://example.org/clinical/> .
            @prefix chr: <https://w3id.org/shexmap/resource/ontology-schema/d285f599-dc2e-4bd0-83f3-df21defa8821/> .
            @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
            @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

            ex:cond1 a chr:ClinicalCondition ;
                chr:hasRecordDate "2026-01-01T00:00:00"^^xsd:dateTime .
        """)
        result = property_completeness(ttl, ctx)
        assert result["pc"] == pytest.approx(1.0)
        assert result["covered_slots"] == result["total_slots"]

    def test_instance_with_no_slots_filled_scores_0(self, tmp_path, ctx):
        """A ClinicalCondition with no properties populated → PC=0.0."""
        ttl = _write_ttl(tmp_path, "bare.ttl", """
            @prefix ex:  <http://example.org/clinical/> .
            @prefix chr: <https://w3id.org/shexmap/resource/ontology-schema/d285f599-dc2e-4bd0-83f3-df21defa8821/> .
            @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

            ex:cond1 a chr:ClinicalCondition .
        """)
        result = property_completeness(ttl, ctx)
        assert result["pc"] == pytest.approx(0.0)
        assert result["covered_slots"] == 0
        assert result["total_slots"] > 0

    def test_pc_is_none_for_class_with_no_declared_slots(self, tmp_path, ctx):
        """A Person instance has no declared sh:property slots → not judged → pc=None."""
        ttl = _write_ttl(tmp_path, "person.ttl", """
            @prefix ex:  <http://example.org/clinical/> .
            @prefix chr: <https://w3id.org/shexmap/resource/ontology-schema/d285f599-dc2e-4bd0-83f3-df21defa8821/> .
            @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

            ex:patient1 a chr:Person .
        """)
        result = property_completeness(ttl, ctx)
        # Person has no sh:property constraints in the schema shapes,
        # so no slots → pc=None (no instances judged).
        assert result["pc"] is None

    def test_partial_coverage_gives_fractional_pc(self, tmp_path, ctx):
        """Two instances, one with slots filled, one without → PC = 0.5."""
        ttl = _write_ttl(tmp_path, "partial.ttl", """
            @prefix ex:  <http://example.org/clinical/> .
            @prefix chr: <https://w3id.org/shexmap/resource/ontology-schema/d285f599-dc2e-4bd0-83f3-df21defa8821/> .
            @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
            @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

            ex:cond1 a chr:ClinicalCondition ;
                chr:hasRecordDate "2026-01-01T00:00:00"^^xsd:dateTime .
            ex:cond2 a chr:ClinicalCondition .
        """)
        result = property_completeness(ttl, ctx)
        assert result["pc"] == pytest.approx(0.5)
        assert result["total_slots"] == 2
        assert result["covered_slots"] == 1
