"""Per-vignette inspection tool — for the thesis defence Q&A.

If an examiner asks "show me how it works on vignette X", run this:

    python scripts/inspect_vignette.py vignette_030

Prints, for the requested vignette:
  - source text excerpt
  - gold KG triple count + sample
  - System A output: triple count, SHACL violations, OWL violations
  - System B output: same
  - System D output if exists
  - Judge scores per variant if computed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from extract import validate_with_shacl  # noqa: E402
from owl_validator import validate_with_owl  # noqa: E402
from prompts import chr_context  # noqa: E402

import rdflib  # noqa: E402

TEXT_DIR = ROOT / "evaluation/corpus/vignettes"
GOLD_DIR = ROOT / "evaluation/corpus/abox_gold"
OUT_GEMINI_SCHEMA = ROOT / "evaluation/outputs/chr/schema/full"
OUT_GEMINI_ONT    = ROOT / "evaluation/outputs/chr/ontology/ontology"
OUT_GPTOSS_SCHEMA = ROOT / "evaluation/outputs_freemodel/chr/schema/full"
OUT_GPTOSS_ONT    = ROOT / "evaluation/outputs_freemodel/chr/ontology/ontology"
JUDGE_FILE = ROOT / "evaluation/outputs/judge_scores.json"
CHR_ONT = ROOT / "evaluation/corpus/tbox/chr_ontology.owl.ttl"


def _safe_parse(ttl_path: Path) -> rdflib.Graph | None:
    if not ttl_path.exists():
        return None
    ttl = ttl_path.read_text()
    if "# LLM call failed" in ttl:
        return None
    g = rdflib.Graph()
    try:
        g.parse(data=ttl, format="turtle")
        return g
    except Exception:
        return None


def _shacl_violations(ttl_path: Path) -> int:
    if not ttl_path.exists():
        return -1
    ttl = ttl_path.read_text()
    if "# LLM call failed" in ttl:
        return -1
    ctx = chr_context("schema")
    _, report = validate_with_shacl(ttl, ctx)
    if "Conforms: True" in report:
        return 0
    return report.count("Constraint Violation")


def _owl_violations(ttl_path: Path) -> int:
    if not ttl_path.exists():
        return -1
    ttl = ttl_path.read_text()
    if "# LLM call failed" in ttl:
        return -1
    ok, report = validate_with_owl(ttl, CHR_ONT)
    if ok:
        return 0
    return report.count("Constraint Violation")


def _judge_for(vid: str, label: str) -> dict | None:
    if not JUDGE_FILE.exists():
        return None
    try:
        d = json.loads(JUDGE_FILE.read_text())
    except Exception:
        return None
    run = d.get(label, {})
    return run.get("vignettes", {}).get(vid)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("vid", help="Vignette ID (e.g. vignette_030)")
    args = ap.parse_args()
    vid = args.vid

    # 1) Source text
    text_path = TEXT_DIR / f"{vid}.txt"
    print(f"=== {vid} — source text ===")
    if text_path.exists():
        text = text_path.read_text()
        # Show first ~30 lines
        for i, line in enumerate(text.splitlines()[:30]):
            print(f"  {line}")
        if len(text.splitlines()) > 30:
            print(f"  ... ({len(text.splitlines())} lines total)")
    else:
        print(f"  (NOT FOUND: {text_path})")

    # 2) Gold KG
    gold_path = GOLD_DIR / f"{vid}_gold_schema.ttl"
    print(f"\n=== Gold KG (schema-track) ===")
    g_gold = _safe_parse(gold_path)
    if g_gold:
        print(f"  {len(g_gold)} triples")
        # Subject types
        types = set(g_gold.objects(None, rdflib.RDF.type))
        print(f"  {len(set(g_gold.subjects()))} unique subjects")
        print(f"  {len(types)} distinct rdf:type values")
    else:
        print(f"  (NOT FOUND or unparseable)")

    # 3) Each system
    print(f"\n=== System outputs ===")
    print(f"  {'Variant':<30} {'triples':>8} {'SHACL viol':>12} {'OWL viol':>10}")
    for label, ttl_path, has_shacl, has_owl in [
        ("Gemini schema A",   OUT_GEMINI_SCHEMA / "a" / f"{vid}.ttl", True,  False),
        ("Gemini schema B",   OUT_GEMINI_SCHEMA / "b" / f"{vid}.ttl", True,  False),
        ("Gemini schema D",   OUT_GEMINI_SCHEMA / "d" / f"{vid}.ttl", True,  False),
        ("Gemini ontology A", OUT_GEMINI_ONT / "a"    / f"{vid}.ttl", False, True),
        ("Gemini ontology B", OUT_GEMINI_ONT / "b"    / f"{vid}.ttl", False, True),
        ("gpt-oss schema A",  OUT_GPTOSS_SCHEMA / "a" / f"{vid}.ttl", True,  False),
        ("gpt-oss schema B",  OUT_GPTOSS_SCHEMA / "b" / f"{vid}.ttl", True,  False),
        ("gpt-oss schema D",  OUT_GPTOSS_SCHEMA / "d" / f"{vid}.ttl", True,  False),
        ("gpt-oss ontology A", OUT_GPTOSS_ONT / "a"   / f"{vid}.ttl", False, True),
        ("gpt-oss ontology B", OUT_GPTOSS_ONT / "b"   / f"{vid}.ttl", False, True),
    ]:
        g = _safe_parse(ttl_path)
        n_tr = len(g) if g else "—"
        n_sh = _shacl_violations(ttl_path) if has_shacl else "—"
        n_ow = _owl_violations(ttl_path) if has_owl else "—"
        print(f"  {label:<30} {str(n_tr):>8} {str(n_sh):>12} {str(n_ow):>10}")

    # 4) Judge scores
    print(f"\n=== LLM-as-judge scores (if available) ===")
    print(f"  {'Variant':<35} {'Faith':>6} {'Comp':>6} {'Hallu':>6}")
    for label in [
        "main/chr/schema/full/b",
        "main/chr/ontology/ontology/b",
        "freemodel/chr/schema/full/b",
        "freemodel/chr/ontology/ontology/b",
    ]:
        scores = _judge_for(vid, label)
        if scores and scores.get("faithfulness") is not None:
            short = (label.replace("main/chr/", "Gemini ")
                          .replace("freemodel/chr/", "gpt-oss ")
                          .replace("/full", "").replace("/ontology", "")
                          .replace("/b", ""))
            print(f"  {short:<35} "
                  f"{scores['faithfulness']:>6} {scores['completeness']:>6} "
                  f"{scores['hallucination']:>6}")
            rat = scores.get("rationale", {})
            for k in ("faithfulness", "completeness", "hallucination"):
                if rat.get(k):
                    print(f"    {k}: {rat[k][:160]}")


if __name__ == "__main__":
    main()
