"""F1 with vs without FHIR-grounded IRI normaliser.

The pipeline's evaluate.py always applies normalize_llm_to_gold before
computing F1. This script computes F1 BOTH WAYS (with and without) so
the thesis can defensibly report:

  - F1 with normaliser  : the headline F1 number
  - F1 without normaliser: lower bound (what F1 would be if the LLM
                           had to produce gold IRI shapes verbatim)

The DIFFERENCE between the two quantifies the IRI normaliser's
contribution to the headline F1.

Output: stdout table + JSON dump at
``evaluation/outputs/f1_normalizer_ablation.json``.
"""
from __future__ import annotations

import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

import rdflib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from iri_normalizer import normalize_llm_to_gold  # noqa: E402

GOLD_DIR = ROOT / "evaluation/corpus/abox_gold"


def _stratum(vid: str) -> str:
    m = re.match(r"vignette_(\d+)", vid)
    if not m:
        return "other"
    return "general" if int(m.group(1)) <= 100 else "complex"


def _f1(gold_set: set, gen_set: set) -> tuple[float, float, float]:
    if not gen_set:
        return (0.0, 0.0, 0.0)
    tp = len(gold_set & gen_set)
    p = tp / len(gen_set)
    r = tp / len(gold_set) if gold_set else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return (p, r, f)


def _worker(args) -> tuple[str, str, str, dict]:
    """Process (vid, model, sys_tag, gen_path_str, gold_path_str).
    Returns (vid, model, sys_tag, {with: {p,r,f1}, without: {p,r,f1}})."""
    vid, model, sys_tag, gen_path_str, gold_path_str = args
    gen_path = Path(gen_path_str)
    gold_path = Path(gold_path_str)
    if not gen_path.exists() or not gold_path.exists():
        return (vid, model, sys_tag, {})
    if "# LLM call failed" in gen_path.read_text():
        return (vid, model, sys_tag, {})
    try:
        gold_g = rdflib.Graph().parse(gold_path.as_posix(), format="turtle")
        gen_g = rdflib.Graph().parse(gen_path.as_posix(), format="turtle")
    except Exception:
        return (vid, model, sys_tag, {})

    gold_set = {(str(s), str(p), str(o)) for s, p, o in gold_g}

    # Without normalizer (raw LLM IRIs)
    gen_raw_set = {(str(s), str(p), str(o)) for s, p, o in gen_g}
    p_raw, r_raw, f_raw = _f1(gold_set, gen_raw_set)

    # With normalizer (LLM IRIs rewritten to gold counterparts)
    gen_norm, _ = normalize_llm_to_gold(gen_g, gold_g)
    gen_norm_set = {(str(s), str(p), str(o)) for s, p, o in gen_norm}
    p_norm, r_norm, f_norm = _f1(gold_set, gen_norm_set)

    return (vid, model, sys_tag, {
        "without": {"p": p_raw, "r": r_raw, "f1": f_raw},
        "with":    {"p": p_norm, "r": r_norm, "f1": f_norm},
    })


def main() -> None:
    print("F1 ablation: WITH vs WITHOUT FHIR-grounded IRI normaliser")
    print("=" * 78)

    jobs = []
    for gold in sorted(GOLD_DIR.glob("vignette_*_gold_schema.ttl")):
        vid = gold.stem.replace("_gold_schema", "")
        for model, root in [
            ("gptoss",  ROOT / "evaluation/outputs/chr/schema/full"),
        ]:
            for sys_tag in ("a", "b"):
                jobs.append((vid, model, sys_tag,
                             str(root / sys_tag / f"{vid}.ttl"),
                             str(gold)))

    results = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        futs = [pool.submit(_worker, j) for j in jobs]
        for fut in as_completed(futs):
            results.append(fut.result())

    # Aggregate per (model, sys, stratum)
    agg: dict = {}
    for vid, model, sys_tag, scores in results:
        if not scores:
            continue
        st = _stratum(vid)
        key = (model, sys_tag, st)
        d = agg.setdefault(key, {"with_f1": [], "without_f1": [],
                                  "with_p": [], "with_r": [],
                                  "without_p": [], "without_r": []})
        d["with_f1"].append(scores["with"]["f1"])
        d["without_f1"].append(scores["without"]["f1"])
        d["with_p"].append(scores["with"]["p"])
        d["with_r"].append(scores["with"]["r"])
        d["without_p"].append(scores["without"]["p"])
        d["without_r"].append(scores["without"]["r"])

    # Print
    print(f"\n{'Model':<10} {'Stratum':<10} {'Sys':<4} "
          f"{'F1 raw':>8} {'F1 norm':>8} {'Δ':>7}")
    print("-" * 78)
    for model in ("gptoss",):
        for stratum in ("general", "complex"):
            for sys_tag in ("a", "b"):
                d = agg.get((model, sys_tag, stratum), {})
                if not d:
                    continue
                raw = mean(d["without_f1"])
                nrm = mean(d["with_f1"])
                print(f"{model:<10} {stratum:<10} {sys_tag:<4} "
                      f"{raw:>8.3f} {nrm:>8.3f} {nrm - raw:>+7.3f}")
        print("-" * 78)

    # JSON dump
    out = {f"{m}_{s}_{st}": {
        "n": len(d["with_f1"]),
        "raw_f1_mean": mean(d["without_f1"]) if d["without_f1"] else 0,
        "norm_f1_mean": mean(d["with_f1"]) if d["with_f1"] else 0,
        "raw_p_mean": mean(d["without_p"]) if d["without_p"] else 0,
        "raw_r_mean": mean(d["without_r"]) if d["without_r"] else 0,
        "norm_p_mean": mean(d["with_p"]) if d["with_p"] else 0,
        "norm_r_mean": mean(d["with_r"]) if d["with_r"] else 0,
    } for (m, s, st), d in agg.items()}
    out_path = ROOT / "evaluation/outputs/f1_normalizer_ablation.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[OK] dumped {out_path}")


if __name__ == "__main__":
    main()
