"""Evaluate System D (LLM looped without verifier) and compare to A vs B.

Answers the reviewer's apples-to-apples question: "what does the verifier
add beyond just running the language model more times?"

For gpt-oss-120b:
  - A: zero-shot
  - B: SHACL-feedback retry loop (existing data)
  - D: 4 sequential calls, no feedback (NEW)

Reports per-system:
  - SHACL pass-rate (schema track)
  - Triple-level F1 against gold
  - Mean / median triples produced

If B > D on SHACL: feedback signal > raw compute.
If B ≈ D on SHACL: it's just more compute scaling.
"""
from __future__ import annotations

import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, median

import rdflib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from evaluate import shacl_conforms  # noqa: E402
from prompts import chr_context  # noqa: E402
from iri_normalizer import normalize_llm_to_gold  # noqa: E402

GOLD_DIR = ROOT / "evaluation/corpus/abox_gold"
OUT_GPTOSS = ROOT / "evaluation/outputs/chr/schema/full"


def _stratum(vid: str) -> str:
    m = re.match(r"vignette_(\d+)", vid)
    if not m:
        return "other"
    return "general" if int(m.group(1)) <= 100 else "complex"


def _f1(gold: set, gen: set) -> float:
    if not gen:
        return 0.0
    tp = len(gold & gen)
    p = tp / len(gen)
    r = tp / len(gold) if gold else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def _worker(args) -> tuple[str, str, str, float | None, bool | None, int]:
    """Process one (vid, gen_path, gold_path, model, sys_tag).
    Returns (vid, model, sys_tag, f1, shacl_ok, n_triples).
    """
    vid, gen_path_str, gold_path_str, model_tag, sys_tag = args
    gen_path = Path(gen_path_str)
    gold_path = Path(gold_path_str)
    if not gen_path.exists() or not gold_path.exists():
        return (vid, model_tag, sys_tag, None, None, 0)
    if "# LLM call failed" in gen_path.read_text():
        return (vid, model_tag, sys_tag, None, None, 0)
    try:
        gold_g = rdflib.Graph().parse(gold_path.as_posix(), format="turtle")
        gen_g = rdflib.Graph().parse(gen_path.as_posix(), format="turtle")
    except Exception:
        return (vid, model_tag, sys_tag, None, None, 0)
    gen_rewritten, _ = normalize_llm_to_gold(gen_g, gold_g)
    gold_set = {(str(s), str(p), str(o)) for s, p, o in gold_g}
    gen_set = {(str(s), str(p), str(o)) for s, p, o in gen_rewritten}
    f1 = _f1(gold_set, gen_set)
    ctx = chr_context("schema")
    ok = shacl_conforms(gen_path, ctx)
    return (vid, model_tag, sys_tag, f1, ok is True, len(gen_g))


def main() -> None:
    # Discover D outputs
    gptoss_d = OUT_GPTOSS / "d"
    print(f"gpt-oss-120b D outputs: {len(list(gptoss_d.glob('vignette_*.ttl'))) if gptoss_d.exists() else 0}")

    # Build all job tuples
    jobs = []
    for gold in sorted(GOLD_DIR.glob("vignette_*_gold_schema.ttl")):
        vid = gold.stem.replace("_gold_schema", "")
        for model_tag, root in [("gptoss", OUT_GPTOSS)]:
            for sys_tag in ("a", "b", "d"):
                gen = root / sys_tag / f"{vid}.ttl"
                jobs.append((vid, str(gen), str(gold), model_tag, sys_tag))

    print(f"\nEvaluating {len(jobs)} (vignette, model, system) combinations...")

    results = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        futs = [pool.submit(_worker, j) for j in jobs]
        for fut in as_completed(futs):
            results.append(fut.result())

    # Aggregate
    agg: dict = {}
    for vid, model, sys_tag, f1, ok, n_trip in results:
        stratum = _stratum(vid)
        key = (model, sys_tag, stratum)
        d = agg.setdefault(key, {"f1s": [], "shacl_pass": 0, "n": 0,
                                  "n_triples": []})
        if f1 is None:
            continue
        d["f1s"].append(f1)
        d["n"] += 1
        d["n_triples"].append(n_trip)
        if ok:
            d["shacl_pass"] += 1

    # Print results
    print("\n" + "=" * 90)
    print("SYSTEM D EVALUATION — compute-fair baseline (4 calls no feedback)")
    print("=" * 90)
    for model in ("gptoss",):
        print(f"\n{model.upper()}")
        print(f"  {'Stratum':<10} {'Sys':<5} {'n':<5} {'F1':>6} {'P':>6} {'R':>6} "
              f"{'SHACL':>8} {'triples':>9}")
        for stratum in ("general", "complex"):
            for sys_tag in ("a", "b", "d"):
                d = agg.get((model, sys_tag, stratum), {})
                if not d or d["n"] == 0:
                    continue
                f1m = mean(d["f1s"]) if d["f1s"] else 0
                shp = 100 * d["shacl_pass"] / d["n"]
                med_t = int(median(d["n_triples"])) if d["n_triples"] else 0
                print(f"  {stratum:<10} {sys_tag:<5} {d['n']:<5} "
                      f"{f1m:>6.3f} {'':>6} {'':>6} "
                      f"{d['shacl_pass']}/{d['n']:<3} ({shp:5.1f}%) "
                      f"{med_t:>9}")
        # Δ B vs D headline
        print(f"\n  {model.upper()} — Δ B vs D (compute-matched comparison)")
        for stratum in ("general", "complex"):
            b = agg.get((model, "b", stratum), {})
            dd = agg.get((model, "d", stratum), {})
            if not b or not dd or b["n"] == 0 or dd["n"] == 0:
                continue
            b_sh = 100 * b["shacl_pass"] / b["n"]
            d_sh = 100 * dd["shacl_pass"] / dd["n"]
            b_f1 = mean(b["f1s"])
            d_f1 = mean(dd["f1s"])
            print(f"  {stratum:<10}  ΔSHACL(B-D) = {b_sh - d_sh:+.1f}pp   "
                  f"ΔF1(B-D) = {b_f1 - d_f1:+.3f}")

    print()


if __name__ == "__main__":
    main()
