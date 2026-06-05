"""Connectivity-stratified analysis (supervisor's ask, 2026-05-22 meeting).

Quote: "see in your data set how much these triples are connected with
each other. I have three maybe four class — one is highly connected,
less connected, more — and based on that you can say let's see SHACL
works on these cases better than the other."

Parallel implementation using ProcessPoolExecutor (6 workers).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

import rdflib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from evaluate import shacl_conforms  # noqa: E402
from prompts import chr_context  # noqa: E402
from owl_validator import validate_with_owl  # noqa: E402
from iri_normalizer import normalize_llm_to_gold  # noqa: E402

GOLD_DIR = ROOT / "evaluation/corpus/abox_gold"
OUT_SCHEMA = ROOT / "evaluation/outputs/chr/schema/full"
OUT_ONTOLOGY = ROOT / "evaluation/outputs/chr/ontology/ontology"
CHR_ONT = ROOT / "evaluation/corpus/tbox/chr_ontology.owl.ttl"


# ---------------------------------------------------------------------------
# Worker functions (must be top-level for multiprocessing).
# ---------------------------------------------------------------------------
def _connectivity_metrics(gold_path_str: str) -> tuple[str, dict]:
    """Connectivity metrics for one gold KG. Returns (vid, metrics)."""
    gold_path = Path(gold_path_str)
    vid = gold_path.stem.replace("_gold_schema", "")
    g = rdflib.Graph()
    g.parse(gold_path.as_posix(), format="turtle")
    entities, out_degree, n_uri_objects = set(), defaultdict(int), 0
    n_triples = 0
    for s, p, o in g:
        n_triples += 1
        if isinstance(s, rdflib.URIRef):
            entities.add(s)
            out_degree[s] += 1
        if isinstance(o, rdflib.URIRef):
            entities.add(o)
            n_uri_objects += 1
    return vid, {
        "n_triples": n_triples,
        "n_entities": len(entities),
        "mean_out_degree": mean(out_degree.values()) if out_degree else 0.0,
        "linked_fraction": n_uri_objects / n_triples if n_triples else 0.0,
    }


def _f1(gold: set, gen: set) -> dict:
    if not gen:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp = len(gold & gen)
    p = tp / len(gen)
    r = tp / len(gold) if gold else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f}


def _schema_worker(args) -> tuple[str, str, str, float | None, bool | None]:
    """For one (vid, system_dir_str): return (vid, model_tag, sys, f1, shacl_ok)."""
    vid, gen_path_str, gold_path_str, model_tag, sys_tag = args
    gen_path = Path(gen_path_str)
    gold_path = Path(gold_path_str)
    if not gen_path.exists() or not gold_path.exists():
        return (vid, model_tag, sys_tag, None, None)
    if "# LLM call failed" in gen_path.read_text():
        return (vid, model_tag, sys_tag, None, None)
    try:
        gold_g = rdflib.Graph().parse(gold_path.as_posix(), format="turtle")
        gen_g = rdflib.Graph().parse(gen_path.as_posix(), format="turtle")
    except Exception:
        return (vid, model_tag, sys_tag, None, None)
    gen_rewritten, _ = normalize_llm_to_gold(gen_g, gold_g)
    gold_set = {(str(s), str(p), str(o)) for s, p, o in gold_g}
    gen_set = {(str(s), str(p), str(o)) for s, p, o in gen_rewritten}
    f1 = _f1(gold_set, gen_set)["f1"]
    ctx = chr_context("schema")
    ok = shacl_conforms(gen_path, ctx)
    return (vid, model_tag, sys_tag, f1, ok is True)


def _owl_worker(args) -> tuple[str, str, str, bool]:
    """For one (vid, gen_path_str, model_tag, sys_tag) on ontology track."""
    vid, gen_path_str, model_tag, sys_tag = args
    gen_path = Path(gen_path_str)
    if not gen_path.exists():
        return (vid, model_tag, sys_tag, False)
    ttl = gen_path.read_text()
    if "# LLM call failed" in ttl:
        return (vid, model_tag, sys_tag, False)
    ok, _ = validate_with_owl(ttl, CHR_ONT)
    return (vid, model_tag, sys_tag, bool(ok))


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main() -> None:
    # Step 1: connectivity metrics for all gold KGs (parallel)
    print("[1/3] Computing connectivity metrics on 200 gold KGs...", flush=True)
    gold_paths = sorted(GOLD_DIR.glob("vignette_*_gold_schema.ttl"))
    metrics_by_vid: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=6) as pool:
        futs = [pool.submit(_connectivity_metrics, str(p)) for p in gold_paths]
        for fut in as_completed(futs):
            vid, m = fut.result()
            metrics_by_vid[vid] = m
    print(f"      Done: {len(metrics_by_vid)} vignettes")

    # Step 2: bucket by n_entities tertile
    sorted_vids = sorted(metrics_by_vid.keys(),
                          key=lambda v: metrics_by_vid[v]["n_entities"])
    n = len(sorted_vids)
    third = n // 3
    buckets = {
        "low":    sorted_vids[:third],
        "medium": sorted_vids[third : 2 * third],
        "high":   sorted_vids[2 * third:],
    }

    print("\n[Bucket ranges by n_entities]")
    for tier in ("low", "medium", "high"):
        vids = buckets[tier]
        sizes = [metrics_by_vid[v]["n_entities"] for v in vids]
        linked = [metrics_by_vid[v]["linked_fraction"] for v in vids]
        print(f"  {tier:<6}  n={len(vids):<3}  "
              f"entities min={min(sizes)} max={max(sizes)} mean={mean(sizes):.1f}  "
              f"linked-fraction mean={mean(linked):.2f}")

    # Step 3a: schema-track F1 + SHACL across 4 system variants × all vignettes (parallel)
    print("\n[2/3] Schema track: F1 + SHACL on 800 graphs...", flush=True)
    schema_jobs = []
    for vid in sorted_vids:
        gold = GOLD_DIR / f"{vid}_gold_schema.ttl"
        for model_tag, root in [("gptoss", OUT_SCHEMA)]:
            for sys_tag in ("a", "b"):
                gen = root / sys_tag / f"{vid}.ttl"
                schema_jobs.append((vid, str(gen), str(gold), model_tag, sys_tag))

    schema_results: list = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        futs = [pool.submit(_schema_worker, j) for j in schema_jobs]
        for fut in as_completed(futs):
            schema_results.append(fut.result())
    print(f"      Done: {len(schema_results)} schema evaluations")

    # Step 3b: ontology-track OWL conformance across 4 system variants × all vignettes
    print("\n[3/3] Ontology track: OWL on 800 graphs...", flush=True)
    owl_jobs = []
    for vid in sorted_vids:
        for model_tag, root in [("gptoss", OUT_ONTOLOGY)]:
            for sys_tag in ("a", "b"):
                gen = root / sys_tag / f"{vid}.ttl"
                owl_jobs.append((vid, str(gen), model_tag, sys_tag))

    owl_results: list = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        futs = [pool.submit(_owl_worker, j) for j in owl_jobs]
        for fut in as_completed(futs):
            owl_results.append(fut.result())
    print(f"      Done: {len(owl_results)} OWL evaluations")

    # Aggregate per bucket
    def bucket_for(vid: str) -> str:
        for tier, vids_in_tier in buckets.items():
            if vid in vids_in_tier:
                return tier
        return "other"

    schema_by_bucket: dict = defaultdict(lambda: defaultdict(
        lambda: {"f1s": [], "shacl_pass": 0, "n": 0}))
    for vid, model_tag, sys_tag, f1, ok in schema_results:
        if f1 is None:
            continue
        b = bucket_for(vid)
        key = f"{model_tag}_{sys_tag}"
        schema_by_bucket[b][key]["f1s"].append(f1)
        schema_by_bucket[b][key]["n"] += 1
        if ok:
            schema_by_bucket[b][key]["shacl_pass"] += 1

    owl_by_bucket: dict = defaultdict(lambda: defaultdict(
        lambda: {"owl_pass": 0, "n": 0}))
    for vid, model_tag, sys_tag, ok in owl_results:
        b = bucket_for(vid)
        key = f"{model_tag}_{sys_tag}"
        owl_by_bucket[b][key]["n"] += 1
        if ok:
            owl_by_bucket[b][key]["owl_pass"] += 1

    # Pretty-print
    print("\n" + "=" * 86)
    print("CONNECTIVITY-STRATIFIED RESULTS — supervisor's connectivity ask · n=200")
    print("=" * 86)
    for tier in ("low", "medium", "high"):
        vids = buckets[tier]
        sizes = [metrics_by_vid[v]["n_entities"] for v in vids]
        linked = [metrics_by_vid[v]["linked_fraction"] for v in vids]
        print(f"\n{tier.upper()} CONNECTIVITY  "
              f"(n={len(vids)}, entities {min(sizes)}–{max(sizes)} mean {mean(sizes):.0f}, "
              f"linked-fraction mean {mean(linked):.2f})")
        for model in ("gptoss",):
            sa = schema_by_bucket[tier][f"{model}_a"]
            sb = schema_by_bucket[tier][f"{model}_b"]
            sa_f1 = mean(sa["f1s"]) if sa["f1s"] else 0
            sb_f1 = mean(sb["f1s"]) if sb["f1s"] else 0
            sa_sh = 100 * sa["shacl_pass"] / sa["n"] if sa["n"] else 0
            sb_sh = 100 * sb["shacl_pass"] / sb["n"] if sb["n"] else 0
            print(f"  {model:<7} schema    "
                  f"F1 A={sa_f1:.3f} B={sb_f1:.3f}   "
                  f"SHACL A={sa_sh:5.1f}% B={sb_sh:5.1f}%   "
                  f"ΔSHACL=+{sb_sh - sa_sh:.1f}pp")
        for model in ("gptoss",):
            oa = owl_by_bucket[tier][f"{model}_a"]
            ob = owl_by_bucket[tier][f"{model}_b"]
            oa_pct = 100 * oa["owl_pass"] / oa["n"] if oa["n"] else 0
            ob_pct = 100 * ob["owl_pass"] / ob["n"] if ob["n"] else 0
            print(f"  {model:<7} ontology  "
                  f"OWL A={oa_pct:5.1f}% B={ob_pct:5.1f}%   "
                  f"ΔOWL=+{ob_pct - oa_pct:.1f}pp")

    # JSON dump
    output = {
        "metric": "n_entities",
        "buckets": {
            tier: {
                "vignettes": buckets[tier],
                "n_entities_mean": mean(
                    metrics_by_vid[v]["n_entities"] for v in buckets[tier]
                ),
                "linked_fraction_mean": mean(
                    metrics_by_vid[v]["linked_fraction"] for v in buckets[tier]
                ),
                "schema": {
                    k: {
                        "f1_mean": mean(v["f1s"]) if v["f1s"] else 0,
                        "n": v["n"],
                        "shacl_pass": v["shacl_pass"],
                    } for k, v in schema_by_bucket[tier].items()
                },
                "ontology": {
                    k: {"n": v["n"], "owl_pass": v["owl_pass"]}
                    for k, v in owl_by_bucket[tier].items()
                },
            } for tier in ("low", "medium", "high")
        },
    }
    out_path = ROOT / "evaluation/outputs/connectivity_buckets.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n[OK] dumped {out_path}")


if __name__ == "__main__":
    main()
