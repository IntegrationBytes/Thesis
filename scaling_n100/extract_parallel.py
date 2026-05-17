"""Parallel wrapper around pipeline/extract.py per-vignette logic.

Runs Systems A/B/C across all vignettes with a thread pool. The OpenRouter
LLM call is I/O-bound (network), so threads are the right primitive — no
GIL contention on the actual API wait. With 5 workers, n=108 vignettes
extracts in ~10-15 min instead of 3+ hours.

Skip-if-exists logic: if the System A output for a vignette already
exists AND parses to a non-empty graph, all three systems are skipped
for that vignette. This makes the script idempotent — safe to re-run.

Usage::

    python scaling_n100/extract_parallel.py \\
        --schema chr --track schema --prompt full --systems a,b,c --workers 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "pipeline"))

from extract import (  # type: ignore  # noqa: E402
    OUTPUTS_ROOT,
    _iter_vignettes,
    _output_path,
    _vignette_id,
    _pick_builder,
    load_schema,
    run_system_a,
    run_system_b,
    run_system_c,
)
from rdflib import Graph  # noqa: E402


def _is_complete(out_path: Path) -> bool:
    """A System output is considered complete if the file exists, parses
    as Turtle, and contains at least one triple. Error-comment outputs
    parse to zero triples and so re-extract."""
    if not out_path.exists():
        return False
    try:
        g = Graph()
        g.parse(out_path.as_posix(), format="turtle")
        return len(g) > 0
    except Exception:
        return False


def process_one(args, vignette_path: Path) -> dict:
    """Run all selected systems on a single vignette. Returns a timing dict."""
    ctx = load_schema(args.schema, args.track)
    vid = _vignette_id(vignette_path)
    snippet = vignette_path.read_text().strip()
    entry: dict = {"vignette": vid}
    systems = set(args.systems.split(","))

    a_path = _output_path(ctx, args.prompt, "a", vid)
    b_path = _output_path(ctx, args.prompt, "b", vid)
    c_path = _output_path(ctx, args.prompt, "c", vid)

    # Skip if all selected systems are already complete.
    if all(_is_complete(_output_path(ctx, args.prompt, s, vid)) for s in systems):
        return {"vignette": vid, "skipped": True}

    # ---- System A ----
    system_a_ttl = ""
    if "a" in systems or "b" in systems or "c" in systems:
        if _is_complete(a_path):
            system_a_ttl = a_path.read_text()
            entry["system_a_skipped"] = True
        else:
            t0 = time.time()
            try:
                system_a_ttl = run_system_a(ctx, snippet, args.prompt)
            except Exception as e:
                system_a_ttl = f"# LLM call failed: {e}"
            entry["system_a_sec"] = round(time.time() - t0, 2)
            entry["system_a_llm_calls"] = 1
            if "a" in systems:
                a_path.parent.mkdir(parents=True, exist_ok=True)
                a_path.write_text(system_a_ttl)

    # ---- System B ----
    if "b" in systems and not _is_complete(b_path):
        t0 = time.time()
        b_cycles = OUTPUTS_ROOT / ctx.name / ctx.track / args.prompt / "b" / "cycles" / vid
        try:
            ttl_b, conforms_b, b_calls = run_system_b(ctx, system_a_ttl, cycles_dir=b_cycles)
        except Exception as e:
            ttl_b, conforms_b, b_calls = system_a_ttl, False, 0
            entry["system_b_error"] = str(e)[:100]
        entry["system_b_sec"] = round(time.time() - t0, 2)
        entry["system_b_conforms"] = conforms_b
        entry["system_b_extra_llm_calls"] = b_calls
        b_path.parent.mkdir(parents=True, exist_ok=True)
        b_path.write_text(ttl_b)

    # ---- System C ----
    if "c" in systems and not _is_complete(c_path):
        t0 = time.time()
        c_cycles = OUTPUTS_ROOT / ctx.name / ctx.track / args.prompt / "c" / "cycles" / vid
        try:
            ttl_c = run_system_c(ctx, system_a_ttl, snippet, cycles_dir=c_cycles)
        except Exception as e:
            ttl_c = system_a_ttl
            entry["system_c_error"] = str(e)[:100]
        entry["system_c_sec"] = round(time.time() - t0, 2)
        entry["system_c_llm_calls"] = 3
        c_path.parent.mkdir(parents=True, exist_ok=True)
        c_path.write_text(ttl_c)

    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", default="chr")
    ap.add_argument("--track", choices=("schema", "ontology"), default="schema")
    ap.add_argument("--prompt", default="full")
    ap.add_argument("--systems", default="a,b,c")
    ap.add_argument("--workers", type=int, default=5,
                    help="Parallel vignette workers (default 5; OpenRouter rate-limits ~5-10).")
    args = ap.parse_args()

    ctx = load_schema(args.schema, args.track)
    _pick_builder(args.prompt, args.track)
    vignettes = _iter_vignettes(None)
    print(f"\nParallel extraction · {args.schema}/{args.track}/{args.prompt}")
    print(f"  vignettes: {len(vignettes)}")
    print(f"  systems:   {args.systems}")
    print(f"  workers:   {args.workers}\n")

    t_start = time.time()
    timings: list[dict] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, args, vp): vp for vp in vignettes}
        completed = 0
        for fut in as_completed(futures):
            vp = futures[fut]
            completed += 1
            try:
                result = fut.result()
                if result.get("skipped"):
                    print(f"[{completed}/{len(vignettes)}] {result['vignette']:<25} SKIPPED (already complete)")
                else:
                    sec_a = result.get("system_a_sec", 0)
                    sec_b = result.get("system_b_sec", 0)
                    sec_c = result.get("system_c_sec", 0)
                    sec_total = sec_a + sec_b + sec_c
                    conf_b = "OK" if result.get("system_b_conforms") else "fail"
                    print(f"[{completed}/{len(vignettes)}] {result['vignette']:<25} "
                          f"A={sec_a:.0f}s B={sec_b:.0f}s({conf_b}) C={sec_c:.0f}s  total={sec_total:.0f}s")
                timings.append(result)
            except Exception as exc:
                print(f"[{completed}/{len(vignettes)}] {vp.name} ERROR: {exc}")
                timings.append({"vignette": _vignette_id(vp), "error": str(exc)[:200]})

    # Persist timing summary alongside the cell outputs.
    timing_path = OUTPUTS_ROOT / ctx.name / ctx.track / args.prompt / "timing.json"
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    timing_path.write_text(json.dumps(timings, indent=2, default=str))

    elapsed = time.time() - t_start
    print(f"\nElapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Timing -> {timing_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
