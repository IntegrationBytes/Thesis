"""System D — compute-fair baseline (no verifier feedback).

Reviewer's question (2026-05-22): "If I just loop the language model four
times without verifier feedback the performance might still improve.
What is the verifier doing?"

System B runs the extractor once, then up to MAX_SHACL_RETRIES=3 retry
calls with SHACL feedback. Worst-case token budget per vignette = 4 LLM
calls. System D matches this budget BUT without feedback: it just calls
the LLM 4 times with the same prompt and keeps the last output.

If System B beats System D on SHACL conformance, the verifier signal is
the cause (not just more compute). If they are equivalent, the loop is
basically inference-time scaling and SHACL adds no real signal.

Outputs land at::

    evaluation/outputs[_freemodel]/chr/schema/full/d/vignette_NNN.ttl

CLI::

    # Gemini schema D:
    python scripts/run_system_d.py --schema chr --track schema --workers 5

    # gpt-oss schema D:
    OPENROUTER_MODEL_OVERRIDE=openai/gpt-oss-120b \\
    OUTPUTS_ROOT_OVERRIDE=$(pwd)/evaluation/outputs_freemodel \\
    python scripts/run_system_d.py --schema chr --track schema --workers 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from extract import (  # noqa: E402
    OUTPUTS_ROOT,
    _iter_vignettes,
    _output_path,
    _vignette_id,
    _pick_builder,
    call_llm_for_turtle,
    load_schema,
    strip_markdown_fences,
)
from prompts import build_full_schema_prompt  # noqa: E402

N_CALLS = 4  # matches System B's max budget (1 initial + 3 retries)


def _is_complete(path: Path) -> bool:
    """Output complete iff exists and parses to >0 triples."""
    if not path.exists():
        return False
    try:
        from rdflib import Graph
        return len(Graph().parse(path.as_posix(), format="turtle")) > 0
    except Exception:
        return False


def process_one(args, vignette_path: Path) -> dict:
    """Run N_CALLS sequential LLM calls on one vignette, save last output."""
    ctx = load_schema(args.schema, args.track)
    vid = _vignette_id(vignette_path)
    snippet = vignette_path.read_text().strip()

    d_path = _output_path(ctx, args.prompt, "d", vid)
    if _is_complete(d_path):
        return {"vignette": vid, "skipped": True}

    builder = _pick_builder(args.prompt, args.track)
    entry: dict = {"vignette": vid, "calls": []}

    last_ttl = ""
    for i in range(N_CALLS):
        t0 = time.time()
        try:
            prompt = builder(ctx, snippet)
            raw = call_llm_for_turtle(prompt)
            last_ttl = strip_markdown_fences(raw)
            entry["calls"].append({
                "i": i,
                "sec": round(time.time() - t0, 2),
                "triples": last_ttl.count("\n"),  # rough
            })
        except Exception as e:
            entry["calls"].append({
                "i": i,
                "error": str(e)[:100],
            })
            if i == 0:
                last_ttl = f"# LLM call failed: {e}"

    d_path.parent.mkdir(parents=True, exist_ok=True)
    d_path.write_text(last_ttl)
    entry["total_sec"] = sum(c.get("sec", 0) for c in entry["calls"])
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", default="chr")
    ap.add_argument("--track", default="schema",
                    choices=("schema", "ontology"))
    ap.add_argument("--prompt", default="full")
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()

    ctx = load_schema(args.schema, args.track)
    _pick_builder(args.prompt, args.track)
    vignettes = _iter_vignettes(None)
    print(f"\nSystem D · {args.schema}/{args.track}/{args.prompt}")
    print(f"  vignettes: {len(vignettes)}")
    print(f"  N_CALLS:   {N_CALLS} per vignette")
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
                    print(f"[{completed}/{len(vignettes)}] {result['vignette']:<25} SKIPPED")
                else:
                    sec = result.get("total_sec", 0)
                    print(f"[{completed}/{len(vignettes)}] {result['vignette']:<25} "
                          f"{N_CALLS} calls = {sec:.0f}s")
                timings.append(result)
            except Exception as exc:
                print(f"[{completed}/{len(vignettes)}] {vp.name} ERROR: {exc}")
                timings.append({"vignette": _vignette_id(vp),
                                 "error": str(exc)[:200]})

    timing_path = (OUTPUTS_ROOT / ctx.name / ctx.track / args.prompt
                   / "timing_d.json")
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    timing_path.write_text(json.dumps(timings, indent=2, default=str))
    print(f"\nElapsed: {time.time() - t_start:.0f}s")
    print(f"Timing -> {timing_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
