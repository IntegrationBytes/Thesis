"""LLM-as-judge post-hoc evaluator.

Rates generated knowledge graphs against source text + gold using a
DIFFERENT-FAMILY LLM (Llama 3.3 70B by default — different family from
the gpt-oss-120b extractor) to reduce same-family judge bias.
Three Likert scores per (source, gold, generated) triple:

    - faithfulness : do generated facts match the source text?
    - completeness : are gold facts covered in the generated graph?
    - hallucination: any generated facts not supported by source/gold?
                     (higher = MORE hallucination — inverse of quality)

The judge is post-hoc: it walks existing extraction outputs and writes
scores to ``evaluation/outputs/judge_scores.json``. No re-extraction.

Usage::

    # Score all outputs for a given (schema, track, prompt, system):
    python pipeline/llm_judge.py --schema chr --track schema \\
        --prompt full --system b --limit 10

    # Score everything with stratification (vignettes 001-100 vs 101-200):
    python pipeline/llm_judge.py --schema chr --track schema \\
        --prompt full --system b
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# ---------------------------------------------------------------------------
# Paths + config.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_PIPELINE_DIR))

VIGNETTES_DIR = _REPO_ROOT / "evaluation" / "corpus" / "vignettes"
GOLD_DIR = _REPO_ROOT / "evaluation" / "corpus" / "abox_gold"
JUDGE_OUTPUT = _REPO_ROOT / "evaluation" / "outputs" / "judge_scores.json"

load_dotenv(_REPO_ROOT / ".env")

# Judge model is intentionally different-family from the
# gpt-oss-120b extractor to reduce same-family bias.
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "meta-llama/llama-3.3-70b-instruct")


JUDGE_SYSTEM_PROMPT = """\
You are an impartial knowledge-graph evaluation judge. You will be shown:
  (1) the SOURCE TEXT — a clinical vignette.
  (2) the GOLD KNOWLEDGE GRAPH — the deterministic ground truth derived
      from the underlying FHIR data (treat as authoritative).
  (3) the GENERATED KNOWLEDGE GRAPH — produced by an LLM extractor.

Rate the generated graph on three dimensions, each on a 1–5 Likert scale.

  FAITHFULNESS (1=many fabricated claims, 5=every triple traces to source)
    Does every generated triple correspond to information that IS in the
    source text (or is trivially derivable from it, e.g. patient label)?

  COMPLETENESS (1=most gold triples missing, 5=all gold information covered)
    Does the generated graph express the same factual content as the gold
    (modulo URI shape and ordering)?

  HALLUCINATION (1=no hallucination, 5=heavy hallucination)
    Are there generated triples that are NEITHER in the source text NOR in
    the gold? Note: this is the INVERSE of quality — higher means worse.

Be a strict judge but charitable to surface variation (URI shape, IRI
minting, label phrasing). Penalise only for content errors.

Return ONLY a JSON object with this exact shape:

{
  "faithfulness": <int 1-5>,
  "completeness": <int 1-5>,
  "hallucination": <int 1-5>,
  "rationale": {
    "faithfulness": "<one sentence>",
    "completeness": "<one sentence>",
    "hallucination": "<one sentence>"
  }
}
"""


# ---------------------------------------------------------------------------
# LLM client.
# ---------------------------------------------------------------------------
def call_judge(source_text: str, gold_ttl: str, generated_ttl: str,
               *, max_attempts: int = 4) -> dict:
    """Send a single judging request and return the parsed JSON.

    On non-JSON or transient failure, retries up to ``max_attempts``. On
    permanent failure returns a dict with all scores=None and an "error"
    key set.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Add it to .env or your environment."
        )
    client = OpenAI(base_url=OPENROUTER_BASE, api_key=api_key)

    user_prompt = (
        "SOURCE TEXT\n-----------\n"
        f"{source_text.strip()}\n\n"
        "GOLD KNOWLEDGE GRAPH (Turtle)\n-----------------------------\n"
        f"{gold_ttl.strip()}\n\n"
        "GENERATED KNOWLEDGE GRAPH (Turtle)\n----------------------------------\n"
        f"{generated_ttl.strip()}"
    )

    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            choices = getattr(response, "choices", None) or []
            if not choices:
                raise RuntimeError("empty response.choices")
            content = (choices[0].message.content or "").strip()
            return _parse_judge_response(content)
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            retryable = any(code in msg for code in (
                "429", "504", "502", "503", "empty response.choices",
            ))
            if attempt < max_attempts and retryable:
                wait = 5 * attempt
                print(f"    [judge] transient error ({msg[:80]}...); retry in {wait}s")
                time.sleep(wait)
                continue
            break

    return {
        "faithfulness": None,
        "completeness": None,
        "hallucination": None,
        "rationale": {},
        "error": str(last_exc) if last_exc else "unknown",
    }


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_judge_response(content: str) -> dict:
    """Pull the first JSON object out of the LLM response."""
    match = _JSON_BLOCK_RE.search(content)
    if not match:
        return {
            "faithfulness": None,
            "completeness": None,
            "hallucination": None,
            "rationale": {},
            "error": f"no JSON found in response: {content[:200]}",
        }
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return {
            "faithfulness": None,
            "completeness": None,
            "hallucination": None,
            "rationale": {},
            "error": f"JSON decode error: {exc}",
        }
    # Clamp to 1..5, accept None
    for key in ("faithfulness", "completeness", "hallucination"):
        v = parsed.get(key)
        if isinstance(v, (int, float)):
            parsed[key] = max(1, min(5, int(v)))
        else:
            parsed[key] = None
    parsed.setdefault("rationale", {})
    return parsed


# ---------------------------------------------------------------------------
# Walker.
# ---------------------------------------------------------------------------
def _resolve_output_dir(schema: str, track: str, prompt: str, system: str,
                       *, freemodel: bool = False) -> Path:
    """Return the directory holding generated TTLs for (schema, track, prompt, system)."""
    base = "outputs_freemodel" if freemodel else "outputs"
    return _REPO_ROOT / "evaluation" / base / schema / track / prompt / system


def _stratum(vid: str) -> str:
    """'general' for vignette_001-100, 'complex' for 101-200, else 'other'."""
    m = re.match(r"vignette_(\d+)", vid)
    if not m:
        return "other"
    n = int(m.group(1))
    if 1 <= n <= 100:
        return "general"
    if 101 <= n <= 200:
        return "complex"
    return "other"


def run_judge(schema: str, track: str, prompt: str, system: str,
              *, freemodel: bool = False,
              limit: int | None = None,
              only_stratum: str | None = None) -> dict:
    """Walk the output dir, judge each (source, gold, generated), return aggregated result."""
    out_dir = _resolve_output_dir(schema, track, prompt, system, freemodel=freemodel)
    if not out_dir.exists():
        print(f"[ERROR] output directory not found: {out_dir}")
        return {}

    generated_paths = sorted(out_dir.glob("vignette_*.ttl"))
    if not generated_paths:
        print(f"[ERROR] no vignette outputs in: {out_dir}")
        return {}

    if limit is not None:
        generated_paths = generated_paths[:limit]

    results = {}
    model_tag = "freemodel" if freemodel else "main"
    judge_label = f"{model_tag}/{schema}/{track}/{prompt}/{system}"
    print(f"[judge] {judge_label} :: {len(generated_paths)} outputs")

    for gen_path in generated_paths:
        vid = gen_path.stem
        stratum = _stratum(vid)
        if only_stratum and stratum != only_stratum:
            continue

        vignette_path = VIGNETTES_DIR / f"{vid}.txt"
        gold_path = GOLD_DIR / f"{vid}_gold_schema.ttl"
        if not (vignette_path.exists() and gold_path.exists()):
            print(f"  [skip] {vid} (missing source or gold)")
            continue

        source_text = vignette_path.read_text()
        gold_ttl = gold_path.read_text()
        generated_ttl = gen_path.read_text()

        # Skip empty / error generations
        if "# LLM call failed" in generated_ttl or len(generated_ttl.strip()) < 50:
            results[vid] = {
                "stratum": stratum,
                "faithfulness": None,
                "completeness": None,
                "hallucination": None,
                "rationale": {},
                "skip_reason": "empty or failed generation",
            }
            continue

        print(f"  [{stratum}] {vid}...", end=" ", flush=True)
        score = call_judge(source_text, gold_ttl, generated_ttl)
        score["stratum"] = stratum
        results[vid] = score

        f, c, h = score.get("faithfulness"), score.get("completeness"), score.get("hallucination")
        print(f"F={f} C={c} H={h}")

    return {
        "label": judge_label,
        "judge_model": JUDGE_MODEL,
        "vignettes": results,
        "summary": _aggregate(results),
    }


def _aggregate(results: dict) -> dict:
    """Mean and stratified means."""
    def mean_of(rows, key):
        vals = [r[key] for r in rows if isinstance(r.get(key), int)]
        return sum(vals) / len(vals) if vals else None

    all_rows = list(results.values())
    general = [r for r in all_rows if r.get("stratum") == "general"]
    complex_ = [r for r in all_rows if r.get("stratum") == "complex"]

    return {
        "n_total": len(all_rows),
        "n_general": len(general),
        "n_complex": len(complex_),
        "overall": {
            "faithfulness": mean_of(all_rows, "faithfulness"),
            "completeness": mean_of(all_rows, "completeness"),
            "hallucination": mean_of(all_rows, "hallucination"),
        },
        "general": {
            "faithfulness": mean_of(general, "faithfulness"),
            "completeness": mean_of(general, "completeness"),
            "hallucination": mean_of(general, "hallucination"),
        },
        "complex": {
            "faithfulness": mean_of(complex_, "faithfulness"),
            "completeness": mean_of(complex_, "completeness"),
            "hallucination": mean_of(complex_, "hallucination"),
        },
    }


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--schema", default="chr")
    parser.add_argument("--track", choices=("schema", "ontology"), default="schema")
    parser.add_argument("--prompt", default="full")
    parser.add_argument("--system", choices=("a", "b", "c"), required=True)
    parser.add_argument(
        "--freemodel", action="store_true",
        help="Legacy flag: score the archived evaluation/outputs_freemodel/ "
             "outputs instead of the main evaluation/outputs/ (gpt-oss-120b).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process at most N vignettes (smoke test).",
    )
    parser.add_argument(
        "--stratum", choices=("general", "complex"), default=None,
        help="Only judge one stratum.",
    )
    parser.add_argument(
        "--output", type=Path, default=JUDGE_OUTPUT,
        help=f"Where to write the result JSON (default: {JUDGE_OUTPUT}).",
    )
    args = parser.parse_args()

    result = run_judge(
        schema=args.schema, track=args.track, prompt=args.prompt,
        system=args.system, freemodel=args.freemodel,
        limit=args.limit, only_stratum=args.stratum,
    )
    if not result:
        sys.exit(1)

    # Merge into existing judge_scores.json if present (additive across runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if args.output.exists():
        try:
            existing = json.loads(args.output.read_text())
        except Exception:
            existing = {}
    existing[result["label"]] = result
    args.output.write_text(json.dumps(existing, indent=2))
    print(f"\n[OK] wrote {args.output}")
    print(f"\nSummary for {result['label']}:")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
