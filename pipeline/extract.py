"""Schema-agnostic extraction runner — A/B/C ablation pipeline.
Runs the A/B/C ablation against any registered schema + track, using a
prompt variant from :mod:`prompts`.
    System A — zero-shot LLM extraction (no SHACL).
    System B — System A output -> SHACL-feedback correction loop (up to ``MAX_SHACL_RETRIES`` rounds).
    System C — System A output -> LLM self-correction loop (no SHACL).
The runner itself knows nothing about "CHR" — it asks :mod:`prompts` for
a :class:`SchemaContext` and builds prompts + validates output against
``ctx.shapes_path`` and ``ctx.tbox_path``. Adding a new schema is "add a
factory in prompts.py" rather than "edit extract.py".
Outputs land under::
    evaluation/outputs/<schema>/<track>/<prompt-variant>/<system>/<vignette>.ttl
Usage::
    # Schema-track full prompt, all systems:
    python pipeline/extract.py --schema chr --track schema --systems a,b,c
    # Ontology-track:
    python pipeline/extract.py --schema chr --track ontology --prompt ontology --systems a,b
    # Single vignette (cheap smoke test):
    python pipeline/extract.py --schema chr --track schema \\
        --systems a --only vignette_001"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable
from dotenv import load_dotenv
from openai import OpenAI
import pyshacl
from rdflib import Graph

# ---------------------------------------------------------------------------
# Schema-agnostic prompts.
# ---------------------------------------------------------------------------
_PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_PIPELINE_DIR))
from prompts import (  # noqa: E402  (path-dependent import by design)
    SchemaContext,
    build_correction_prompt,
    build_full_schema_prompt,
    build_ontology_prompt,
    build_self_correction_prompt,
    chr_context,
)

# ---------------------------------------------------------------------------
# Paths + config.
# ---------------------------------------------------------------------------
PROJECT_ROOT = _PIPELINE_DIR.parent
load_dotenv(PROJECT_ROOT / ".env")

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
# Model can be overridden via env var for multi-model replication runs
# (e.g. open-weight rerun for supervisor's free-model request).
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL_OVERRIDE", "google/gemini-2.0-flash-001")
MAX_SHACL_RETRIES = 3
CORPUS_ROOT = PROJECT_ROOT / "evaluation" / "corpus"
# Outputs root can be redirected via env var so a free-model rerun
# doesn't overwrite the Gemini-baseline outputs.
OUTPUTS_ROOT = Path(os.getenv("OUTPUTS_ROOT_OVERRIDE",
                              str(PROJECT_ROOT / "evaluation" / "outputs")))

# Map prompt-variant name -> (builder, tracks it supports).
# None in the track-set means "any track".
_PROMPT_BUILDERS: dict[str, tuple[Callable[..., str], set[str] | None]] = {
    "full":     (build_full_schema_prompt, None),
    "ontology": (build_ontology_prompt,   {"ontology"}),
}


# ---------------------------------------------------------------------------
# Schema loading (extend with new factories in prompts.py as needed).
# ---------------------------------------------------------------------------
def load_schema(name: str, track: str) -> SchemaContext:
    if name == "chr":
        return chr_context(track)
    raise ValueError(
        f"unknown schema {name!r}; supported: 'chr' "
        f"(extend in prompts.py to add more)"
    )


# ---------------------------------------------------------------------------
# LLM client.
# ---------------------------------------------------------------------------
def call_llm_for_turtle(prompt: str, *, max_attempts: int = 6) -> str:
    """Send prompt to the LLM via OpenRouter and return the raw response.
    OpenRouter proxies Google AI Studio and regularly returns 429
    (rate-limit) or 504 (upstream aborted) during bursty runs. Rather
    than let those failures silently poison the batch — each failure
    becomes a ``# LLM call failed:`` comment that parses to zero triples —
    retry up to ``max_attempts`` with linear backoff.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Add it to .env or your environment."
        )
    client = OpenAI(base_url=OPENROUTER_BASE, api_key=api_key)
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            # OpenRouter occasionally returns a 200 with no .choices (empty
            # list, None, or missing entirely) when the upstream provider
            # silently drops the completion — typically on rate-limit
            # overflow that the proxy didn't translate into an HTTP error.
            # Treat this as a transient failure so the retry loop engages
            # instead of raising `'NoneType' object is not subscriptable`.
            choices = getattr(response, "choices", None) or []
            if not choices:
                raise RuntimeError(
                    "empty response.choices (OpenRouter upstream dropped completion); "
                    f"id={getattr(response, 'id', '?')}"
                )
            msg_obj = getattr(choices[0], "message", None)
            content = getattr(msg_obj, "content", None) if msg_obj else None
            return content or ""
        except Exception as exc:
            msg = str(exc)
            retryable = any(code in msg for code in (
                "429", "504", "502", "503", "empty response.choices",
            ))
            if attempt < max_attempts and retryable:
                # Exponential-style backoff for sustained rate limits during
                # n=100 runs against OpenRouter's shared Gemini quota:
                # 15s, 30s, 60s, 120s, 240s. Total worst-case = 7m45s before
                # the call gives up — comfortable margin for upstream rate
                # limits that typically clear within a few minutes.
                wait = 15 * (2 ** (attempt - 1))
                print(f"    [LLM] transient error ({msg[:80]}...); "
                      f"retry {attempt}/{max_attempts - 1} in {wait}s")
                time.sleep(wait)
                last_exc = exc
                continue
            raise
    # Exhausted retries — raise the last exception so the caller sees it.
    assert last_exc is not None
    raise last_exc


def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences + optional language tag from LLM output."""
    if "```" not in text:
        return text.strip()
    parts = text.split("```")
    block = parts[1].strip()
    lines = block.splitlines()
    if lines and lines[0].strip().lower() in ("turtle", "ttl", "rdf"):
        block = "\n".join(lines[1:])
    return block.strip()


# ---------------------------------------------------------------------------
# SHACL validation — inference='none' on both tracks.
# ---------------------------------------------------------------------------
# Why 'none': chr:hasPatient declares three rdfs:domain values
# (ClinicalVisit, MedicalProcedure, Measurement). With rdfs-inference on,
# pyshacl cross-types every focus node of hasPatient as all three classes,
# which makes every class-targeted shape fire on every visit — a flood of
# spurious violations. Skipping inference is correct for a closed-world
# SHACL check on extracted ABoxes.
def validate_with_shacl(ttl_content: str, ctx: SchemaContext) -> tuple[bool, str]:
    """Run pyshacl and return (conforms, human-readable results_text)."""
    try:
        data_graph = Graph()
        data_graph.parse(data=ttl_content, format="turtle")
    except Exception as parse_err:
        return False, f"Turtle parse error (not valid RDF): {parse_err}"
    try:
        shapes_graph = Graph().parse(ctx.shapes_path.as_posix(), format="turtle")
        ont_graph = Graph().parse(ctx.tbox_path.as_posix(), format="turtle")
        conforms, _, results_text = pyshacl.validate(
            data_graph,
            shacl_graph=shapes_graph,
            ont_graph=ont_graph,
            inference="none",
            abort_on_first=False,
            meta_shacl=False,
            advanced=True,
            js=False,
        )
        return conforms, results_text
    except Exception as shacl_err:
        return False, f"SHACL engine error: {shacl_err}"


def filter_violations_only(results_text: str) -> str:
    """Drop Warning/Info blocks from a pyshacl text report.

    Each result block starts with 'Constraint Violation'; we keep blocks
    whose Severity is sh:Violation.
    """
    sections = re.split(r"\n(?=Constraint )", results_text)
    kept = [s for s in sections if "sh:Warning" not in s and "sh:Info" not in s]
    return "\n".join(kept).strip()


# ---------------------------------------------------------------------------
# Systems A / B / C.
# ---------------------------------------------------------------------------
def _pick_builder(prompt_variant: str, track: str):
    """Return the prompt builder for ``(variant, track)`` or raise."""
    if prompt_variant not in _PROMPT_BUILDERS:
        raise ValueError(
            f"unknown prompt variant {prompt_variant!r}; "
            f"available: {sorted(_PROMPT_BUILDERS)}"
        )
    builder, allowed_tracks = _PROMPT_BUILDERS[prompt_variant]
    if allowed_tracks is not None and track not in allowed_tracks:
        raise ValueError(
            f"prompt variant {prompt_variant!r} not valid on track "
            f"{track!r} (allowed: {sorted(allowed_tracks)})"
        )
    return builder


def run_system_a(ctx: SchemaContext, snippet_text: str, prompt_variant: str) -> str:
    """Zero-shot extraction with the chosen prompt variant."""
    builder = _pick_builder(prompt_variant, ctx.track)
    prompt = builder(ctx, snippet_text)
    raw = call_llm_for_turtle(prompt)
    return strip_markdown_fences(raw)


def run_system_b(ctx: SchemaContext, system_a_ttl: str, cycles_dir: Path | None = None) -> tuple[str, bool, int]:
    """SHACL correction loop over a prior System A output.
    Returns ``(final_ttl, conforms, extra_llm_calls)``. ``extra_llm_calls``
    is the number of correction calls (0 if System A was already
    conformant). Only Violation-level results are fed back to the LLM —
    Warning-level results are stripped so the model doesn't hallucinate
    triples to satisfy advisories.
    When ``cycles_dir`` is provided, every intermediate TTL and the SHACL
    report that drove the next correction are persisted there as
    ``cycle_00.ttl`` → ``cycle_NN.ttl`` + ``violations_NN.txt`` +
    ``summary.json``. This is the per-cycle trace Remzi asked for at the
    22 April supervisor meeting.
    """
    ttl_content = system_a_ttl
    llm_calls = 0
    trace: list[dict] = []

    def _record(cycle: int, ttl: str, conforms: bool, results_text: str) -> None:
        if cycles_dir is None:
            return
        cycles_dir.mkdir(parents=True, exist_ok=True)
        (cycles_dir / f"cycle_{cycle:02d}.ttl").write_text(ttl)
        if not conforms:
            (cycles_dir / f"violations_{cycle:02d}.txt").write_text(results_text)
        trace.append({
            "cycle": cycle,
            "conforms": conforms,
            "results_summary": results_text[:500],
        })

    conforms, results_text = validate_with_shacl(ttl_content, ctx)
    _record(0, ttl_content, conforms, results_text)
    if conforms:
        if cycles_dir is not None:
            (cycles_dir / "summary.json").write_text(
                json.dumps({"system": "b", "trace": trace,
                            "final_conforms": True, "llm_calls": 0}, indent=2))
        return ttl_content, True, llm_calls

    # Distinguish parse failures from real constraint violations. A parse
    # failure is *not* SHACL-conformance; treating it as such silently
    # masks unparseable LLM output. Instead, feed the parse error back as
    # violations_text so the correction prompt asks the LLM to produce
    # syntactically valid Turtle.
    is_parse_error = results_text.startswith("Turtle parse error")
    if is_parse_error:
        violations_text = (
            f"Turtle syntax error — rewrite the file as valid Turtle.\n\n"
            f"{results_text}"
        )
    else:
        violations_text = filter_violations_only(results_text)
        if not violations_text or "Constraint" not in violations_text:
            # Only warnings remained — treat as conforming for System B purposes.
            if cycles_dir is not None:
                (cycles_dir / "summary.json").write_text(
                    json.dumps({"system": "b", "trace": trace,
                                "final_conforms": True, "llm_calls": 0,
                                "note": "warnings only"}, indent=2))
            return ttl_content, True, llm_calls

    for attempt in range(1, MAX_SHACL_RETRIES + 1):
        print(f"    SHACL retry {attempt}/{MAX_SHACL_RETRIES}...")
        prompt = build_correction_prompt(ctx, ttl_content, violations_text)
        raw = call_llm_for_turtle(prompt)
        llm_calls += 1
        ttl_content = strip_markdown_fences(raw)
        conforms, results_text = validate_with_shacl(ttl_content, ctx)
        _record(attempt, ttl_content, conforms, results_text)
        if conforms:
            if cycles_dir is not None:
                (cycles_dir / "summary.json").write_text(
                    json.dumps({"system": "b", "trace": trace,
                                "final_conforms": True,
                                "llm_calls": llm_calls}, indent=2))
            return ttl_content, True, llm_calls
        is_parse_error = results_text.startswith("Turtle parse error")
        if is_parse_error:
            violations_text = (
                f"Turtle syntax error — rewrite the file as valid Turtle.\n\n"
                f"{results_text}"
            )
            continue
        violations_text = filter_violations_only(results_text)
        if not violations_text or "Constraint" not in violations_text:
            if cycles_dir is not None:
                (cycles_dir / "summary.json").write_text(
                    json.dumps({"system": "b", "trace": trace,
                                "final_conforms": True,
                                "llm_calls": llm_calls,
                                "note": "warnings only after retry"},
                               indent=2))
            return ttl_content, True, llm_calls

    if cycles_dir is not None:
        (cycles_dir / "summary.json").write_text(
            json.dumps({"system": "b", "trace": trace,
                        "final_conforms": False,
                        "llm_calls": llm_calls,
                        "note": "retry budget exhausted"}, indent=2))
    return ttl_content, False, llm_calls


def run_system_c(
    ctx: SchemaContext, system_a_ttl: str, snippet_text: str,
    cycles_dir: Path | None = None,
) -> str:
    """LLM self-correction loop — no SHACL.

    When ``cycles_dir`` is set, writes ``cycle_00.ttl`` (the System A input)
    through ``cycle_NN.ttl`` (final pass) plus ``summary.json``. Parallel
    to System B's per-cycle trace; same per-cycle artefact layout.
    """
    ttl_content = system_a_ttl
    trace: list[dict] = []
    if cycles_dir is not None:
        cycles_dir.mkdir(parents=True, exist_ok=True)
        (cycles_dir / "cycle_00.ttl").write_text(ttl_content)
        trace.append({"cycle": 0, "source": "system_a"})

    for attempt in range(1, MAX_SHACL_RETRIES + 1):
        print(f"    Self-correction pass {attempt}/{MAX_SHACL_RETRIES}...")
        prompt = build_self_correction_prompt(ctx, ttl_content, snippet_text)
        raw = call_llm_for_turtle(prompt)
        ttl_content = strip_markdown_fences(raw)
        if cycles_dir is not None:
            (cycles_dir / f"cycle_{attempt:02d}.ttl").write_text(ttl_content)
            trace.append({"cycle": attempt, "source": "self_correction"})

    if cycles_dir is not None:
        (cycles_dir / "summary.json").write_text(
            json.dumps({"system": "c", "trace": trace,
                        "passes": MAX_SHACL_RETRIES}, indent=2))
    return ttl_content


# ---------------------------------------------------------------------------
# Vignette discovery.
# ---------------------------------------------------------------------------
def _vignette_id(path: Path) -> str:
    """``Path('.../vignette_001.txt')`` -> ``'vignette_001'``."""
    return path.stem


def _iter_vignettes(only: str | None) -> list[Path]:
    vignettes_dir = CORPUS_ROOT / "vignettes"
    if not vignettes_dir.exists():
        raise FileNotFoundError(
            f"corpus vignettes directory missing: {vignettes_dir}"
        )
    paths = sorted(vignettes_dir.glob("vignette_*.txt"))
    if only:
        paths = [p for p in paths if _iter_matches(p, only)]
        if not paths:
            raise ValueError(f"--only {only!r} matched no vignettes")
    return paths


def _iter_matches(path: Path, only: str) -> bool:
    """Match ``--only vignette_001`` against full filenames."""
    return _vignette_id(path) == only or path.stem == only


# ---------------------------------------------------------------------------
# Output paths.
# ---------------------------------------------------------------------------
def _output_path(
    ctx: SchemaContext, prompt_variant: str, system: str, vid: str
) -> Path:
    """evaluation/outputs/<schema>/<track>/<variant>/<system>/<vid>.ttl"""
    return (
        OUTPUTS_ROOT
        / ctx.name
        / ctx.track
        / prompt_variant
        / system
        / f"{vid}.ttl"
    )


# ---------------------------------------------------------------------------
# Runner.
# ---------------------------------------------------------------------------
def run(
    schema: str,
    track: str,
    systems: set[str],
    prompt_variant: str,
    only: str | None,
) -> dict:
    ctx = load_schema(schema, track)
    _pick_builder(prompt_variant, track)  # validates variant × track early

    print(f"\nExtract pipeline")
    print(f"  schema:  {schema}")
    print(f"  track:   {track}")
    print(f"  prompt:  {prompt_variant}")
    print(f"  systems: {','.join(sorted(systems))}")
    print(f"  shapes:  {ctx.shapes_path.relative_to(PROJECT_ROOT)}")
    print(f"  tbox:    {ctx.tbox_path.relative_to(PROJECT_ROOT)}")

    vignettes = _iter_vignettes(only)
    print(f"  vignettes: {len(vignettes)} "
          f"({', '.join(_vignette_id(v) for v in vignettes)})\n")

    batch_start = time.time()
    timing: list[dict] = []

    for vignette_path in vignettes:
        vid = _vignette_id(vignette_path)
        snippet = vignette_path.read_text().strip()
        print(f"=== {vid} ({vignette_path.name}) ===")
        entry: dict = {"vignette": vid}

        # ---- System A ----
        system_a_ttl = ""
        if systems & {"a", "b", "c"}:
            print("  [System A] zero-shot extraction...")
            t0 = time.time()
            try:
                system_a_ttl = run_system_a(ctx, snippet, prompt_variant)
            except Exception as e:
                print(f"  [System A] ERROR: {e}")
                system_a_ttl = f"# LLM call failed: {e}"
            t_a = time.time() - t0
            entry["system_a_sec"] = round(t_a, 2)
            entry["system_a_llm_calls"] = 1

            if "a" in systems:
                out = _output_path(ctx, prompt_variant, "a", vid)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(system_a_ttl)
                print(f"  [System A] wrote {out.relative_to(PROJECT_ROOT)}"
                      f"  ({t_a:.1f}s)")

        # ---- System B ----
        if "b" in systems:
            print("  [System B] SHACL correction loop...")
            t0 = time.time()
            b_cycles_dir = (
                OUTPUTS_ROOT / ctx.name / ctx.track / prompt_variant
                / "b" / "cycles" / vid
            )
            try:
                system_b_ttl, conforms_b, b_calls = run_system_b(
                    ctx, system_a_ttl, cycles_dir=b_cycles_dir,
                )
            except Exception as e:
                print(f"  [System B] ERROR: {e}")
                system_b_ttl, conforms_b, b_calls = system_a_ttl, False, 0
            t_b = time.time() - t0
            entry["system_b_sec"] = round(t_b, 2)
            entry["system_b_conforms"] = conforms_b
            entry["system_b_extra_llm_calls"] = b_calls

            out = _output_path(ctx, prompt_variant, "b", vid)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(system_b_ttl)
            status = "SHACL-VALID" if conforms_b else "best-effort"
            print(f"  [System B] wrote {out.relative_to(PROJECT_ROOT)}"
                  f"  ({status}, {t_b:.1f}s, +{b_calls} LLM calls)")

        # ---- System C ----
        if "c" in systems:
            print("  [System C] LLM self-correction (no SHACL)...")
            t0 = time.time()
            c_cycles_dir = (
                OUTPUTS_ROOT / ctx.name / ctx.track / prompt_variant
                / "c" / "cycles" / vid
            )
            try:
                system_c_ttl = run_system_c(
                    ctx, system_a_ttl, snippet, cycles_dir=c_cycles_dir,
                )
            except Exception as e:
                print(f"  [System C] ERROR: {e}")
                system_c_ttl = system_a_ttl
            t_c = time.time() - t0
            entry["system_c_sec"] = round(t_c, 2)
            entry["system_c_llm_calls"] = MAX_SHACL_RETRIES

            out = _output_path(ctx, prompt_variant, "c", vid)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(system_c_ttl)
            print(f"  [System C] wrote {out.relative_to(PROJECT_ROOT)}"
                  f"  ({t_c:.1f}s)")

        timing.append(entry)
        print()

    # ---- Persist timing ----
    elapsed = time.time() - batch_start
    timing_out = (
        OUTPUTS_ROOT / ctx.name / ctx.track / prompt_variant / "timing.json"
    )
    timing_out.parent.mkdir(parents=True, exist_ok=True)
    timing_out.write_text(json.dumps(timing, indent=2))
    print(f"Timing -> {timing_out.relative_to(PROJECT_ROOT)}")
    print(f"Total elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    return {
        "elapsed_sec": round(elapsed, 2),
        "timing_path": str(timing_out),
        "vignettes": len(vignettes),
    }


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def _parse_systems(raw: str) -> set[str]:
    picks = {s.strip().lower() for s in raw.split(",") if s.strip()}
    bad = picks - {"a", "b", "c"}
    if bad:
        raise argparse.ArgumentTypeError(
            f"invalid system(s): {sorted(bad)} (choose from a, b, c)"
        )
    if not picks:
        raise argparse.ArgumentTypeError("must choose at least one system")
    return picks


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--schema", default="chr",
        help="Schema name (default: chr; extend via prompts.py).",
    )
    parser.add_argument(
        "--track", choices=("schema", "ontology"), default="schema",
        help="Encoding track (default: schema).",
    )
    parser.add_argument(
        "--prompt", choices=tuple(_PROMPT_BUILDERS), default="full",
        help="Prompt variant (default: full).",
    )
    parser.add_argument(
        "--systems", type=_parse_systems, default={"a", "b", "c"},
        help="Comma-separated systems to run (default: a,b,c).",
    )
    parser.add_argument(
        "--only", default=None,
        help="Only process this vignette id (e.g. vignette_001).",
    )
    args = parser.parse_args()

    run(
        schema=args.schema,
        track=args.track,
        systems=args.systems,
        prompt_variant=args.prompt,
        only=args.only,
    )


if __name__ == "__main__":
    main()
