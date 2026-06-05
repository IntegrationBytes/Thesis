"""Per-shape violation breakdown — supervisor's exact 22 May ask.

Quote: "show us the cases where SHACL actually helps."

For each SHACL shape (schema track) and each OWL violation type
(ontology track), count violations on System A vs B outputs across the
n=200 corpus. The result is two ranked lists:

  - SCHEMA TRACK: per-shape (A_violations, B_violations, fix_rate)
                  sortable by fix rate to show which shapes the retry
                  loop reliably fixes vs which resist correction.

  - ONTOLOGY TRACK: per-OWL-violation-type
                    (DisjointnessClash / FunctionalPropertyConflict /
                     RangeViolation) similar breakdown.

Stratified by connectivity bucket so we can also answer "where does
SHACL help most — on low-connectivity or high-connectivity inputs?"

Output: stdout table + JSON dump at
``evaluation/outputs/per_shape_violations.json``.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from extract import validate_with_shacl  # noqa: E402
from owl_validator import validate_with_owl  # noqa: E402
from prompts import chr_context  # noqa: E402

GOLD_DIR = ROOT / "evaluation/corpus/abox_gold"
CHR_ONT = ROOT / "evaluation/corpus/tbox/chr_ontology.owl.ttl"

# Vignette → bucket map (from connectivity_analysis output)
_BUCKETS_JSON = ROOT / "evaluation/outputs/connectivity_buckets.json"
_VID_TO_BUCKET: dict[str, str] = {}
if _BUCKETS_JSON.exists():
    _data = json.loads(_BUCKETS_JSON.read_text())
    for tier, info in _data["buckets"].items():
        for vid in info["vignettes"]:
            _VID_TO_BUCKET[vid] = tier

_SHAPE_RE = re.compile(r"Source Shape:\s*(\S+)")
_OWL_TYPE_RE = re.compile(r"^(\w+(?:Clash|Conflict|Violation))", re.MULTILINE)


def _shape_counts_schema(args) -> tuple[str, str, str, Counter]:
    """Per-vignette: return (vid, model, sys, Counter[shape]) for SHACL violations."""
    vid, ttl_path_str, model, sys_tag = args
    ttl_path = Path(ttl_path_str)
    counter: Counter = Counter()
    if not ttl_path.exists():
        return (vid, model, sys_tag, counter)
    ttl = ttl_path.read_text()
    if "# LLM call failed" in ttl:
        return (vid, model, sys_tag, counter)
    ctx = chr_context("schema")
    _, report = validate_with_shacl(ttl, ctx)
    if "Conforms: True" in report:
        return (vid, model, sys_tag, counter)
    for m in _SHAPE_RE.finditer(report):
        counter[m.group(1)] += 1
    return (vid, model, sys_tag, counter)


def _owl_counts(args) -> tuple[str, str, str, Counter]:
    """Per-vignette: return (vid, model, sys, Counter[owl_violation_type])."""
    vid, ttl_path_str, model, sys_tag = args
    ttl_path = Path(ttl_path_str)
    counter: Counter = Counter()
    if not ttl_path.exists():
        return (vid, model, sys_tag, counter)
    ttl = ttl_path.read_text()
    if "# LLM call failed" in ttl:
        return (vid, model, sys_tag, counter)
    ok, report = validate_with_owl(ttl, CHR_ONT)
    if ok:
        return (vid, model, sys_tag, counter)
    for m in _OWL_TYPE_RE.finditer(report):
        counter[m.group(1)] += 1
    return (vid, model, sys_tag, counter)


def _stratum(vid: str) -> str:
    return _VID_TO_BUCKET.get(vid, "other")


def main() -> None:
    # ---- Schema track ----
    print("[1/2] Schema track — per-shape violation counts (4 system variants × n=200)")
    schema_jobs = []
    for vid_path in sorted(GOLD_DIR.glob("vignette_*_gold_schema.ttl")):
        vid = vid_path.stem.replace("_gold_schema", "")
        for model, root in [
            ("gptoss", ROOT / "evaluation/outputs/chr/schema/full"),
        ]:
            for sys_tag in ("a", "b"):
                schema_jobs.append((vid, str(root / sys_tag / f"{vid}.ttl"), model, sys_tag))

    schema_results: list = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        futs = [pool.submit(_shape_counts_schema, j) for j in schema_jobs]
        for fut in as_completed(futs):
            schema_results.append(fut.result())

    # Aggregate per (model, sys, bucket) -> Counter[shape]
    schema_agg: dict = defaultdict(Counter)
    for vid, model, sys_tag, counter in schema_results:
        b = _stratum(vid)
        for shape, n in counter.items():
            schema_agg[(model, sys_tag, b)][shape] += n
            schema_agg[(model, sys_tag, "all")][shape] += n

    # ---- Ontology track ----
    print("[2/2] Ontology track — per-OWL-type violation counts")
    owl_jobs = []
    for vid_path in sorted(GOLD_DIR.glob("vignette_*_gold_schema.ttl")):
        vid = vid_path.stem.replace("_gold_schema", "")
        for model, root in [
            ("gptoss", ROOT / "evaluation/outputs/chr/ontology/ontology"),
        ]:
            for sys_tag in ("a", "b"):
                owl_jobs.append((vid, str(root / sys_tag / f"{vid}.ttl"), model, sys_tag))

    owl_results: list = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        futs = [pool.submit(_owl_counts, j) for j in owl_jobs]
        for fut in as_completed(futs):
            owl_results.append(fut.result())

    owl_agg: dict = defaultdict(Counter)
    for vid, model, sys_tag, counter in owl_results:
        b = _stratum(vid)
        for vt, n in counter.items():
            owl_agg[(model, sys_tag, b)][vt] += n
            owl_agg[(model, sys_tag, "all")][vt] += n

    # ---- Print tables ----
    print("\n" + "=" * 86)
    print("SCHEMA TRACK — SHACL shapes ranked by fix rate (overall, n=200)")
    print("=" * 86)
    for model in ("gptoss",):
        print(f"\n{model.upper()}")
        a_total = schema_agg.get((model, "a", "all"), Counter())
        b_total = schema_agg.get((model, "b", "all"), Counter())
        all_shapes = set(a_total) | set(b_total)
        rows = []
        for shape in all_shapes:
            a_n = a_total.get(shape, 0)
            b_n = b_total.get(shape, 0)
            fix_pct = (1 - b_n / a_n) * 100 if a_n else (0 if b_n == 0 else -1)
            rows.append((shape, a_n, b_n, fix_pct))
        rows.sort(key=lambda r: r[3], reverse=True)
        print(f"  {'Shape':<55} {'A':>5} {'B':>5} {'fix%':>7}")
        for shape, a, b, pct in rows:
            short = shape.rsplit("/", 1)[-1].rsplit("#", 1)[-1][:55]
            print(f"  {short:<55} {a:>5} {b:>5} {pct:>6.1f}%")

    print("\n" + "=" * 86)
    print("ONTOLOGY TRACK — OWL violation types (overall, n=200)")
    print("=" * 86)
    for model in ("gptoss",):
        print(f"\n{model.upper()}")
        a_total = owl_agg.get((model, "a", "all"), Counter())
        b_total = owl_agg.get((model, "b", "all"), Counter())
        all_types = set(a_total) | set(b_total)
        rows = []
        for vt in all_types:
            a_n = a_total.get(vt, 0)
            b_n = b_total.get(vt, 0)
            fix_pct = (1 - b_n / a_n) * 100 if a_n else (0 if b_n == 0 else -1)
            rows.append((vt, a_n, b_n, fix_pct))
        rows.sort(key=lambda r: r[3], reverse=True)
        print(f"  {'Violation type':<40} {'A':>5} {'B':>5} {'fix%':>7}")
        for vt, a, b, pct in rows:
            print(f"  {vt:<40} {a:>5} {b:>5} {pct:>6.1f}%")

    # ---- Bucket × shape (schema track — full matrix would be too wide) ----
    print("\n" + "=" * 86)
    print("GPT-OSS-120B SCHEMA — top 5 most-violated shapes, fix-rate per connectivity bucket")
    print("=" * 86)
    a_overall = schema_agg.get(("gptoss", "a", "all"), Counter())
    top5 = [s for s, _ in a_overall.most_common(5)]
    for shape in top5:
        short = shape.rsplit("/", 1)[-1].rsplit("#", 1)[-1][:50]
        print(f"\n  {short}")
        print(f"    {'Bucket':<10} {'A':>4} {'B':>4} {'fix%':>7}")
        for bucket in ("low", "medium", "high"):
            a_n = schema_agg.get(("gptoss", "a", bucket), Counter()).get(shape, 0)
            b_n = schema_agg.get(("gptoss", "b", bucket), Counter()).get(shape, 0)
            fix_pct = (1 - b_n / a_n) * 100 if a_n else 0
            print(f"    {bucket:<10} {a_n:>4} {b_n:>4} {fix_pct:>6.1f}%")

    # ---- JSON dump ----
    out = {
        "schema": {
            f"{m}_{s}_{b}": dict(c)
            for (m, s, b), c in schema_agg.items()
        },
        "ontology": {
            f"{m}_{s}_{b}": dict(c)
            for (m, s, b), c in owl_agg.items()
        },
    }
    out_path = ROOT / "evaluation/outputs/per_shape_violations.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[OK] dumped {out_path}")


if __name__ == "__main__":
    main()
