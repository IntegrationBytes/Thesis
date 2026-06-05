"""Parallel OWL conformance survey across A / B(OWL) / B(SHACL archived).

Used for the supervisor's ontology track comparison. Spawns one worker
per file so OWL closure can run across CPU cores.

Usage::
    python scripts/owl_conformance_survey.py
"""
from __future__ import annotations

import glob
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from owl_validator import validate_with_owl  # noqa: E402

CHR_ONT = ROOT / "evaluation/corpus/tbox/chr_ontology.owl.ttl"


def _check_one(path_str: str) -> tuple[str, str, bool]:
    """Worker: returns (path, stratum, ok)."""
    path = Path(path_str)
    vid = path.stem
    m = re.match(r"vignette_(\d+)", vid)
    if not m:
        return (path_str, "other", False)
    n = int(m.group(1))
    stratum = "general" if n <= 100 else "complex"
    ttl = path.read_text()
    if "# LLM call failed" in ttl:
        return (path_str, stratum, False)
    ok, _ = validate_with_owl(ttl, CHR_ONT)
    return (path_str, stratum, ok)


def main() -> None:
    targets = [
        ("gpt-oss-120b ontology",
         "evaluation/outputs/chr/ontology/ontology"),
    ]
    systems = [
        ("A (zero-shot)", "a"),
        ("B (OWL-driven retry)", "b"),
        ("B_shacl (SHACL-driven retry — archived)", "b_shacl"),
    ]

    # Gather all (label, system_label, files) tasks
    jobs: list[tuple[str, str, list[str]]] = []
    for label, root in targets:
        for system_label, system_dir in systems:
            pattern = str(ROOT / root / system_dir / "vignette_*.ttl")
            files = sorted(glob.glob(pattern))
            jobs.append((label, system_label, files))

    results: dict[tuple[str, str], dict[str, dict[str, int]]] = {}
    for label, system_label, files in jobs:
        key = (label, system_label)
        results[key] = {
            "general": {"pass": 0, "n": 0},
            "complex": {"pass": 0, "n": 0},
        }
        if not files:
            continue
        print(f"  [{label} / {system_label}] {len(files)} files...", flush=True)
        with ProcessPoolExecutor(max_workers=6) as pool:
            futs = [pool.submit(_check_one, f) for f in files]
            for fut in as_completed(futs):
                _, stratum, ok = fut.result()
                if stratum in ("general", "complex"):
                    results[key][stratum]["n"] += 1
                    if ok:
                        results[key][stratum]["pass"] += 1

    print("\n" + "=" * 80)
    print("OWL CONFORMANCE — A vs B(OWL-driven) vs B(SHACL-archived) · n=200")
    print("=" * 80)
    for label, _ in targets:
        print(f"\n{label}")
        for system_label, _ in systems:
            r = results[(label, system_label)]
            g = r["general"]
            c = r["complex"]
            g_pct = 100 * g["pass"] / g["n"] if g["n"] else 0
            c_pct = 100 * c["pass"] / c["n"] if c["n"] else 0
            print(
                f"  {system_label:<42}  "
                f"general {g['pass']:>3}/{g['n']:<3} ({g_pct:5.1f}%)   "
                f"complex {c['pass']:>3}/{c['n']:<3} ({c_pct:5.1f}%)"
            )


if __name__ == "__main__":
    main()
