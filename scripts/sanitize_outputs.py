"""Post-extraction Turtle sanitiser pass (report §IV-B).

gpt-oss-120b occasionally emits IRI prefix shorthand whose local name
contains characters disallowed by the Turtle grammar (most commonly '+'
and an additional ':' from ISO-8601 timezone suffixes, e.g.
``ex:visit_2019_03_15+01:00``). This script walks every extractor output
.ttl under the outputs root, and for any file that fails to parse it
applies ``pipeline.ttl_sanitizer.sanitize_ttl`` and rewrites the file if
(and only if) the sanitised text parses.

The pass is idempotent: already-valid (already-sanitised) files are left
untouched. Run it once after a fresh extraction and before evaluation:

    python scripts/sanitize_outputs.py                      # default outputs root
    OUTPUTS_ROOT_OVERRIDE=... python scripts/sanitize_outputs.py

It reports the salvage rate (parseable-after / unparseable-before).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from rdflib import Graph  # noqa: E402
from ttl_sanitizer import sanitize_ttl  # noqa: E402


def _parses(text: str) -> bool:
    try:
        Graph().parse(data=text, format="turtle")
        return True
    except Exception:
        return False


def main() -> None:
    outputs_root = Path(
        os.getenv("OUTPUTS_ROOT_OVERRIDE", str(ROOT / "evaluation" / "outputs"))
    )
    ttl_files = sorted(outputs_root.rglob("vignette_*.ttl"))
    # Skip the per-cycle trace files; only sanitise the final system outputs.
    ttl_files = [f for f in ttl_files if "cycles" not in f.parts]

    n_total = len(ttl_files)
    n_unparseable = 0
    n_salvaged = 0
    n_changed = 0
    still_broken: list[str] = []

    for f in ttl_files:
        text = f.read_text()
        if text.startswith("# LLM call failed"):
            continue  # genuine API failure, not a syntax issue
        if _parses(text):
            continue  # already valid; idempotent no-op
        n_unparseable += 1
        fixed = sanitize_ttl(text)
        if _parses(fixed):
            n_salvaged += 1
            if fixed != text:
                f.write_text(fixed)
                n_changed += 1
        else:
            still_broken.append(str(f.relative_to(outputs_root)))

    print(f"Outputs root        : {outputs_root}")
    print(f"Final .ttl outputs  : {n_total}")
    print(f"Unparseable before  : {n_unparseable}")
    print(f"Salvaged by sanitiser: {n_salvaged}")
    print(f"Rewritten this run  : {n_changed}  (0 => already sanitised / idempotent)")
    if n_unparseable:
        print(f"Salvage rate        : {n_salvaged / n_unparseable * 100:.0f}%")
    if still_broken:
        print(f"Still unparseable   : {len(still_broken)} (discarded at eval time)")
        for s in still_broken[:20]:
            print(f"  - {s}")


if __name__ == "__main__":
    main()
