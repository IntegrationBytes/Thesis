"""Statistical analysis over evaluation results.

Reports descriptive statistics with honest uncertainty bounds and effect
sizes, delegating the numerics to ``scipy.stats`` through
:mod:`pipeline.stats_core`.

Output: a single ``evaluation/outputs/stats.json`` with the numeric
results, plus a markdown report at
``evaluation/outputs/stats_summary.md``.

Usage:
    .venv/bin/python pipeline/stats.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.stats_core import _per_cell_block
from pipeline.stats_report import _render_markdown

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "outputs" / "chr"
OUT_JSON = PROJECT_ROOT / "evaluation" / "outputs" / "stats.json"
OUT_MD = PROJECT_ROOT / "evaluation" / "outputs" / "stats_summary.md"

# Evaluation cells available: (track, prompt) -> eval.json path.
# Schema track only — the ontology track reports SHACL conformance directly
# from b/cycles/*/summary.json (no gold ABoxes exist for the synthea corpus
# on the ontology track, so triple-level F1 is not computable there).
CELLS: list[tuple[str, str, Path]] = [
    ("schema", "full", RESULTS_DIR / "schema" / "full" / "eval.json"),
]


def _load_cell(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> None:
    cells: list[dict] = []
    for track, prompt, path in CELLS:
        data = _load_cell(path)
        if data is None:
            print(f"[WARN] missing cell: {path}", file=sys.stderr)
            continue
        cells.append(_per_cell_block(track, prompt, data))
        print(f"[OK] loaded cell {track}/{prompt} (n={len(data['per_vignette'])})")

    out = {"cells": cells}
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_JSON.relative_to(PROJECT_ROOT)}")

    md = _render_markdown(cells)
    OUT_MD.write_text(md)
    print(f"Wrote {OUT_MD.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
