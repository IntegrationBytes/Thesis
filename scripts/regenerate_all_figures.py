"""One-shot regenerator for every thesis figure.

Runs each per-figure script in sequence. Idempotent — overwrites
existing PNGs. Used for reproducibility and for re-rendering after
data refreshes.

CLI::

    python scripts/regenerate_all_figures.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The five figures in the report, in order. These read the committed
# gpt-oss-120b outputs under evaluation/outputs/ (see README section 3).
SCRIPTS = [
    "make_percycle_v3.py",                # Figure 1 (per-cycle dynamics)
    "make_n200_decoupling_plot.py",       # Figure 2 (decoupling)
    "make_pipeline_e2e_figure.py",        # Figure 3 (single-case walkthrough)
    "make_violation_resolution_figure.py",# Figure 4 (per-violation, both tracks)
    "make_owl_process_figure.py",         # Figure 5 (ontology-track loop)
]


def main() -> None:
    print("Regenerating all thesis figures...\n")
    failures: list[str] = []
    for script in SCRIPTS:
        script_path = ROOT / "scripts" / script
        if not script_path.exists():
            print(f"  [skip] {script}: not found")
            continue
        print(f"  Running {script}...", flush=True)
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            failures.append(script)
            print(f"    FAILED ({result.returncode})")
            print(result.stderr[-400:])
        else:
            # Print the "Wrote ..." line if present
            for line in result.stdout.splitlines():
                if line.startswith(("Wrote", "[OK]")):
                    print(f"    {line}")

    print("\n=== Summary ===")
    if failures:
        print(f"  FAILED: {failures}")
        sys.exit(1)
    print(f"  All {len(SCRIPTS)} figures regenerated successfully.")


if __name__ == "__main__":
    main()
