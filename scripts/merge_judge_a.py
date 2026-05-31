"""Merge the four System A judge runs into judge_scores.json.

The four parallel runs each wrote their own JSON to
evaluation/outputs/judge_a_partial/. This script reads them and adds
their entries to the main judge_scores.json (which currently holds
the B variants).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "evaluation" / "outputs" / "judge_scores.json"
PARTIAL_DIR = ROOT / "evaluation" / "outputs" / "judge_a_partial"

partial_files = [
    "gemini_schema_a.json",
    "gemini_ontology_a.json",
    "gptoss_schema_a.json",
    "gptoss_ontology_a.json",
]

main = json.loads(MAIN.read_text())
print(f"Existing keys in main: {list(main.keys())}")

added = []
for fname in partial_files:
    p = PARTIAL_DIR / fname
    if not p.exists():
        print(f"  [MISS] {fname} not found, skipping")
        continue
    part = json.loads(p.read_text())
    for label, payload in part.items():
        # Sanity: count how many vignettes have non-None scores
        v = payload.get("vignettes", {})
        good = sum(1 for vid, sc in v.items()
                   if isinstance(sc.get("faithfulness"), int))
        total = len(v)
        if good == 0:
            print(f"  [FAIL] {label}: 0/{total} valid scores — refusing to merge")
            continue
        main[label] = payload
        added.append((label, good, total))
        print(f"  [OK]  {label}: {good}/{total} valid")

if not added:
    print("\nNothing to merge. Exiting.")
    raise SystemExit(1)

MAIN.write_text(json.dumps(main, indent=2))
print(f"\nWrote {MAIN}")
print(f"All keys now: {sorted(main.keys())}")

# Print summary for each added key
print("\n=== Summary of newly added cells ===")
for label, good, total in added:
    summ = main[label]["summary"]
    print(f"\n{label}  (n={summ['n_total']})")
    for sub in ("general", "complex"):
        s = summ[sub]
        print(f"  {sub:8s}  F={s['faithfulness']:.2f}  C={s['completeness']:.2f}  H={s['hallucination']:.2f}")
