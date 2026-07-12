"""Examiner-requested experiment: run the OWL reasoner on the SCHEMA-track ABox.

Instead of a separately generated ontology ABox (which uses the SULO Role
pattern), reuse the SAME generated graph that the schema track validates with
SHACL (flat chr: predicates), merge it with the CHR TBox + SULO, run OWL-RL
closure, and report what ADDITIONAL error types the reasoner surfaces.

Output: evaluation/outputs/owl_on_schema_abox.json (+ console summary)
"""
import sys, json
sys.path.insert(0, "pipeline")
from pathlib import Path
from collections import Counter
from owl_validator import validate_with_owl

CHR_ONT = Path("evaluation/corpus/tbox/chr_ontology.owl.ttl")
ERRTYPES = ("DisjointnessClash", "FunctionalPropertyConflict", "RangeViolation")
result = {}

for sysname in ("a", "b"):
    d = Path(f"evaluation/outputs/chr/schema/full/{sysname}")
    files = sorted(d.glob("vignette_*.ttl"))
    consistent = inconsistent = parsefail = 0
    files_with = Counter()      # files affected by each error type
    instances = Counter()       # total instances of each error type
    examples = {}
    per_file = {}
    for f in files:
        ok, rep = validate_with_owl(f.read_text(), CHR_ONT)
        if "parse error" in rep.lower() or "closure error" in rep.lower():
            parsefail += 1; per_file[f.stem] = "PARSE_FAIL"; continue
        if ok:
            consistent += 1; per_file[f.stem] = "consistent"; continue
        inconsistent += 1
        hits = []
        for et in ERRTYPES:
            n = rep.count(et)
            if n:
                files_with[et] += 1; instances[et] += n; hits.append(f"{et}x{n}")
                if et not in examples:
                    for line in rep.splitlines():
                        if et in line:
                            examples[et] = line.strip()[:140]; break
        per_file[f.stem] = ";".join(hits)
    result[sysname] = {
        "n_files": len(files), "consistent": consistent,
        "inconsistent": inconsistent, "parse_fail": parsefail,
        "files_with_error_type": dict(files_with),
        "error_instances": dict(instances),
        "examples": examples,
    }
    print(f"=== System {sysname.upper()} : SCHEMA-track ABox under OWL-RL (CHR TBox + SULO) ===")
    print(f"  files={len(files)}  consistent={consistent}  inconsistent={inconsistent}  parse/closure-fail={parsefail}")
    print(f"  files affected by each OWL error type: {dict(files_with)}")
    print(f"  total instances: {dict(instances)}")
    for et, m in examples.items():
        print(f"    e.g. {et}: {m}")
    print()

out = Path("evaluation/outputs/owl_on_schema_abox.json")
out.write_text(json.dumps(result, indent=2))
print(f"[OK] wrote {out}")
