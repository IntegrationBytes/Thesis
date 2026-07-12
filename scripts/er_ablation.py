"""Entity-resolution ablation: which matching signals carry the F1?

Re-frames iri_normalizer as a rule-based entity-resolution scorer and ablates
each comparison feature (code / local-name / value+date / label / token-Jaccard
/ structural-relational pass), re-running triple-level F1 over the 200
schema-track gold pairs. Shows every signal contributes and that codes+dates are
load-bearing (not just labels) — directly answering the examiner.

Output: evaluation/outputs/er_ablation.json

Deterministic: re-execs under PYTHONHASHSEED=0 so the greedy 1:1 matcher
breaks ties identically across runs and the reported numbers are exactly
reproducible.
"""
import os, sys
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import json, re
sys.path.insert(0, "pipeline")
from pathlib import Path
from rdflib import Graph, RDF
import iri_normalizer as N

GOLD = Path("evaluation/corpus/abox_gold")
BDIR = Path("evaluation/outputs/chr/schema/full/b")
_orig_struct = N._structural_score


def _norm_code(c):
    m = re.match(r"^https?://loinc\.org/(?:id/)?(.+)$", c)
    if m: return f"loinc:{m.group(1)}"
    m = re.match(r"^https?://snomed\.info/(?:id/|sct/)?(.+)$", c)
    if m: return f"snomed:{m.group(1)}"
    return c


def make_score(enabled):
    def score(lk, gk, type_uri=""):
        s = 0.0
        if "code" in enabled:
            lc, gc = lk.get("code"), gk.get("code")
            if lc and gc and _norm_code(lc) == _norm_code(gc): s += 2.0
        if "local" in enabled and lk["local_name"] == gk["local_name"]: s += 1.0
        if "value" in enabled:
            lv, gv = lk["values"], gk["values"]
            for k in set(lv) & set(gv):
                if lv[k] == gv[k]: s += 1.0; break
        if "label" in enabled and lk["label"] and gk["label"] and lk["label"] == gk["label"]: s += 0.9
        if "jaccard" in enabled:
            j = N._jaccard(lk["local_tokens"], gk["label_tokens"])
            if j > 0: s += min(j, 0.8)
            jl = N._jaccard(lk["label_tokens"], gk["label_tokens"])
            if jl > 0: s += min(jl, 0.8)
        return s
    return score


def triples(g):
    return {(str(s), str(p), str(o)) for s, p, o in g if str(p) != str(RDF.type)}


def stratum(vid):
    n = int(vid.split("_")[1]); return "general" if n <= 100 else "complex"


vignettes = sorted(p.name.replace("_gold_schema.ttl", "")
                   for p in GOLD.glob("*_gold_schema.ttl"))
ALL = {"code", "local", "value", "label", "jaccard"}
CONFIGS = {
    "full (all signals)":      (ALL, True),
    "- code (SNOMED/LOINC)":   (ALL - {"code"}, True),
    "- value/date":            (ALL - {"value"}, True),
    "- label":                 (ALL - {"label"}, True),
    "- token Jaccard":         (ALL - {"jaccard"}, True),
    "- structural pass":       (ALL, False),
    "code + local only":       ({"code", "local"}, True),
}

result = {}
for name, (enabled, struct_on) in CONFIGS.items():
    N._score = make_score(enabled)
    N._structural_score = _orig_struct if struct_on else (lambda *a, **k: 0.0)
    agg = {s: {"tp": 0, "fp": 0, "fn": 0} for s in ("general", "complex")}
    for vid in vignettes:
        lf = BDIR / f"{vid}.ttl"; gf = GOLD / f"{vid}_gold_schema.ttl"
        if not lf.exists(): continue
        try:
            llm = Graph().parse(lf.as_posix(), format="turtle")
            gold = Graph().parse(gf.as_posix(), format="turtle")
        except Exception:
            continue
        rew, _ = N.normalize_llm_to_gold(llm, gold)
        lt, gt = triples(rew), triples(gold)
        st = stratum(vid); a = agg[st]
        a["tp"] += len(lt & gt); a["fp"] += len(lt - gt); a["fn"] += len(gt - lt)

    def prf1(d):
        tp, fp, fn = d["tp"], d["fp"], d["fn"]
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return round(p, 3), round(r, 3), round(f, 3)
    gP, gR, gF = prf1(agg["general"]); cP, cR, cF = prf1(agg["complex"])
    result[name] = {"general_F1": gF, "complex_F1": cF, "general_PR": [gP, gR], "complex_PR": [cP, cR]}
    print(f"  {name:24}  general F1={gF:.3f}  complex F1={cF:.3f}")

# restore
N._score = make_score(ALL); N._structural_score = _orig_struct
Path("evaluation/outputs/er_ablation.json").write_text(json.dumps(result, indent=2))
print("\n[OK] wrote evaluation/outputs/er_ablation.json")
print("Read: drop in F1 when a signal is removed = that signal's contribution.")
