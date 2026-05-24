"""Per-cycle plot v2 — comparable-scale (reviewer's request).

Reviewer feedback (2026-05-22): the previous plot used dual axes (mean
triples on left, mean violations on right). They asked for "correctness
in an absolute sense and normalized for the count, two metrics on the
same scale".

This version plots, per retry cycle:
  - Triple-level F1 against gold      (0–1, schema track only)
  - SHACL pass-rate across vignettes  (0–1, share of graphs that conform)

Both axes share the [0, 1] scale, so the divergence between
"conformance climbs" and "F1 holds steady" is visible at a glance — the
decoupling finding made visual.

Output: report/figures/fig_percycle_v2.png
"""
from __future__ import annotations

import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import rdflib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from iri_normalizer import normalize_llm_to_gold  # noqa: E402

GOLD_DIR = ROOT / "evaluation/corpus/abox_gold"


def _stratum(vid: str) -> str:
    m = re.match(r"vignette_(\d+)", vid)
    if not m:
        return "other"
    return "general" if int(m.group(1)) <= 100 else "complex"


def _f1(gold: set, gen: set) -> float:
    if not gen:
        return 0.0
    tp = len(gold & gen)
    p = tp / len(gen)
    r = tp / len(gold) if gold else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def _process_cycle(args) -> tuple[str, int, bool, float, int]:
    """For one (vid, cycle_idx, ttl_path, gold_path, conforms_flag) entry:
    return (stratum, cycle, conforms, f1, n_triples)."""
    vid, cycle_idx, ttl_path, gold_path, conforms_flag = args
    stratum = _stratum(vid)
    ttl_p = Path(ttl_path)
    gold_p = Path(gold_path)
    if not ttl_p.exists() or not gold_p.exists():
        return (stratum, cycle_idx, False, 0.0, 0)
    try:
        gen_g = rdflib.Graph().parse(ttl_p.as_posix(), format="turtle")
        gold_g = rdflib.Graph().parse(gold_p.as_posix(), format="turtle")
    except Exception:
        return (stratum, cycle_idx, conforms_flag, 0.0, 0)
    gen_rewritten, _ = normalize_llm_to_gold(gen_g, gold_g)
    gold_set = {(str(s), str(p), str(o)) for s, p, o in gold_g}
    gen_set = {(str(s), str(p), str(o)) for s, p, o in gen_rewritten}
    f1 = _f1(gold_set, gen_set)
    return (stratum, cycle_idx, conforms_flag, f1, len(gen_g))


def collect_jobs(cycles_root: Path, gold_dir: Path) -> list:
    """Walk cycles_root → list of (vid, cycle_idx, ttl_path, gold_path, conforms_flag)."""
    jobs = []
    for cycles_dir in sorted(cycles_root.glob("vignette_*")):
        if not cycles_dir.is_dir():
            continue
        vid = cycles_dir.name
        gold = gold_dir / f"{vid}_gold_schema.ttl"
        if not gold.exists():
            continue
        summary = cycles_dir / "summary.json"
        if not summary.exists():
            continue
        try:
            s = json.loads(summary.read_text())
        except Exception:
            continue
        trace = s.get("trace", [])
        # Some traces only include cycles where validation ran;
        # we also need cycle_00 (initial A output) which may not be in trace
        for i, t in enumerate(trace):
            ttl_path = cycles_dir / f"cycle_{i:02d}.ttl"
            if not ttl_path.exists():
                continue
            jobs.append((
                vid, i, str(ttl_path), str(gold), bool(t.get("conforms")),
            ))
    return jobs


def main() -> None:
    sources = [
        ("Gemini schema",
         ROOT / "evaluation/outputs/chr/schema/full/b/cycles"),
        ("gpt-oss schema",
         ROOT / "evaluation/outputs_freemodel/chr/schema/full/b/cycles"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)

    for ax, (title, cycles_root) in zip(axes, sources):
        print(f"\nProcessing {title}...", flush=True)
        jobs = collect_jobs(cycles_root, GOLD_DIR)
        print(f"  {len(jobs)} (vignette, cycle) entries")

        # Parallel processing
        results: list = []
        with ProcessPoolExecutor(max_workers=6) as pool:
            futs = [pool.submit(_process_cycle, j) for j in jobs]
            for fut in as_completed(futs):
                results.append(fut.result())

        # Aggregate per (stratum, cycle_idx)
        agg: dict = {}
        for stratum, cycle, conforms, f1, _ in results:
            key = (stratum, cycle)
            agg.setdefault(key, {"conforms": [], "f1s": []})
            agg[key]["conforms"].append(conforms)
            agg[key]["f1s"].append(f1)

        # Cycles range
        max_cycle = max(c for _, c in agg.keys())
        cycles = list(range(max_cycle + 1))

        # Plot per stratum
        for stratum, color, marker in [("general", "#1f6fb4", "o"),
                                        ("complex", "#c44e2e", "s")]:
            pass_rate, f1_mean = [], []
            for c in cycles:
                d = agg.get((stratum, c), {"conforms": [], "f1s": []})
                pr = (sum(d["conforms"]) / len(d["conforms"])
                       if d["conforms"] else 0)
                fm = mean(d["f1s"]) if d["f1s"] else 0
                pass_rate.append(pr)
                f1_mean.append(fm)
            # SHACL pass rate (solid)
            ax.plot(cycles, pass_rate, marker=marker, color=color, ls="-",
                    lw=1.6, ms=5,
                    label=f"{stratum} · SHACL pass-rate")
            # F1 (dashed)
            ax.plot(cycles, f1_mean, marker=marker, color=color, ls="--",
                    lw=1.6, ms=5, alpha=0.7,
                    label=f"{stratum} · F1 (normalised)")

        ax.set_title(title, fontsize=10)
        ax.set_xticks(cycles)
        ax.set_xlabel("retry cycle", fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", ls=":", alpha=0.4)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("rate / score  (both on [0, 1])", fontsize=9)
    axes[0].legend(loc="center right", fontsize=7.5, framealpha=0.9)

    fig.suptitle(
        "Per-cycle convergence — both metrics on comparable [0, 1] scale "
        "(reviewer's ask)",
        fontsize=10.5, y=1.02,
    )
    fig.tight_layout()

    out = ROOT / "report/figures/fig_percycle_v2.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
