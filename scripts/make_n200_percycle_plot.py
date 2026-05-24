"""Per-cycle plot stratified by general (1-100) vs complex (101-200).

Shows monotonic convergence (RQ1) on the SHACL retry loop, with the
complex stratum's bigger starting violation count making the
loop's effectiveness more visible.

Output: report/figures/fig_percycle_n200.png
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rdflib

ROOT = Path(__file__).resolve().parent.parent


def _stratum(vid: str) -> str:
    m = re.match(r"vignette_(\d+)", vid)
    if not m:
        return "other"
    return "general" if int(m.group(1)) <= 100 else "complex"


def collect_cycle_means(cycles_root: Path) -> dict:
    """Return {stratum: {cycle_idx: {'triples_mean', 'viol_mean', 'n'}}}."""
    out: dict[str, dict[int, dict]] = {"general": {}, "complex": {}}
    if not cycles_root.exists():
        return out

    for cycles_dir in sorted(cycles_root.glob("vignette_*")):
        if not cycles_dir.is_dir():
            continue
        st = _stratum(cycles_dir.name)
        if st == "other":
            continue
        summary = cycles_dir / "summary.json"
        if not summary.exists():
            continue
        try:
            s = json.loads(summary.read_text())
        except Exception:
            continue
        trace = s.get("trace", [])
        for i, t in enumerate(trace):
            ttl_path = cycles_dir / f"cycle_{i:02d}.ttl"
            n_triples = 0
            if ttl_path.exists():
                try:
                    g = rdflib.Graph()
                    g.parse(ttl_path.as_posix(), format="turtle")
                    n_triples = len(g)
                except Exception:
                    n_triples = 0
            rs = t.get("results_summary", "")
            n_viol = rs.count("Constraint Violation") if "Constraint Violation" in rs else 0

            bucket = out[st].setdefault(i, {"triples": [], "violations": []})
            bucket["triples"].append(n_triples)
            bucket["violations"].append(n_viol)

    # Reduce to means + n
    reduced: dict[str, dict[int, dict]] = {"general": {}, "complex": {}}
    for st in ("general", "complex"):
        for i, b in out[st].items():
            ts = b["triples"]
            vs = b["violations"]
            reduced[st][i] = {
                "triples_mean": sum(ts) / len(ts) if ts else 0,
                "viol_mean": sum(vs) / len(vs) if vs else 0,
                "n": len(ts),
            }
    return reduced


def main() -> None:
    sources = [
        ("Gemini · schema",  ROOT / "evaluation/outputs/chr/schema/full/b/cycles"),
        ("Gemini · ontology", ROOT / "evaluation/outputs/chr/ontology/ontology/b/cycles"),
        ("gpt-oss · schema",  ROOT / "evaluation/outputs_freemodel/chr/schema/full/b/cycles"),
        ("gpt-oss · ontology", ROOT / "evaluation/outputs_freemodel/chr/ontology/ontology/b/cycles"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 5.6), sharex=True)
    axes = axes.flatten()

    for ax, (title, cycles_root) in zip(axes, sources):
        data = collect_cycle_means(cycles_root)
        ax2 = ax.twinx()

        # Determine max cycle index across both strata
        max_cycle = max(
            max(data["general"].keys(), default=-1),
            max(data["complex"].keys(), default=-1),
        )
        cycles = list(range(0, max_cycle + 1))

        # general (solid) and complex (dashed)
        for st, marker, ls, color_tri, color_viol in [
            ("general", "o", "-", "#1f6fb4", "#c44e2e"),
            ("complex", "s", "--", "#0a3e6c", "#7a2a18"),
        ]:
            tri = [data[st].get(i, {}).get("triples_mean", 0) for i in cycles]
            vio = [data[st].get(i, {}).get("viol_mean", 0)    for i in cycles]
            ax.plot(cycles, tri, marker=marker, ls=ls, color=color_tri,
                    label=f"{st} triples", lw=1.4, ms=4)
            ax2.plot(cycles, vio, marker=marker, ls=ls, color=color_viol,
                     label=f"{st} violations", lw=1.4, ms=4)

        ax.set_title(title, fontsize=10)
        ax.set_xticks(cycles)
        ax.set_xlabel("retry cycle", fontsize=9)
        ax.set_ylabel("mean triples", color="#1f6fb4", fontsize=8.5)
        ax2.set_ylabel("mean violations", color="#c44e2e", fontsize=8.5)
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelcolor="#1f6fb4", labelsize=8)
        ax2.tick_params(axis="y", labelcolor="#c44e2e", labelsize=8)

    # Single combined legend at figure level
    handles_solid = [
        plt.Line2D([0], [0], marker="o", color="#1f6fb4", lw=1.4, ms=4,
                   ls="-",  label="general (1-100)"),
        plt.Line2D([0], [0], marker="s", color="#0a3e6c", lw=1.4, ms=4,
                   ls="--", label="complex (101-200)"),
    ]
    fig.legend(handles=handles_solid, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.02), fontsize=9, frameon=False)

    fig.suptitle(
        "Per-cycle convergence of System B (SHACL retry loop), stratified · n=200",
        fontsize=10.5, y=1.06,
    )
    fig.tight_layout()

    out = ROOT / "report/figures/fig_percycle_n200.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
