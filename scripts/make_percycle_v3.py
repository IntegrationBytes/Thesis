"""Per-cycle convergence — schema | ontology, both models, n=200.

Two panels:
  (a) Schema track: SHACL pass-rate (solid) and triple-level F1 (dashed)
      per cycle, by model.
  (b) Ontology track: OWL pass-rate per cycle, by model. F1 is not
      reported on the ontology track because gold IRIs are schema-form.

Both axes are [0, 1] so the divergence between "structure climbs" and
"content stays flat" is visible at a glance.

Output: report/figures/fig_percycle_v2.png  (overwrites the old single-track plot)
"""
from __future__ import annotations

import json
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


def _prf(gold: set, gen: set) -> tuple[float, float, float]:
    """Return (precision, recall, F1)."""
    if not gen:
        return (0.0, 0.0, 0.0)
    tp = len(gold & gen)
    p = tp / len(gen)
    r = tp / len(gold) if gold else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return (p, r, f1)


def _process_schema(args) -> tuple[int, bool, float, float, float]:
    """schema track: compute conforms + (P, R, F1) per cycle."""
    cycle_idx, ttl_path, gold_path, conforms_flag = args
    ttl_p = Path(ttl_path)
    gold_p = Path(gold_path)
    if not ttl_p.exists() or not gold_p.exists():
        return (cycle_idx, False, 0.0, 0.0, 0.0)
    try:
        gen_g = rdflib.Graph().parse(ttl_p.as_posix(), format="turtle")
        gold_g = rdflib.Graph().parse(gold_p.as_posix(), format="turtle")
    except Exception:
        return (cycle_idx, conforms_flag, 0.0, 0.0, 0.0)
    gen_rewritten, _ = normalize_llm_to_gold(gen_g, gold_g)
    gold_set = {(str(s), str(p), str(o)) for s, p, o in gold_g}
    gen_set = {(str(s), str(p), str(o)) for s, p, o in gen_rewritten}
    p, r, f1 = _prf(gold_set, gen_set)
    return (cycle_idx, conforms_flag, p, r, f1)


def collect_schema_jobs(cycles_root: Path) -> tuple[list, dict, int]:
    """Returns (jobs, per_vignette_trace, n_vignettes).
    per_vignette_trace[vid] = list of conforms flags per cycle.
    """
    jobs = []
    per_v: dict[str, list[bool]] = {}
    n = 0
    for cycles_dir in sorted(cycles_root.glob("vignette_*")):
        if not cycles_dir.is_dir():
            continue
        vid = cycles_dir.name
        gold = GOLD_DIR / f"{vid}_gold_schema.ttl"
        if not gold.exists():
            continue
        summary = cycles_dir / "summary.json"
        if not summary.exists():
            continue
        try:
            s = json.loads(summary.read_text())
        except Exception:
            continue
        n += 1
        trace = s.get("trace", [])
        per_v[vid] = [bool(t.get("conforms")) for t in trace]
        for i, t in enumerate(trace):
            ttl_path = cycles_dir / f"cycle_{i:02d}.ttl"
            if not ttl_path.exists():
                continue
            jobs.append((vid, i, str(ttl_path), str(gold), bool(t.get("conforms"))))
    return jobs, per_v, n


def _process_schema_v(args) -> tuple[str, int, bool, float, float, float]:
    """schema track with vignette id."""
    vid, cycle_idx, ttl_path, gold_path, conforms_flag = args
    cycle, conf, p, r, f1 = _process_schema((cycle_idx, ttl_path, gold_path, conforms_flag))
    return (vid, cycle, conf, p, r, f1)


def collect_ontology_traces(cycles_root: Path) -> tuple[dict, int]:
    """Returns ({vid: [conforms_per_cycle]}, n_vignettes)."""
    per_v: dict[str, list[bool]] = {}
    for cycles_dir in sorted(cycles_root.glob("vignette_*")):
        if not cycles_dir.is_dir():
            continue
        summary = cycles_dir / "summary.json"
        if not summary.exists():
            continue
        try:
            s = json.loads(summary.read_text())
        except Exception:
            continue
        per_v[cycles_dir.name] = [bool(t.get("conforms")) for t in s.get("trace", [])]
    return per_v, len(per_v)


def compute_schema_cumulative(cycles_root: Path):
    """Returns (cycles, pass_rate, p_mean, r_mean, f1_mean)."""
    jobs, per_v_conf, n = collect_schema_jobs(cycles_root)
    print(f"  {len(jobs)} schema (vignette, cycle) entries · n={n} vignettes")

    # Compute (P, R, F1) for each (vid, cycle)
    p_table: dict[tuple[str, int], float] = {}
    r_table: dict[tuple[str, int], float] = {}
    f1_table: dict[tuple[str, int], float] = {}
    with ProcessPoolExecutor(max_workers=6) as pool:
        futs = [pool.submit(_process_schema_v, j) for j in jobs]
        for fut in as_completed(futs):
            vid, c, conf, p_v, r_v, f1_v = fut.result()
            p_table[(vid, c)] = p_v
            r_table[(vid, c)] = r_v
            f1_table[(vid, c)] = f1_v

    max_cycle = max((c for _, c in f1_table.keys()), default=0)
    cycles = list(range(max_cycle + 1))

    first_pass: dict[str, int | None] = {}
    for vid, confs in per_v_conf.items():
        idx = next((i for i, c in enumerate(confs) if c), None)
        first_pass[vid] = idx

    # Pre-compute max cycle traced per vignette
    traced_max: dict[str, int] = {}
    for (v, c) in f1_table.keys():
        if c > traced_max.get(v, -1):
            traced_max[v] = c

    pass_rate, p_mean, r_mean, f1_mean = [], [], [], []
    for k in cycles:
        passed = sum(1 for vid, fp in first_pass.items()
                     if fp is not None and fp <= k)
        pass_rate.append(passed / n)
        ps, rs, fs = [], [], []
        for vid in per_v_conf.keys():
            fp = first_pass[vid]
            traced = traced_max.get(vid, 0)
            state_c = min(k, fp if fp is not None else traced)
            ps.append(p_table.get((vid, state_c), 0.0))
            rs.append(r_table.get((vid, state_c), 0.0))
            fs.append(f1_table.get((vid, state_c), 0.0))
        p_mean.append(mean(ps) if ps else 0.0)
        r_mean.append(mean(rs) if rs else 0.0)
        f1_mean.append(mean(fs) if fs else 0.0)
    return cycles, pass_rate, p_mean, r_mean, f1_mean


def compute_ontology_cumulative(cycles_root: Path) -> tuple[list[int], list[float]]:
    """Cumulative OWL pass-rate per cycle."""
    per_v, n = collect_ontology_traces(cycles_root)
    max_cycle = max((len(v) - 1 for v in per_v.values() if v), default=0)
    cycles = list(range(max_cycle + 1))
    first_pass: dict[str, int | None] = {}
    for vid, confs in per_v.items():
        idx = next((i for i, c in enumerate(confs) if c), None)
        first_pass[vid] = idx
    pass_rate = []
    for k in cycles:
        passed = sum(1 for fp in first_pass.values()
                     if fp is not None and fp <= k)
        pass_rate.append(passed / n)
    return cycles, pass_rate


def main() -> None:
    GEMINI_BLUE = "#1f6fb4"
    GPTOSS_ORANGE = "#c44e2e"
    F1_GREY = "#9aa0a6"   # F1 lines de-emphasised in grey
    BAND_GREY = "#eef0f3"

    sources_schema = [
        ("gpt-oss-120b", GPTOSS_ORANGE, "s",
         ROOT / "evaluation/outputs/chr/schema/full/b/cycles"),
    ]
    sources_ontology = [
        ("gpt-oss-120b", GPTOSS_ORANGE, "s",
         ROOT / "evaluation/outputs/chr/ontology/ontology/b/cycles"),
    ]

    fig, (ax_s, ax_o) = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True,
                                      gridspec_kw={"width_ratios": [1.05, 1.0]})

    # ===== PANEL A: schema track =====
    schema_results: list[tuple] = []
    for label, color, marker, root in sources_schema:
        print(f"\n[schema] {label}…", flush=True)
        cycles, pass_rate, p_mean, r_mean, f1_mean = compute_schema_cumulative(root)
        schema_results.append((label, color, marker, cycles, pass_rate, p_mean, r_mean, f1_mean))

    for label, color, marker, cycles, pass_rate, p_mean, r_mean, f1_mean in schema_results:
        # SHACL pass-rate — solid, full saturation, heavy weight (the headline)
        ax_s.plot(cycles, pass_rate, marker=marker, color=color, ls="-",
                  lw=2.6, ms=8,
                  label=f"{label}  ·  SHACL pass-rate")
        # Precision — same colour, dotted, no marker fill
        ax_s.plot(cycles, p_mean, marker=marker, color=color, ls=":",
                  lw=1.4, ms=5, alpha=0.85,
                  markerfacecolor="white", markeredgewidth=1.2,
                  label=f"{label}  ·  Precision")
        # Recall — same colour, dash-dot
        ax_s.plot(cycles, r_mean, marker=marker, color=color, ls="-.",
                  lw=1.4, ms=5, alpha=0.85,
                  markerfacecolor="white", markeredgewidth=1.2,
                  label=f"{label}  ·  Recall")
        # F1 — same colour, dashed (slightly heavier than P/R)
        ax_s.plot(cycles, f1_mean, marker=marker, color=color, ls="--",
                  lw=1.8, ms=6, alpha=0.95,
                  markerfacecolor="white", markeredgewidth=1.5,
                  label=f"{label}  ·  $F_1$")

        # SHACL endpoint labels (k=0 → k=last)
        k_last = cycles[-1]
        ax_s.annotate(f"{pass_rate[0]*100:.0f}%", xy=(0, pass_rate[0]),
                       xytext=(-0.15, pass_rate[0] - 0.05),
                       fontsize=9.5, color=color, fontweight="bold", ha="right")
        ax_s.annotate(f"{pass_rate[-1]*100:.0f}%", xy=(k_last, pass_rate[-1]),
                       xytext=(k_last + 0.12, pass_rate[-1]),
                       fontsize=9.5, color=color, fontweight="bold",
                       ha="left", va="center")

        # P / R / F1 endpoint labels at k=last (small, italic, same colour)
        for series_label, series in [("P", p_mean), ("R", r_mean), ("F$_1$", f1_mean)]:
            v = series[-1]
            ax_s.annotate(f"{series_label} {v:.2f}", xy=(k_last, v),
                           xytext=(k_last + 0.12, v),
                           fontsize=7.5, color=color, fontweight="bold",
                           ha="left", va="center", style="italic")

    ax_s.set_title("(a)  Schema track  ·  SHACL climbs; P, R, $F_1$ stay flat",
                   fontsize=11.5, loc="left", pad=10, fontweight="bold")
    ax_s.set_xlabel("Retry cycle  k", fontsize=10)
    ax_s.set_ylabel("Rate / score  ([0, 1] scale)", fontsize=10)
    ax_s.set_ylim(0, 1.05)
    ax_s.grid(axis="y", ls=":", alpha=0.35)
    ax_s.set_axisbelow(True)

    # Legend at top of panel — outside the data
    leg = ax_s.legend(loc="lower right", fontsize=8.5, framealpha=0.95,
                       ncol=2, columnspacing=1.0)
    leg.get_frame().set_edgecolor("#cbd5e1")

    # ===== PANEL B: ontology track =====
    ontology_results = []
    for label, color, marker, root in sources_ontology:
        print(f"\n[ontology] {label}…", flush=True)
        cycles, pass_rate = compute_ontology_cumulative(root)
        ontology_results.append((label, color, marker, cycles, pass_rate))

    for label, color, marker, cycles, pass_rate in ontology_results:
        ax_o.plot(cycles, pass_rate, marker=marker, color=color, ls="-",
                  lw=2.4, ms=7.5,
                  label=f"{label}  ·  OWL pass-rate")
        k0 = pass_rate[0]
        kf = pass_rate[-1]
        k_last = cycles[-1]
        ax_o.annotate(f"{k0*100:.0f}%", xy=(0, k0),
                       xytext=(-0.15, k0),
                       fontsize=9.5, color=color, fontweight="bold",
                       ha="right", va="center")
        ax_o.annotate(f"{kf*100:.0f}%", xy=(k_last, kf),
                       xytext=(k_last + 0.12, kf),
                       fontsize=9.5, color=color, fontweight="bold",
                       ha="left", va="center")

    ax_o.set_title("(b)  Ontology track  ·  OWL-RL closure over SULO + CHR + data",
                   fontsize=11.5, loc="left", pad=10, fontweight="bold")
    ax_o.set_xlabel("Retry cycle  k", fontsize=10)
    ax_o.set_ylim(0, 1.05)
    ax_o.grid(axis="y", ls=":", alpha=0.35)
    ax_o.set_axisbelow(True)
    leg2 = ax_o.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
    leg2.get_frame().set_edgecolor("#cbd5e1")

    # Force integer cycle ticks + small padding so endpoint labels fit
    for ax in (ax_s, ax_o):
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xlim(-0.4, 3.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.tight_layout(pad=1.5)
    out = ROOT / "report/figures/fig_percycle_v2.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
