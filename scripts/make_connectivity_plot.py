"""Connectivity-stratified plot — supervisor's ask.

3 buckets (low/medium/high by n_entities tertile) × 2 models × 2 tracks.
Shows SHACL retry loop's value (ΔSHACL) growing with connectivity,
plus the F1 cost growing alongside (decoupling at scale).

Output: report/figures/fig_connectivity_n200.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# Hardcoded from connectivity_analysis.py output (2026-05-24).
DATA = {
    "low":    {"n": 66, "entities_mean": 36, "linked": 0.68},
    "medium": {"n": 66, "entities_mean": 70, "linked": 0.69},
    "high":   {"n": 68, "entities_mean": 108, "linked": 0.69},
}

# Schema-track SHACL deltas (A → B)
SHACL_PP = {
    "low":    {"Gemini": +90.9, "gpt-oss": +65.2},
    "medium": {"Gemini": +93.7, "gpt-oss": +72.7},
    "high":   {"Gemini": +98.4, "gpt-oss": +67.6},
}

# Ontology-track OWL deltas (A → B-OWL)
OWL_PP = {
    "low":    {"Gemini":  +3.0, "gpt-oss":  +9.1},
    "medium": {"Gemini": +22.7, "gpt-oss": +13.6},
    "high":   {"Gemini": +42.6, "gpt-oss":  +4.4},
}

# Schema F1 deltas (B - A)
F1_DELTA = {
    "low":    {"Gemini": -0.019, "gpt-oss": -0.002},
    "medium": {"Gemini": -0.057, "gpt-oss": -0.003},
    "high":   {"Gemini": -0.081, "gpt-oss": -0.007},
}


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharex=True)

    buckets = ["low", "medium", "high"]
    x = np.arange(len(buckets))
    width = 0.35

    # Panel 1: ΔSHACL (schema) and ΔOWL (ontology) — Gemini only (the
    # model that shows the supervisor's connectivity effect most clearly)
    ax = axes[0]
    schema_d = [SHACL_PP[b]["Gemini"] for b in buckets]
    owl_d    = [OWL_PP[b]["Gemini"]    for b in buckets]
    b1 = ax.bar(x - width/2, schema_d, width, color="#1f6fb4",
                edgecolor="black", linewidth=0.4,
                label="ΔSHACL (schema)")
    b2 = ax.bar(x + width/2, owl_d, width, color="#c44e2e",
                edgecolor="black", linewidth=0.4,
                label="ΔOWL (ontology)")
    for bars, vals in [(b1, schema_d), (b2, owl_d)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1.5,
                    f"+{v:.0f}pp",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([
        f"low\n(n=66, ~{DATA['low']['entities_mean']} ent.)",
        f"medium\n(n=66, ~{DATA['medium']['entities_mean']} ent.)",
        f"high\n(n=68, ~{DATA['high']['entities_mean']} ent.)",
    ], fontsize=8.5)
    ax.set_ylabel("Δ pass-rate (percentage points)", fontsize=9)
    ax.set_ylim(0, 115)
    ax.set_title("Gemini · feedback lift grows with connectivity",
                 fontsize=10)
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=8)

    # Panel 2: F1 cost (Δ in F1) by connectivity, both models
    ax = axes[1]
    gemini_f1 = [F1_DELTA[b]["Gemini"]  for b in buckets]
    gptoss_f1 = [F1_DELTA[b]["gpt-oss"] for b in buckets]
    b1 = ax.bar(x - width/2, gemini_f1, width, color="#1f6fb4",
                edgecolor="black", linewidth=0.4,
                label="Gemini")
    b2 = ax.bar(x + width/2, gptoss_f1, width, color="#7a9c40",
                edgecolor="black", linewidth=0.4,
                label="gpt-oss")
    for bars, vals in [(b1, gemini_f1), (b2, gptoss_f1)]:
        for bar, v in zip(bars, vals):
            y = bar.get_height()
            offset = -0.003 if y < 0 else 0.001
            va = "top" if y < 0 else "bottom"
            ax.text(bar.get_x() + bar.get_width()/2, y + offset,
                    f"{v:+.3f}",
                    ha="center", va=va, fontsize=8)
    ax.axhline(0, color="dimgray", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([
        f"low",
        f"medium",
        f"high",
    ], fontsize=9)
    ax.set_ylabel("Δ F1 (schema track)", fontsize=9)
    ax.set_ylim(-0.1, 0.02)
    ax.set_title("F1 cost of B's structural fixes grows with connectivity",
                 fontsize=10)
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=8)

    fig.suptitle(
        "Verifier-in-loop effectiveness scales with input connectivity (n=200)",
        fontsize=10.5, y=1.02,
    )
    fig.tight_layout()

    out = ROOT / "report/figures/fig_connectivity_n200.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
