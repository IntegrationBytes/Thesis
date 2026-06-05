"""Decoupling figure: ΔSHACL vs ΔF1 (B - A) across stratum.

The thesis finding: structural conformance and content fidelity are
INDEPENDENT axes. ΔSHACL moves +66 to +69pp while ΔF1 moves −0.008 to
−0.001. The decoupling ratio (ΔSHACL / |ΔF1|) is in the tens of
thousands.

Output: report/figures/fig_decoupling_n200.png
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Recomputed from n=200 stratified eval (gpt-oss-120b, clean).
# dF1 values are the exact unrounded means so the printed decoupling
# ratios are accurate (bar labels round to 3 decimals for display).
# Schema track only (F1 is valid here; ontology track has no F1).
ROWS = [
    # (model, stratum, dF1, dSHACL_pp)
    ("gpt-oss-120b", "general", -0.00767, 69.0),
    ("gpt-oss-120b", "complex", -0.00136, 66.0),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    ax2 = ax.twinx()

    x_labels = [f"{m}\n{st}" for m, st, *_ in ROWS]
    x = np.arange(len(ROWS))
    bar_w = 0.35

    dshacl = [r[3] for r in ROWS]
    df1    = [r[2] for r in ROWS]

    b1 = ax.bar(x - bar_w/2, dshacl, bar_w,
                color="#1f6fb4", edgecolor="black", linewidth=0.4,
                label="ΔSHACL (pp)")
    b2 = ax2.bar(x + bar_w/2, df1, bar_w,
                 color="#c44e2e", edgecolor="black", linewidth=0.4,
                 label="ΔF1")

    # Labels on each bar
    for b, v in zip(b1, dshacl):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5,
                f"+{v:.0f}pp", ha="center", va="bottom", fontsize=8,
                color="#1f6fb4")
    for b, v in zip(b2, df1):
        y = b.get_height()
        offset = -0.008 if y < 0 else 0.003
        va = "top" if y < 0 else "bottom"
        ax2.text(b.get_x() + b.get_width()/2, y + offset,
                 f"{v:+.3f}", ha="center", va=va, fontsize=8,
                 color="#c44e2e")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel("Δ SHACL pass-rate (percentage points)", color="#1f6fb4", fontsize=9)
    ax2.set_ylabel("Δ Triple-level F1", color="#c44e2e", fontsize=9)
    ax.tick_params(axis="y", labelcolor="#1f6fb4")
    ax2.tick_params(axis="y", labelcolor="#c44e2e")
    ax.set_ylim(0, 85)
    ax2.set_ylim(-0.020, 0.005)
    ax2.axhline(0, color="dimgray", lw=0.5)
    ax.set_title("Structural vs content quality: independent axes (n=200, schema track)",
                 fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

    # Decoupling-ratio footer
    ratios = [f"{r[0]} {r[1]}: ΔSH/|ΔF1| = {r[3]/max(abs(r[2]), 0.0001):.0f}×"
              for r in ROWS]
    fig.text(0.5, -0.05,
             "Decoupling ratios:  " + "    ".join(ratios),
             ha="center", fontsize=7.5, color="dimgray")

    fig.tight_layout()
    out = Path("report/figures/fig_decoupling_n200.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
