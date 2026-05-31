"""Flagship decoupling figure — per-case scatter + connectivity bars.

Left panel (per-clinical-case):
  F1(A) vs F1(B) for all 200 clinical cases, both models. Diagonal y=x
  marks "no F1 change." Points cluster around the diagonal even when
  SHACL went 0→1, demonstrating the decoupling at the case level.

Right panel (connectivity groups):
  Mean ΔSHACL (large +%) and ΔF1 (near-zero) per ground-truth-KG
  entity group, Gemini schema track. Shows the decoupling holds
  across case complexity.

Output: report/figures/fig_decoupling_n200.png
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report/figures/fig_decoupling_n200.png"

GEMINI_BLUE = "#1f6fb4"
GPTOSS_ORANGE = "#c44e2e"
GREY = "#9ca3af"


def load_pairs(eval_path: Path) -> list[tuple[str, float, float, bool, bool]]:
    """Returns (vignette, f1_a, f1_b, shacl_a, shacl_b) per vignette."""
    d = json.load(open(eval_path))
    out = []
    for v in d["per_vignette"]:
        a, b = v["a"], v["b"]
        if not (a["output_present"] and b["output_present"]):
            continue
        out.append((v["vignette"], a["f1"], b["f1"],
                    bool(a["shacl_conforms"]), bool(b["shacl_conforms"])))
    return out


def main() -> None:
    gemini = load_pairs(ROOT / "evaluation/outputs/chr/schema/full/eval.json")
    gptoss = load_pairs(ROOT / "evaluation/outputs_freemodel/chr/schema/full/eval.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.0),
                                    gridspec_kw={"width_ratios": [1.05, 1.0]})

    # ===== LEFT: per-vignette F1(A) vs F1(B) scatter =====
    ax1.plot([0, 1], [0, 1], "--", color=GREY, lw=1.0, zorder=1,
             label="$F_1(B) = F_1(A)$  (no content change)")

    # Plot in 4 buckets: model × (SHACL fixed vs unchanged)
    def split(pairs):
        fixed = [(a, b) for _, a, b, sa, sb in pairs if sb and not sa]
        unchanged = [(a, b) for _, a, b, sa, sb in pairs if sa == sb]
        return fixed, unchanged

    g_fix, g_un = split(gemini)
    o_fix, o_un = split(gptoss)

    sz = 24
    if g_fix:
        ax1.scatter(*zip(*g_fix), s=sz, c=GEMINI_BLUE, alpha=0.55,
                    edgecolors="none", label=f"Gemini · SHACL 0→1 (n={len(g_fix)})",
                    zorder=3)
    if g_un:
        ax1.scatter(*zip(*g_un), s=sz, facecolors="none", edgecolors=GEMINI_BLUE,
                    lw=0.9, label=f"Gemini · SHACL unchanged (n={len(g_un)})",
                    zorder=3)
    if o_fix:
        ax1.scatter(*zip(*o_fix), s=sz, c=GPTOSS_ORANGE, alpha=0.55,
                    edgecolors="none", label=f"gpt-oss · SHACL 0→1 (n={len(o_fix)})",
                    zorder=2)
    if o_un:
        ax1.scatter(*zip(*o_un), s=sz, facecolors="none", edgecolors=GPTOSS_ORANGE,
                    lw=0.9, label=f"gpt-oss · SHACL unchanged (n={len(o_un)})",
                    zorder=2)

    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(0, 1.0)
    ax1.set_aspect("equal")
    ax1.set_xlabel("$F_1$  ·  System A (zero-shot)", fontsize=10)
    ax1.set_ylabel("$F_1$  ·  System B (SHACL retry)", fontsize=10)
    ax1.set_title("(a)  Per-clinical-case triple-level F$_1$ (n=200 × 2 models)",
                  fontsize=11, loc="left", pad=8)
    ax1.legend(fontsize=8, loc="lower right", framealpha=0.92, frameon=True)
    ax1.grid(True, alpha=0.25, linestyle=":")

    # Annotation: mean ΔF1 numbers, big and visible
    mean_d_gem = sum(b - a for _, a, b, *_ in gemini) / len(gemini)
    mean_d_gpt = sum(b - a for _, a, b, *_ in gptoss) / len(gptoss)
    ann = (f"mean ΔF$_1$:  Gemini {mean_d_gem:+.3f}    gpt-oss {mean_d_gpt:+.3f}\n"
           "→ points cluster on the diagonal regardless of SHACL outcome")
    ax1.text(0.02, 0.96, ann, transform=ax1.transAxes, fontsize=8.8,
             ha="left", va="top",
             bbox=dict(boxstyle="round,pad=0.35", fc="#fef3c7", ec="#a16207",
                       lw=0.7, alpha=0.95))

    # ===== RIGHT: connectivity tertile bars (Gemini schema) =====
    buckets = json.load(open(ROOT / "evaluation/outputs/connectivity_buckets.json"))["buckets"]
    labels = ["low\n(≈36)", "medium\n(≈70)", "high\n(≈108)"]
    keys = ["low", "medium", "high"]

    d_shacl = []
    d_f1 = []
    for k in keys:
        s = buckets[k]["schema"]
        a = s["gemini_a"]
        b = s["gemini_b"]
        d_shacl.append(100 * (b["shacl_pass"] / b["n"] - a["shacl_pass"] / a["n"]))
        d_f1.append(b["f1_mean"] - a["f1_mean"])

    x = np.arange(3)
    w = 0.4

    ax2b = ax2.twinx()
    b1 = ax2.bar(x - w/2, d_shacl, w, color=GEMINI_BLUE,
                  edgecolor="black", lw=0.5,
                  label="Δ SHACL pass-rate")
    b2 = ax2b.bar(x + w/2, d_f1, w, color=GPTOSS_ORANGE,
                   edgecolor="black", lw=0.5,
                   label="Δ Triple-level F$_1$")

    for bar, val in zip(b1, d_shacl):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 1.2,
                  f"+{val:.1f}%", ha="center", va="bottom",
                  fontsize=9, color=GEMINI_BLUE, fontweight="bold")
    for bar, val in zip(b2, d_f1):
        ax2b.text(bar.get_x() + bar.get_width()/2,
                   val + (-0.005 if val < 0 else 0.003),
                   f"{val:+.3f}", ha="center",
                   va=("top" if val < 0 else "bottom"),
                   fontsize=9, color=GPTOSS_ORANGE, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9.5)
    ax2.set_xlabel("Ground-truth KG entity group (mean entity count)", fontsize=10)
    ax2.set_ylabel("Δ SHACL pass-rate (%)",
                    color=GEMINI_BLUE, fontsize=10)
    ax2b.set_ylabel("Δ Triple-level F$_1$",
                     color=GPTOSS_ORANGE, fontsize=10)
    ax2.set_ylim(0, 110)
    ax2b.set_ylim(-0.12, 0.04)
    ax2b.axhline(0, color=GREY, lw=0.6, linestyle=":")
    ax2.tick_params(axis="y", labelcolor=GEMINI_BLUE)
    ax2b.tick_params(axis="y", labelcolor=GPTOSS_ORANGE)
    ax2.set_title("(b)  Decoupling scales with complexity (Gemini · schema)",
                  fontsize=11, loc="left", pad=8)

    # Joint legend for the right panel
    handles = [b1, b2]
    ax2.legend([b1, b2], ["Δ SHACL pass-rate (%)", "Δ Triple-level F$_1$"],
                loc="lower right", fontsize=8.5, framealpha=0.92)

    fig.tight_layout(pad=1.2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")
    print(f"Gemini mean ΔF1 = {mean_d_gem:+.4f}")
    print(f"gpt-oss mean ΔF1 = {mean_d_gpt:+.4f}")


if __name__ == "__main__":
    main()
