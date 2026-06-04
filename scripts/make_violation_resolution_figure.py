"""Violation-resolution figure — per-shape A→B counts, both models.

Two side-by-side panels (Gemini | gpt-oss). Each row is one SHACL
shape; bars show cycle-0 violations (light, hollow) and cycle-3
violations (solid). Annotation shows raw counts and the fix rate.

Output: report/figures/fig_violation_resolution.png
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report/figures/fig_violation_resolution.png"
DATA = ROOT / "evaluation/outputs/per_shape_violations.json"

GEMINI_BLUE = "#1f6fb4"
GPTOSS_ORANGE = "#c44e2e"


def short_shape(name: str) -> str:
    if name == "[":
        return "(unnamed)"
    return name.rsplit(":", 1)[-1].rsplit("/", 1)[-1].replace("Shape", "")[:34]


def model_data(d: dict, m_key: str) -> dict[str, tuple[int, int]]:
    a = d.get(f"{m_key}_a_all", {})
    b = d.get(f"{m_key}_b_all", {})
    out: dict[str, tuple[int, int]] = {}
    for sh in set(a) | set(b):
        if sh == "[":
            continue
        out[sh] = (a.get(sh, 0), b.get(sh, 0))
    return out


def draw_panel(ax, m_key: str, color: str, title: str) -> None:
    d = json.load(open(DATA))["schema"]
    md = model_data(d, m_key)
    # Filter: keep shapes with A >= 1 (real cycle-0 violations)
    items = [(sh, a, b) for sh, (a, b) in md.items() if a >= 1]
    items.sort(key=lambda r: r[1], reverse=True)

    labels = [short_shape(sh) for sh, _, _ in items]
    a_vals = [a for _, a, _ in items]
    b_vals = [b for _, _, b in items]

    y = np.arange(len(items))
    h = 0.66

    # A: hollow rectangle behind
    ax.barh(y, a_vals, h, facecolor=color, alpha=0.22,
            edgecolor=color, lw=1.0, label="cycle-0 (A)")
    # B: solid foreground (clipped to A so B <= A always visible)
    ax.barh(y, b_vals, h, color=color, edgecolor="black", lw=0.4,
            label="cycle-3 (B)")

    # Annotation: "A→B  (fix%)" at the right end of the A bar
    xmax = max(a_vals + [1]) * 1.02
    for i, (a, b) in enumerate(zip(a_vals, b_vals)):
        fix = 100 * (1 - b / a) if a else 0
        txt = f"{a}→{b}  ({fix:.0f}%)"
        ax.text(a + xmax * 0.015, y[i], txt, va="center",
                fontsize=8.6, color="#374151", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, xmax * 1.18)
    ax.set_xlabel("SHACL violations across n=200 vignettes", fontsize=9.5)
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.3, ls=":")
    ax.set_title(title, fontsize=10.5, loc="left", pad=8, color=color,
                 fontweight="bold")
    ax.legend(loc="lower right", fontsize=8.2, framealpha=0.92)


def main() -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.6))

    # Note: per-shape data is keyed by the historical "gemini" label but
    # is now sourced from outputs/chr/... (gpt-oss-120b data per current
    # codebase layout — see per_shape_violation_analysis.py).
    draw_panel(ax, "gemini", GPTOSS_ORANGE,
               "gpt-oss-120b · SHACL violations · cycle 0 (A) vs cycle 3 (B)")

    fig.tight_layout(pad=1.2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
