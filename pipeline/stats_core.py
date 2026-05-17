"""Statistical calculation helpers for evaluation results.

The stats reported in the thesis are deliberately simple. At the cell size
of n = 100 (Synthea-derived schema track) the small-sample machinery —
paired bootstrap intervals, exact sign tests, Clopper–Pearson on
proportions — gives interpretable answers without an extra metric.
We intentionally do NOT report Cliff's delta or other rank-based effect
sizes: at this scale the mean delta + its CI + an exact sign-test p-value
already tell the whole story.
"""

from __future__ import annotations
import json
from pathlib import Path
from statistics import median
from typing import Sequence
import numpy as np
from scipy.stats import binomtest, bootstrap

# All bootstrap CIs in the project use this resample count and seed so the
# numbers in the report, the JSON, and the plots agree exactly.
N_BOOTSTRAP = 10_000
RNG_SEED = 20260423


# ---------------------------------------------------------------------------
# Thin wrappers around scipy so call-sites read like English. Each wrapper
# handles the empty / degenerate cases scipy would raise on.
# ---------------------------------------------------------------------------
def _mean(xs: Sequence[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def _bootstrap_ci(
    xs: Sequence[float], *, alpha: float = 0.05, n_boot: int = N_BOOTSTRAP
) -> tuple[float, float]:
    """Percentile bootstrap 95% CI on the mean.

    Uses :func:`scipy.stats.bootstrap` with ``method="percentile"``.
    BCa would be more accurate but the bias correction is unstable at
    the sample sizes we run here, so the percentile interval is the
    honest reporting choice — just remember it underestimates CI
    width at small n.
    """
    if len(xs) < 2:
        # scipy.bootstrap requires at least 2 samples; report a degenerate CI.
        val = float(xs[0]) if xs else float("nan")
        return val, val
    res = bootstrap(
        (np.asarray(xs, dtype=float),),
        np.mean,
        n_resamples=n_boot,
        confidence_level=1 - alpha,
        method="percentile",
        random_state=np.random.default_rng(RNG_SEED),
    )
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def _sign_test_p(paired: Sequence[float]) -> float:
    """Exact two-sided sign test p-value on paired differences.

    Ties are dropped (standard convention). Returns 1.0 if every
    paired difference is a tie. Delegates to
    :func:`scipy.stats.binomtest` under H0: p = 0.5.
    """
    non_zero = [d for d in paired if d != 0]
    n = len(non_zero)
    if n == 0:
        return 1.0
    pos = sum(1 for d in non_zero if d > 0)
    return float(binomtest(pos, n, p=0.5, alternative="two-sided").pvalue)


def _clopper_pearson(k: int, n: int, *, alpha: float = 0.05) -> tuple[float, float]:
    """Exact Clopper–Pearson 95% CI on a binomial proportion.

    Delegates to :meth:`scipy.stats.BinomTestResult.proportion_ci`
    with ``method="exact"``.
    """
    if n == 0:
        return float("nan"), float("nan")
    ci = binomtest(k, n).proportion_ci(confidence_level=1 - alpha, method="exact")
    return float(ci.low), float(ci.high)


# ---------------------------------------------------------------------------
# Evaluation aggregation.
# ---------------------------------------------------------------------------
def _is_present(pv: dict, sys_: str) -> bool:
    """Did the system actually produce output for this vignette?

    Older eval.json files (regenerated post-fix) carry an explicit
    ``output_present`` flag per system. For backward compatibility, fall
    back to "non-zero total_triples or shacl result is not None" — that
    matches the legacy behaviour where missing files silently aggregated
    as zeros.
    """
    sys_block = pv.get(sys_, {})
    if "output_present" in sys_block:
        return bool(sys_block["output_present"])
    # Legacy fallback: a SHACL result of None means the file could not be
    # read; total_triples == 0 might be a real empty graph or a missing one.
    return sys_block.get("shacl_conforms") is not None or sys_block.get("total_triples", 0) > 0


def _collect_present(per_vignette: list[dict], sys_: str, metric: str) -> list[float]:
    """Collect a metric from vignettes where the system actually ran."""
    xs: list[float] = []
    for pv in per_vignette:
        if not _is_present(pv, sys_):
            continue
        v = pv.get(sys_, {}).get(metric)
        if v is None:
            continue
        if isinstance(v, bool):
            xs.append(1.0 if v else 0.0)
        else:
            xs.append(float(v))
    return xs


def _collect_f1_comparable(per_vignette: list[dict], sys_: str) -> list[float]:
    """Collect F1 only for vignettes where exact IRI overlap is meaningful
    AND the system produced output."""
    xs: list[float] = []
    for pv in per_vignette:
        if not pv.get("f1_comparable", True):
            continue
        if not _is_present(pv, sys_):
            continue
        v = pv.get(sys_, {}).get("f1")
        if v is not None:
            xs.append(float(v))
    return xs


def _per_cell_block(track: str, prompt: str, data: dict) -> dict:
    per_v = data["per_vignette"]
    f1_n = sum(1 for pv in per_v if pv.get("f1_comparable", True))
    block: dict = {
        "track": track,
        "prompt": prompt,
        "n": len(per_v),
        "f1_n": f1_n,
        "systems": {},
    }

    for sys_ in ("a", "b", "c"):
        f1s = _collect_f1_comparable(per_v, sys_)
        ocs = _collect_present(per_v, sys_, "oc")
        tns = _collect_present(per_v, sys_, "total_triples")
        # SHACL pass-rate denominator is the count of vignettes where the
        # system actually produced output. Missing runs are excluded —
        # they're a missing-data problem, not a SHACL failure.
        present_count = sum(1 for pv in per_v if _is_present(pv, sys_))
        shacl_pass_count = sum(
            1
            for pv in per_v
            if _is_present(pv, sys_) and pv.get(sys_, {}).get("shacl_conforms") is True
        )
        valids = [
            1.0 if pv.get(sys_, {}).get("valid") else 0.0
            for pv in per_v
            if _is_present(pv, sys_)
        ]

        # PC: collect only vignettes where the system ran AND pc was computable
        pcs = [
            float(pv[sys_]["pc"])
            for pv in per_v
            if _is_present(pv, sys_) and pv.get(sys_, {}).get("pc") is not None
        ]

        f1_mean = _mean(f1s)
        f1_lo, f1_hi = _bootstrap_ci(f1s)
        oc_mean = _mean(ocs)
        pc_mean = _mean(pcs) if pcs else float("nan")
        pc_lo, pc_hi = _bootstrap_ci(pcs) if len(pcs) >= 2 else (float("nan"), float("nan"))
        tn_mean = _mean(tns)
        cp_lo, cp_hi = _clopper_pearson(shacl_pass_count, present_count)

        block["systems"][sys_] = {
            "f1_values": f1s,
            "f1_comparable_count": len(f1s),
            "f1_mean": f1_mean,
            "f1_median": float(median(f1s)) if f1s else float("nan"),
            "f1_min": min(f1s) if f1s else float("nan"),
            "f1_max": max(f1s) if f1s else float("nan"),
            "f1_boot95": [f1_lo, f1_hi],
            "oc_mean": oc_mean,
            "pc_values": pcs,
            "pc_mean": pc_mean,
            "pc_boot95": [pc_lo, pc_hi],
            "total_triples_mean": tn_mean,
            "parseable_count": int(sum(valids)),
            "present_count": present_count,
            "shacl_pass_count": shacl_pass_count,
            "shacl_pass_cp95": [cp_lo, cp_hi],
        }

    # Paired comparisons: B-A, C-A, C-B — aligned on vignette order.
    # Both systems must be present for the same vignette AND the gold must
    # be f1_comparable, otherwise the pair is dropped (not zero-padded).
    block["paired"] = {}
    for (x, y) in (("b", "a"), ("c", "a"), ("c", "b")):
        dxs: list[float] = []
        for pv in per_v:
            if not pv.get("f1_comparable", True):
                continue
            if not (_is_present(pv, x) and _is_present(pv, y)):
                continue
            xf = pv.get(x, {}).get("f1")
            yf = pv.get(y, {}).get("f1")
            if xf is None or yf is None:
                continue
            dxs.append(float(xf) - float(yf))
        if not dxs:
            block["paired"][f"{x}-{y}"] = None
            continue
        lo, hi = _bootstrap_ci(dxs)
        n_pos = sum(1 for d in dxs if d > 0)
        n_neg = sum(1 for d in dxs if d < 0)
        n_tie = sum(1 for d in dxs if d == 0)
        block["paired"][f"{x}-{y}"] = {
            "deltas": dxs,
            "mean_delta": _mean(dxs),
            "ci95": [lo, hi],
            "sign_p": _sign_test_p(dxs),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "n_tie": n_tie,
        }

    return block


