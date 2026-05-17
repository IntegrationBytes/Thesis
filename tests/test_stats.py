"""Unit tests for pipeline/stats_core.py scipy-backed helpers.

The numerics themselves are scipy's — these tests verify that our thin
wrappers (1) handle empty / degenerate inputs without crashing, (2) seed
the bootstrap RNG so repeated calls return byte-identical CIs, and (3)
agree with hand-checked reference values at simple n's.
"""
from __future__ import annotations

import math

from pipeline.stats_core import (
    _bootstrap_ci,
    _clopper_pearson,
    _is_present,
    _mean,
    _sign_test_p,
)


# ---------------------------------------------------------------------------
# _mean — trivial but guards the empty-list path.
# ---------------------------------------------------------------------------
class TestMean:
    def test_basic(self):
        assert _mean([1.0, 2.0, 3.0]) == 2.0

    def test_empty_is_nan(self):
        assert math.isnan(_mean([]))


# ---------------------------------------------------------------------------
# _bootstrap_ci — seeded, so repeated calls on the same input match exactly.
# ---------------------------------------------------------------------------
class TestBootstrapCi:
    def test_degenerate_single_sample(self):
        """scipy.bootstrap requires ≥2 samples — we fall back to a degenerate CI."""
        lo, hi = _bootstrap_ci([0.42])
        assert lo == 0.42
        assert hi == 0.42

    def test_empty_is_nan(self):
        lo, hi = _bootstrap_ci([])
        assert math.isnan(lo) and math.isnan(hi)

    def test_constant_input_has_zero_width(self):
        """All samples identical → bootstrap CI collapses to a point."""
        lo, hi = _bootstrap_ci([0.5, 0.5, 0.5, 0.5, 0.5])
        assert lo == 0.5
        assert hi == 0.5

    def test_seeded_output_is_reproducible(self):
        """Two back-to-back calls on the same input must match byte-for-byte."""
        xs = [0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6]
        assert _bootstrap_ci(xs) == _bootstrap_ci(xs)

    def test_ci_brackets_the_mean(self):
        xs = [0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6]
        mean = sum(xs) / len(xs)
        lo, hi = _bootstrap_ci(xs)
        assert lo <= mean <= hi


# ---------------------------------------------------------------------------
# _sign_test_p — exact binomial under H0: p = 0.5.
# ---------------------------------------------------------------------------
class TestSignTestP:
    def test_all_ties_returns_one(self):
        """Every paired difference is zero — no evidence of effect."""
        assert _sign_test_p([0, 0, 0, 0]) == 1.0

    def test_empty_returns_one(self):
        assert _sign_test_p([]) == 1.0

    def test_all_positive_small_n(self):
        """n=5 positives, 0 negatives. P(X=5 or X=0 | n=5, p=0.5) = 2/32."""
        p = _sign_test_p([1, 1, 1, 1, 1])
        assert p == 0.0625

    def test_balanced_is_one(self):
        """2 pos, 2 neg — two-sided p=1.0 under H0: p=0.5."""
        p = _sign_test_p([1, -1, 1, -1])
        assert p == 1.0

    def test_ties_are_dropped(self):
        """Ties (zeros) drop out before counting pos vs neg."""
        p_with_ties = _sign_test_p([1, 1, 1, 1, 1, 0, 0, 0])
        p_without = _sign_test_p([1, 1, 1, 1, 1])
        assert p_with_ties == p_without


# ---------------------------------------------------------------------------
# _is_present — distinguishes "system did not run" from "system ran and
# produced an empty graph". The eval pipeline writes an explicit flag;
# legacy data falls back to a heuristic.
# ---------------------------------------------------------------------------
class TestIsPresent:
    def test_explicit_flag_true(self):
        assert _is_present({"a": {"output_present": True}}, "a") is True

    def test_explicit_flag_false(self):
        assert _is_present({"a": {"output_present": False}}, "a") is False

    def test_legacy_with_shacl_result_is_present(self):
        """Legacy eval.json (no output_present): shacl_conforms set means present."""
        assert _is_present({"a": {"shacl_conforms": True, "total_triples": 0}}, "a") is True
        assert _is_present({"a": {"shacl_conforms": False, "total_triples": 5}}, "a") is True

    def test_legacy_with_no_signals_is_absent(self):
        """Legacy: no shacl result and no triples — system did not run."""
        assert _is_present({"a": {"shacl_conforms": None, "total_triples": 0}}, "a") is False

    def test_missing_system_is_absent(self):
        assert _is_present({}, "a") is False


# ---------------------------------------------------------------------------
# _clopper_pearson — exact binomial CI via scipy.stats.binomtest.
# ---------------------------------------------------------------------------
class TestClopperPearson:
    def test_zero_n_is_nan(self):
        lo, hi = _clopper_pearson(0, 0)
        assert math.isnan(lo) and math.isnan(hi)

    def test_zero_successes_lower_bound_is_zero(self):
        """k=0/n=8: lower bound is exactly 0, upper < 1."""
        lo, hi = _clopper_pearson(0, 8)
        assert lo == 0.0
        assert 0.0 < hi < 1.0

    def test_all_successes_upper_bound_is_one(self):
        """k=8/n=8: upper bound is exactly 1, lower > 0."""
        lo, hi = _clopper_pearson(8, 8)
        assert hi == 1.0
        assert 0.0 < lo < 1.0

    def test_half_successes_brackets_half(self):
        lo, hi = _clopper_pearson(4, 8)
        assert lo < 0.5 < hi

    def test_ci_widens_with_smaller_n(self):
        """Same point estimate (1/2 and 4/8 both = 0.5), smaller n → wider CI."""
        lo1, hi1 = _clopper_pearson(1, 2)
        lo2, hi2 = _clopper_pearson(4, 8)
        assert (hi1 - lo1) > (hi2 - lo2)
