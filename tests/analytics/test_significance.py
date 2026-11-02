"""Tests for analytics.significance — hand-computed expected values."""

from __future__ import annotations

import pytest

from llmdiff.analytics.significance import (
    benjamini_hochberg,
    bonferroni,
    cohens_d,
    correct_pvalues,
    holm,
    paired_t_test,
    sign_test,
    wilcoxon_signed_rank,
)


def test_sign_test_all_wins_small_n():
    # 3 wins out of 3 non-zero diffs: exact two-sided p = 2 * (1/8) = 0.25
    assert sign_test([0, 0, 0], [1, 2, 3]) == pytest.approx(0.25)


def test_sign_test_no_difference():
    assert sign_test([1, 2, 3], [1, 2, 3]) == 1.0


def test_sign_test_unequal_length_raises():
    with pytest.raises(ValueError):
        sign_test([1, 2], [1])


def test_wilcoxon_all_positive():
    assert wilcoxon_signed_rank([0.0] * 10, [1.0] * 10) < 0.01


def test_wilcoxon_no_effect():
    a = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.55]
    assert wilcoxon_signed_rank(a, list(reversed(a))) > 0.5


def test_paired_t_test_constant_shift():
    assert paired_t_test([0.50] * 20, [0.55] * 20) == 0.0


def test_paired_t_test_no_effect():
    a = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.55, 0.45, 0.65]
    assert paired_t_test(a, list(a)) == 1.0


def test_cohens_d():
    assert cohens_d([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert cohens_d([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]) > 0


def test_bonferroni():
    assert bonferroni([0.01, 0.02, 0.03]) == [0.03, 0.06, 0.09]
    assert bonferroni([0.5]) == [0.5]


def test_holm_monotonic():
    adj = holm([0.01, 0.02, 0.03, 0.5])
    assert adj[0] == pytest.approx(0.04)
    assert adj[0] <= adj[1] <= adj[2] <= adj[3]


def test_benjamini_hochberg():
    adj = benjamini_hochberg([0.01, 0.02, 0.03, 0.5])
    # smallest p (0.01) is rank 1 of 4 -> 0.01 * 4 / 1 = 0.04
    assert adj[0] == pytest.approx(0.04)
    assert adj[3] == pytest.approx(0.5)


def test_correct_pvalues_dispatch():
    assert correct_pvalues([0.01, 0.02], "bonferroni") == [0.02, 0.04]
    with pytest.raises(ValueError):
        correct_pvalues([0.01], "nope")
