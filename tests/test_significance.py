"""Williams test: a published worked example, scipy's t distribution and simulation."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from metaeval.coefficients import pearson
from metaeval.significance import student_t_sf, williams, williams_test


def test_student_t_tail_matches_scipy():
    for df in (0.5, 1, 2, 2.5, 3, 5, 13, 30, 97, 288, 1000, 50_000):
        # With many degrees of freedom the log-gamma terms cancel and cost some digits.
        tolerance = 1e-10 if df <= 1000 else 1e-8
        for t in np.linspace(-12.0, 12.0, 97):
            assert student_t_sf(float(t), df) == pytest.approx(stats.t.sf(t, df), rel=tolerance)
    assert student_t_sf(math.inf, 5) == 0.0
    assert student_t_sf(-math.inf, 5) == 1.0
    assert math.isnan(student_t_sf(math.nan, 5))
    assert math.isnan(student_t_sf(1.0, 0))


def test_williams_published_worked_example():
    # Diedenhofen and Musch (2015), "cocor: A Comprehensive Solution for the Statistical
    # Comparison of Correlations", PLOS ONE: r.jk = .1038, r.jh = .3213, r.kh = .0257,
    # n = 291 gives, for williams1959, t = -2.7743, df = 288, p-value = 0.0059 (two-sided).
    result = williams_test(0.1038, 0.3213, 0.0257, 291)
    assert result.t == pytest.approx(-2.7743, abs=5e-5)
    assert result.df == 288
    assert result.p_value == pytest.approx(0.0059, abs=5e-5)
    assert result.alternative == "two-sided"
    # One-sided in the direction of the data: the other correlation is the larger one.
    reverse = williams_test(0.3213, 0.1038, 0.0257, 291, alternative="greater")
    assert reverse.t == pytest.approx(2.7743, abs=5e-5)
    assert reverse.p_value == pytest.approx(result.p_value / 2)
    against = williams_test(0.1038, 0.3213, 0.0257, 291, alternative="greater")
    assert against.p_value == pytest.approx(1 - result.p_value / 2)


def test_williams_keeps_its_level_under_the_null():
    # Two metrics that are equally noisy copies of the human score: the rejection rate
    # at the 5 % level should be close to 5 %.
    rng = np.random.default_rng(17)
    trials, n, rejections = 4000, 30, 0
    for _ in range(trials):
        human = rng.normal(size=n)
        shared = rng.normal(size=n)
        a = human + 0.6 * shared + rng.normal(size=n)
        b = human + 0.6 * shared + rng.normal(size=n)
        r = np.corrcoef([human, a, b])
        rejections += williams_test(r[0, 1], r[0, 2], r[1, 2], n).p_value < 0.05
    assert 0.035 < rejections / trials < 0.065


def test_williams_has_power_when_one_metric_is_better():
    rng = np.random.default_rng(18)
    trials, n, rejections = 500, 40, 0
    for _ in range(trials):
        human = rng.normal(size=n)
        a = human + 0.3 * rng.normal(size=n)
        b = human + 1.5 * rng.normal(size=n)
        r = np.corrcoef([human, a, b])
        rejections += (
            williams_test(r[0, 1], r[0, 2], r[1, 2], n, alternative="greater").p_value < 0.05
        )
    assert rejections / trials > 0.9


def test_williams_undefined_cases():
    assert math.isnan(williams_test(0.5, 0.4, 0.3, 3).p_value)
    assert math.isnan(williams_test(math.nan, 0.4, 0.3, 30).t)
    # Identical metrics: the variance term vanishes.
    assert math.isnan(williams_test(1.0, 1.0, 1.0, 30).t)
    with pytest.raises(ValueError, match="alternative"):
        williams_test(0.5, 0.4, 0.3, 30, alternative="less")


def test_williams_on_score_arrays():
    rng = np.random.default_rng(19)
    human = rng.normal(size=(12, 20))
    a = human + rng.normal(size=(12, 20))
    b = human + 2.0 * rng.normal(size=(12, 20))
    b[0, :5] = np.nan
    system = williams(a, b, human, grouping="system")
    complete = ~np.isnan(b)
    means = [np.array([s[i, complete[i]].mean() for i in range(12)]) for s in (a, b, human)]
    expected = williams_test(
        pearson(means[0], means[2]), pearson(means[1], means[2]), pearson(means[0], means[1]), 12
    )
    assert (system.n, system.df, system.grouping) == (12, 9, "system")
    assert system.t == pytest.approx(expected.t)
    assert system.p_value == pytest.approx(expected.p_value)
    pooled = williams(a, b, human, grouping="pooled", alternative="greater")
    assert (pooled.n, pooled.grouping, pooled.alternative) == (12 * 20 - 5, "pooled", "greater")
    assert pooled.r_a == pytest.approx(pearson(a[complete], human[complete]))
    assert pooled.r_ab == pytest.approx(pearson(a[complete], b[complete]))
    assert pooled.to_dict()["t"] == pytest.approx(pooled.t)
    assert williams(a[:3, :1], a[:3, :1], human[:3, :1], grouping="system").to_dict()["t"] is None
    with pytest.raises(ValueError, match="'system' or 'pooled'"):
        williams(a, b, human, grouping="by_item")
