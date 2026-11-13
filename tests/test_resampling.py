"""Bootstrap intervals, the paired bootstrap and the permutation test."""

from __future__ import annotations

import math

import numpy as np
import pytest

from metaeval.grouping import GROUPINGS, correlate
from metaeval.resampling import (
    RESAMPLE,
    bootstrap,
    paired_bootstrap,
    percentile_interval,
    permutation_test,
    resample_statistic,
)


def scores(seed: int, shape=(8, 25), noise=1.0):
    rng = np.random.default_rng(seed)
    quality = rng.normal(size=(shape[0], 1)) + rng.normal(size=shape)
    human = np.clip(np.rint(3 + quality), 1, 5)
    metric = quality + noise * rng.normal(size=shape)
    return metric, human


def test_resample_statistic_draws_systems_then_items_with_one_generator():
    array = np.arange(12.0).reshape(3, 4)
    seen = []
    resample_statistic(lambda a: seen.append(a.copy()) or 0.0, [array], n_resamples=3, seed=5)
    rng = np.random.default_rng(5)
    for sample in seen:
        rows = rng.integers(0, 3, size=3)
        columns = rng.integers(0, 4, size=4)
        np.testing.assert_array_equal(sample, array[rows][:, columns])

    seen.clear()
    resample_statistic(
        lambda a: seen.append(a.copy()) or 0.0, [array], resample="items", n_resamples=2, seed=5
    )
    rng = np.random.default_rng(5)
    for sample in seen:
        np.testing.assert_array_equal(sample, array[:, rng.integers(0, 4, size=4)])

    seen.clear()
    resample_statistic(
        lambda a: seen.append(a.copy()) or 0.0, [array], resample="systems", n_resamples=2, seed=5
    )
    rng = np.random.default_rng(5)
    for sample in seen:
        np.testing.assert_array_equal(sample, array[rng.integers(0, 3, size=3)])


def test_resample_statistic_resamples_all_arrays_alike_and_stacks():
    first = np.arange(12.0).reshape(3, 4)
    stacked = np.stack([first, first + 100.0])
    values = resample_statistic(
        lambda a, b: float(np.abs(b[0] - a).max() + np.abs(b[1] - a - 100.0).max()),
        [first, stacked],
        n_resamples=20,
        seed=1,
    )
    np.testing.assert_array_equal(values, np.zeros(20))
    with pytest.raises(ValueError, match="unknown resample"):
        resample_statistic(lambda a: 0.0, [first], resample="rows")


def test_percentile_interval():
    values = np.arange(1.0, 101.0)
    low, high, undefined = percentile_interval(values, 0.9)
    assert (low, high, undefined) == (pytest.approx(5.95), pytest.approx(95.05), 0)
    low, high, undefined = percentile_interval(np.array([1.0, math.nan, 3.0]), 0.5)
    assert (low, high, undefined) == (pytest.approx(1.5), pytest.approx(2.5), 1)
    low, high, undefined = percentile_interval(np.array([math.nan, math.nan]))
    assert math.isnan(low) and math.isnan(high) and undefined == 2
    with pytest.raises(ValueError, match="confidence"):
        percentile_interval(values, 1.0)


@pytest.mark.parametrize("resample", list(RESAMPLE))
@pytest.mark.parametrize("grouping", list(GROUPINGS))
def test_bootstrap_is_deterministic_under_a_seed(resample, grouping):
    metric, human = scores(1)
    arguments = dict(coefficient="kendall_b", grouping=grouping, resample=resample, n_resamples=60)
    first = bootstrap(metric, human, seed=3, **arguments)
    again = bootstrap(metric, human, seed=3, **arguments)
    other = bootstrap(metric, human, seed=4, **arguments)
    assert first == again
    assert (first.low, first.high) != (other.low, other.high)
    assert first.value == correlate(metric, human, coefficient="kendall_b", grouping=grouping).value
    assert first.low <= first.value <= first.high
    assert (first.resample, first.n_resamples, first.seed) == (resample, 60, 3)
    assert first.confidence == 0.95
    assert first.to_dict()["low"] == first.low


def test_bootstrap_matches_a_plain_reimplementation():
    metric, human = scores(2, shape=(5, 12))
    result = bootstrap(
        metric, human, coefficient="spearman", grouping="by_item", n_resamples=40, seed=9
    )
    rng = np.random.default_rng(9)
    values = []
    for _ in range(40):
        rows = rng.integers(0, 5, size=5)
        columns = rng.integers(0, 12, size=12)
        m, h = metric[rows][:, columns], human[rows][:, columns]
        values.append(correlate(m, h, coefficient="spearman", grouping="by_item").value)
    defined = np.array([value for value in values if not math.isnan(value)])
    assert result.n_undefined == 40 - defined.size
    assert (result.low, result.high) == pytest.approx(tuple(np.quantile(defined, [0.025, 0.975])))


def test_bootstrap_counts_undefined_resamples():
    # Two systems: half of the system resamples draw the same system twice, which
    # leaves nothing to correlate at system level.
    metric = np.array([[0.1, 0.2, 0.3], [0.5, 0.6, 0.9]])
    human = np.array([[1.0, 2.0, 2.0], [3.0, 3.0, 4.0]])
    result = bootstrap(
        metric, human, coefficient="pearson", grouping="system", resample="systems", n_resamples=200
    )
    assert 60 < result.n_undefined < 140
    assert (result.low, result.high) == (1.0, 1.0)
    constant = bootstrap(
        np.ones((2, 3)), human, coefficient="pearson", grouping="system", n_resamples=5
    )
    assert math.isnan(constant.value) and math.isnan(constant.low)
    assert constant.n_undefined == 5
    assert constant.to_dict()["value"] is None


def test_bootstrap_interval_covers_the_true_correlation():
    # One system, items resampled: the classical bootstrap of a Pearson correlation.
    # Bivariate normal with correlation 0.6; a 90 % percentile interval should cover
    # the true value in roughly 90 % of the simulated data sets.
    rng = np.random.default_rng(23)
    trials, covered = 200, 0
    for trial in range(trials):
        x = rng.normal(size=(1, 80))
        y = 0.6 * x + 0.8 * rng.normal(size=(1, 80))
        interval = bootstrap(
            x,
            y,
            coefficient="pearson",
            grouping="pooled",
            resample="items",
            n_resamples=200,
            confidence=0.9,
            seed=trial,
        )
        covered += interval.low <= 0.6 <= interval.high
    assert 0.82 <= covered / trials <= 0.96


def test_more_items_give_a_narrower_interval():
    arguments = dict(coefficient="kendall_b", grouping="pooled", resample="items", n_resamples=150)
    small = bootstrap(*scores(3, shape=(6, 15)), **arguments)
    large = bootstrap(*scores(3, shape=(6, 240)), **arguments)
    assert large.high - large.low < 0.6 * (small.high - small.low)


def test_paired_bootstrap():
    metric, human = scores(5, noise=0.5)
    rng = np.random.default_rng(6)
    worse = metric + 4.0 * rng.normal(size=metric.shape)
    arguments = dict(coefficient="kendall_b", grouping="by_item", n_resamples=200, seed=2)
    result = paired_bootstrap(metric, worse, human, **arguments)
    assert (
        result.value_a
        == correlate(metric, human, coefficient="kendall_b", grouping="by_item").value
    )
    assert result.difference == pytest.approx(result.value_a - result.value_b)
    assert result.difference > 0 and result.low > 0
    assert result.p_value == pytest.approx(2 / 201)
    # The same draws with the metrics exchanged mirror the result.
    mirrored = paired_bootstrap(worse, metric, human, **arguments)
    assert mirrored.difference == pytest.approx(-result.difference)
    assert (mirrored.low, mirrored.high) == pytest.approx((-result.high, -result.low))
    assert mirrored.p_value == pytest.approx(result.p_value)
    assert result == paired_bootstrap(metric, worse, human, **arguments)
    # A metric against itself: no difference in any resample.
    same = paired_bootstrap(metric, metric, human, **arguments)
    assert (same.difference, same.low, same.high, same.p_value) == (0.0, 0.0, 0.0, 1.0)
    assert same.to_dict()["p_value"] == 1.0


def test_paired_bootstrap_with_nothing_defined():
    human = np.array([[1.0, 2.0], [2.0, 3.0]])
    result = paired_bootstrap(
        np.ones((2, 2)), human, human, coefficient="pearson", grouping="pooled", n_resamples=5
    )
    assert math.isnan(result.difference) and math.isnan(result.p_value)
    assert result.n_undefined == 5


def test_permutation_test_detects_a_better_metric():
    metric, human = scores(7, noise=0.4)
    rng = np.random.default_rng(8)
    worse = 50.0 + 20.0 * (metric + 3.0 * rng.normal(size=metric.shape))
    arguments = dict(coefficient="pearson", grouping="pooled", n_resamples=300, seed=1)
    result = permutation_test(metric, worse, human, **arguments)
    assert result.difference > 0
    assert result.p_value == pytest.approx(1 / 301)
    assert result == permutation_test(metric, worse, human, **arguments)
    one_sided = permutation_test(metric, worse, human, alternative="greater", **arguments)
    assert one_sided.p_value == pytest.approx(1 / 301)
    reverse = permutation_test(worse, metric, human, alternative="greater", **arguments)
    assert reverse.p_value > 0.99
    same = permutation_test(metric, metric, human, **arguments)
    assert (same.difference, same.p_value) == (0.0, 1.0)
    assert result.to_dict()["resample"] == "both"


@pytest.mark.parametrize("resample", list(RESAMPLE))
def test_permutation_test_keeps_its_level_under_the_null(resample):
    # Two metrics that are equally noisy readings of the same quality.
    rejections, trials = 0, 120
    for trial in range(trials):
        rng = np.random.default_rng(1000 + trial)
        quality = rng.normal(size=(6, 12))
        human = quality + 0.5 * rng.normal(size=(6, 12))
        a = quality + rng.normal(size=(6, 12))
        b = 3.0 * (quality + rng.normal(size=(6, 12))) + 10.0
        result = permutation_test(
            a,
            b,
            human,
            coefficient="pearson",
            grouping="pooled",
            resample=resample,
            n_resamples=99,
            seed=trial,
        )
        rejections += result.p_value <= 0.05
    assert rejections / trials < 0.13


def test_permutation_test_undefined():
    human = np.array([[1.0, 2.0], [2.0, 3.0]])
    result = permutation_test(
        np.ones((2, 2)),
        np.ones((2, 2)),
        human,
        coefficient="pearson",
        grouping="pooled",
        n_resamples=5,
    )
    assert math.isnan(result.difference) and math.isnan(result.p_value)
    assert result.n_undefined == 5


def test_invalid_arguments():
    metric, human = scores(1)
    with pytest.raises(ValueError, match="unknown coefficient"):
        bootstrap(metric, human, coefficient="tau", grouping="system")
    with pytest.raises(ValueError, match="unknown grouping"):
        bootstrap(metric, human, coefficient="pearson", grouping="segment")
    with pytest.raises(ValueError, match="unknown resample"):
        bootstrap(metric, human, coefficient="pearson", grouping="system", resample="rows")
    with pytest.raises(ValueError, match="at least 1"):
        paired_bootstrap(
            metric, metric, human, coefficient="pearson", grouping="system", n_resamples=0
        )
    with pytest.raises(ValueError, match="alternative"):
        permutation_test(
            metric, metric, human, coefficient="pearson", grouping="system", alternative="less"
        )
    with pytest.raises(ValueError, match="differ in shape"):
        paired_bootstrap(metric, metric[:, :3], human, coefficient="pearson", grouping="system")
