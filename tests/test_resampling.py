"""Bootstrap intervals, the paired bootstrap and the permutation test."""

from __future__ import annotations

import math

import numpy as np
import pytest

from metaeval.grouping import GROUPINGS, correlate
from metaeval.resampling import (
    PERMUTE,
    PERMUTE_BY_GROUPING,
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
    arguments = dict(coefficient="pearson", grouping="pooled", n_resamples=200, seed=1)
    result = permutation_test(metric, worse, human, **arguments)
    assert result.difference > 0
    # Eight systems allow 2^8 patterns of whole systems, two of which reproduce the
    # observed difference, so p stays a little above the smallest possible 1/201.
    assert 1 / 201 <= result.p_value < 0.05
    assert result.unit == "systems, items"
    assert result == permutation_test(metric, worse, human, **arguments)
    # Twenty-five items allow so many patterns that no permutation reaches the difference.
    by_items = permutation_test(metric, worse, human, unit="items", **arguments)
    assert by_items.p_value == pytest.approx(1 / 201)
    one_sided = permutation_test(
        metric, worse, human, unit="items", alternative="greater", **arguments
    )
    assert one_sided.p_value == pytest.approx(1 / 201)
    reverse = permutation_test(worse, metric, human, alternative="greater", **arguments)
    assert reverse.p_value > 0.99
    same = permutation_test(metric, metric, human, **arguments)
    assert (same.difference, same.p_value) == (0.0, 1.0)
    assert result.to_dict()["unit"] == "systems, items"


def test_permutation_unit_follows_the_grouping():
    metric, human = scores(9, shape=(6, 10), noise=0.5)
    rng = np.random.default_rng(10)
    other = metric + rng.normal(size=metric.shape)
    arguments = dict(coefficient="pearson", n_resamples=60, seed=3)
    assert PERMUTE_BY_GROUPING == {
        "system": ("systems",),
        "by_item": ("systems",),
        "by_system": ("items",),
        "pooled": ("systems", "items"),
    }
    for grouping, units in PERMUTE_BY_GROUPING.items():
        automatic = permutation_test(metric, other, human, grouping=grouping, **arguments)
        single = [
            permutation_test(metric, other, human, grouping=grouping, unit=unit, **arguments)
            for unit in units
        ]
        assert automatic.unit == ", ".join(units)
        # With several units the larger p value is reported.
        assert automatic.p_value == max(result.p_value for result in single)
        assert automatic.n_resamples == 60
    by_outputs = permutation_test(
        metric, other, human, grouping="system", unit="outputs", **arguments
    )
    assert by_outputs.unit == "outputs"
    assert set(PERMUTE) == {"systems", "items", "outputs"}


def test_permutation_draws_whole_systems_items_or_outputs(monkeypatch):
    # Record what reaches the statistic: with whole systems exchanged every row of the
    # first array is either metric a's row or metric b's row, and so on.
    from metaeval import resampling

    a = np.arange(12.0).reshape(3, 4)
    b = a + 100.0
    human = np.arange(12.0).reshape(3, 4) % 5
    seen = []

    def spy(x, h, coefficient, grouping, undefined):
        seen.append(x.copy())
        return 0.0

    monkeypatch.setattr(resampling, "statistic", spy)
    monkeypatch.setattr(resampling, "_standardize", lambda scores: scores)
    for unit in ("systems", "items", "outputs"):
        seen.clear()
        permutation_test(
            a, b, human, coefficient="pearson", grouping="pooled", unit=unit, n_resamples=20
        )
        from_b = [sample >= 100.0 for sample in seen[2:]]
        assert any(mask.any() and not mask.all() for mask in from_b)
        for mask in from_b:
            if unit == "systems":
                assert (mask.all(axis=1) | ~mask.any(axis=1)).all()
            elif unit == "items":
                assert (mask.all(axis=0) | ~mask.any(axis=0)).all()
        if unit == "outputs":
            assert any(
                not (mask.all(axis=1) | ~mask.any(axis=1)).all()
                and not (mask.all(axis=0) | ~mask.any(axis=0)).all()
                for mask in from_b
            )
        # The two arrays of a permutation are complementary.
        for first, second in zip(seen[2::2], seen[3::2], strict=True):
            np.testing.assert_array_equal(first + second, a + b)


def equally_good_metrics(rng: np.random.Generator, shape=(10, 8), per_system=1.0):
    """Two metrics with the same noise; each has its own error for every system."""
    quality = rng.normal(size=(shape[0], 1)) + rng.normal(size=shape)
    human = quality + 0.5 * rng.normal(size=shape)
    metrics = [
        quality + rng.normal(size=shape) + per_system * rng.normal(size=(shape[0], 1))
        for _ in range(2)
    ]
    return metrics[0], metrics[1], human


def rejection_rate(unit: str, grouping: str, per_system: float, trials: int = 60) -> float:
    rejections = 0
    for trial in range(trials):
        a, b, human = equally_good_metrics(
            np.random.default_rng(500 + trial), per_system=per_system
        )
        result = permutation_test(
            a,
            b,
            human,
            coefficient="pearson",
            grouping=grouping,
            unit=unit,
            n_resamples=59,
            seed=trial,
        )
        rejections += result.p_value <= 0.05
    return rejections / trials


def test_exchanging_outputs_rejects_too_often_when_metrics_err_per_system():
    # Two equally good metrics, each with its own error per system. At system level a
    # test at the 5 % level should reject about one time in twenty. Exchanging single
    # outputs averages the two metrics within every system, so the permuted differences
    # are far too small and it rejects many times more often; exchanging whole systems,
    # the default at this level, keeps its level.
    assert rejection_rate("outputs", "system", per_system=1.0) > 0.25
    assert rejection_rate("auto", "system", per_system=1.0) < 0.12
    assert rejection_rate("auto", "by_item", per_system=1.0) < 0.12


def test_exchanging_outputs_is_fine_without_errors_per_system():
    assert rejection_rate("outputs", "system", per_system=0.0) < 0.12
    assert rejection_rate("auto", "system", per_system=0.0) < 0.12


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
    assert result.n_undefined == 10  # five permutations for each of the two units


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
    with pytest.raises(ValueError, match="unknown unit"):
        permutation_test(
            metric, metric, human, coefficient="pearson", grouping="system", unit="both"
        )
    with pytest.raises(ValueError, match="differ in shape"):
        paired_bootstrap(metric, metric[:, :3], human, coefficient="pearson", grouping="system")
