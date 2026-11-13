"""Groupings: a closed-form grid where the four levels disagree, and edge cases."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from scipy import stats

from metaeval.coefficients import COEFFICIENTS
from metaeval.grouping import GROUPINGS, Correlation, _merge_near_ties, correlate

# Three systems (rows) by four items (columns). With i the system and j the item,
# human = i + j + 1 and metric = i - j + 4: the metric follows the system effect and
# runs against the item effect.
HUMAN = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]], dtype=float)
METRIC = np.array([[4, 3, 2, 1], [5, 4, 3, 2], [6, 5, 4, 3]], dtype=float)


def value(metric, human, coefficient, grouping, **kwargs) -> float:
    return correlate(metric, human, coefficient=coefficient, grouping=grouping, **kwargs).value


@pytest.mark.parametrize("coefficient", sorted(COEFFICIENTS))
def test_grid_system_level_is_perfect(coefficient):
    # Both sets of system means are 2.5, 3.5, 4.5.
    result = correlate(METRIC, HUMAN, coefficient=coefficient, grouping="system")
    assert result.value == pytest.approx(1.0)
    assert (result.n_points, result.n_groups, result.n_undefined) == (3, 1, 0)


@pytest.mark.parametrize("coefficient", sorted(COEFFICIENTS))
def test_grid_by_item_is_perfect(coefficient):
    # Within an item both scores rise by one per system.
    result = correlate(METRIC, HUMAN, coefficient=coefficient, grouping="by_item")
    assert result.value == pytest.approx(1.0)
    assert (result.n_points, result.n_groups, result.n_undefined) == (12, 4, 0)


@pytest.mark.parametrize("coefficient", sorted(COEFFICIENTS))
def test_grid_by_system_is_reversed(coefficient):
    # Within a system the human score rises and the metric falls with the item.
    result = correlate(METRIC, HUMAN, coefficient=coefficient, grouping="by_system")
    expected = 0.0 if coefficient == "pairwise_accuracy" else -1.0
    assert result.value == pytest.approx(expected)
    assert (result.n_points, result.n_groups, result.n_undefined) == (12, 3, 0)


def test_grid_pooled_closed_form():
    # Pearson: cov(i - j, i + j) = var(i) - var(j) = 2/3 - 5/4 and both variances are
    # var(i) + var(j) = 23/12, so r = -7/23.
    assert value(METRIC, HUMAN, "pearson", "pooled") == pytest.approx(-7 / 23)
    # Kendall: for two outputs the product of the differences is di^2 - dj^2. Counting
    # the 66 pairs by (|di|, |dj|) gives C = 18, D = 32, 8 pairs tied in the metric
    # only (di = dj), 8 tied in the human score only (di = -dj) and none tied in both.
    assert value(METRIC, HUMAN, "kendall_b", "pooled") == pytest.approx(-14 / 58)
    assert value(METRIC, HUMAN, "kendall_c", "pooled") == pytest.approx(2 * -14 * 6 / (144 * 5))
    assert value(METRIC, HUMAN, "pairwise_accuracy", "pooled") == pytest.approx(18 / 66)
    result = correlate(METRIC, HUMAN, coefficient="kendall_b", grouping="pooled")
    assert (result.n_points, result.n_groups, result.n_undefined) == (12, 1, 0)


def test_by_item_is_by_system_of_the_transpose():
    rng = np.random.default_rng(2)
    metric = rng.integers(0, 4, size=(5, 7)).astype(float)
    human = rng.integers(1, 6, size=(5, 7)).astype(float)
    for coefficient in COEFFICIENTS:
        by_item = correlate(metric, human, coefficient=coefficient, grouping="by_item")
        by_system = correlate(metric.T, human.T, coefficient=coefficient, grouping="by_system")
        assert by_item.n_groups == by_system.n_groups
        assert by_item.n_undefined == by_system.n_undefined
        if math.isnan(by_item.value):
            assert math.isnan(by_system.value)
        else:
            assert by_item.value == pytest.approx(by_system.value)


def test_grouped_value_is_the_mean_of_per_group_coefficients():
    rng = np.random.default_rng(4)
    metric = rng.normal(size=(6, 9))
    human = rng.integers(1, 6, size=(6, 9)).astype(float)
    for name, function in COEFFICIENTS.items():
        per_item = [function(metric[:, j], human[:, j]) for j in range(9)]
        per_system = [function(metric[i], human[i]) for i in range(6)]
        assert value(metric, human, name, "by_item") == pytest.approx(np.nanmean(per_item))
        assert value(metric, human, name, "by_system") == pytest.approx(np.nanmean(per_system))
        assert value(metric, human, name, "pooled") == pytest.approx(
            function(metric.ravel(), human.ravel())
        )
        assert value(metric, human, name, "system") == pytest.approx(
            function(metric.mean(axis=1), human.mean(axis=1))
        )


SCIPY_REFERENCE = {
    "pearson": lambda x, y: stats.pearsonr(x, y).statistic,
    "spearman": lambda x, y: stats.spearmanr(x, y).statistic,
    "kendall_b": lambda x, y: stats.kendalltau(x, y, variant="b").statistic,
    "kendall_c": lambda x, y: stats.kendalltau(x, y, variant="c").statistic,
}


def reference(metric, human, coefficient: str, grouping: str) -> float:
    """The four groupings written out plainly on top of scipy, for comparison."""
    function = SCIPY_REFERENCE[coefficient]
    both = np.isfinite(metric) & np.isfinite(human)

    def one(x, y) -> float:
        if len(x) < 2:
            return math.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(function(x, y))

    if grouping == "pooled":
        return one(metric[both], human[both])
    if grouping == "system":
        rows = [i for i in range(metric.shape[0]) if both[i].any()]
        # Rounded so that equal means stay equal whatever the order of summation.
        x = [round(float(metric[i, both[i]].mean()), 12) for i in rows]
        y = [round(float(human[i, both[i]].mean()), 12) for i in rows]
        return one(x, y)
    if grouping == "by_item":
        metric, human, both = metric.T, human.T, both.T
    per_group = [one(metric[i, both[i]], human[i, both[i]]) for i in range(metric.shape[0])]
    defined = [group for group in per_group if not math.isnan(group)]
    return float(np.mean(defined)) if defined else math.nan


def test_groupings_match_a_scipy_reference_with_missing_scores():
    rng = np.random.default_rng(8)
    compared = 0
    for _ in range(40):
        shape = (int(rng.integers(2, 7)), int(rng.integers(2, 9)))
        metric = np.round(rng.normal(size=shape), 1)
        human = rng.integers(1, 5, size=shape).astype(float)
        metric[rng.random(shape) < 0.15] = np.nan
        human[rng.random(shape) < 0.15] = np.nan
        for coefficient in SCIPY_REFERENCE:
            for grouping in GROUPINGS:
                expected = reference(metric, human, coefficient, grouping)
                ours = value(metric, human, coefficient, grouping)
                # The rounding of the reference's system means moves Pearson a little.
                tolerance = 1e-9 if grouping == "system" else 1e-12
                assert ours == pytest.approx(expected, abs=tolerance, nan_ok=True)
                compared += not math.isnan(expected)
    assert compared > 500


def test_reordering_systems_and_items_changes_nothing():
    rng = np.random.default_rng(6)
    metric = np.round(rng.normal(size=(5, 8)), 1)
    human = rng.integers(1, 6, size=(5, 8)).astype(float)
    rows, columns = rng.permutation(5), rng.permutation(8)
    for coefficient in COEFFICIENTS:
        for grouping in GROUPINGS:
            before = value(metric, human, coefficient, grouping)
            after = value(metric[rows][:, columns], human[rows][:, columns], coefficient, grouping)
            assert before == pytest.approx(after, nan_ok=True)


def test_undefined_group_is_skipped_or_counted_as_zero():
    # Item 0 separates the systems; on item 1 the humans tie all three outputs.
    human = np.array([[1, 2], [2, 2], [3, 2]], dtype=float)
    metric = np.array([[0.1, 0.5], [0.2, 0.4], [0.3, 0.6]])
    skipped = correlate(metric, human, coefficient="kendall_b", grouping="by_item")
    assert skipped.value == pytest.approx(1.0)
    assert (skipped.n_groups, skipped.n_undefined, skipped.undefined) == (2, 1, "skip")
    zero = correlate(metric, human, coefficient="kendall_b", grouping="by_item", undefined="zero")
    assert zero.value == pytest.approx(0.5)
    assert (zero.n_groups, zero.n_undefined, zero.undefined) == (2, 1, "zero")
    # Pairwise accuracy is defined there: the metric ties none of the three tied pairs.
    accuracy = correlate(metric, human, coefficient="pairwise_accuracy", grouping="by_item")
    assert accuracy.value == pytest.approx(0.5)
    assert (accuracy.n_groups, accuracy.n_undefined) == (2, 0)


def test_all_groups_undefined_gives_nan_or_zero():
    human = np.full((3, 4), 2.0)
    metric = np.arange(12, dtype=float).reshape(3, 4)
    skipped = correlate(metric, human, coefficient="spearman", grouping="by_system")
    assert math.isnan(skipped.value)
    assert (skipped.n_groups, skipped.n_undefined) == (3, 3)
    assert value(metric, human, "spearman", "by_system", undefined="zero") == 0.0
    pooled = correlate(metric, human, coefficient="spearman", grouping="pooled", undefined="zero")
    assert math.isnan(pooled.value)
    assert (pooled.n_groups, pooled.n_undefined) == (1, 1)


def test_only_outputs_with_both_scores_are_used():
    nan = float("nan")
    human = np.array([[1, 2, 3, nan], [2, nan, 4, 5], [nan, nan, nan, nan]])
    metric = np.array([[1, 2, 9, 7], [4, 5, nan, 8], [1, 2, 3, 4]], dtype=float)
    # Paired outputs: system 0 on items 0, 1, 2; system 1 on items 0 and 3.
    pooled = correlate(metric, human, coefficient="pearson", grouping="pooled")
    assert pooled.n_points == 5
    assert pooled.value == pytest.approx(COEFFICIENTS["pearson"]([1, 2, 9, 4, 8], [1, 2, 3, 2, 5]))
    # System means over paired outputs only; system 2 has none and drops out.
    system = correlate(metric, human, coefficient="pearson", grouping="system")
    assert system.n_points == 2
    assert system.value == pytest.approx(COEFFICIENTS["pearson"]([4.0, 6.0], [2.0, 3.5]))
    # Only item 0 has two paired outputs; the other items cannot be correlated.
    by_item = correlate(metric, human, coefficient="kendall_b", grouping="by_item")
    assert (by_item.n_groups, by_item.n_points, by_item.n_undefined) == (1, 2, 0)
    assert by_item.value == pytest.approx(1.0)
    by_system = correlate(metric, human, coefficient="kendall_b", grouping="by_system")
    assert (by_system.n_groups, by_system.n_points) == (2, 5)
    assert by_system.value == pytest.approx(1.0)


def test_no_paired_outputs_is_undefined():
    nan = float("nan")
    human = np.array([[1.0, nan], [nan, 2.0]])
    metric = np.array([[nan, 1.0], [2.0, nan]])
    for grouping in GROUPINGS:
        result = correlate(metric, human, coefficient="pairwise_accuracy", grouping=grouping)
        assert math.isnan(result.value)
        assert result.n_points == 0


def test_system_means_that_differ_by_rounding_are_tied():
    # Systems 0 and 1 received the same seven mean ratings in a different order, so
    # their true means are equal, although the float sums can differ in the last bit.
    thirds_a = np.array([10, 15, 12, 11, 10, 10, 15]) / 3.0
    thirds_b = np.array([12, 15, 11, 15, 10, 10, 10]) / 3.0
    human = np.vstack([thirds_a, thirds_b, np.full(7, 1.0)])
    metric = np.vstack([np.full(7, 0.5), np.full(7, 0.5), np.full(7, 0.1)])
    assert value(metric, human, "pairwise_accuracy", "system") == 1.0
    assert value(metric, human, "kendall_b", "system") == pytest.approx(1.0)


def test_merge_near_ties():
    base = 3.952380952380952
    above = np.nextafter(base, 10.0)
    values = np.array([above, 1.0, base, 2.0, np.nextafter(above, 10.0), 2.0 + 1e-9])
    merged = _merge_near_ties(values)
    assert merged[0] == merged[2] == merged[4] == base
    assert merged[1] == 1.0
    # A difference of 1e-9 is far above rounding noise and is kept.
    assert merged[3] == 2.0
    assert merged[5] == 2.0 + 1e-9
    np.testing.assert_array_equal(_merge_near_ties(np.array([0.0, 0.0, 0.0])), np.zeros(3))
    np.testing.assert_array_equal(_merge_near_ties(np.array([5.0])), np.array([5.0]))


def test_correlation_to_dict():
    result = correlate(METRIC, HUMAN, coefficient="pearson", grouping="pooled")
    assert isinstance(result, Correlation)
    assert result.label == GROUPINGS["pooled"]
    assert result.to_dict() == {
        "coefficient": "pearson",
        "grouping": "pooled",
        "value": pytest.approx(-7 / 23),
        "n_points": 12,
        "n_groups": 1,
        "n_undefined": 0,
        "undefined": "skip",
        "label": GROUPINGS["pooled"],
    }
    undefined = correlate(np.ones((2, 2)), HUMAN[:2, :2], coefficient="pearson", grouping="pooled")
    assert undefined.to_dict()["value"] is None


def test_invalid_arguments():
    with pytest.raises(ValueError, match="unknown coefficient"):
        correlate(METRIC, HUMAN, coefficient="tau", grouping="system")
    with pytest.raises(ValueError, match="unknown grouping"):
        correlate(METRIC, HUMAN, coefficient="pearson", grouping="segment")
    with pytest.raises(ValueError, match="undefined must be"):
        correlate(METRIC, HUMAN, coefficient="pearson", grouping="system", undefined="drop")
    with pytest.raises(ValueError, match="two-dimensional"):
        correlate(METRIC[0], HUMAN[0], coefficient="pearson", grouping="system")
    with pytest.raises(ValueError, match="differ in shape"):
        correlate(METRIC, HUMAN[:, :3], coefficient="pearson", grouping="system")
    with pytest.raises(ValueError, match="infinite"):
        correlate(
            METRIC, np.where(HUMAN == 6, np.inf, HUMAN), coefficient="pearson", grouping="system"
        )
    with pytest.raises(TypeError):
        correlate(METRIC, HUMAN)  # coefficient and grouping have no default
