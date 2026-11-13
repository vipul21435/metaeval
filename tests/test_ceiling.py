"""Human ceiling by leave-one-annotator-out."""

from __future__ import annotations

import math

import numpy as np
import pytest

from metaeval.ceiling import human_ceiling, leave_one_out_means, metric_against_rest
from metaeval.grouping import GROUPINGS, correlate

NAN = float("nan")


def panel(seed: int = 0, annotators: int = 4, shape=(5, 9)) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    quality = rng.normal(size=shape)
    ratings = np.stack(
        [
            np.clip(np.rint(3 + quality + 0.7 * rng.normal(size=shape)), 1, 5)
            for _ in range(annotators)
        ]
    )
    metric = quality + rng.normal(size=shape)
    return ratings, metric


def test_leave_one_out_means_by_hand():
    ratings = np.array([[[1.0, 2.0]], [[3.0, NAN]], [[5.0, 4.0]]])
    rest = leave_one_out_means(ratings)
    np.testing.assert_array_equal(rest, [[[4.0, 4.0]], [[3.0, 3.0]], [[2.0, 2.0]]])
    lonely = leave_one_out_means(np.array([[[1.0, NAN]], [[NAN, NAN]]]))
    np.testing.assert_array_equal(lonely, [[[NAN, NAN]], [[1.0, NAN]]])


@pytest.mark.parametrize("grouping", list(GROUPINGS))
def test_ceiling_is_each_annotator_against_the_mean_of_the_others(grouping):
    ratings, metric = panel()
    ceiling = human_ceiling(ratings, coefficient="kendall_b", grouping=grouping)
    expected = []
    against = []
    for index in range(4):
        others = np.delete(ratings, index, axis=0).mean(axis=0)
        expected.append(
            correlate(ratings[index], others, coefficient="kendall_b", grouping=grouping).value
        )
        against.append(correlate(metric, others, coefficient="kendall_b", grouping=grouping).value)
    assert ceiling.per_annotator == pytest.approx(tuple(expected))
    assert ceiling.value == pytest.approx(np.mean(expected))
    assert (ceiling.coefficient, ceiling.grouping) == ("kendall_b", grouping)
    assert metric_against_rest(
        metric, ratings, coefficient="kendall_b", grouping=grouping
    ) == pytest.approx(np.mean(against))
    assert ceiling.to_dict()["per_annotator"] == pytest.approx(expected)


def test_identical_annotators_reach_one_and_noise_lowers_the_ceiling():
    ratings, _ = panel(1)
    same = np.stack([ratings[0]] * 3)
    assert human_ceiling(same, coefficient="spearman", grouping="pooled").value == pytest.approx(
        1.0
    )
    noisy = human_ceiling(ratings, coefficient="spearman", grouping="pooled").value
    assert 0.2 < noisy < 0.95


def test_undefined_annotators_are_left_out():
    ratings, _ = panel(2, annotators=3)
    ratings[2] = 3.0
    ceiling = human_ceiling(ratings, coefficient="pearson", grouping="pooled")
    assert math.isnan(ceiling.per_annotator[2])
    assert ceiling.value == pytest.approx(np.mean(ceiling.per_annotator[:2]))
    assert ceiling.to_dict()["per_annotator"][2] is None
    flat = human_ceiling(np.full((2, 2, 3), 2.0), coefficient="pearson", grouping="pooled")
    assert math.isnan(flat.value) and flat.to_dict()["value"] is None


def test_invalid_arguments():
    ratings, metric = panel()
    with pytest.raises(ValueError, match="annotators, systems, items"):
        human_ceiling(ratings[0], coefficient="pearson", grouping="pooled")
    with pytest.raises(ValueError, match="at least two annotators"):
        human_ceiling(ratings[:1], coefficient="pearson", grouping="pooled")
    with pytest.raises(ValueError, match="unknown coefficient"):
        human_ceiling(ratings, coefficient="tau", grouping="pooled")
    with pytest.raises(ValueError, match="unknown grouping"):
        human_ceiling(ratings, coefficient="pearson", grouping="segment")
    with pytest.raises(ValueError, match="differ in their"):
        metric_against_rest(metric[:, :3], ratings, coefficient="pearson", grouping="pooled")
