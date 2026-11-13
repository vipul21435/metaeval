"""Tie calibration: hand-worked cases and a brute-force search over thresholds."""

from __future__ import annotations

import math

import numpy as np
import pytest

from metaeval.coefficients import CALIBRATED, calibrate_ties
from metaeval.grouping import GROUPINGS, STATISTICS, correlate
from metaeval.resampling import bootstrap


def brute_force(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Best threshold by trying every candidate on every pair, group by group."""
    groups = []
    for row_x, row_y in zip(x, y, strict=True):
        keep = ~np.isnan(row_x) & ~np.isnan(row_y)
        if keep.sum() >= 2:
            groups.append((row_x[keep], row_y[keep]))
    candidates = {0.0}
    for gx, _ in groups:
        candidates |= {abs(a - b) for i, a in enumerate(gx) for b in gx[i + 1 :]}

    def accuracy(epsilon: float) -> float:
        per_group = []
        for gx, gy in groups:
            right = total = 0
            for i in range(len(gx)):
                for j in range(i + 1, len(gx)):
                    total += 1
                    metric_tied = abs(gx[i] - gx[j]) <= epsilon
                    human_tied = gy[i] == gy[j]
                    if metric_tied or human_tied:
                        right += metric_tied and human_tied
                    else:
                        right += (gx[i] - gx[j]) * (gy[i] - gy[j]) > 0
            per_group.append(right / total)
        return float(np.mean(per_group))

    scored = sorted((round(-accuracy(epsilon), 12), epsilon) for epsilon in candidates)
    return scored[0][1], -scored[0][0], accuracy(0.0)


def test_calibration_turns_a_near_tie_into_a_tie():
    # The humans tie the first two outputs and the metric separates them by 0.5.
    # Without calibration 5 of 6 pairs are right; with threshold 0.5 all 6 are.
    epsilon, accuracy, plain, groups, pairs = calibrate_ties([[1.0, 1.5, 5.0, 9.0]], [[1, 1, 2, 3]])
    assert (epsilon, accuracy, groups, pairs) == (0.5, 1.0, 1, 6)
    assert plain == pytest.approx(5 / 6)


def test_calibration_weighs_gained_ties_against_lost_orderings():
    # Threshold 1 makes the two human ties right (+2) and loses the concordant pair
    # (1, 2), whose metric scores also differ by 1 (-1): 5 of 6. Larger thresholds lose more.
    epsilon, accuracy, plain, _, _ = calibrate_ties([[1.0, 2.0, 3.0, 4.0]], [[1, 1, 2, 2]])
    assert (epsilon, accuracy) == (1.0, pytest.approx(5 / 6))
    assert plain == pytest.approx(4 / 6)


def test_calibration_keeps_zero_when_ties_do_not_help():
    epsilon, accuracy, plain, _, _ = calibrate_ties([[1.0, 2.0, 3.0, 4.0]], [[1, 2, 3, 4]])
    assert (epsilon, accuracy, plain) == (0.0, 1.0, 1.0)
    # A metric that already ties what the humans tie keeps its exact ties at zero.
    epsilon, accuracy, plain, _, _ = calibrate_ties([[1.0, 1.0, 3.0]], [[2, 2, 5]])
    assert (epsilon, accuracy, plain) == (0.0, 1.0, 1.0)


def test_one_threshold_serves_all_groups():
    # Group 0 would like threshold 1 (ties the humans tie), group 1 has a concordant
    # pair at distance 1 that it would lose. Each group weighs one half.
    x = np.array([[1.0, 2.0, np.nan], [1.0, 2.0, 9.0]])
    y = np.array([[3.0, 3.0, np.nan], [1.0, 2.0, 3.0]])
    epsilon, accuracy, plain, groups, pairs = calibrate_ties(x, y)
    assert (groups, pairs) == (2, 4)
    assert plain == pytest.approx(0.5 * 0 + 0.5 * 1)
    # Threshold 1: group 0 is right (1/1), group 1 keeps 2 of 3 pairs.
    assert (epsilon, accuracy) == (1.0, pytest.approx(0.5 * 1 + 0.5 * 2 / 3))


def test_calibration_matches_brute_force():
    rng = np.random.default_rng(3)
    for _ in range(120):
        groups, n = int(rng.integers(1, 5)), int(rng.integers(2, 9))
        x = np.round(rng.normal(size=(groups, n)), 1)
        y = rng.integers(1, 4, size=(groups, n)).astype(float)
        x[rng.random((groups, n)) < 0.15] = np.nan
        y[rng.random((groups, n)) < 0.1] = np.nan
        epsilon, accuracy, plain, _, pairs = calibrate_ties(x, y)
        if pairs == 0:
            assert math.isnan(epsilon) and math.isnan(accuracy)
            continue
        expected_epsilon, expected_accuracy, expected_plain = brute_force(x, y)
        assert accuracy == pytest.approx(expected_accuracy, abs=1e-12)
        assert plain == pytest.approx(expected_plain, abs=1e-12)
        assert epsilon == expected_epsilon
        assert accuracy >= plain - 1e-12


def test_calibrated_accuracy_through_correlate_and_bootstrap():
    assert STATISTICS[-1] == CALIBRATED == "pairwise_accuracy_calibrated"
    rng = np.random.default_rng(5)
    quality = rng.normal(size=(6, 14))
    human = np.clip(np.rint(3 + quality), 2, 4)
    metric = quality + 0.3 * rng.normal(size=(6, 14))
    for grouping in GROUPINGS:
        plain = correlate(metric, human, coefficient="pairwise_accuracy", grouping=grouping)
        result = correlate(metric, human, coefficient=CALIBRATED, grouping=grouping)
        assert result.value >= plain.value - 1e-12
        assert (result.n_points, result.n_groups) == (plain.n_points, plain.n_groups)
        assert result.n_undefined == 0
        assert result.epsilon is not None and result.epsilon >= 0
        assert result.to_dict()["epsilon"] == result.epsilon
        assert "epsilon" not in plain.to_dict()
    # Heavily tied human scores: a continuous metric gains from calibration.
    by_item = correlate(metric, human, coefficient=CALIBRATED, grouping="by_item")
    assert by_item.epsilon > 0
    assert (
        by_item.value
        > correlate(metric, human, coefficient="pairwise_accuracy", grouping="by_item").value
    )
    interval = bootstrap(metric, human, coefficient=CALIBRATED, grouping="by_item", n_resamples=40)
    assert interval.value == by_item.value
    assert interval.low <= interval.value <= interval.high
    empty = correlate(
        np.full((2, 2), np.nan), human[:2, :2], coefficient=CALIBRATED, grouping="pooled"
    )
    assert math.isnan(empty.value) and empty.epsilon is None


def test_calibration_refuses_too_many_pairs(monkeypatch):
    from metaeval import coefficients

    monkeypatch.setattr(coefficients, "_PAIR_LIMIT", 10)
    with pytest.raises(ValueError, match="too many pairs"):
        calibrate_ties(np.zeros((1, 6)), np.zeros((1, 6)))
    with pytest.raises(ValueError, match="two-dimensional"):
        calibrate_ties(np.zeros(6), np.zeros(6))
