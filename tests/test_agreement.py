"""Ordinal agreement and the scale audit: hand-worked values and sklearn's kappa."""

from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score

from metaeval.agreement import (
    adjacent_agreement,
    exact_agreement,
    mean_signed_difference,
    quadratic_weighted_kappa,
    scale_audit,
    to_scale,
)

NAN = float("nan")


def test_quadratic_weighted_kappa_by_hand():
    assert quadratic_weighted_kappa([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    # Reversed scores: mean squared difference 8/3 against an expected 2/3 + 2/3.
    assert quadratic_weighted_kappa([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    # Independent scores: observed 0.5, expected 0.25 + 0.25.
    assert quadratic_weighted_kappa([1, 1, 2, 2], [1, 2, 1, 2]) == pytest.approx(0.0)
    # A grader that is always one point above: 1 / (2/3 + 2/3 + 1) of disagreement.
    assert quadratic_weighted_kappa([2, 3, 4], [1, 2, 3]) == pytest.approx(1 - 1 / (4 / 3 + 1))
    assert math.isnan(quadratic_weighted_kappa([2, 2], [2, 2]))
    assert math.isnan(quadratic_weighted_kappa([], []))
    # A constant grader against varying humans is defined and equals zero.
    assert quadratic_weighted_kappa([3, 3, 3, 3], [1, 2, 4, 5]) == pytest.approx(0.0)


def test_quadratic_weighted_kappa_matches_sklearn():
    rng = np.random.default_rng(12)
    for _ in range(100):
        n = int(rng.integers(5, 80))
        low = int(rng.integers(0, 3))
        high = low + int(rng.integers(1, 6))
        human = rng.integers(low, high + 1, size=n)
        grader = np.clip(human + rng.integers(-2, 3, size=n), low, high)
        if len(set(human)) == 1 and len(set(grader)) == 1:
            continue
        expected = cohen_kappa_score(
            grader, human, weights="quadratic", labels=list(range(low, high + 1))
        )
        assert quadratic_weighted_kappa(grader, human) == pytest.approx(expected, abs=1e-12)


def test_simple_agreement_measures():
    grader = np.array([[1, 2, 3], [5, 3, NAN]])
    human = np.array([[1, 3, 5], [4, NAN, 2]])
    # Paired outputs: (1,1) (2,3) (3,5) (5,4).
    assert exact_agreement(grader, human) == pytest.approx(1 / 4)
    assert adjacent_agreement(grader, human) == pytest.approx(3 / 4)
    assert mean_signed_difference(grader, human) == pytest.approx((0 - 1 - 2 + 1) / 4)
    assert adjacent_agreement([1.0, 2.0], [1.5, 3.0], step=0.5) == pytest.approx(0.5)
    for function in (exact_agreement, adjacent_agreement, mean_signed_difference):
        assert math.isnan(function([NAN], [1.0]))
    with pytest.raises(ValueError, match="differ in shape"):
        exact_agreement([1, 2], [1, 2, 3])
    with pytest.raises(ValueError, match="infinite"):
        exact_agreement([1, math.inf], [1, 2])


def test_to_scale_rounds_half_up_and_clips():
    np.testing.assert_array_equal(
        to_scale([0.2, 1.49, 1.5, 2.5, 3.5, 4.51, 7.0], 1, 5), [1, 1, 2, 3, 4, 5, 5]
    )
    np.testing.assert_array_equal(to_scale([1.2, 1.25, 1.8], 1, 2, step=0.5), [1.0, 1.5, 2.0])
    with pytest.raises(ValueError, match="high > low"):
        to_scale([1.0], 3, 3)


def test_scale_audit_by_hand():
    # Ten outputs on a 1-5 scale. The grader avoids the ends of the scale.
    human = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5], dtype=float)
    grader = np.array([2, 3, 3, 3, 3, 3, 3, 4, 4, 3.4])
    audit = scale_audit(grader, human, low=1, high=5)
    assert (audit.n, audit.low, audit.high, audit.step) == (10, 1.0, 5.0, 1.0)
    # On the scale the grader reads 2 3 3 3 3 3 3 4 4 3: equal to the human on outputs
    # 4, 5 and 7; within one point on 0, 2, 3, 4, 5, 6, 7, 8.
    assert audit.exact == pytest.approx(3 / 10)
    assert audit.adjacent == pytest.approx(8 / 10)
    assert audit.mean_signed_difference == pytest.approx((31.4 - 30) / 10)
    assert audit.mean_grader == pytest.approx(3.14)
    assert audit.mean_human == pytest.approx(3.0)
    assert audit.sd_human == pytest.approx(math.sqrt(2.0))
    assert audit.sd_ratio == pytest.approx(float(np.std(grader)) / math.sqrt(2.0))
    assert audit.sd_ratio < 0.5
    assert (audit.min_grader, audit.max_grader, audit.min_human, audit.max_human) == (2, 4, 1, 5)
    assert (audit.distinct_grader, audit.distinct_human) == (4, 5)
    assert audit.distribution == (
        (1.0, 0.0, 0.2),
        (2.0, 0.1, 0.2),
        (3.0, 0.7, 0.2),
        (4.0, 0.2, 0.2),
        (5.0, 0.0, 0.2),
    )
    assert (audit.central_grader, audit.central_human) == (0.7, 0.2)
    # Tied pairs: the grader has six 3s and two 4s, 15 + 1 of 45; the humans 5 of 45.
    assert audit.tie_share_grader == pytest.approx(16 / 45)
    assert audit.tie_share_human == pytest.approx(5 / 45)
    assert audit.off_scale == pytest.approx(1 / 10)
    on_scale = np.array([2, 3, 3, 3, 3, 3, 3, 4, 4, 3])
    assert audit.quadratic_weighted_kappa == pytest.approx(
        cohen_kappa_score(on_scale, human.astype(int), weights="quadratic", labels=[1, 2, 3, 4, 5])
    )
    as_dict = audit.to_dict()
    assert as_dict["distribution"][2] == {"score": 3.0, "grader": 0.7, "human": 0.2}
    assert as_dict["exact"] == audit.exact


def test_scale_audit_defaults_and_edge_cases():
    human = np.array([[2.0, 3.0, 4.0, NAN]])
    grader = np.array([[2.0, 3.0, 5.0, 1.0]])
    audit = scale_audit(grader, human)
    # Bounds come from the human scores; the grader's 5 is clipped to 4 on the scale.
    assert (audit.low, audit.high, audit.n) == (2.0, 4.0, 3)
    assert audit.exact == 1.0
    assert audit.off_scale == pytest.approx(1 / 3)
    even = scale_audit([1, 2, 3, 4], [1, 2, 3, 4], low=1, high=4)
    assert (even.central_grader, even.central_human) == (0.5, 0.5)
    constant = scale_audit([3, 3, 3], [3, 3, 3], low=1, high=5)
    assert math.isnan(constant.sd_ratio) and math.isnan(constant.quadratic_weighted_kappa)
    assert constant.to_dict()["sd_ratio"] is None
    assert math.isnan(scale_audit([3], [3], low=1, high=5).tie_share_grader)
    with pytest.raises(ValueError, match="no output has both"):
        scale_audit([NAN], [1.0])
    with pytest.raises(ValueError, match="high > low"):
        scale_audit([3, 3], [3, 3])
